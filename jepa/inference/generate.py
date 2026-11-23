"""Text generation and decoded latent plans for a trained JEPA checkpoint.

**Generation** (default): each block re-encodes the current context, samples a
committed future from the predictor, and generates ``k`` tokens
autoregressively. The decoder sees the last ``sliding_window`` tokens, so
grammar comes from the token context and semantics from the committed future.

**Decoded plans** (``--plan-depth N``): encode the prompt once, plan ``N``
blocks ahead in latent space, and decode each plan into text without
re-encoding anything. Conditions: a π-argmax rollout, ``--samples`` π-sample
rollouts, the μ path (measurement's baseline), and with ``--baseline-model``
the π-argmax rollout of another checkpoint (the P = 1 baseline) decoded by its
own decoder. Prompts are the first ``--num-prompts`` validation chunks long
enough for ``--prompt-length + N * k`` tokens, in a seeded random order (the
rest of each chunk is the reference continuation), plus any ``--prompt``.

Caveats when reading decoded plans:

- The decoder never saw μ hints: it is trained on committed candidates, so the
  μ path is out of distribution for it. The fair blurry comparison is the
  P = 1 model's own decoder (``--baseline-model``).
- Token sampling adds variance. Plans decode greedily by default
  (``--temperature 0``) so differences come from the plan; show sampled
  decodes separately.
- Pick samples honestly: prompts are seeded and fixed. If showing a subset,
  say how it was chosen.

Usage:
    python -m jepa generate --prompt "Once upon a time"
    python -m jepa generate --model runs/jepa --prompt "Once upon a time" \\
        --steps 3 --temperature 0.7
    python -m jepa generate --plan-depth 4 --baseline-model runs/p1 \\
        --json runs/jepa/plans.json --html runs/jepa/plans.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from ..data.tokenizer import eos_token_id, load_tokenizer
from ..evaluation.measure import mu_path
from ..models.commitment import PiArgmax, PiSample, rollout
from ..models.jepa import JepaModel
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from ..runtime import select_device
from ..training.data import load_datasets

DEFAULT_MODEL = str(DEFAULT_RUN_DIR)
DEFAULT_TOKENIZER = str(TOKENIZER_DIR)
DEFAULT_STEPS = 1
DEFAULT_TEMPERATURE = 0.8
DEFAULT_SAMPLES = 3
DEFAULT_NUM_PROMPTS = 8
DEFAULT_PROMPT_LENGTH = 64
PLAN_CONDITIONS = ("pi_argmax", "pi_sample", "mu")


@torch.no_grad()
def generate(
    model: JepaModel,
    tokenizer,
    prompt: str,
    steps: int,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int | None = 50,
    commit_temperature: float = 1.0,
    commit_top_k: int | None = 0,
    device: torch.device,
) -> tuple[str, list[int]]:
    """Generate ``steps * k`` tokens from ``prompt``.

    Each block re-encodes the current context, samples a committed future, and
    generates ``k`` tokens autoregressively. The decoder sees the last
    ``sliding_window`` tokens and the committed future, so grammar comes from
    the token context and semantics from the JEPA prediction.
    """
    encoded = tokenizer([prompt], return_tensors="pt", padding=False)
    context_ids = encoded["input_ids"].to(device)

    if context_ids.numel() == 0:
        raise ValueError("The prompt contains no tokens.")

    max_context = getattr(model.config.encoder, "max_position_embeddings", None)
    if max_context is not None:
        context_ids = context_ids[:, -max_context:]

    k = model.pred_horizon
    eos_id = eos_token_id(tokenizer)
    generated_ids: list[int] = []

    for _ in range(steps):
        enc_ids = context_ids
        if max_context is not None:
            enc_ids = enc_ids[:, -max_context:]

        # Re-encode the current context so the committed future is grounded in
        # the tokens that have actually been generated.
        latents = model.encode(enc_ids)
        z_t = latents[:, -1, :]

        # The predictor reads only the last latent (z_t).
        futures = model.predictor(z_t)
        committed = futures.commit(
            PiSample(temperature=commit_temperature, top_k=commit_top_k)
        ).future

        new_ids = model.decoder.generate(
            z_t=z_t,
            committed=committed,
            context_ids=context_ids,
            k=k,
            temperature=temperature,
            top_k=top_k,
        )
        token_ids = new_ids[0].tolist()

        if eos_id is not None and eos_id in token_ids:
            generated_ids.extend(token_ids[: token_ids.index(eos_id)])
            break

        generated_ids.extend(token_ids)
        context_ids = torch.cat([context_ids, new_ids], dim=1)
        if max_context is not None:
            context_ids = context_ids[:, -max_context:]

    return tokenizer.decode(generated_ids), generated_ids


@torch.no_grad()
def decode_plan(
    model: JepaModel,
    z_t: torch.Tensor,
    path: torch.Tensor,
    context_ids: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> list[list[int]]:
    """Decode a latent path ``(1, depth, D)`` into ``depth`` blocks of ``k`` ids.

    Block ``i`` is decoded with the hint (start = path latent ``i - 1``, or
    ``z_t`` for the first block; committed = path latent ``i``). The token
    context is ``context_ids`` plus every token decoded so far; nothing is
    re-encoded.
    """
    blocks = []
    start = z_t
    for depth in range(path.shape[-2]):
        committed = path[:, depth]
        new_ids = model.decoder.generate(
            z_t=start,
            committed=committed,
            context_ids=context_ids,
            k=model.pred_horizon,
            temperature=temperature,
            top_k=top_k,
        )
        blocks.append(new_ids[0].tolist())
        context_ids = torch.cat([context_ids, new_ids], dim=1)
        start = committed
    return blocks


@torch.no_grad()
def plan_prompt(
    model: JepaModel,
    prompt_ids: list[int],
    *,
    depth: int,
    samples: int = DEFAULT_SAMPLES,
    conditions: tuple[str, ...] = PLAN_CONDITIONS,
    temperature: float = 0.0,
    top_k: int | None = None,
    commit_temperature: float = 1.0,
    commit_top_k: int | None = 0,
    generator_seed: int = 0,
) -> list[dict]:
    """Plan ``depth`` blocks ahead from a prompt and decode every plan.

    The prompt is encoded once; ``z_t`` is its last latent. Each condition in
    ``conditions`` gives a latent path: ``pi_argmax`` (one rollout),
    ``pi_sample`` (``samples`` rollouts, seeded by ``generator_seed``) and
    ``mu`` (measurement's μ baseline). Every path is decoded block by block
    with :func:`decode_plan`, so each record holds ``depth`` blocks of exactly
    ``k`` token ids (an EOS does not stop a plan).

    Returns one ``{"condition", "sample", "token_ids"}`` dict per decoded
    plan, in the order of ``conditions``.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1; got {depth!r}")
    unknown = set(conditions) - set(PLAN_CONDITIONS)
    if unknown:
        raise ValueError(
            f"Unknown plan conditions {sorted(unknown)}; expected {PLAN_CONDITIONS}"
        )
    if not prompt_ids:
        raise ValueError("The prompt contains no tokens.")
    device = next(model.parameters()).device
    context_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    max_context = model.config.encoder.max_position_embeddings
    z_t = model.encode(context_ids[:, -max_context:])[:, -1]

    generator = torch.Generator(device=device)
    generator.manual_seed(generator_seed)
    sampler = PiSample(commit_temperature, commit_top_k, generator=generator)
    paths = []
    for condition in conditions:
        if condition == "pi_argmax":
            paths.append((condition, 0, rollout(model.predictor, z_t, PiArgmax(), depth).future))
        elif condition == "pi_sample":
            paths += [
                (condition, sample, rollout(model.predictor, z_t, sampler, depth).future)
                for sample in range(samples)
            ]
        else:
            paths.append((condition, 0, mu_path(model.predictor, z_t, depth)))

    return [
        {
            "condition": condition,
            "sample": sample,
            "token_ids": decode_plan(
                model, z_t, path, context_ids, temperature=temperature, top_k=top_k
            ),
        }
        for condition, sample, path in paths
    ]


def validation_prompts(
    dataset, *, count: int, prompt_length: int, continuation: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """``(prompt, reference)`` id pairs from a seeded order of validation chunks.

    Takes the first ``count`` chunks, in a ``seed``-ed random order, that hold
    at least ``prompt_length + continuation`` tokens. No other selection.
    """
    generator = torch.Generator().manual_seed(seed)
    prompts = []
    for index in torch.randperm(len(dataset), generator=generator).tolist():
        if len(prompts) == count:
            break
        ids = list(dataset[index]["input_ids"])
        if len(ids) >= prompt_length + continuation:
            prompts.append(
                (ids[:prompt_length], ids[prompt_length : prompt_length + continuation])
            )
    return prompts


def _blocks(ids: list[int], k: int) -> list[list[int]]:
    return [ids[i : i + k] for i in range(0, len(ids), k)]


def plans_html(plans: dict) -> str:
    """A small side-by-side page: one table per prompt, one column per block."""
    context = plans["context"]
    prompts: dict[int, list[dict]] = {}
    for record in plans["records"]:
        prompts.setdefault(record["prompt_index"], []).append(record)
    header = "".join(f"<th>block {i}</th>" for i in range(1, context["depth"] + 1))
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Decoded plans</title>",
        "<style>body{font-family:sans-serif}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:4px;vertical-align:top;"
        "white-space:pre-wrap}</style>",
        f"<p>{html.escape(json.dumps(context))}</p>",
    ]
    for index, records in prompts.items():
        first = records[0]
        rows = []
        if first["reference"] is not None:
            rows.append(("reference", first["reference"]))
        rows += [
            (f"{r['condition']} #{r['sample']}", r["blocks"]) for r in records
        ]
        body = "".join(
            f"<tr><th>{html.escape(label)}</th>"
            + "".join(f"<td>{html.escape(text)}</td>" for text in blocks)
            + "</tr>"
            for label, blocks in rows
        )
        parts.append(
            f"<h2>Prompt {index} ({first['source']})</h2>"
            f"<p>{html.escape(first['prompt'])}</p>"
            f"<table><tr><th>condition</th>{header}</tr>{body}</table>"
        )
    return "\n".join(parts) + "\n"


def _load_model(path, tokenizer, device) -> JepaModel:
    model = JepaModel.from_pretrained(str(path)).to(device).eval()
    vocab_size = model.config.encoder.vocab_size
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but {path} expects "
            f"{vocab_size}. Use the tokenizer that trained this model."
        )
    return model


def run_plans(args, tokenizer, model, device) -> None:
    """``--plan-depth`` mode: decode plans for every prompt and write them."""
    k = model.pred_horizon
    depth = args.plan_depth
    baseline = None
    if args.baseline_model is not None:
        baseline = _load_model(args.baseline_model, tokenizer, device)
        if baseline.pred_horizon != k:
            raise ValueError(
                f"--baseline-model has pred_horizon {baseline.pred_horizon}, "
                f"but {args.model} has {k}; plans would not line up."
            )

    prompts = []  # (source, prompt text, prompt ids, reference ids | None)
    if args.num_prompts > 0:
        _, validation = load_datasets(args.fineweb_dir)
        for prompt_ids, reference in validation_prompts(
            validation, count=args.num_prompts, prompt_length=args.prompt_length,
            continuation=depth * k, seed=args.seed,
        ):
            prompts.append(
                ("validation", tokenizer.decode(prompt_ids), prompt_ids, reference)
            )
    for text in args.prompt:
        ids = tokenizer([text], return_tensors="pt", padding=False)["input_ids"]
        prompts.append(("prompt", text, ids[0].tolist(), None))
    if not prompts:
        raise ValueError("No prompts: pass --prompt or --num-prompts > 0.")

    options = dict(
        depth=depth, temperature=args.temperature, top_k=args.top_k,
        commit_temperature=args.commit_temperature,
        commit_top_k=args.commit_top_k, generator_seed=args.seed,
    )
    records = []
    for index, (source, text, prompt_ids, reference) in enumerate(prompts):
        plans = plan_prompt(model, prompt_ids, samples=args.samples, **options)
        if baseline is not None:
            plans += [
                {**plan, "condition": "baseline"}
                for plan in plan_prompt(
                    baseline, prompt_ids, samples=0, conditions=("pi_argmax",),
                    **options,
                )
            ]
        reference_text = (
            None if reference is None
            else [tokenizer.decode(block) for block in _blocks(reference, k)]
        )
        for plan in plans:
            records.append({
                "prompt_index": index,
                "source": source,
                "prompt": text,
                "reference": reference_text,
                **plan,
                "blocks": [tokenizer.decode(block) for block in plan["token_ids"]],
            })

    plans = {
        "context": {
            "model": str(args.model),
            "baseline_model": None if baseline is None else str(args.baseline_model),
            "depth": depth, "pred_horizon": k, "samples": args.samples,
            "temperature": args.temperature, "top_k": args.top_k,
            "commit_temperature": args.commit_temperature,
            "commit_top_k": args.commit_top_k, "seed": args.seed,
            "prompt_length": args.prompt_length,
            "selection": "first --num-prompts validation chunks long enough, "
                         "in a seeded random order; no other selection",
        },
        "records": records,
    }
    for index, (_, text, _, _) in enumerate(prompts):
        print(f"\n=== prompt {index}: {text!r}")
        rows = [r for r in records if r["prompt_index"] == index]
        if rows[0]["reference"] is not None:
            print(f"{'reference':<14}{' | '.join(rows[0]['reference'])!r}")
        for r in rows:
            label = f"{r['condition']} #{r['sample']}"
            print(f"{label:<14}{' | '.join(r['blocks'])!r}")
    if args.json is not None:
        args.json.write_text(json.dumps(plans, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    if args.html is not None:
        args.html.write_text(plans_html(plans))
        print(f"wrote {args.html}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jepa generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint directory")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer directory trained by `python -m jepa train-tokenizer`",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=f"Token sampling temperature; 0 means greedy (default: "
        f"{DEFAULT_TEMPERATURE}, or 0 with --plan-depth).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff; 0 disables it.",
    )
    parser.add_argument(
        "--commit-temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling a committed future from the "
        "win-probability distribution (pi); 0 commits by π-argmax.",
    )
    parser.add_argument(
        "--commit-top-k",
        type=int,
        default=0,
        help="Top-k cutoff for committed-future sampling; 0 disables it.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Prediction steps")
    parser.add_argument(
        "--prompt", action="append", default=[],
        help="Prompt text (repeatable with --plan-depth)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")

    plan = parser.add_argument_group("decoded plans")
    plan.add_argument(
        "--plan-depth", type=int, default=None,
        help="Decode latent plans this many blocks deep instead of generating.",
    )
    plan.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help="π-sample rollouts per prompt.",
    )
    plan.add_argument(
        "--baseline-model", type=Path, default=None,
        help="Second checkpoint (the P = 1 baseline) whose π-argmax plan is "
        "decoded by its own decoder.",
    )
    plan.add_argument("--fineweb-dir", type=Path, default=FINEWEB_DIR)
    plan.add_argument(
        "--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS,
        help="Seeded validation chunks used as prompts; 0 uses only --prompt.",
    )
    plan.add_argument(
        "--prompt-length", type=int, default=DEFAULT_PROMPT_LENGTH,
        help="Prompt tokens taken from each validation chunk.",
    )
    plan.add_argument("--json", type=Path, default=None, help="Write the records as JSON.")
    plan.add_argument("--html", type=Path, default=None, help="Write a side-by-side page.")
    args = parser.parse_args(argv)

    plan_mode = args.plan_depth is not None
    if plan_mode:
        if args.plan_depth < 1:
            parser.error("--plan-depth must be at least 1")
        if args.samples < 0 or args.num_prompts < 0 or args.prompt_length < 1:
            parser.error(
                "--samples and --num-prompts must be >= 0, --prompt-length >= 1"
            )
        if args.seed is None:
            args.seed = 0
        if args.temperature is None:
            args.temperature = 0.0
    else:
        for flag in ("baseline_model", "json", "html"):
            if getattr(args, flag) is not None:
                parser.error(f"--{flag.replace('_', '-')} needs --plan-depth")
        if len(args.prompt) != 1:
            parser.error("exactly one --prompt is required without --plan-depth")
        if args.steps < 1:
            parser.error("--steps must be at least 1")
        if args.temperature is None:
            args.temperature = DEFAULT_TEMPERATURE

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    model = _load_model(args.model, tokenizer, device)

    k = model.pred_horizon
    print(f"Loaded {args.model} on {device} (k={k} tokens per prediction step)")

    if plan_mode:
        run_plans(args, tokenizer, model, device)
        return

    prompt = args.prompt[0]
    if not prompt.strip():
        raise ValueError("--prompt cannot be empty.")

    steps = args.steps

    generated_text, generated_ids = generate(
        model,
        tokenizer,
        prompt,
        steps,
        temperature=args.temperature,
        top_k=args.top_k,
        commit_temperature=args.commit_temperature,
        commit_top_k=args.commit_top_k,
        device=device,
    )

    print()
    print(prompt + generated_text)
    print(
        f"\n[{len(generated_ids)} tokens generated "
        f"over {steps} prediction step(s)]"
    )


if __name__ == "__main__":
    main()
