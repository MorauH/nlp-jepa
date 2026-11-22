"""Interactive generator for a trained JEPA encoder + predictor + decoder.

Each block re-encodes the current context, samples a committed future from the
predictor, and generates ``k`` tokens autoregressively. The decoder sees the
last ``sliding_window`` tokens, so grammar comes from the token context and
semantics from the committed future.

Usage:
    python -m jepa generate --prompt "Once upon a time"
    python -m jepa generate --model runs/jepa --prompt "Once upon a time" \\
        --steps 3 --temperature 0.7
"""

from __future__ import annotations

import argparse

import torch

from ..data.tokenizer import eos_token_id, load_tokenizer
from ..models.commitment import PiSample
from ..models.jepa import JepaModel
from ..paths import DEFAULT_RUN_DIR, TOKENIZER_DIR
from ..runtime import select_device

DEFAULT_MODEL = str(DEFAULT_RUN_DIR)
DEFAULT_TOKENIZER = str(TOKENIZER_DIR)
DEFAULT_STEPS = 1


@torch.no_grad()
def generate(
    model: JepaModel,
    tokenizer,
    prompt: str,
    steps: int,
    *,
    temperature: float = 0.8,
    top_k: int | None = 50,
    direction_temperature: float = 1.0,
    direction_top_k: int | None = 0,
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
        latents = model.encoder(enc_ids)
        z_t = latents[:, -1, :]

        # The predictor reads only the last latent (z_t).
        futures = model.predictor(z_t)
        committed = futures.commit(
            PiSample(temperature=direction_temperature, top_k=direction_top_k)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jepa generate", description=__doc__
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
        default=0.8,
        help="Sampling temperature; 0 means greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff; 0 disables it.",
    )
    parser.add_argument(
        "--direction-temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling a committed future from the "
        "win-probability head (pi); 0 picks the most likely one.",
    )
    parser.add_argument(
        "--direction-top-k",
        type=int,
        default=0,
        help="Top-k cutoff for committed-future sampling; 0 disables it.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Prediction steps")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be at least 1")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    model = JepaModel.from_pretrained(args.model).to(device).eval()

    vocab_size = model.config.encoder.vocab_size
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but the checkpoint expects "
            f"{vocab_size}. Use the tokenizer that trained this model."
        )

    k = model.pred_horizon
    print(f"Loaded {args.model} on {device} (k={k} tokens per prediction step)")

    prompt = args.prompt
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
        direction_temperature=args.direction_temperature,
        direction_top_k=args.direction_top_k,
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
