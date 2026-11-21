"""Interactive generator for a trained JEPA encoder + predictor + decoder.

Each block re-encodes the current context, samples a latent waypoint from the
predictor, and generates ``k`` tokens autoregressively. The decoder sees the
last ``sliding_window`` tokens, so grammar comes from the token context and
semantics from the JEPA waypoint.

Usage:
    python -m jepa generate
    python -m jepa generate --model runs/jepa --steps 3 --temperature 0.7
"""

from __future__ import annotations

import argparse

import torch

from ..data.tokenizer import eos_token_id, load_tokenizer
from ..models.stack import JepaDecodeModel
from ..paths import DEFAULT_RUN_DIR, TOKENIZER_DIR
from ..runtime import select_device

DEFAULT_MODEL = str(DEFAULT_RUN_DIR)
DEFAULT_TOKENIZER = str(TOKENIZER_DIR)


def ask_int(prompt: str, *, minimum: int = 1) -> int:
    """Ask until the user enters an integer >= ``minimum``."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < minimum:
            print(f"Please enter an integer >= {minimum}.")
            continue
        return value


@torch.no_grad()
def sample_direction(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """Sample one candidate per batch row from ``scores``.

    candidates: (batch, prediction_count, target_dim)
    scores:     (batch, prediction_count) win-probability logits (pi)
    Returns:    (batch, target_dim)
    """
    if temperature <= 0:
        indices = scores.argmax(dim=-1)
    else:
        logits = scores / temperature

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        probabilities = torch.softmax(logits, dim=-1)
        indices = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    rows = torch.arange(candidates.shape[0], device=candidates.device)
    return candidates[rows, indices]


@torch.no_grad()
def generate(
    model: JepaDecodeModel,
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

    Each block re-encodes the current context, samples a latent waypoint, and
    generates ``k`` tokens autoregressively. The decoder sees the last
    ``sliding_window`` tokens and the latent waypoint, so grammar comes from
    the token context and semantics from the JEPA plan.
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

        # Re-encode the current context so the waypoint is grounded in the
        # tokens that have actually been generated.
        latents = model.encoder(enc_ids)
        z_t = latents[:, -1, :]

        # The predictor reads only the last latent (z_t).
        candidates, scores, _ = model.predictor(latents, return_all_timesteps=False)
        z_future = sample_direction(
            candidates,
            scores,
            temperature=direction_temperature,
            top_k=direction_top_k,
        )

        new_ids = model.decoder.generate(
            z_t=z_t,
            z_future=z_future,
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
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps, ...")
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
        help="Temperature for sampling a future direction from the "
        "win-probability head; 0 picks the most likely one.",
    )
    parser.add_argument(
        "--direction-top-k",
        type=int,
        default=0,
        help="Top-k cutoff for direction sampling; 0 disables it.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Prediction steps")
    parser.add_argument("--prompt", default=None, help="Prompt text")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    args = parser.parse_args(argv)

    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    model = JepaDecodeModel.from_pretrained(args.model).to(device).eval()

    vocab_size = model.config.encoder.vocab_size
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer)} tokens but the checkpoint expects "
            f"{vocab_size}. Use the tokenizer that trained this model."
        )

    k = model.pred_horizon
    print(f"Loaded {args.model} on {device} (k={k} tokens per prediction step)")

    prompt = args.prompt
    while prompt is None or not prompt.strip():
        prompt = input("Prompt: ")
        if not prompt.strip():
            print("Prompt cannot be empty.")

    steps = args.steps
    if steps is None:
        steps = ask_int(
            f"Number of prediction steps (1 step = {k} tokens): ", minimum=1
        )

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
