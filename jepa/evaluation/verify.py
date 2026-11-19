"""Collapse and latent-usage checks for a trained JEPA decoder.

Run from the repository root, e.g.::

    python -m jepa verify --model runs/jepa

It loads a validation batch, runs the frozen encoder,
and reports:

* representation health (scale, mean, effective rank, temporal cosine),
* predictor candidate diversity and score-head calibration,
* the `win` histogram,
* a teacher-forced hint ablation: predicted vs shuffled vs null vs raw
  block-end latent, overall and per block offset.

Expected ranges are printed alongside the results.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from ..data.tokenizer import load_tokenizer
from ..models.stack import JepaDecodeModel, build_block_hints, select_waypoints
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR

DEFAULT_MODEL = DEFAULT_RUN_DIR
DEFAULT_TOKENIZER = TOKENIZER_DIR
DEFAULT_FINEWEB_DIR = FINEWEB_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jepa verify", description=__doc__
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--fineweb-dir",
        type=Path,
        default=DEFAULT_FINEWEB_DIR,
        help="Prepared token chunks used for validation (the default).",
    )
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cuda",
        help="cpu or cuda.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_batch(args: argparse.Namespace, tokenizer):
    dataset = load_from_disk(str(args.fineweb_dir))["validation"]
    n = min(args.num_examples, len(dataset))
    sequences = [
        list(dataset[i]["input_ids"])[: args.max_length] for i in range(n)
    ]
    max_len = max(len(sequence) for sequence in sequences)
    labels = torch.tensor(
        [
            sequence + [tokenizer.pad_token_id] * (max_len - len(sequence))
            for sequence in sequences
        ]
    )
    attention_mask = torch.tensor(
        [
            [1] * len(sequence) + [0] * (max_len - len(sequence))
            for sequence in sequences
        ]
    )
    return labels, attention_mask


def fmt(value) -> str:
    return f"{float(value):.4f}"


def flag_std(value: float) -> str:
    if value < 0.1:
        return "COLLAPSE"
    if 0.8 <= value <= 1.1:
        return "OK"
    return "WARN"


def flag_cos(value: float) -> str:
    if value > 0.95:
        return "TOO SMOOTH"
    if 0.5 <= value <= 0.85:
        return "OK"
    return "WARN"


def flag_candidate_cos(value: float) -> str:
    if value > 0.9:
        return "COLLAPSED"
    if value < 0.7:
        return "diverse"
    return "WARN"


def flag_score(value: float, log_p: float) -> str:
    if value < log_p - 0.1:
        return "OK"
    return "UNIFORM"


def flag_win(top_share: float, prediction_count: int) -> str:
    if prediction_count <= 1:
        return "COLLAPSED"
    if top_share > max(0.25, 2.5 / prediction_count):
        return "CONCENTRATED"
    return "OK"


def flag_rank(value: float) -> str:
    if value < 5:
        return "SUSPICIOUS"
    if value < 15:
        return "LOW"
    return "OK"


def flag_delta(value: float) -> str:
    if value < 0.05:
        return "COPYING"
    if value < 0.2:
        return "WARN"
    return "OK"


def retrieval_metrics(query, keys, *, max_pool: int = 4096, seed: int = 0):
    """Top-1/top-5/mean-rank retrieval of the matching future latent."""
    n = query.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n > max_pool:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator)[:max_pool].to(query.device)
        query = query[perm]
        keys = keys[perm]
        n = max_pool

    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)
    similarity = query @ keys.T
    target = torch.arange(n, device=similarity.device)

    top1 = (similarity.argmax(dim=-1) == target).float().mean().item()
    k = min(5, n)
    top5 = (
        (similarity.topk(k, dim=-1).indices == target[:, None])
        .any(dim=-1)
        .float()
        .mean()
        .item()
    )
    ranks = (similarity > similarity[target, target][:, None]).sum(dim=-1) + 1
    return top1, top5, ranks.float().mean().item(), n


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or args.max_length, args.max_length
    )
    model = JepaDecodeModel.from_pretrained(str(args.model)).to(device).eval()
    if len(tokenizer) != model.config.encoder.vocab_size:
        raise ValueError(
            f"{args.model} was trained with vocab_size="
            f"{model.config.encoder.vocab_size}, but {args.tokenizer} has "
            f"{len(tokenizer)} tokens. Pass the matching --tokenizer."
        )

    labels, attention_mask = load_batch(args, tokenizer)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    k = model.pred_horizon
    prediction_count = model.config.predictor.prediction_count
    log_p = math.log(prediction_count)
    window = model.decoder.sliding_window

    print("=" * 78)
    print(f"checkpoint: {args.model}")
    print(
        f"device: {device} | examples: {labels.shape[0]} x {labels.shape[1]} "
        f"| k: {k} | candidates: {prediction_count} | decoder window: {window}"
    )
    print("=" * 78)

    with torch.no_grad():
        latents = model.encoder(labels, attention_mask=attention_mask)
        jepa_out, decoder_out = model(
            labels=labels,
            attention_mask=attention_mask,
            training=True,
            loss_chunk_size=args.loss_chunk_size,
        )

        # ---------------------------------------------------------------
        # Representation health
        # ---------------------------------------------------------------
        z = latents[attention_mask.bool()].float()
        mu = z.mean(dim=0)
        z_centered = z - mu
        std = z_centered.std(dim=0, unbiased=False)
        covariance = (z_centered.T @ z_centered) / max(1, z.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        total = eigenvalues.sum().clamp_min(1e-12)
        effective_rank = torch.exp(
            -(eigenvalues / total * torch.log(eigenvalues / total + 1e-12)).sum()
        )

        seq_len = labels.shape[1]
        horizon_len = seq_len - k
        z_t = latents[:, :horizon_len, :]
        z_future = latents[:, k:, :]
        valid = attention_mask[:, :horizon_len].bool() & attention_mask[:, k:].bool()
        temporal_cos = F.cosine_similarity(
            F.normalize(z_t[valid], dim=-1),
            F.normalize(z_future[valid], dim=-1),
            dim=-1,
        )

        preds, scores = model.predictor(
            latents[:, :-k, :], return_all_timesteps=True
        )
        distances = (
            F.normalize(preds, dim=-1)
            - F.normalize(z_future, dim=-1).unsqueeze(2)
        ).pow(2).sum(-1)
        delta = (preds - latents[:, :-k, :].unsqueeze(2)).norm(dim=-1)
        if prediction_count > 1:
            pairwise = F.cosine_similarity(
                F.normalize(preds, dim=-1).unsqueeze(3),
                F.normalize(preds, dim=-1).unsqueeze(2),
                dim=-1,
            )
            off_diagonal = 1 - torch.eye(prediction_count, device=device)
            candidate_cos = (
                (pairwise[valid] * off_diagonal).sum()
                / (valid.sum() * prediction_count * (prediction_count - 1))
            )
        else:
            candidate_cos = torch.ones((), device=device)

        best_d = distances.min(dim=-1).values[valid].mean()
        score_std = scores.std()
        loss_score = jepa_out["stats"]["loss_score"]

        print("\n--- representation health ---")
        print(
            f"per-dim std:        {fmt(std.mean())}  "
            f"[{flag_std(float(std.mean()))}; healthy ~0.8-1.1]"
        )
        print(
            f"mean vector norm:   {fmt(mu.norm())}  "
            "[lower is better; ~21 with tiny std means mean-dominated]"
        )
        print(
            f"effective rank:     {fmt(effective_rank)} / {z.shape[1]}  "
            f"[{flag_rank(float(effective_rank))}; >15 okay]"
        )
        print(
            f"cos(z_t, z_t+k):    {fmt(temporal_cos.mean())}  "
            f"[{flag_cos(float(temporal_cos.mean()))}; 0.5-0.85 healthy]"
        )

        print("\n--- predictor / score head ---")
        print(
            f"candidate pair cos: {fmt(candidate_cos)}  "
            f"[{flag_candidate_cos(float(candidate_cos))}; <0.7 diverse]"
        )
        print(
            f"delta norm / z:     {fmt(delta.mean() / z_t.norm(dim=-1).mean())}  "
            f"[{flag_delta(float(delta.mean() / z_t.norm(dim=-1).mean()))}; >0.2 meaningful]"
        )
        print(f"best_d:             {fmt(best_d)}  [lower is better; <0.1 easy, >0.3 hard]")
        print(
            f"score logit std:    {fmt(score_std)}  "
            "[>1 discriminative; <0.5 near uniform]"
        )
        print(
            f"loss_score:         {fmt(loss_score)}  vs log(P)={fmt(log_p)}  "
            f"[{flag_score(float(loss_score), log_p)}]"
        )

        # ---------------------------------------------------------------
        # Oracle vs selected waypoint
        # ---------------------------------------------------------------
        selected = select_waypoints(preds, scores)
        best_index = distances.argmin(dim=-1)
        oracle = preds.gather(
            2,
            best_index[..., None, None].expand(-1, -1, 1, preds.shape[-1]),
        ).squeeze(2)

        query_copy = z_t[valid]
        query_selected = selected[valid]
        query_oracle = oracle[valid]
        retrieval_keys = z_future[valid]

        def norm_distance(query):
            return (
                (F.normalize(query, dim=-1) - F.normalize(retrieval_keys, dim=-1))
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

        d_copy = float(norm_distance(query_copy))
        d_selected = float(norm_distance(query_selected))
        d_oracle = float(norm_distance(query_oracle))

        print("\n--- oracle vs selected waypoint ---")
        print(
            f"{'query':12s}{'distance':>10s}{'top1':>10s}"
            f"{'top5':>10s}{'mean_rank':>12s}"
        )
        retrieval_pool = 0
        for name, query in (
            ("copy z_t", query_copy),
            ("selected", query_selected),
            ("oracle best", query_oracle),
        ):
            top1, top5, mean_rank, retrieval_pool = retrieval_metrics(
                query, retrieval_keys, seed=args.seed
            )
            print(
                f"{name:12s}{fmt(norm_distance(query)):>10s}"
                f"{fmt(top1):>10s}{fmt(top5):>10s}{fmt(mean_rank):>12s}"
            )

        print(
            f"retrieval pool: {retrieval_pool} "
            f"(chance top1 ~{1 / max(1, retrieval_pool):.5f})"
        )
        print(f"oracle improvement over copy:   {fmt(d_copy - d_oracle)}")
        print(f"selected improvement over copy: {fmt(d_copy - d_selected)}")

        oracle_good = d_oracle < d_copy - 0.1
        selected_good = d_selected < d_copy - 0.1
        if oracle_good and not selected_good:
            print(
                "SCORER PROBLEM: the candidate set is better than copy, "
                "but the selected waypoint is not."
            )
        elif oracle_good and selected_good:
            print("HINT USEFUL: the selected waypoint is better than copy.")
        elif not oracle_good:
            print(
                "CANDIDATE SET WEAK: the oracle best candidate is not better "
                "than copy; improve the representation/predictor/data."
            )
        else:
            print("INCONCLUSIVE: the oracle/selected differences are small.")
        if d_selected - d_oracle < 0.05 and selected_good:
            print("selection is close to the oracle.")

        # ---------------------------------------------------------------
        # win histogram
        # ---------------------------------------------------------------
        win = jepa_out["stats"]["win"][valid]
        win_counts = torch.bincount(win, minlength=prediction_count)
        win_shares = win_counts.float() / win_counts.sum().clamp_min(1)
        top_share = win_shares.max()
        print("\n--- win histogram (oracle argmin) ---")
        print("counts:", win_counts.tolist())
        print("shares:", [round(float(x), 3) for x in win_shares])
        print(
            f"top share: {fmt(top_share)}  "
            f"[{flag_win(float(top_share), prediction_count)}; uniform ~{1/prediction_count:.3f}]"
        )

        # ---------------------------------------------------------------
        # Hint ablation
        # ---------------------------------------------------------------
        predicted = selected
        z_t_hint, future_predicted = build_block_hints(
            latents, k, predicted_future=predicted
        )
        _, future_real = build_block_hints(latents, k)
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(predicted.shape[0], generator=generator).to(device)
        _, future_shuffled = build_block_hints(
            latents, k, predicted_future=predicted[perm]
        )
        future_null = torch.zeros_like(future_predicted)

        input_ids = labels[:, :-1]
        targets = labels[:, 1:]
        decoder_mask = attention_mask[:, :-1]
        positions = torch.arange(seq_len - 1, device=device)
        offsets = ((positions % k) + 1).unsqueeze(0).expand(labels.shape[0], -1)

        def eval_hint(future, name: str):
            hint = model.decoder._hint(z_t_hint, future)
            hidden = model.decoder._run(input_ids, hint, decoder_mask)
            logits = model.decoder.lm_head(hidden)
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="none",
            ).view(labels.shape[0], -1)
            target_valid = targets != tokenizer.pad_token_id
            values = per_token[target_valid]
            offset_values = offsets[target_valid]
            overall = values.mean()
            per_offset = [
                values[offset_values == o].mean() for o in range(1, k + 1)
            ]
            return overall, per_offset

        results = {
            "predicted": eval_hint(future_predicted, "predicted"),
            "shuffled": eval_hint(future_shuffled, "shuffled"),
            "null": eval_hint(future_null, "null"),
            "real block": eval_hint(future_real, "real block"),
        }

        header = "hint".ljust(12) + "overall".rjust(9) + "".join(
            f"off{o}".rjust(9) for o in range(1, k + 1)
        )
        print("\n--- decoder hint ablation ---")
        print(header)
        for name, (overall, per_offset) in results.items():
            row = name.ljust(12) + fmt(overall).rjust(9) + "".join(
                fmt(value).rjust(9) for value in per_offset
            )
            print(row)

        predicted_overall = results["predicted"][0]
        shuffled_overall = results["shuffled"][0]
        null_overall = results["null"][0]
        real_overall = results["real block"][0]
        predicted_offsets = results["predicted"][1]
        shuffled_offsets = results["shuffled"][1]
        real_offsets = results["real block"][1]

        print("\n--- interpretation ---")
        gap = float(predicted_overall - shuffled_overall)
        if gap < -0.05:
            print(
                f"decoder USES the waypoint: predicted is {abs(gap):.4f} CE "
                "better than shuffled"
            )
        elif abs(gap) < 0.02:
            print(
                "decoder IGNORES the waypoint: predicted and shuffled are "
                "within noise"
            )
        else:
            print(f"weak/inconclusive waypoint effect: gap={gap:.4f}")

        semantic_offsets = [
            o
            for o in range(1, k)
            if predicted_offsets[o - 1] < shuffled_offsets[o - 1] - 0.05
        ]
        if semantic_offsets:
            print(
                "semantic guidance on offsets "
                + ", ".join(str(o) for o in semantic_offsets)
            )
        else:
            print("no clear semantic guidance on offsets 1..k-1")

        if real_offsets[-1] + 0.5 < predicted_offsets[-1]:
            print(
                "raw block-end latent leaks the last token: "
                f"real off{k}={fmt(real_offsets[-1])} vs "
                f"predicted={fmt(predicted_offsets[-1])}"
            )

        print("\n--- loss context ---")
        print(f"JEPA loss:    {fmt(jepa_out['loss'])}")
        print(f"decoder loss: {fmt(decoder_out['loss'])}")
        print(
            "The absolute decoder loss is not comparable to leaky checkpoints; "
            "judge it with the ablation above."
        )

    print("\n--- reference guide ---")
    print("per-dim std:        healthy ~0.8-1.1; <0.1 collapse; >1.5 too large")
    print("mean vector norm:   lower is better; ~21 with tiny std means mean-dominated")
    print("effective rank:     >15 okay; <5 suspicious; sample-size sensitive (use >=8 x 256)")
    print("cos(z_t, z_t+k):    0.5-0.85 healthy; >0.95 too smooth/copyable")
    print("candidate pair cos: <0.7 diverse; >0.9 candidates collapsed")
    print("delta/z:            >0.2 meaningful movement; <0.05 predictor copying")
    print("best_d:             lower is better; <0.1 easy, >0.3 hard")
    print("score logit std:    >1 discriminative; <0.5 near uniform")
    print("loss_score:         below log(P)-0.1 meaningful; near log(P) = uniform")
    print("win share:          uniform ~1/P; >25% concentrated; >50% collapse")
    print("oracle vs selected: oracle << copy and selected ~ copy = scorer problem;")
    print("                    oracle and selected << copy = hint useful;")
    print("                    oracle ~ copy = candidate set weak")
    print("retrieval:          higher top1/top5, lower mean rank; chance ~1/N")
    print("hint ablation:      predicted < shuffled-0.05 = uses hint;")
    print("                    predicted ~ shuffled ~ null = ignores hint;")
    print("                    real off_k << predicted off_k = last-token leak")


if __name__ == "__main__":
    main()
