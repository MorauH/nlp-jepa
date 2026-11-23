"""Smoke tests for the composed JEPA stack."""

import pytest
import torch
import torch.nn.functional as F

from jepa import (
    CommitmentLossConfig,
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    JepaModel,
    PredictorConfig,
    VISRegConfig,
    build_block_hints,
)


def tiny_config() -> JepaConfig:
    encoder = EncoderConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    predictor = PredictorConfig.from_encoder(
        encoder,
        prediction_count=4,
        num_hidden_layers=1,
    )
    decoder = DecoderConfig.from_encoder(
        encoder,
        sliding_window=8,
        num_hidden_layers=1,
    )
    return JepaConfig.from_encoder(
        encoder,
        predictor=predictor,
        decoder=decoder,
        pred_horizon=2,
        visreg=VISRegConfig(num_slices=16),
    )


def expected_decoder_loss(model, labels, attention_mask, policy_name):
    """Reference decoder loss built from the submodules directly."""
    from jepa.models.commitment import Hindsight, PiArgmax

    k = model.pred_horizon
    with torch.no_grad():
        latents = model.encode(labels, attention_mask)
        futures = model.predictor(latents[:, :-k, :])
        if policy_name == "hindsight":
            committed = futures.commit(Hindsight(), latents[:, k:, :]).future
        else:
            committed = futures.commit(PiArgmax()).future
        z_t, hint, valid = build_block_hints(
            latents, k, committed=committed.detach()
        )
        loss_mask = (
            valid.unsqueeze(0)
            & attention_mask[:, :-1].bool()
            & attention_mask[:, 1:].bool()
        )
        total, count = model.decoder.token_loss(
            z_t.detach(),
            hint.detach(),
            labels,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
    return total / count.clamp_min(1), futures


def test_full_stack_forward():
    model = JepaModel(tiny_config())
    labels = torch.randint(1, 64, (2, 12))
    attention_mask = torch.ones_like(labels)

    jepa_out = model.jepa_loss(labels, attention_mask=attention_mask)
    decoder_out = model.decoder_loss(labels, attention_mask=attention_mask)

    assert jepa_out.loss.ndim == 0
    assert decoder_out.loss.ndim == 0
    assert set(decoder_out.terms) == {"token"}
    assert "tokens" in decoder_out.diagnostics
    assert {"scale", "shape", "center", "move", "pi", "mu", "diversity"} <= set(
        jepa_out.terms
    )


def test_predictor_is_a_pure_function_of_a_latent():
    """Each latent's output depends on that latent and nothing else."""
    from jepa import PredictorModel

    torch.manual_seed(0)
    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder, prediction_count=4, num_hidden_layers=1
        )
    ).eval()
    latents = torch.randn(2, 6, 32)

    # mu starts at exactly 0; give it non-zero output weights so the test
    # exercises it.
    torch.nn.init.normal_(predictor.mu_head[-1].weight)

    with torch.no_grad():
        futures = predictor(latents)
        candidates, pi, mu = futures
        assert candidates.shape == (2, 6, 4, 32)
        assert pi.shape == (2, 6, 4)
        assert mu.shape == (2, 6, 32)
        # The single-latent path (leading dims (B,)) matches every position.
        for t in range(6):
            step = predictor(latents[:, t])
            torch.testing.assert_close(
                step.candidates, candidates[:, t], atol=1e-5, rtol=1e-4
            )
            torch.testing.assert_close(step.pi, pi[:, t], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(step.mu, mu[:, t], atol=1e-5, rtol=1e-4)

        # No leading dimensions at all: (D,) -> (P, D), (P,), (D,).
        single = predictor(latents[0, 0])
        assert single.candidates.shape == (4, 32)
        assert single.pi.shape == (4,)
        assert single.mu.shape == (32,)
        torch.testing.assert_close(
            single.candidates, candidates[0, 0], atol=1e-5, rtol=1e-4
        )

        # Changing one latent changes only that position.
        shifted = latents.clone()
        shifted[:, 3] += 1.0
        shifted_futures = predictor(shifted)
        keep = [0, 1, 2, 4, 5]
        torch.testing.assert_close(
            shifted_futures.candidates[:, keep], candidates[:, keep]
        )
        torch.testing.assert_close(shifted_futures.pi[:, keep], pi[:, keep])
        torch.testing.assert_close(shifted_futures.mu[:, keep], mu[:, keep])
        assert not torch.allclose(
            shifted_futures.candidates[:, 3], candidates[:, 3]
        )
        assert not torch.allclose(shifted_futures.mu[:, 3], mu[:, 3])


def test_mu_head_starts_at_zero_and_is_detached():
    """mu starts at 0 and its loss never reaches z_t (it is an instrument)."""
    from jepa import PredictorModel

    torch.manual_seed(0)
    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder, prediction_count=4, num_hidden_layers=1
        )
    )
    latents = torch.randn(2, 5, 32, requires_grad=True)
    mu = predictor(latents).mu
    assert torch.count_nonzero(mu) == 0

    torch.nn.init.normal_(predictor.mu_head[-1].weight)
    mu = predictor(latents).mu
    mu.pow(2).sum().backward()
    assert latents.grad is None
    assert predictor.mu_head[0].weight.grad is not None


def test_commitment_loss_matches_reference():
    """The dot-product distance, pi, mu and diversity equal the direct formulas."""
    from jepa.models.commitment import Futures
    from jepa.models.losses import CommitmentLoss

    torch.manual_seed(0)
    loss_fn = CommitmentLoss(CommitmentLossConfig(diversity_w=0.5))
    cand = torch.randn(2, 5, 4, 16, requires_grad=True)
    tgt = torch.randn(2, 5, 16)
    scores = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5)
    mask[1, 3:] = 0

    mean = torch.randn(2, 5, 16)
    out = loss_fn(Futures(cand, scores, mean), tgt, mask=mask)

    valid = mask.bool()
    d_ref = (
        F.normalize(cand, dim=-1) - F.normalize(tgt, dim=-1).unsqueeze(2)
    ).pow(2).sum(-1)
    best_ref = (d_ref.min(-1).values * mask).sum() / mask.sum()
    torch.testing.assert_close(out.diagnostics["hindsight_d"], best_ref.detach())

    dirs = F.normalize(cand[valid], dim=-1)
    sim = torch.bmm(dirs, dirs.transpose(1, 2))
    off_diag = 1.0 - torch.eye(4)
    div_ref = (F.relu(sim - 0.5) * off_diag).sum() / (dirs.shape[0] * 4 * 3)
    torch.testing.assert_close(out.terms["diversity"], div_ref.detach())

    def masked(values):
        return (values * mask).sum() / mask.sum()

    # pi: cross-entropy against the relaxed WTA assignment r.
    d = d_ref.detach()
    temperature = 0.1 * masked(d.mean(-1))
    r = 0.95 * F.softmax(-d / temperature, dim=-1) + 0.05 / 4
    pi_ref = masked(-(r * F.log_softmax(scores, dim=-1)).sum(-1))
    torch.testing.assert_close(out.terms["pi"], pi_ref)
    pi_argmax_index = scores.argmax(-1, keepdim=True)
    torch.testing.assert_close(
        out.diagnostics["pi_argmax_d"],
        masked(d.gather(-1, pi_argmax_index).squeeze(-1)),
    )

    # mu: squared error onto the normalized target; its cosine distance.
    tgt_unit = F.normalize(tgt, dim=-1)
    torch.testing.assert_close(
        out.terms["mu"], masked((mean - tgt_unit).pow(2).sum(-1))
    )
    torch.testing.assert_close(
        out.diagnostics["mu_norm"], masked(mean.norm(dim=-1))
    )
    torch.testing.assert_close(
        out.diagnostics["mu_distance"],
        masked((F.normalize(mean, dim=-1) - tgt_unit).pow(2).sum(-1)),
    )


# ---------------------------------------------------------------------------
# Decoder: tied embeddings and per-layer hints
# ---------------------------------------------------------------------------


def test_decoder_tied_embeddings_round_trip(tmp_path):
    from safetensors.torch import load_file

    from jepa import DecoderModel

    model = JepaModel(tiny_config())
    decoder = model.decoder
    assert decoder.lm_head.weight is decoder.embedding.weight
    # The decoder table stays separate from the encoder's.
    assert decoder.embedding.weight is not model.encoder.embedding.weight

    labels = torch.randint(1, 64, (2, 12))
    z_t = torch.zeros(2, 11, 32)
    committed = torch.zeros(2, 11, 32)
    with torch.no_grad():
        before, count = decoder.token_loss(z_t, committed, labels)
    assert int(count) == 2 * 11

    full_dir = tmp_path / "full"
    model.save_pretrained(full_dir)
    # Safetensors cannot store aliases: only the embedding copy is written.
    saved = load_file(full_dir / "model.safetensors")
    assert "decoder.embedding.weight" in saved
    assert "decoder.lm_head.weight" not in saved

    reloaded, info = JepaModel.from_pretrained(
        full_dir, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert reloaded.decoder.lm_head.weight is reloaded.decoder.embedding.weight
    assert torch.equal(
        reloaded.decoder.embedding.weight, decoder.embedding.weight
    )
    with torch.no_grad():
        after, _ = reloaded.decoder.token_loss(z_t, committed, labels)
    torch.testing.assert_close(after, before)

    standalone_dir = tmp_path / "decoder"
    decoder.save_pretrained(standalone_dir)
    standalone, info = DecoderModel.from_pretrained(
        standalone_dir, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert standalone.lm_head.weight is standalone.embedding.weight
    assert torch.equal(standalone.embedding.weight, decoder.embedding.weight)


def test_hint_reinjected_before_every_layer():
    """Each layer's zero-initialized hint projection can reach the output."""
    from jepa import DecoderModel

    torch.manual_seed(0)
    decoder = DecoderModel(
        DecoderConfig.from_encoder(
            tiny_config().encoder,
            sliding_window=8,
            num_hidden_layers=2,
        )
    ).eval()
    assert len(decoder.hint_proj) == 2
    assert all(
        torch.count_nonzero(projection.weight) == 0
        for projection in decoder.hint_proj
    )

    labels = torch.randint(1, 64, (1, 7))
    z_t = torch.randn(1, 6, 32)
    committed = torch.randn(1, 6, 32)
    with torch.no_grad():
        plain, _ = decoder.token_loss(z_t, committed, labels)

    # A fresh decoder is input-only injection: the per-layer projections are
    # exactly zero.  Giving any one layer a non-zero projection changes the
    # loss, so every layer re-injects the hint before itself.
    for index, projection in enumerate(decoder.hint_proj):
        torch.nn.init.normal_(projection.weight)
        with torch.no_grad():
            layered, _ = decoder.token_loss(z_t, committed, labels)
        assert not torch.allclose(plain, layered), f"layer {index} is inert"
        with torch.no_grad():
            projection.weight.zero_()


def test_fresh_decoder_loss_is_near_uniform():
    import math

    from jepa import DecoderModel

    vocab_size = 32000
    torch.manual_seed(0)
    encoder = EncoderConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    decoder = DecoderModel(
        DecoderConfig.from_encoder(encoder, sliding_window=8, num_hidden_layers=1)
    ).eval()
    assert decoder.lm_head.weight is decoder.embedding.weight

    labels = torch.randint(1, vocab_size, (4, 33))
    z_t = torch.randn(4, 32, 64)
    committed = torch.randn(4, 32, 64)
    with torch.no_grad():
        total, count = decoder.token_loss(z_t, committed, labels)
    assert abs((total / count).item() - math.log(vocab_size)) < 1.0


def test_hint_and_generate_path_agree():
    from jepa import DecoderModel

    torch.manual_seed(0)
    decoder = DecoderModel(
        DecoderConfig.from_encoder(
            tiny_config().encoder,
            sliding_window=8,
            num_hidden_layers=1,
        )
    ).eval()
    z_t = torch.randn(1, 32)
    committed = torch.randn(1, 32)
    context_ids = torch.randint(1, 64, (1, 6))

    with torch.no_grad():
        greedy = decoder.generate(z_t, committed, context_ids, k=1, temperature=0.0)

        # Score every possible next token through token_loss and keep only the
        # last target position.  Its input and hint are exactly what generate()
        # sees, so the lowest cross-entropy token is the greedy one.
        steps = context_ids.shape[1]
        keep = torch.zeros(1, steps, dtype=torch.bool)
        keep[:, -1] = True
        scores = []
        for token in range(1, 64):
            labels = torch.cat(
                [context_ids, torch.tensor([[token]])], dim=1
            )
            total, _ = decoder.token_loss(
                z_t[:, None].expand(1, steps, -1),
                committed[:, None].expand(1, steps, -1),
                labels,
                loss_mask=keep,
            )
            scores.append(total.item())

    assert int(greedy[0, 0]) == int(torch.tensor(scores).argmin()) + 1


# ---------------------------------------------------------------------------
# Full stack
# ---------------------------------------------------------------------------


def test_full_stack_forward_with_chunked_decoder_loss():
    model = JepaModel(tiny_config())
    model.config.decoder.loss_chunk_size = 4
    labels = torch.randint(1, 64, (2, 12))
    jepa_out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels))
    decoder_out = model.decoder_loss(
        labels, attention_mask=torch.ones_like(labels)
    )
    assert torch.isfinite(jepa_out.loss) and torch.isfinite(decoder_out.loss)

    (jepa_out.loss + decoder_out.loss).backward()
    grads = dict(model.named_parameters())
    for name in (
        "predictor.hidden_expansion.weights",
        "predictor.transformer.decoders.0.attn.w_v.weight",
        "predictor.delta_head.weights",
        "predictor.pi_head.weights",
        "predictor.mu_head.2.weight",
        "decoder.hint_proj.0.weight",
        "decoder.embedding.weight",
        "encoder.embedding.weight",
    ):
        grad = grads[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_decoder_loss_does_not_update_encoder_or_predictor():
    model = JepaModel(tiny_config())
    labels = torch.randint(1, 64, (2, 12))
    decoder_out = model.decoder_loss(
        labels, attention_mask=torch.ones_like(labels)
    )
    decoder_out.loss.backward()

    assert model.encoder.embedding.weight.grad is None
    assert all(p.grad is None for p in model.predictor.parameters())
    # The decoder's own (tied) table does receive the decoder gradient.
    assert model.decoder.embedding.weight.grad is not None
    assert model.decoder.lm_head.weight.grad is not None


def test_build_block_hints_never_leaks_future():
    """Hints come from the target's own block; the tail block is masked out."""
    for seq_len, horizon in ((12, 2), (13, 5), (8, 3)):
        latents = torch.zeros(1, seq_len, 4)
        for j in range(seq_len):
            latents[0, j, 0] = j + 1
        committed = torch.zeros(1, seq_len - horizon, 4)
        for j in range(seq_len - horizon):
            committed[0, j, 0] = j + 1

        z_t, hint, valid = build_block_hints(latents, horizon, committed=committed)
        assert z_t.shape[1] == seq_len - 1
        assert hint.shape[1] == seq_len - 1
        assert valid.shape == (seq_len - 1,)

        positions = torch.arange(seq_len - 1)
        block_start = (positions // horizon) * horizon
        last_valid = seq_len - horizon - 1
        # The last partial block has no committed future and is masked out.
        assert not valid[-1]
        assert torch.equal(valid, block_start <= last_valid)

        for i in range(seq_len - 1):
            if not valid[i]:
                assert int(block_start[i]) > last_valid
                continue
            # Both hint parts come from the same block start, never from a
            # later block, so neither can see target position i + 1.
            z_pos = int(z_t[0, i, 0].item()) - 1
            committed_pos = int(hint[0, i, 0].item()) - 1
            assert z_pos == committed_pos == int(block_start[i]) <= i


def test_decoder_loss_masks_the_last_partial_block():
    """Targets with no committed future do not contribute to the loss."""
    from jepa.models.commitment import Hindsight, PiArgmax

    torch.manual_seed(0)
    model = JepaModel(tiny_config())
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)

    # Positions 0..10 predict tokens 1..11; block_start <= 12 - 2 - 1 = 9, so
    # 10 of the 11 targets count.
    with torch.no_grad():
        for policy in (None, PiArgmax(), Hindsight()):
            out = model.decoder_loss(labels, attention_mask=mask, policy=policy)
            assert int(out.diagnostics["tokens"]) == 2 * 10

        # The last token is only predicted by the excluded position and only
        # feeds latents no hint uses, so changing it leaves the loss untouched.
        changed = labels.clone()
        changed[:, -1] = (changed[:, -1] % 63) + 1
        for policy in (PiArgmax(), Hindsight()):
            before = model.decoder_loss(
                labels, attention_mask=mask, policy=policy
            ).loss
            after = model.decoder_loss(
                changed, attention_mask=mask, policy=policy
            ).loss
            torch.testing.assert_close(before, after)


class CrashingDataset:
    """Wrap a dataset and raise after ``crash_after`` example reads.

    Simulates a crash so a mid-epoch checkpoint is left on disk.  It exposes
    the Arrow table so ``example_lengths`` does not read every row up front.
    """

    def __init__(self, dataset, crash_after):
        self.dataset = dataset
        self.data = dataset.data
        self.crash_after = crash_after
        self.calls = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError("simulated crash")
        return self.dataset[index]


def test_trainer_saves_mid_epoch_checkpoints(tmp_path):
    """save_every_steps leaves a loadable checkpoint on disk mid-epoch."""
    import json
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(10)]}
    )
    trainer = JepaTrainer(
        model=JepaModel(tiny_config()),
        config=TrainConfig(
            output_dir=str(tmp_path),
            stage="jepa",
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
            num_workers=0,
            save_every_steps=2,
        ),
        train_dataset=CrashingDataset(data, crash_after=5),
        eval_dataset=data.select(range(4)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )

    # 10 examples / batch 2 = 5 steps.  The crash hits the third batch, after
    # the mid-epoch save at step 2.
    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer.train()

    history = json.loads((tmp_path / "history_jepa.json").read_text())
    checkpoint = history["last_checkpoint"]
    assert checkpoint["global_step"] == 2
    assert checkpoint["total_steps"] == 5
    assert not checkpoint.get("end_of_epoch")
    JepaModel.from_pretrained(str(tmp_path))

    # Every save also records the position needed to continue the run.
    state = torch.load(
        tmp_path / "trainer_state_jepa.pt", map_location="cpu", weights_only=True
    )
    assert (state["epoch"], state["epoch_step"], state["global_step"]) == (0, 2, 2)
    assert state["total_steps"] == 5
    assert state["optimizer"]["state"]


def test_resume_continues_the_optimizer_schedule_and_epoch(tmp_path):
    """--resume continues global_step, Adam state, LR and the epoch order."""
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(10)]}
    )

    def build(model, output_dir, *, resume=False, crash_after=None):
        history = None
        if resume:
            history_path = Path(output_dir) / "history_jepa.json"
            if history_path.exists():
                history = json.loads(history_path.read_text())
        train_dataset = (
            data if crash_after is None else CrashingDataset(data, crash_after)
        )
        return JepaTrainer(
            model=model,
            config=TrainConfig(
                output_dir=str(output_dir),
                stage="jepa",
                learning_rate=1e-3,
                batch_size=2,
                epochs=2,
                device="cpu",
                num_workers=0,
                resume=resume,
                save_every_steps=2,
            ),
            train_dataset=train_dataset,
            eval_dataset=data.select(range(4)),
            tokenizer=SimpleNamespace(pad_token_id=0),
            history=history,
        )

    def run(trainer, *, crash_after=None):
        """Run and record the LR each optimizer step used and the Adam counts."""
        lrs, adam_steps = [], []
        original_step = torch.optim.AdamW.step

        def recording_step(self, closure=None):
            lrs.append(self.param_groups[0]["lr"])
            out = original_step(self, closure)
            if self.state:
                adam_steps.append(
                    max(int(v["step"]) for v in self.state.values() if "step" in v)
                )
            return out

        torch.optim.AdamW.step = recording_step
        try:
            if crash_after is None:
                trainer.train()
            else:
                with pytest.raises(RuntimeError, match="simulated crash"):
                    trainer.train()
        finally:
            torch.optim.AdamW.step = original_step
        return lrs, adam_steps, trainer.history

    # Reference: an uninterrupted 2-epoch run over the same seed and data.
    torch.manual_seed(2024)
    reference = JepaModel(tiny_config())
    ref_lrs, ref_steps, ref_history = run(build(reference, tmp_path / "reference"))
    assert len(ref_lrs) == 10
    assert ref_steps == list(range(1, 11))  # Adam's per-parameter step counts
    assert ref_lrs[-1] < ref_lrs[0]  # cosine decay, not a warmup restart

    # Crash after 6 steps: epoch 1 plus the first batch of epoch 2.  Twelve
    # example reads is exactly the batch after the step-6 save.
    torch.manual_seed(2024)
    interrupted = JepaModel(tiny_config())
    interrupted_dir = tmp_path / "run"
    interrupted_lrs, interrupted_steps, interrupted_history = run(
        build(interrupted, interrupted_dir, crash_after=12),
        crash_after=12,
    )
    assert interrupted_lrs == ref_lrs[:6]
    assert interrupted_steps[-1] == 6
    assert len(interrupted_history["train_loss"]) == 1
    state = torch.load(
        interrupted_dir / "trainer_state_jepa.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert (state["epoch"], state["epoch_step"], state["global_step"]) == (1, 1, 6)

    resumed = JepaModel.from_pretrained(str(interrupted_dir))
    resumed_lrs, resumed_steps, history = run(
        build(resumed, interrupted_dir, resume=True)
    )

    # global_step continues: the first resumed step is step 6 of the schedule,
    # so the LR and the scheduled shape are the uninterrupted ones.
    assert resumed_lrs == ref_lrs[6:]
    assert resumed_steps == list(range(7, 11))

    # Optimizer state was restored, not rebuilt: the resumed run reaches the
    # same weights as the uninterrupted run.
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in resumed.named_parameters():
        torch.testing.assert_close(
            parameter,
            reference_parameters[name],
            atol=1e-6,
            rtol=1e-5,
        )

    # History continues: once per epoch, including the epoch that was
    # interrupted and resumed, with the epoch means of the uninterrupted run.
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    for index in range(2):
        assert history["train_loss"][index] == pytest.approx(
            ref_history["train_loss"][index], rel=1e-5
        )


def test_resume_without_a_trainer_state_fails(tmp_path):
    """A checkpoint without trainer_state_<stage>.pt cannot be resumed."""
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    trainer = JepaTrainer(
        model=JepaModel(tiny_config()),
        config=TrainConfig(
            output_dir=str(tmp_path),
            stage="jepa",
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
            num_workers=0,
            resume=True,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    with pytest.raises(FileNotFoundError, match="trainer_state_jepa.pt"):
        trainer.train()


def test_decoder_loss_policy_is_an_argument_not_the_module_mode():
    """policy=None uses train_policy; PiArgmax() is the inference condition."""
    from jepa.models.commitment import PiArgmax

    torch.manual_seed(0)
    model = JepaModel(tiny_config())
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)

    with torch.no_grad():
        # Module mode is irrelevant: the default is the training condition.
        for mode in (model.train, model.eval):
            mode()
            actual = model.decoder_loss(labels, attention_mask=mask).loss
            expected, _ = expected_decoder_loss(model, labels, mask, "hindsight")
            torch.testing.assert_close(actual, expected)

            # Explicit π-argmax is the inference condition, in either mode.
            actual = model.decoder_loss(
                labels, attention_mask=mask, policy=PiArgmax()
            ).loss
            expected, _ = expected_decoder_loss(model, labels, mask, "pi_argmax")
            torch.testing.assert_close(actual, expected)

        # train_policy="pi_argmax" changes only the default.
        model.config.decoder.train_policy = "pi_argmax"
        actual = model.decoder_loss(labels, attention_mask=mask).loss
        expected, _ = expected_decoder_loss(model, labels, mask, "pi_argmax")
        torch.testing.assert_close(actual, expected)


def test_candidates_are_rms_normalized():
    from jepa import PredictorModel

    torch.manual_seed(0)
    predictor = PredictorModel(
        PredictorConfig.from_encoder(tiny_config().encoder, prediction_count=4)
    )
    candidates = predictor(torch.randn(2, 5, 32) * 7.0).candidates
    rms = candidates.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)


def test_loss_outputs_are_weighted_sums_of_their_terms():
    """loss == sum(weights[name] * terms[name]) for both modules."""
    from jepa.models.commitment import Futures
    from jepa.models.losses import CommitmentLoss, VISReg

    torch.manual_seed(0)
    visreg = VISReg(
        VISRegConfig(num_slices=4, scale_w=0.5, shape_w=2.0, center_w=3.0)
    )
    out = visreg(torch.randn(8, 6))
    assert set(out.terms) == set(visreg.weights) == {"scale", "shape", "center"}
    expected = sum(visreg.weights[name] * out.terms[name] for name in out.terms)
    torch.testing.assert_close(out.loss, expected)

    commitment = CommitmentLoss(
        CommitmentLossConfig(
            move_w=0.5, pi_w=2.0, mu_w=3.0, diversity_w=4.0
        )
    )
    futures = Futures(
        torch.randn(2, 3, 4, 6), torch.randn(2, 3, 4), torch.randn(2, 3, 6)
    )
    out = commitment(futures, torch.randn(2, 3, 6))
    assert set(out.terms) == set(commitment.weights) == {
        "move",
        "pi",
        "mu",
        "diversity",
    }
    expected = sum(
        commitment.weights[name] * out.terms[name] for name in out.terms
    )
    torch.testing.assert_close(out.loss, expected)


def test_temperature_scale_anneals_and_is_final_without_progress():
    from jepa.models.commitment import Futures
    from jepa.models.losses import CommitmentLoss

    loss_fn = CommitmentLoss(
        CommitmentLossConfig(
            temperature=0.1, temperature_start=1.0, anneal_fraction=0.5
        )
    )
    futures = Futures(
        torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2), torch.randn(1, 1, 4)
    )
    target = torch.randn(1, 1, 4)

    cases = (
        (torch.tensor(0.0), 1.0),
        (torch.tensor(0.25), 0.1**0.5),
        (torch.tensor(0.5), 0.1),
        (torch.tensor(0.9), 0.1),
        (None, 0.1),
    )
    for progress, expected in cases:
        temperature = loss_fn(
            futures, target, progress=progress
        ).diagnostics["temperature_scale"]
        assert temperature.item() == pytest.approx(expected, rel=1e-5)


def test_diversity_margin_ignores_dissimilar_pairs():
    from jepa.models.commitment import Futures
    from jepa.models.losses import CommitmentLoss

    torch.manual_seed(0)
    # Two candidates at cosine 0.3, two at ~0.99.
    a = F.normalize(torch.randn(16), dim=0)
    b = F.normalize(torch.randn(16), dim=0)
    b = F.normalize(b - (b @ a) * a, dim=0)
    far = torch.stack([a, 0.3 * a + (1 - 0.09) ** 0.5 * b])
    near = torch.stack([a, F.normalize(a + 0.05 * b, dim=0)])

    def div(cands, margin):
        crit = CommitmentLoss(
            CommitmentLossConfig(diversity_w=1.0, diversity_margin=margin)
        )
        futures = Futures(
            cands[None, None],
            torch.zeros(1, 1, 2),
            torch.randn(1, 1, 16),
        )
        out = crit(futures, torch.randn(1, 1, 16))
        return out.terms["diversity"].item()

    assert div(far, 0.0) == pytest.approx(0.3, abs=1e-4)
    assert div(far, 0.5) == 0.0
    assert div(near, 0.5) > 0.4


def test_trainer_anneals_temperature(tmp_path):
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    config = tiny_config()
    config.commitment.temperature_start = 1.0
    config.commitment.anneal_fraction = 0.5
    model = JepaModel(config)
    seen = []
    original = model.commitment_loss.forward

    def record(futures, target, mask=None, progress=None):
        seen.append(None if progress is None else progress.item())
        return original(futures, target, mask=mask, progress=progress)

    model.commitment_loss.forward = record
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(8)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path),
            stage="jepa",
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
            num_workers=0,
            save_every_steps=1,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    ).train()

    # Four training steps get an increasing 0-dim progress tensor; the single
    # validation batch gets a 0-dim tensor 1.0, i.e. the final temperature.
    assert seen[:4] == [0.0, 0.25, 0.5, 0.75]
    assert seen[4:] == [1.0]
    assert history["train_jepa_temperature_scale"][0] == pytest.approx(
        (1.0 + 0.1**0.5 + 0.1 + 0.1) / 4, rel=1e-5
    )
    assert history["val_jepa_temperature_scale"][0] == pytest.approx(0.1, rel=1e-5)


# ---------------------------------------------------------------------------
# Two stages: the trainer owns freezing
# ---------------------------------------------------------------------------


def test_decoder_stage_trains_only_the_decoder(tmp_path):
    """Decoder stage: no grads or updates for the backbone, eval mode, no JEPA loss."""
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    model = JepaModel(tiny_config())

    backbone = list(model.encoder.parameters()) + list(model.predictor.parameters())
    before_backbone = [p.detach().clone() for p in backbone]
    before_decoder = [p.detach().clone() for p in model.decoder.parameters()]

    training_modes = []
    model.encoder.register_forward_hook(
        lambda module, *args: training_modes.append(module.training)
    )
    model.predictor.register_forward_hook(
        lambda module, *args: training_modes.append(module.training)
    )

    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path),
            stage="decoder",
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
            num_workers=0,
            save_every_steps=1,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    ).train()

    # The trainer sets the stage modules' modes; the frozen backbone stays in
    # eval mode.  There is no top-level model.train()/eval() dance.
    assert model.decoder.training
    assert not model.encoder.training and not model.predictor.training
    assert training_modes and not any(training_modes)

    assert all(p.grad is None for p in backbone)
    assert all(torch.equal(p, q) for p, q in zip(backbone, before_backbone))
    assert any(
        not torch.equal(p, q) for p, q in zip(model.decoder.parameters(), before_decoder)
    )

    # The decoder stage logs no JEPA keys and reports perplexity.
    train_keys = {
        key.removeprefix("train_") for key in history if key.startswith("train_")
    }
    val_keys = {
        key.removeprefix("val_") for key in history if key.startswith("val_")
    }
    assert {"loss", "decoder_token"} <= train_keys
    assert {"loss", "decoder_token", "perplexity"} <= val_keys
    assert not any("jepa" in key for key in history)
    assert all("jepa" not in key for key in history["last_checkpoint"])

    # The full checkpoint reloads, including the trained decoder.
    reloaded = JepaModel.from_pretrained(str(tmp_path))
    assert not any("_orig_mod." in key for key in reloaded.state_dict())
    for trained, saved in zip(model.decoder.parameters(), reloaded.decoder.parameters()):
        assert torch.equal(trained, saved)


def test_train_cli_decoder_stage_writes_loadable_checkpoint(tmp_path, monkeypatch):
    """`train --stage decoder` main(): frozen backbone, fresh decoder, saves."""
    import json
    from types import SimpleNamespace

    from datasets import Dataset, DatasetDict
    from jepa.training import train

    class FakeTokenizer(SimpleNamespace):
        model_max_length = 32

        def __len__(self):
            return 64

    torch.manual_seed(0)
    base = tmp_path / "base"
    base_model = JepaModel(tiny_config())
    base_model.save_pretrained(base)

    rows = [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]
    data_dir = tmp_path / "data"
    DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": rows}),
            "validation": Dataset.from_dict({"input_ids": rows[:2]}),
        }
    ).save_to_disk(str(data_dir))

    monkeypatch.setattr(
        train, "load_training_tokenizer", lambda path: FakeTokenizer(pad_token_id=0)
    )

    # The decoder stage writes beside --from, never over the JEPA checkpoint.
    jepa_history = {"train_jepa_loss": [1.0], "last_checkpoint": {"global_step": 7}}
    (base / "history_jepa.json").write_text(json.dumps(jepa_history))
    out = tmp_path / "base-decoder"
    train.main(
        [
            "--stage", "decoder",
            "--from", str(base),
            "--fineweb-dir", str(data_dir),
            "--tokenizer", str(tmp_path / "unused-tokenizer"),
            "--batch-size", "2",
            "--epochs", "1",
            "--device", "cpu",
        ]
    )

    assert out.is_dir()
    assert json.loads((base / "history_jepa.json").read_text()) == jepa_history
    assert not (base / "history_decoder.json").exists()
    assert (out / "trainer_state_decoder.pt").exists()
    history = json.loads((out / "history_decoder.json").read_text())
    # 4 examples / batch 2 = 2 steps: the epoch-end checkpoint is on disk.
    assert history["last_checkpoint"]["global_step"] == 2
    assert history["last_checkpoint"]["end_of_epoch"]
    train_keys = {
        key.removeprefix("train_") for key in history if key.startswith("train_")
    }
    val_keys = {
        key.removeprefix("val_") for key in history if key.startswith("val_")
    }
    assert "decoder_token" in train_keys
    assert {"decoder_token", "perplexity"} <= val_keys
    assert not any("jepa" in key for key in history)

    reloaded = JepaModel.from_pretrained(str(out))
    assert not any("_orig_mod." in key for key in reloaded.state_dict())

    # The encoder was loaded from the checkpoint and never updated; the decoder
    # is fresh (the base model's decoder is not reused).
    assert torch.equal(
        reloaded.encoder.embedding.weight, base_model.encoder.embedding.weight
    )
    assert not torch.equal(
        reloaded.decoder.embedding.weight, base_model.decoder.embedding.weight
    )

    # With no epochs left, --resume still has to load *that* decoder, not attach
    # a fresh one from --from: the returned model is the saved checkpoint's.
    saved_decoder = [p.detach().clone() for p in reloaded.decoder.parameters()]
    loaded = train.main(
        [
            "--stage", "decoder",
            "--resume",
            "--from", str(base),
            "--fineweb-dir", str(data_dir),
            "--tokenizer", str(tmp_path / "unused-tokenizer"),
            "--batch-size", "2",
            "--epochs", "1",
            "--device", "cpu",
        ]
    )
    for trained, saved in zip(loaded.model.decoder.parameters(), saved_decoder):
        assert torch.equal(trained, saved)

    # One more epoch continues from that decoder and appends to the history.
    train.main(
        [
            "--stage", "decoder",
            "--resume",
            "--from", str(base),
            "--fineweb-dir", str(data_dir),
            "--tokenizer", str(tmp_path / "unused-tokenizer"),
            "--batch-size", "2",
            "--epochs", "2",
            "--device", "cpu",
        ]
    )
    resumed = JepaModel.from_pretrained(str(out))
    assert not torch.equal(
        resumed.decoder.embedding.weight, reloaded.decoder.embedding.weight
    )
    assert not torch.equal(
        resumed.decoder.embedding.weight, base_model.decoder.embedding.weight
    )
    assert torch.equal(
        resumed.encoder.embedding.weight, base_model.encoder.embedding.weight
    )
    history = json.loads((out / "history_decoder.json").read_text())
    token_key = next(
        key
        for key in history
        if key.startswith("train_") and key.endswith("decoder_token")
    )
    assert len(history[token_key]) == 2
    assert json.loads((base / "history_jepa.json").read_text()) == jepa_history


def test_decoder_stage_defaults_to_a_sibling_and_refuses_the_source(tmp_path):
    """--output-dir defaults beside --from; reusing --from for a fresh run fails."""
    from pathlib import Path

    from jepa.training import train

    args = train.parse_args(["--stage", "decoder", "--from", "runs/jepa"])
    assert args.output_dir == Path("runs/jepa-decoder")

    # Resuming may point at the source (legacy decoder-in-JEPA-dir runs).
    args = train.parse_args(
        [
            "--stage", "decoder", "--resume",
            "--from", "runs/jepa", "--output-dir", "runs/jepa",
        ]
    )
    assert args.output_dir == Path("runs/jepa")

    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--stage", "decoder",
                "--from", "runs/jepa", "--output-dir", "runs/jepa",
            ]
        )


def test_jepa_stage_never_runs_the_decoder(tmp_path):
    """JEPA stage: decoder untouched and never called."""
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    model = JepaModel(tiny_config())

    before_encoder = [p.detach().clone() for p in model.encoder.parameters()]
    before_decoder = {n: p.detach().clone() for n, p in model.decoder.named_parameters()}
    decoder_calls = []
    model.decoder.register_forward_hook(lambda *args: decoder_calls.append(1))

    labels = torch.randint(1, 64, (2, 12))
    jepa_out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels))
    assert jepa_out.loss.ndim == 0

    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path),
            stage="jepa",
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
            num_workers=0,
            save_every_steps=1,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    ).train()

    assert not decoder_calls
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(torch.equal(p, before_decoder[n]) for n, p in model.decoder.named_parameters())
    assert any(
        not torch.equal(p, q) for p, q in zip(model.encoder.parameters(), before_encoder)
    )
    # Only JEPA losses are logged; the checkpoint still has a (loadable) decoder.
    assert set(history) >= {"train_loss", "train_jepa_loss", "val_jepa_loss"}
    assert not any("decoder" in key or "perplexity" in key for key in history)
    reloaded = JepaModel.from_pretrained(str(tmp_path))
    assert all(
        torch.equal(p, before_decoder[n]) for n, p in reloaded.decoder.named_parameters()
    )


def test_jepa_stage_logs_every_term_and_diagnostic(tmp_path):
    """Every JEPA term/diagnostic is in the history and sums to the total."""
    import json
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    model = JepaModel(tiny_config())

    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path),
            stage="jepa",
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
            num_workers=0,
            save_every_steps=1,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    ).train()

    history = json.loads((tmp_path / "history_jepa.json").read_text())
    terms = set(model.visreg.weights) | set(model.commitment_loss.weights)
    diagnostics = {
        "hindsight_d",
        "pi_argmax_d",
        "mu_norm",
        "mu_distance",
        "temperature_scale",
        "assignment_entropy",
    }
    assert terms == {"scale", "shape", "center", "move", "pi", "mu", "diversity"}

    for prefix in ("train", "val"):
        for name in terms | diagnostics:
            key = f"{prefix}_jepa_{name}"
            assert isinstance(history[key], list) and len(history[key]) == 1, key
        # Diagnostics are separate from the loss keys.
        assert not any(name in terms for name in diagnostics)

    weights = {**model.visreg.weights, **model.commitment_loss.weights}
    weighted = sum(
        weights[name] * history[f"val_jepa_{name}"][0] for name in terms
    )
    assert weighted == pytest.approx(history["val_jepa_loss"][0], rel=1e-5)
    assert not any("decoder" in key or "perplexity" in key for key in history)
