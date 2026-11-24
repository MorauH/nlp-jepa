# Two-stage training, no joint training

Superseded by ADR 0002 (the decoder loss trains the encoder through the hint z_t). With `decoder_w = 0`, the default for now, training is still this two-stage recipe.

The JEPA (encoder + predictor) is trained first, then a decoder is trained against the frozen JEPA. Joint training was removed: it fed the decoder hints that kept moving while the JEPA was still training, and it required detached hints, per-stage flags on the model and weighted mixing of the two losses. Two stages give the decoder stable hints from its first step and keep each training step running only one objective. The cost is that the decoder and the JEPA can no longer adapt to each other.
