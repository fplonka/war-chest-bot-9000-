# Isolated TF32 trainer profile

We were testing the user's requested fast, idiomatic Ampere matrix-multiplication mode before changing production. PyTorch 2.12 on the target box defaulted to `highest` float32 matmul precision with TF32 disabled. This repeated the frozen 30-step trainer profile after selecting `high`, which permits TF32 internally while keeping parameters, activations, loss reductions, and optimizer state in float32.

Nothing material changed in isolation. The full step averaged 73.06 ms versus 72.32 ms in the earlier strict-FP32 profile. Preparation and copies were 56.90 ms, forward was 3.30 ms, and backward, clipping, and Adam were 10.38 ms. This is a useful negative result: model GEMMs are only a small fraction of the isolated step, so TF32 cannot fix host row expansion by itself. Its value, if any, must be tested under live solve contention, where the integrated profile measured much longer GPU queues.

At this point production still used strict PyTorch FP32 matmuls; no training code or checkpoint format had changed. The correctness policy allowed bounded numerical differences from fast GPU GEMMs, with invariants and learning behavior carrying the confidence burden rather than exact agreement with a scalar CPU oracle.
