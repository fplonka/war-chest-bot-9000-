# Device-batch integrated launch failure

We were trying the first one-minute live trainer with compact replay expansion on GPU 1 and 128 actors per builder.

The process failed during the pre-clock Triton warm-up, before a snapshot, game, solve, or optimizer step. The Rust solve services had initialized CUDA contexts for devices 0 and 1, the batch tensors were allocated on `cuda:1`, but Triton's launch still selected the process's current device 0 and rejected the device-1 pointers. The standalone parity test had not exposed this because no two-device solve service was active around it.

No performance or learning conclusion can be drawn from this attempt. At this point the device kernels still passed real replay parity; the integration needed to set PyTorch's current CUDA device explicitly before starting the services and compiling Triton. The retry must use a new run directory so this failed launch remains identifiable.
