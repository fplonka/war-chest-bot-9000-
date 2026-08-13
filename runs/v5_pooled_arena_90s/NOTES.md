# v5_pooled_arena_90s

We intended to test the arena planner that reuses static tower scratch for CFR
state. This trace did not run that CUDA layout: the edited `device.rs` was
accidentally copied to `engine/src/device.rs` on the remote machine instead of
replacing `engine/src/gpu/device.rs`. The serializer and its new two-gibibyte
one-job isolation rule were active, but the device still allocated the old
sequential arena. Missing `v5_arena_plan` telemetry exposed the mismatch.

This is therefore a deployment failure, not a candidate performance result.
The trace reached 92,160 solves in 90.009 seconds, or 1,023.9 solves/s, and
drained cleanly. Its 82 pre-stop "oversize" solves mostly reflect the new
two-gibibyte isolation counter, while actual four-gibibyte jobs still took the
old device layout and memory guard. Sampled peak memory was 22,361 MiB on GPU 0
and 22,059 MiB on GPU 1, with no exact fallbacks or dropped solves.

At this point the local implementation and commit are unchanged and all prior
tests remain valid, but the live performance question is unanswered. The next
run must copy `device.rs` to the exact `engine/src/gpu/device.rs` path, remove
the stray file, rebuild, confirm arena-plan telemetry appears, and only then
judge the pooled allocator.
