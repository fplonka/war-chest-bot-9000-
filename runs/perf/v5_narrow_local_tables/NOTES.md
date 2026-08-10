# v5_narrow_local_tables

We were testing whether the now-dominant sparse sweeps would benefit from
storing four job-local immutable index tables as `u16`: per-node support sizes,
draw successors, and the two reverse-reach source arrays. Global offsets and
all mutable CFR/value state stayed `u32`/`f32`, and waves that could not prove
the local bound retained a `u32` fallback. All 64 jobs in the deterministic
production tape qualified for the narrow path, and all 17 CUDA library tests
passed.

The result was a clear negative on two interleaved 20-second one-card pairs.
The narrow build measured 561.8 and 555.6 solves/s, averaging 558.7/s. The
unchanged `u32` binary measured 577.9 and 574.1 solves/s, averaging 576.0/s.
That is a 3.0% regression. The uniform-width branch and conversion/load path in
the hot loops cost more than the smaller upload and table footprint saved.

The experiment was reverted. At this point the compressed network head leaves
reach, backpropagation, and readout at 54.5% of aged kernel time, but simply
narrowing their local index reads is not the way to reduce it. Any future
16-bit table attempt should use separately compiled narrow kernels with no hot
runtime branch, and only after a profile shows immutable-table bandwidth rather
than mutable CFR traffic is the limiting part of those kernels.

