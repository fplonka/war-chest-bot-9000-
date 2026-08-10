# v5_f16cur_final_cap0_aged

We were testing whether storing the per-iteration current CFR strategy in
float16 would speed the now-dominant reach and backpropagation sweeps and shrink
late cell arenas below allocator power-of-two boundaries. Regrets, strategy
sums, the materialised final strategy, reaches, values, root values, and
outputs remained float32. The precise diagnostic path stayed entirely float32.
All fast and precise CUDA library tests passed.

On two interleaved 20-second runs of the identical one-card tape, the candidate
averaged 570.8 solves/s versus 574.6/s for float32 current strategy, a 0.7%
regression. The three-minute learned-checkpoint live stream also finished
slower overall: 184,320 solves in 180.06 seconds, or 1,023.7/s, versus 187,392
and 1,040.5/s for the matching whale-affinity control. It completed 636 games,
used 326 large-search routes, and hit 1,503 node limits, with no exact fallback
or dropped result. Peak memory was 20,313 and 22,123 MiB.

The live trajectory was not uniformly worse. It started a harder path and was
9,216 solves behind at 120 seconds, then produced 51,200 solves in the final
minute versus 45,056 in the control, narrowing the final gap to 3,072. That is
interesting steady-state noise, but it is not a clean implementation speedup:
the deterministic tape was slightly slower and the run completed fewer games
than the float32 control. Keeping a new approximate CFR state format would then
require a learning ladder without having first earned a performance win.

The experiment was reverted. At this point scalar half conversion is not a
useful way to accelerate sparse state access on Ampere. A future state-
compression attempt needs vectorised storage/access or a measured allocator-
class win large enough to overcome the direct-kernel regression.
