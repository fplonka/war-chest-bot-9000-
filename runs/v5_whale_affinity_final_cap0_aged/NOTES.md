# v5_whale_affinity_final_cap0_aged

We were checking the memory and speed effect of routing every common 4 GiB
search to one retained-buffer lane instead of letting successive large searches
grow all five lanes. This replayed the same learned checkpoint, true zero
horizon payoff, two-card five-lane settings, seed, and three-minute live stream
as `v5_headfull16_final_cap0_aged`. No optimizer ran.

The run completed without an allocation failure after 314 large-search routes.
Peak memory was 20,441 MiB on GPU 0 and 16,043 MiB on GPU 1, down from
23,225/22,315 MiB in the earlier run and safely below the 24 GiB limit. The
sampler was restarted about 50 seconds into the run because the first sampler's
timestamp contained the delimiter used by `sed`; the reported peaks cover the
aged portion that matters and the malformed sampler produced no usable rows.

The safety margin was not free. The stream stopped admission at 187,392 solves
in 180.10 seconds, or 1,040.5 solves/s, versus 196,608 and 1,091.7/s in the
control. Counts were essentially equal through 120 seconds, then the final
60-second window produced about 752 solves/s versus 921/s. The live game
trajectories were not identical: this run hit 1,556 node caps versus 1,361 and
finished 732 games versus 871, so the tail difference is not a clean isolated
speed measurement. There were no exact fallbacks or dropped results, and drain
took 46.7 seconds.

At this point lane affinity fixes the observed five-lane retained-memory failure
and all 16 CUDA library tests pass. It may serialize too much expensive tail
work, however. Before adopting it as the final scheduler, compare a less
restrictive retirement policy on frozen large-job work or in another aged
stream; the project still needs both unattended memory safety and materially
higher mature throughput.
