This run asked whether the pre-v5 two-GPU service could satisfy the real
training gate when generation and optimizer time were counted together.  It
used random drafts, depth 2, 64 iterations, changing 384/64/64 weights, and
3,072-game batches, but still used the old resident/tick CUDA architecture and
a two-million-node cap.

It did not produce a valid golden result.  The process stopped after two very
large ReBeL batches, at 1,301.9 seconds rather than the requested 30-minute
deadline.  From the ReBeL transition at 121.6 seconds it completed 537,665
fresh solves in 1,180.3 wall-clock seconds, or about 455.5 solves/s.  The 2,100
logged optimizer steps supplied 537,600 solves of optimizer credit at the 4:1
ratio, so balanced throughput was also about 455.5 solves/s.  There were no
reported dropped games, and 3,678 solver builds reported hitting the configured
node cap.  The important negative result is that large epoch barriers both
miss the throughput target and make the nominal deadline meaningless even
when the generator's own counter briefly reports more than 600 solves/s.

At this point v5 existed only as unvalidated local code.  This run therefore
records the old architecture's end-to-end failure; it is not evidence about
v5 correctness or speed and must not be used as the final golden run.
