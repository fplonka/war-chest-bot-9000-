# v5_st_prio

## What we were trying

Whether giving the trainer a high-priority CUDA stream helps, measured in the
mature state rather than the opening. `--train-stream-priority -1` already
existed and had been recorded as making little difference, but that measurement
(`runs/v5_sweep_priority_gate`) was taken on a run with no Greedy warm-up whose
generation was limited elsewhere. In the mature state the trainer's contention
with solve waves on the card it shares is the binding cost: an optimizer step
there takes about 240 ms against the 72-101 ms it takes alone, and the same
Python thread is what drains finished solves back from Rust.

## What we learned

It is worth about 14%: 420,250 solves in the eight-minute budget against 366,852
and 371,974 for two runs of the same configuration without it. Accumulated
training wall time per reporting interval fell from roughly 8.7 seconds to 2.2,
so the mechanism is the intended one. Training kept up throughout — the run
ended 616 rows in debt against a 1,024 allowance — so the priority is not being
bought by starving the trainer of the work it owes.

The earlier note was not wrong about its own run; it was measuring a state where
something else was the limit. That is the argument for `tools/v5_steady.sh`
existing at all.

Two things that did *not* work, tested in the same harness on the same day:
twelve lanes per card ran out of card memory within ten minutes once the trainer
shared it, and 64 outstanding solves per builder instead of 32 collapsed the run
to 425 solves/s. Both had measured *better* on the generation-only stream, which
has no trainer and never runs long enough to feel the host memory that a deeper
in-flight window retains. Neither is a production setting.

## State of the project at this point

Ten lanes per card, 36 builders, 128 game actors each, 32 solves in flight per
builder, jemalloc preloaded. `--train-stream-priority -1` is now the default in
`tools/v5_gate.sh`. The workload here is harsher than a golden run's, because
the horizon payoff never helps games finish, so these numbers compare only to
each other.
