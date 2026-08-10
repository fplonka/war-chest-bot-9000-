# v5_steady_base

## What we were trying

This is the first run of a new measurement, not a candidate for anything. The
two benchmarks in use had each been misleading in an expensive way: a
180-second generation-only stream that improved 31% while the thirty-minute
golden run did not move at all, and the golden run itself, which is faithful
but costs 32 minutes and varies about ±5% between identical repeats.

`tools/v5_steady.sh` is the compromise. It is a real `train.py` run with the
real trainer, but it skips the Greedy warm-up, starts from a late checkpoint,
and pins the horizon payoff at zero, so the expensive workload is present from
about ninety seconds in rather than after ten minutes. Eight minutes each.

## What we learned

The baseline reads **707.1 balanced solves per second** — far below any
golden-run average, and that is the point. A golden run spends much of its
twenty-five minutes in the cheap opening; this spends almost all of eight
minutes in the part that actually limits a long run.

A repeat of the same configuration read 764.3. Most of that gap is drain: the
first run overran its deadline by 46 seconds and the repeat by none, and the
drain counts against the headline. Compared at equal cumulative solves the two
agree within 2%, so the harness itself is quiet enough to resolve the effects
worth chasing — which the golden run was not.

The immediate result from it: giving the trainer a high-priority CUDA stream
(`--train-stream-priority -1`) completed 420,250 solves in the same eight
minutes against 366,852 and 371,974 for the two baselines, about 14% more work.
That flag had been measured before and recorded as making little difference,
but that was in a run with no warm-up whose generation was limited elsewhere.
In the mature state the trainer's contention with solve waves on the card it
shares is the binding cost, and priority is aimed exactly at it.

## State of the project at this point

Ten lanes per card, 36 builders, 128 game actors each, 32 solves in flight per
builder, jemalloc preloaded, `taskset -c 0-35`. The workload here is harsher
than the golden run's because the horizon payoff never helps games finish, so
the numbers are not comparable to a golden run's headline — only to each other.
