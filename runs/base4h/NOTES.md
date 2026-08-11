# base4h — four hours of the default configuration

The first long run through the new experiment loop, and the first run of any
length on this box since `gpu_golden8` (30 minutes). Default config, 240
minutes, 8 snapshots, dump requested. Commit `c239edb`.

## What we were trying

Two things. Produce the strongest network we have and the replay dump every
offline measurement needs; and find out what happens after the half hour that is
all anyone has ever run here. `gpu_golden8` was still climbing steeply when its
clock ran out, so the honest reading of it was "we do not know where this
levels off".

## What we learned

**It levels off, and much earlier than the run.** The spread of the value
targets — how much the network's predictions actually differentiate positions —
climbs steadily for the first ~100 minutes and then stops:

| minute | 5 | 30 | 60 | 100 | 150 | 200 | 240 |
|---|---:|---:|---:|---:|---:|---:|---:|
| target spread | 0.22 | 0.36 | 0.52 | 0.58 | 0.59 | 0.60 | 0.60 |

The last 140 minutes moved it by 0.02. Whether *strength* plateaus with it is
the question the ladder answers, and that is still running as this is written —
target spread growing is not the same thing as playing better, and the two
coming apart would itself be the finding.

**Throughput halves over the run.** Instantaneous solves per second, computed
from the cumulative counter rather than the run's own (cumulative, and therefore
lagging) `solves_per_s` column:

| minutes | 0–15 | 15–60 | 60–120 | 120–180 | 180–240 |
|---|---:|---:|---:|---:|---:|
| solves/s | 1045 | 838 | 539 | 541 | 564 |

It halves by minute 60 and then holds. And it halves *while every solve is
getting cheaper*: configs per decision fall from 22.0 to 11.1 and target configs
per row from 53 to 22 over the same window. Work per solve down by half,
throughput down by half. Something that is not the per-solve cost is setting the
rate.

A first suspect was the trainer throttling generation through
`train_gen_ratio=4.0`, but `optimizer_debt` is 0 in every epoch of the run,
which is what it looks like when the trainer pays off its debt and waits. The
ratio sitting at exactly 4.00 is the trainer training precisely to the debt, not
evidence of a cap.

The stronger suspect is the CPU tree build. It is paid once per solve and does
not care how many CFR iterations follow it; `docs/PERF.md` puts it around
20–30 ms, and 36 builder threads at 25 ms is a ceiling near 1,400 solves/s —
the right order for what we see. Two observations point the same way. Both cards
sat at 31%/69% utilisation mid-run. And in the three 6-minute probes run
immediately afterwards, T=64 and T=32 produced **bit-identical** solve counts
(166,912), with T=16 slightly *lower* — three configurations whose CFR work
differs fourfold, all stopped at the same place. A 2x2 over builder threads
(36 vs 64) and iterations (64 vs 16) is queued to settle it.

If that is right it changes the plan, because it means cheap solves buy nothing
until the tree build is off the critical path, and the `iters` experiment as
conceived would have measured nothing.

## What went wrong

**The dump was lost, and the report was never written.** `check_alive` fires on
any epoch generating under 50 solves/s, and the last epoch of a run generates
zero *by design* — admission stops and the pipeline drains. It raised
`SystemExit` after the final snapshot but before the buffer dump and
`report.write`, so a 240-minute run produced no `buf.npz` and no page. The
checkpoints and `log.json` survived, and the page was regenerated afterwards
from the log. Fixed in `ad0ff20`: the liveness check is skipped once draining
has begun. Every run that set `dump_buffer` before that commit lost its dump the
same way.

## State of the project at this point

This is the first night the new `config.py` / `exp.py` / `ladder.py` loop has run
on CUDA at all. Getting to a first run needed four fixes, each of which stopped
a GPU run dead: `device="cuda"` carries no index and `torch.cuda.set_device`
rejects it; the `gpu` cargo feature had not compiled since `mc_mix` was removed;
`box.sh` built the extension without that feature; and `--features gpu` replaces
maturin's pyproject feature list rather than extending it, so the build silently
produced a cffi module with no `gpu_start` in it.

Two measurements from the same night bear on how to read this run.
`runs/solvererr_g8` finds that CFR iteration count is not what limits target
quality — the bias at T=64 is 350x below the network's own error — and that the
value network is not antisymmetric by roughly 13% of the value signal. And the
frozen-truth-set idea turned out to rank whichever network built the set, so the
ladder is still the only instrument that compares two runs.
