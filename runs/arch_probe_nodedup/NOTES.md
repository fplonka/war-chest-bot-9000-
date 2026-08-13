# arch_probe_nodedup

## What we were trying

Repeat the `arch_probe_published` trainer probe with one change: `make_batch`
no longer deduplicates config rows before the holding tower. It builds `phi`
for every config and passes an identity `inv`.

## What we learned

The dedup was the single most expensive thing in a training step and it bought
nothing. `np.unique` over the config key measured 123.6 ms against 1.2 ms for
building `phi` directly, on the same cores the generation workers want. The
substitution is exact rather than approximate: with `inv` the identity, `crow`
in `Mlp.forward` collapses to `seg // 2` and the two gathers become no-ops, so
the tower sees the inputs it saw before. Checked directly --
`phi_old[inv_old] == phi_new[inv_new]`.

End to end, against `arch_probe_published` on the same box:

| run | solves | ReBeL wall | end-to-end |
|---|---:|---:|---:|
| arch_probe_published | 38,799 | 248.5 s | 156.1 solves/s |
| arch_probe_nodedup | 42,031 | 215.5 s | 195.0 solves/s |

The 165 optimizer steps consumed 168,960 rows, or 42,240 solves of credit at
the fixed ratio of four, so balanced throughput was also 195.0 solves/s and
there was no training debt. No game was dropped. There were 256 genuine
200,000-node solver fallbacks across the four ReBeL batches. The final log and
snapshot landed at 251.3 seconds, 11.3 seconds after the nominal four-minute
process budget, confirming that the epoch barrier still cannot enforce the
deadline.

Read that as suggestive, not measured. The two runs are one sample each and
they are not perfectly matched: this one took four ReBeL batches where the
other took three, and the generator-side cumulative counter went the other way
(250.0 -> 211.3), which is not what a pure trainer speedup should do. The
exactness of the substitution and the 123.6 ms are solid; the 25% is not, and
anyone quoting it should run the pair again.

## State of the project at this point

Still far from the 1,200 solves/s goal -- roughly 195 against 1,200, so a
factor of six. `docs/GPU_ARCHITECTURE.md` argues that closing it needs the v5
replacement rather than more of this kind of change, and nothing measured here
contradicts that. This one was worth doing because it is small, exact, and
removes CPU work from the cores generation is competing for.
