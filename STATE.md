# fastfit — optimizer step cost tracks its arithmetic

Goal: an optimizer step whose cost tracks its arithmetic, not fixed overhead.
Measured baseline (task facts): 172 ms/step at 1024 rows, 111 ms/step at 256
rows; at batch 256 the optimizer takes 8.4 s of every 10 s epoch.

Plan:
1. Baseline profile on the box: `go out=fastfit_prof minutes=4` with
   WARCHEST_TRAIN_PROFILE=1; read the per-step breakdown (sample/prepare/
   forward/backward/gpu).
2. Remove the overhead at its source (syncs, Python-side gather, unpinned
   transfers, allocations), keeping the epoch record's metrics.
3. Gate 1: steps/s at batch 256 >= 3x baseline in the same short run.
4. Gate 2: 30-min default run, 200 games vs bots/sweep3_b256 (packed from the
   existing runs/sweep3_b256 at git 9c22119; arch unchanged since), cfr=dcfr
   both seats, expected ~0.50+, solves/s vs the 200.5/s of the reference.

Box: 2x RTX 3090 24GB (cuda:0,1 search; cuda:1 trains).
Reference: runs/sweep3_b256 (batch 256, 200.5 solves/s, W675/B686/D135).
