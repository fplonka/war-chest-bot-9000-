# smoke-base

Six-minute tooling smoke on the golden8 player (`minutes=6 warm_minutes=2`), after restoring the tree to `7de5469` and putting `exp.py` / report / a snapshot-vs-Greedy ladder in as the only launch path. The batch `gpu_gen_data` route is gone; the stream is the only GPU generator.

Horizon stayed 0 through ReBeL. About 2,200 balanced solves/s, debt 0. Ladder (40 random-draft games vs Greedy): init 13–12–15, then +105 / +220 / **+233** (final 27–3–10). Healthy for six minutes. The first judge died on `cfr_b`, which golden8 `eval_match` does not have; that call was dropped and the ladder re-run.
