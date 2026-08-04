# Working in this repo

A verified War Chest rules engine in Rust, plus a ReBeL agent trained on top of
it. Start with `README.md` for the layout, `docs/REBEL.md` for how the agent
works and what has been measured about it, and `docs/PERF.md` for how the
generation loop got fast.

## Every training run gets a NOTES.md

When you finish a run, write `runs/<name>/NOTES.md` before moving on. Future
readers — people and agents — will find the numbers but not the reasoning, and
the reasoning is the part that is expensive to reconstruct.

Keep it short, in plain prose, and cover:

- **What we were trying.** The question the run was meant to answer.
- **What we learned.** Including when the answer was "nothing changed" — a
  negative result is a result, and knowing something was already tried is worth
  as much as knowing it worked.
- **State of the project at this point.** What was true when the run happened,
  so the numbers can be read in context later.

Write for someone who does not already know the jargon. Say "how far each square
is from the nearest piece", not "per-hex distance channels".

## Measure, do not guess

The habit that has paid off most here: a ten-minute training run's score wanders
by about ±0.05 on its own, which is larger than most changes worth making.
Comparing two training runs will usually tell you nothing.

Prefer measurements that have no noise in them:

- `train/offline.py` fits candidate networks to a frozen dump of training data.
  Same data, same targets, so architectures can be compared exactly.
- `train/diagnose.py` asks how learnable a dump's targets are at all, without
  training anything.
- `engine/examples/solvererr.rs` measures how wrong the search's answers are, by
  solving the same position to convergence and comparing.
- `engine/examples/featstats.rs` measures the real range of a feature, which is
  how two features were found to be silently pinned at their maximum.

When you do compare builds by speed, drive both with an all-zero network so they
play identical games — `docs/PERF.md` explains why.

## Be willing to be wrong

Several confident hypotheses in this project's history were wrong, and the
measurements said so: that the network needed more capacity, that a convolution
would help on a hex board, that blending in game outcomes was necessary. Each
was recorded rather than quietly dropped, because the next person will otherwise
have the same idea. If a doc states something a new measurement contradicts,
correct the doc and say what changed.
