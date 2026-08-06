# arch_policy_warm — policy head + warm start (KILLED, no results)

## What happened
Fourth run of the chain, scheduled last: --iters 32 --policy 0.3 --warm 15
(half T, because warm start buys iterations and was tested where that can pay).
It was killed intentionally at t=812.7s (~13.5 min, epoch 35, rebel phase) on
2026-08-06 ~21:24 so the ladders could run immediately. Only its init snapshot
(snap_00.pt, t=307s) exists; nothing from this run is used in any ladder.

## Config
Common arch config (see arch_base/NOTES.md) plus --iters 32 --policy 0.3 --warm 15.

## Why it was killed
Planned: the run chain was arch_base -> arch_policy -> arch_dcfr ->
arch_policy_warm; the first three finished, and the last was dropped in favour
of running ladder_arch and ladder_arch_finals. Note from the session: --warm 15
was the one number borrowed from ReBeL (chosen for a converged head) rather than
measured here, so a result would have needed that caveat anyway.

## State of the project
Nothing lost; the only snapshot is the 5-minute init which adds nothing beyond
the other three runs' inits.
