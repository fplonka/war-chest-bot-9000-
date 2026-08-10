# v5_final_cap0_aged

We were measuring the steady-state workload that real training eventually
creates, without optimizer contention. Earlier fixed-stream controls silently
used the engine's diagnostic horizon marker payoff of 0.15. This run explicitly
used the true-game payoff of zero and the final checkpoint from
`v5_fast16_l5_warm_gate`, which had beaten both its post-Greedy initializer and
Greedy in the ladder. It otherwise used the production two-card, five-lane,
36-builder stream settings. Weights stayed fixed and no optimizer ran.

The stream completed 178,176 solves in 180.16 seconds before stopping, or
989.0 solves/s. The mature 120--180 second interval ran at about 820 solves/s,
and the final 30 seconds were about 818/s. It completed 646 games, routed 275
large searches, hit the solver node cap 1,558 times, and had no exact fallback
or dropped work. Draining the admitted tail took another 46.0 seconds, much
longer than on the post-Greedy diagnostic-payoff workload.

Average GPU use was 59.0% and 54.4%. Peak memory was 23,733 and 22,567 MiB, so
five retained lane buffers can approach the 24 GiB limit on the workload that
actually matters. This result explains why the real trainer settles near
900/s even though the historical fixed stream exceeded 1,300/s: learned play
at the true horizon payoff produces a substantially harder late-search mix.
Future speed work and memory guards should use this checkpoint/payoff pair;
the old 0.15-payoff stream remains useful only for matched historical A/Bs.
