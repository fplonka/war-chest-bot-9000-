# Student of Games, and where the device port stands

The engine follows Student of Games (`papers/SoG_2112.03178.pdf`), not ReBeL,
in four of the things the paper specifies. The fifth is not in, and the
throughput work has not touched it either way.

**Sound re-solving is absent.** The paper's re-solving guarantee comes from a
gadget that constrains the opponent's counterfactual values at the root of each
re-solve: the opponent may terminate and take the value the previous search
promised them, which is what stops a fresh solve from exploiting information the
previous one had. There is no gadget here. Beliefs are carried forward
correctly, but every decision builds a solver with zeroed regrets and nothing
retained, which is ReBeL-style unsafe re-solving. So the paper's bound does not
apply to this agent, whatever the four mechanisms below do.

What follows is what *is* in. The first two are held to the CPU solver by
`cuda_parity`; the mixture, the simultaneity and the averaging weight are not
pinned by any test.

## What the paper asks for, and what answers it

**A counterfactual value-and-policy network**, `f(β) = (v, p)`. The second
readout has the same shape as the first: value is a config vector dotted with a
situation vector, `v(c) = <f(c), h>`, and policy is `logit(c, a) = <f_p(c),
e(a)>`. `f_p` is a third head off the config encoder that already produces `f`
and `g`; `e(a)` describes one action — its kind, the coin slot it spends, the
three squares it names — against the board it is played on. Both land on the
`(config, action)` cells the tree already indexes for its strategy, so the
policy needs no table of its own and costs one dot product per cell.

An action's private content is *whether it is legal*, and the tree carries that
as the legal cells, which is what lets one public description serve every
config at a node. `train/test_parity.py::policy_parity` is the only thing that
pins `cfg_p` and the action encoder's place in the weight blob.

**Expansion by `π_select = ½·π_PUCT + ½·π_CFR`,** where `π_CFR` is the
average strategy rather than the current iterate, as the paper says. PUCT is a maximisation, so
its half is a point mass on the argmax and sampling the mixture is a coin flip.
Two arenas laid out like `cur` carry it: `prior` and `visits` (incremented as a
trajectory passes, which is also the paper's virtual loss). Q is re-formed from
the value arena the backward sweep just wrote. Q is divided by the opponent's reach
mass at the node — without it a node behind an unlikely opponent line looks
worthless beside its own siblings instead of being compared with them.

The prior is filled at the start of an expansion phase, not inside `grow`: a
node is expanded before the batch carrying its board vector has run. Only
expanded nodes ever need one, because a leaf has no action list and a
trajectory stops there.

**The regret update phase**: "simultaneous updates, regret-matching+, and
linearly-weighted policy averaging". `Cfr::SOG` is `alpha = inf, beta = -inf,
gamma = 1`. The gamma needs care — a sum decayed by `(t/(t+1))^gamma` weights
iterate `j` by `(j+1)^gamma`, so *linear* is 1, not the 2 `Cfr::PLUS` carries.
`step` traverses both players against one reach profile.

## The CFR loop on the device

A solve's tree, arenas and network state stay on the card, and one call is a
whole GT-CFR iteration: reach forward from the root beliefs, the network at
every leaf, the terminals, backpropagation and the regret update for both
players, the average strategy, and the expansion trajectories. The host keeps
growth, because growth is the game's rules. What crosses per iteration is the
handful of leaves the expansion sampled. See `docs/REDESIGN.md` for what that
costs and what to do about it.

### Two differences from the host loop, both deliberate

An expansion phase's simulations all run before the host grows any of them, so
two can land on the same leaf and the second is dropped. On the host each
simulation grows its leaf before the next starts, so a later one can walk
*through* what an earlier added. The visit counts a trajectory leaves behind --
the paper's virtual loss -- are what make the collision rare rather than usual.

And a device solve's random stream is read off the game's rather than drawn
from it. The host spends draws inside `sample_leaf`; if the device spent one
too, the two paths would sample different actions afterwards even when told to
grow nothing, and nothing could be compared.

### What holds it honest

`cuda_parity` has four tests. `the_network_agrees` checks the trunk and the
config encoder call by call against the CPU network. `the_cfr_loop_agrees_on_a_fixed_tree`
runs real self-play through both backends with `expand = 0`, so both solve the
tree `Solver::new` built and every number is of the same thing. With growth on
the trees part company at the first repeated leaf, so
`growth_on_the_device_produces_sane_targets` checks the scale instead, and
`a_solve_does_not_depend_on_the_round_it_rides_in` pins that a solve's answer
does not depend on who shares its round. That last one is what caught the two
real bugs of the port, both the same shape -- a slot reused without being
forgotten.

## What a solve is worth arguing about

**A solve's cost varies 26x** and cannot be predicted from its root: three
solves with an identical root support of 124 came to 203k, 1.32M and 1.73M
strategy cells. Supports grow away from the root by up to 10.7x, so
`config_cap` bounding the root bounds nothing else, and where they grow depends
on which lines the search follows -- which is not known when the solve starts.
A scheduler has to react to cost as it accrues, not sort by it up front.

**An expansion adds about twenty-one nodes**, because `k = infinity` adds every
public child and everything forced beneath it. So tree size is `expansions x 21`
and the `nodes` budget is a ceiling, not a target.
