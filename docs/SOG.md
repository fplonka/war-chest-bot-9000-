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
Three arenas laid out like `cur` carry it: `prior`, `visits` (incremented as a
trajectory passes, which is also the paper's virtual loss), and `qval`, the
action value `backprop` already forms. Q is divided by the opponent's reach
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

**The CFR loop on the device.** Done. See below: a solve's tree, its arenas
and its network state stay on the card, and one call is a whole GT-CFR
iteration -- the sweeps, the network, the regret update, the average strategy
and the expansion trajectories. The host keeps growth, which is the game rules.

## The device

**The wall was never arithmetic.** Throughput pinned at ~380k network rows/s
whatever the thread count, with the CPUs at 13% and the cards at 20%: ~3 KB
crossed the bus per join row per iteration and 87% of it was data the card had
just produced. So a solve's board vectors, `f`, `g` and belief index stay with
the backend, a round shards by *solve* so every call of one reaches the backend
holding its state, and the pooled block and the head never leave.

That fixed the traffic and left three further walls, each smaller than the last
and each a different shape.

**A round is the unit, not a solve.** Three stages of the leaf pass launched
once per solve, because `f`, `g`, `p` and `jp` are resident per solve. A round
holds thirty-odd of them, so the cards idled between thousands of small kernels
a second: 61% of the pass was launches, at ~52 µs a call against the 5-10 µs a
launch normally takes. The solves' arrays travel as an array of device pointers
now and every stage is one launch. Two things hid inside that number.
`CudaSlice::clone` allocates and copies *on the device*, so the per-solve part
list had been duplicating every resident `f` and `g` once per CFR iteration.
And the readout took a list of cells, which the host built at twelve million
entries a round to say what the offsets in `coff` already said.

**The whole CFR loop moved.** The arenas an iteration reads — reach, values,
regrets, the iterate, the strategy sum, the action values, the visit counts —
are 4.9 to 33 MB over three real solves. Copying them for an expansion phase
that runs on the host is 153 GB/s at the target rate against roughly 50 across
two PCIe links, so the sweeps could never move on their own: either the
expansion went with them or neither did.

Both went. A solve keeps its tree, its arenas and its network state on the card,
and one call runs a whole GT-CFR iteration there: reach forward from the root
beliefs, the network at every leaf, the terminals, backpropagation and the
regret update for both players, the average strategy, and the expansion
trajectories. What crosses per iteration is the handful of leaves the expansion
sampled.

The host still grows, because growth is the game rules. Two rounds an iteration
carry that: the trunk and the config encoder over what the last growth added,
then the tree delta and the iteration. `Gate::submit_all` is what lets a thread
raise both calls of a round at once.

A level's nodes never depend on each other and neither do two solves, so one
launch covers a level of the whole round — `blockIdx.y` is the solve, and a
solve with a shallower tree simply has no work at the deeper levels. Each
solve's forty-odd arrays reach the kernels through one descriptor rather than
forty arrays of pointers; the field order is `Card::describe` against `struct
Tree` in `kernels.cu`, positional because every field is eight bytes wide.

**Fewer, larger launches.** What was left was the round doing in many launches
what it can do in one. A growth touches a tail of each of thirty arrays and a
round holds thirty-odd solves, which is a thousand stream operations a round —
more host time than every kernel of the iteration together; they travel
concatenated now, one buffer up and one kernel to place the pieces. The two
traversers were two passes over everything and are one: the beliefs and the
pooling do not depend on which seat is asking, the join and the readout run
over a batch of twice the rows rather than twice over one, and value
backpropagation runs both at once off `blockIdx.z`. The average strategy folded
into the reach sweep, which is where the reach it needs already is. And the
expansion is a warp a solve rather than a thread — a trajectory is sequential,
but each step sums an opponent's whole reach and scans a legal row.

**A slot gives its pages back.** A gate slot is reused by whichever solve takes
it next, and a solve's cost varies twenty-six fold, so a slot that kept the
largest tree it had ever served needed the worst case in every slot at once
rather than what is in flight. At 144 threads that filled a 24 GB card.
Allocation is stream-ordered, so releasing returns the pages to a pool the
other slots draw from.

### Two differences from the host loop, both deliberate

An expansion phase's simulations all run before the host grows any of them, so
two can land on the same leaf and the second is dropped. On the host each
simulation grows its leaf before the next starts, so a later one can walk
*through* what an earlier added. The visit counts a trajectory leaves behind —
the paper's virtual loss — are what makes the collision rare rather than usual.

And a device solve's random stream is read off the game's rather than drawn
from it. The host spends draws inside `sample_leaf`; if the device spent one
too, the two paths would sample different actions afterwards even when told to
grow nothing, and nothing could be compared.

### What holds it honest

`cuda_parity` has three tests. `the_network_agrees` checks the trunk and the
config encoder call by call against the CPU network, because those are the two
passes whose answers still cross the bus. `the_cfr_loop_agrees_on_a_fixed_tree`
runs real self-play through both backends with `expand = 0` — neither side
grows, so both solve the tree `Solver::new` built and every number is of the
same thing. Over eight iterations the targets agree to `9.8e-6` and the policy
to `5.2e-3`. With growth on the trees part company at the first repeated leaf,
so `growth_on_the_device_produces_sane_targets` checks the scale instead: a run
whose regrets or reaches were carried over from the solve before blows up there
long before it produces a plausible spread. That test is what caught the two
real bugs of the port, both the same shape — a slot reused without being
forgotten.

## Numbers to size against

Measured at `nodes=8192, expand=8, iters=64` over real roots.

* An expansion adds ~17 nodes, so tree size is set by `iters × expand` and the
  `nodes` budget barely binds; growth finishes near iteration 38 of 64.
* Per solve: 10,175 trunk rows, 422k join row-queries, 41.9M readout configs =
  312 GFLOP. **150 solves/s is 47 TFLOP/s** against ~71 of FP32 across two
  3090s, so it needs the tensor cores.
* The network is 93% of a solve — but the CFR sweeps are 73% of what is *left*
  once it moves, and that residual caps the host near **146 solves/s** until
  they move too. Both ends are tight at 150.
* The flat gather is *slower* than the tree walk on a CPU (0.7x). It buys
  parallelism with work and there is nobody on a host to spend it, so both
  implementations stay.
* **A solve's cost varies 26x.** Strategy cells per solve run 67k to 1.73M over
  eight real roots — 8 to 227 cells a node — and the single largest solve of the
  eight carries 40% of all the cells. `config_cap` bounds the *root* belief;
  interior supports still multiply through every draw, so a tree that reaches
  past a round boundary explodes where a shallow one does not.

  Two consequences for the device. Memory: 35 MB of cell arenas for the largest
  solve, ~390 MB for thirty-six at the mean but 1.2 GB if they are all large,
  on a 24 GB card. And load: the sweeps are O(cells), so a round holding one
  large solve and thirty-five small ones is as slow as its largest member. The
  architecture at `f5f4c05^` learned the same lesson twice — reserved lanes for
  oversized work, and returning outsized buffers rather than retaining them.
* **A solve's cost cannot be predicted from its root.** The obvious way to feed
  reserved lanes is to route by root belief size. It does not work: three solves
  with an identical root support of 124 came to 203k, 1.32M and 1.73M cells, an
  8.5x spread, while the *largest* root of the eight (278) produced middling
  trees. Supports do grow away from the root — up to 10.7x, so `config_cap`
  bounding the root bounds nothing else — but where they grow depends on which
  lines the search chooses to follow, which is not known when the solve starts.
  A scheduler therefore has to react to cost as it accrues, not sort by it up
  front.
