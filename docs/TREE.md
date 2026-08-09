# The CPU-to-GPU tree contract

Version 3 ("WCJ3"). Tree building, rules and live belief tracking stay on
the CPU; the GPU receives a **completed flat tree** and runs CFR over it.
The arrays below are what `Job::to_bytes` writes and `gpu/layout.rs`
uploads; the version bumps when any of them changes shape or meaning.

Changes from v2: the arrays nothing read are gone (`psi`, per-member config
ids/counts, `cmap_key`, `action_pays/fdown`, `draw_steps`, the uploaded
initial reach — the device seeds and propagates its own), and the reverse
(gather) transition tables are new. The GPU's forward sweep *gathers*: every
child config sums its own (parent, weight) entries in the CPU's accumulation
order, so every output has exactly one writer and no reduction crosses a
solve.

Runtime metadata (depth, iteration count, CFR parameters, the snapshot list)
travels with each request, not in the tables.

## Conventions

- Little-endian; f32 floats; u8/u32/i32 as stated; `u32::MAX` is the "none"
  sentinel for optional node indices.
- Nodes are numbered in build order (parents before children); `bfs_order` /
  `level_start` give the sweeps' level slices.
- Configs are numbered per node per player; a "cell" is `config x action`
  for the acting player of a decision node, in `soff` order.

## Uploaded arrays

### Tree structure, per node

| array | dtype | length | meaning |
|---|---|---|---|
| `node_kind` | u8 | nodes | 0 decision, 1 chance (draw), 2 leaf |
| `node_player` | u8 | nodes | acting player (chance: the drawing player) |
| `node_child_start` | u32 | nodes + 1 | children of `i` are `node_child[start[i]..start[i+1]]` |
| `node_child` | u32 | children | public-child node ids (one per observation group) |
| `node_parent` | u32 | nodes | each node's parent; `MAX` for the root |
| `obs_off` | u32 | nodes + 1 | offset of node `i`'s segment in `obs_start` |
| `obs_start` | u32 | segments | per-child action-group boundaries |
| `obs_act` | u32 | actions | action ids in the node's action list |
| `obs_child` | u32 | actions | public child index of each action |
| `legal` | bits | cells | per-cell legality mask |
| `trans` | i32 | cells | per-cell successor config in the child, or -1 |

### Chance transitions (forward CSR, for the backward sweep)

| array | dtype | length | meaning |
|---|---|---|---|
| `draw_off` | u32 | nodes + 1 | node `i`'s entries in `draw_to`/`draw_p` |
| `draw_row_off` | u32 | nodes + 1 | node `i`'s row boundaries in `draw_row_start` |
| `draw_row_start` | u32 | rows + 1 per chance node | per-parent-config entry spans |
| `draw_to` | u32 | entries | child config per draw outcome |
| `draw_p` | f32 | entries | outcome probability (rows sum to 1) |

### Reverse (gather) transitions, for the forward sweep

Rows are the *child's* me-player configs. Entry order within a row is the
CPU's accumulation order (actions in observation order, then parent configs
ascending; for chance, parent configs ascending then outcomes), which is
what keeps GPU results bit-stable and within float noise of the CPU.

| array | dtype | length | meaning |
|---|---|---|---|
| `rev_row_of` | u32 | nodes | node `j`'s first row, when its parent is a decision node (`MAX` otherwise) |
| `rev_start` | u32 | rows + 1 | entry spans |
| `rev_src` | u32 | entries | parent config index |
| `rev_cell` | u32 | entries | global strategy cell (`soff`-indexed) to weight by |
| `rvd_row_of` | u32 | nodes | as `rev_row_of`, for chance parents |
| `rvd_start` | u32 | rows + 1 | entry spans |
| `rvd_src` | u32 | entries | parent config index |
| `rvd_p` | f32 | entries | draw probability |

### Config support and arena offsets

| array | dtype | length | meaning |
|---|---|---|---|
| `cfg_off` | u32 | 2 * nodes + 1 | player `p` of node `i` has `cfg_off[2i+p+1] - cfg_off[2i+p]` configs |
| `reach_off` | u32 | nodes + 1 | node `i`'s reach block (player 0 then 1) |
| `soff` | u32 | nodes + 1 | node `i`'s strategy cells |
| `voff` | u32 | nodes + 1 | node `i`'s value block, `max(nc0, nc1)` wide (derived host-side) |
| `act_off` | u32 | nodes + 1 | cumulative action counts (derived host-side) |

### Leaves and features

| array | dtype | length | meaning |
|---|---|---|---|
| `leaf_rows` | u32 | network leaves | node ids in batch order |
| `term_leaves` | u32 | terminal leaves | node ids; scored from the game |
| `terminal_utility` | f32 | terminal leaves | utility for the leaf's acting player |
| `leaf_coff` | u32 | 2 * rows + 1 | spans into `leaf_cidx` |
| `leaf_cidx` | u32 | leaf configs | config-table rows per leaf per player |
| `leaf_xpub` | f32 x PUBFEAT | rows | the frozen public encoding per leaf |
| `cphi` | f32 x CFEAT | ncfg | config feature vectors (leaf-referenced only) |
| `ids` | u8 | NTYPE | the draft's unit ids, player-major slot order |
| `bfs_order`, `level_start` | u32 | nodes, levels + 1 | the sweeps' level slices |
| `root`, `carried` | f32 | supports | root beliefs; carried root vectors for Phase 2 |

## Returned

Trip 1: the reference strategy (`avg`, cell layout) and per-carried-root
per-player root values. Trip 2: the carried beliefs at the exit leaf, one
per kept snapshot except the final one.

## Operating model

Unchanged in spirit from the plan: the solver state lives on the device for
the life of the solve; a tick advances every live solve one step (iterate /
value / carry by its own stage); admission uploads a batch and runs the
network builds as batched GEMM chains; the walk costs two round trips per
generation solve and one per evaluation solve. `docs/GPU_ARCHITECTURE.md`
explains why this WIP operating model should be replaced for the 1,200/s
training target.
