# The CPU-to-GPU tree contract

Version 1 — frozen with the pre-CUDA plan. Tree building, rules and live
belief tracking stay on the CPU; the GPU receives a **completed flat tree**
and runs CFR iterations over it. This document is the specification every
uploaded array is written against. The arrays below are exactly the ones the
CPU solver already builds (`engine/src/search.rs`); the version bumps when
any of them changes shape or meaning.

Runtime metadata (depth, leaf rule, widths, rank, iteration count, the
snapshot list) is not part of the contract — it travels with each request.

## Conventions

- All arrays little-endian; f32 for floats; u8/u16/u32 as stated.
- `NONE = 255` is the sentinel everywhere an index may be absent.
- Nodes are numbered in build order (parents before children, which is also
  the order the reaches and values propagate).
- Configs are numbered per node per player; a "cell" is `config x action`
  for the acting player of a decision node.

## Uploaded arrays

### Tree structure, per node

| array | dtype | length | meaning |
|---|---|---|---|
| `node_kind` | u8 | nodes | 0 decision, 1 chance (draw), 2 leaf |
| `node_player` | u8 | nodes | acting player (chance: the drawing player) |
| `node_leaf` | u8 | nodes | 1 for leaves |
| `node_child_start` | u32 | nodes + 1 | children of node `i` are `node_child[start[i]..start[i+1]]` |
| `node_child` | u32 | children | public-child node ids (one per observation group) |
| `obs_start` | u32 | nodes + 1 | actions of node `i` are `obs_act[obs_start[i]..obs_start[i+1]]` |
| `obs_act` | u32 | actions | action ids in the node's action list |
| `obs_child` | u32 | actions | public child index of each action |
| `legal` | bits | cells | per-cell legality mask (`config x action`) |
| `trans` | i32 | cells | per-cell successor config index in the child, or -1 |
| `action_pays` | i8 | actions | coin slot an action spends (-1 = spends nothing) |
| `action_fdown` | u8 | actions | 1 if the spent coin goes face-down |

### Chance transitions (CSR over the drawing player's configs)

| array | dtype | length | meaning |
|---|---|---|---|
| `draw_start` | u32 | nodes + 1 | row `ci` of node `i` is `draw_to[draw_start[i]..draw_start[i+1]]` |
| `draw_to` | u32 | entries | child config index per draw outcome |
| `draw_p` | f32 | entries | draw probability per outcome (rows sum to 1) |
| `draw_steps` | u8 | nodes | how many of the game's draws a chance node covers |

Private Warrior Priest draws are chance nodes like any other; their children
carry `pending_coin` inside the config (see below).

### Config support, per node per player

| array | dtype | length | meaning |
|---|---|---|---|
| `cfg_off` | u32 | 2 * nodes + 1 | player `p` of node `i` spans `cfg_id[cfg_off[2i+p]..cfg_off[2i+p+1]]` |
| `cfg_id` | u32 | configs | global config-table id of each support member |
| `cfg_hand` | u8 x5 | configs | hand counts per slot |
| `cfg_fd` | u8 x5 | configs | face-down counts per slot |
| `cfg_pending` | i8 | configs | pending forced-play slot (-1 = none; present only while a WP forced play is open, absent at every leaf) |
| `reach_off` | u32 | nodes + 1 | node `i`'s reaches are `reach[reach_off[i]..reach_off[i] + ncfg0 + ncfg1]`, player 0 first |
| `reach` | f32 | configs | reach probability per config (unnormalised) |

### Strategy arenas

| array | dtype | length | meaning |
|---|---|---|---|
| `soff` | u32 | nodes + 1 | decision node `i`'s strategy cells are `regret[soff[i]..soff[i+1]]` |
| `regret` | f32 | cells | accumulated regret, per cell |
| `inst` | f32 | cells | instantaneous regret of the last traversal (for predictive CFR) |
| `cur` | f32 | cells | current regret-matching strategy |
| `avg` | f32 | cells x snaps | the per-iterate average strategies: snapshot `t` of node `i` is `avg[t * ncells + soff[i] .. + cells_i]` |
| `snap_iters` | u32 | snaps | the exact requested iteration numbers (0 and the final one always present; the GPU must not assume powers of two) |

### Leaves

| array | dtype | length | meaning |
|---|---|---|---|
| `leaf_rows` | u32 | non-terminal leaves | node ids in batch order |
| `term_leaves` | u32 | terminal leaves | node ids; scored from the game, not the network |
| `leaf_xpub` | f32 x PUBFEAT | leaves | the expanded public encoding per leaf (the frozen `write_public_features` layout) |
| `leaf_cidx` | u32 | leaf configs | each leaf's config ids into the global config table, player 0 then 1 |
| `leaf_coff` | u32 | 2 x leaves + 1 | spans into `leaf_cidx` |
| `terminal_utility` | f32 | terminal leaves | utility for the player to act (the opponent's reach factor is applied on GPU) |

### Config table

| array | dtype | length | meaning |
|---|---|---|---|
| `cphi` | f32 x CFEAT | distinct configs | the config feature vectors, deduplicated per solve |
| `cmap_key` | u64 | distinct configs | the dedup keys (counts + seat), so the GPU can map support members to rows |

## Returned arrays

| array | dtype | meaning |
|---|---|---|
| `leaf_vals` | f32 | per-leaf per-config counterfactual values under the traversal, in `leaf_coff` layout |
| `node_vals` | f32 | per-node per-config values (the backprop result), in `cfg_off` layout |
| `avg` | f32 | the reference strategy (CFR average at the requested iterations), same layout as uploaded |
| `conv` | f32 x2 | NashConv and zero-sum residual per solve, when requested |

Generation (TurboReBeL) additionally returns the per-iterate average snapshots
and the carried-belief roots for the next solve.

## Operating model (aligned with the CUDA plan)

The GPU service owns one device and keeps a **live set** of solves resident
(expect 100-250). A tick advances every live solve by one CFR iteration:
~6 phase kernels plus the shared-weight GEMMs. Training solves make two
round trips: trip 1 returns the targets for the carried beliefs and the
reference strategy (the walk acts on it); trip 2, when the walk leaves the
tree, returns the carried beliefs at the exit leaf. Eval solves make one
trip. The CPU keeps the game — rules, tree build, feature encoding, belief
tracking — exactly as today. All arrays above are uploaded once per solve
and stay resident until the solve finishes.

## Sizing

`engine/examples/treesize.rs` builds depth-2/3/4 trees from collected roots
(plan section 6's 1,000-root sample) with an all-zero network and reports
median/p95/p99 of nodes, leaves, action cells, configs, upload bytes, and CPU
build time. The live GPU pool is sized from p99, not the average; the tree
arrays are re-uploaded per solve, and the pool must hold the worst observed
solve plus the network's resident weights.
