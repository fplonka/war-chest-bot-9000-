# The CPU-to-GPU tree contract

Version 8 (`WCJ8`). Rules, public-tree construction, and live belief tracking
stay on the CPU. A GPU request contains one completed sparse tree, its root
beliefs, the previous solve's carried beliefs, and enough network-shape metadata
to reserve the complete device arena before admission. There is no older job
reader or compatibility adapter in the runtime.

`PackedJob::to_bytes` in `engine/src/serialize.rs` is the authoritative field
order. Counts prefix every array; integers and `f32` values are little-endian.
The version changes whenever a field changes shape or meaning.

## Runtime metadata

Each request carries depth, iteration count, whether intermediate snapshots are
kept, the four CFR parameters, the kept snapshot iteration numbers, and the
network's `dims`, which is `MODEL_TAG = [5]` and nothing else. `dims` is an
admission input rather than a shape: the arena reservation is computed from the
v5 constants and the job's own counts, in the three regions
`serialize::device_arena_bytes` and `gpu::device::arena_layout` agree on —
persistent network outputs (`f`, `g`, `P`, `join_p(P)`), then one region shared
by the trunk scratch and the CFR state, which are never live at the same time.
The trunk runs in chunks of `TRUNK_CHUNK_ROWS = 512` canonical rows, because a
row carries 37 hex tokens through eight residual blocks — about 85 KiB of
working tensors — and sizing that for a whole wave is what used to route mature
waves onto the exclusive one-job lane.

## Conventions

- Nodes are in build order, with every parent before its children.
- `node_kind` is 0 for a decision, 1 for a collapsed draw run, and 2 for a leaf.
- Config supports are local to a node and player. `cfg_off` records only their
  counts; only network-row configs are interned in `cphi`.
- A strategy cell exists only for a legal `(config, action)` pair. Node `i` owns
  `soff[i]..soff[i+1]`, and acting-player config rows are described by the
  global `legal_off` CSR.
- `u32::MAX` is the no-parent/no-row/no-transition sentinel where documented.

## Uploaded arrays

### Public tree

| array | dtype | meaning |
|---|---|---|
| `node_kind`, `node_player` | u8 | kind and acting player, one per node |
| `node_child_start`, `node_child` | u32 | public-child CSR |
| `obs_off`, `obs_start` | u32 | per-node public-observation group boundaries |
| `obs_act`, `obs_child` | u32 | actions in observation order and action-to-local-child map |
| `node_parent` | u32 | parent node, or `MAX` at the root |

### Sparse legal strategy cells

| array | dtype | meaning |
|---|---|---|
| `legal_row_of` | u32 | first config row of a decision node in `legal_off`, or `MAX` |
| `legal_off` | u32 | global CSR boundaries; one row per acting-player config |
| `legal_action` | u32 | node-local public action for every legal cell |
| `legal_child` | u32 | wave-patched public child node for every legal cell |
| `legal_trans` | u32 | child-local successor config, or `MAX` |
| `cell_row` | u32 | node-local source config for every legal cell |
| `soff` | u32 | per-node boundaries in all mutable sparse strategy arenas |

This representation has no dense legality bitset, dense successor table,
instantaneous-regret arena, or persistent normalized-average arena. Regret,
current strategy, reach-weighted strategy sum, and retained snapshots are the
only per-cell mutable state.

### Draw transitions

| array | dtype | meaning |
|---|---|---|
| `draw_off` | u32 | per-node spans in `draw_to` and `draw_p` |
| `draw_to`, `draw_p` | u32, f32 | successor config and probability per outcome |
| `draw_row_off`, `draw_row_start` | u32 | per-node CSR rows for source configs |

Consecutive private draws by the same player are collapsed into one chance
node. The idle player's support passes through unchanged.

### Reverse reach gathers

The forward reach sweep gathers into each child config so every output has one
writer.

| array | dtype | meaning |
|---|---|---|
| `rev_row_of`, `rev_start` | u32 | decision-parent gather rows |
| `rev_src`, `rev_cell` | u32 | parent config and sparse strategy cell per entry |
| `rvd_row_of`, `rvd_start` | u32 | chance-parent gather rows |
| `rvd_src`, `rvd_p` | u32, f32 | parent config and draw probability per entry |

Entry order follows the CPU oracle's action/config order. Production CUDA uses
fast FP32 kernels and cuBLAS, so correctness is bounded by numerical tolerance
and probability/index invariants rather than bitwise equality with CPU BLAS.

### Supports and arenas

| array | dtype | meaning |
|---|---|---|
| `cfg_off` | u32 | config-count prefix sum for both players at every node |
| `reach_off` | u32 | per-node reach blocks, player 0 then player 1 |
| `bfs_order`, `level_start` | u32 | level slices for fused forward/backward sweeps |

Value offsets and task lists are derived while jobs are packed into a wave; they
are not serialized per job.

### Network rows and exits

| array | dtype | meaning |
|---|---|---|
| `leaf_rows` | u32 | non-terminal leaves, first in network batch order |
| `term_leaves`, `terminal_utility` | u32, f32 | terminal leaves and acting-player utility |
| `leaf_coff`, `leaf_cidx` | u32 | network-row/player spans into interned configs |
| `snap_coff` | u32 | both-player config spans for every possible walk exit |
| `leaf_raw` | u8 | compact public rows, expanded on the GPU |
| `card_feat` | f32 | the solve-wide printed-card facts, `NTYPE * CARD_FEATS` |
| `cphi`, `cplayer` | f32, u8 | interned config counts, and which player owns each — the config encoder reads that player's five card tokens |

The serialized tail is the two root belief vectors followed by zero or more
pairs of carried-root vectors.

## Wave packing and return values

Compatible jobs are patched into one wave-global index space. Proven local
indices use the narrow path; jobs that do not fit use exact wide indices. The
host computes table bytes and the complete power-of-two-rounded FP32 arena
reservation before admission. Memory-tail jobs drain the card's ordinary lanes,
release their reusable wave buffers, and run alone; they are counted as
`oversize_routes`, never silently dropped.

One completion returns the final sparse reference strategy, per-carried-root
values for both players, and carried beliefs for each kept snapshot except the
final reference. The actor retains a separate compact `WalkTree`; no solve state
remains resident while the public game walks the result.
