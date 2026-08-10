//! The CPU-to-GPU tree contract, as bytes.
//!
//! One job = one solve: everything the GPU service needs to run the CFR
//! iterations of a solved subgame, plus the beliefs its Phase 2 must value.
//! The byte format is the WCJ6 TREE.md contract (`docs/TREE.md`) and
//! the same bytes serve three consumers: the GPU service (uploaded to the
//! device), the torch CFR specification (`train/cfr_spec.py`, the executable
//! oracle for the kernels), and the oracle tests (CPU -> bytes -> CPU).
//!
//! The solver's arenas (`regret`, `inst`, `cur`, `sum_strat`, `avg`, the
//! snapshots) are **not** uploaded: the service initialises them exactly as
//! `Solver::new` does (uniform current strategy, zero regrets, the
//! reach-weighted uniform strategy seed, snapshot 0 = the uniform average).
//! Initial reach is not uploaded either: the wave seeds the root beliefs and
//! runs the same sparse forward gather before any consumer reads it.
//!
//! Layout: little-endian, `u32` counts prefix every array, `f32` floats,
//! `NONE = 255` where the solver uses it. Everything below mirrors
//! `docs/TREE.md`; see that file for the meaning of each array.

use crate::actions::Action;
use crate::net::V3Layout;
use crate::rebel::Config;
use crate::rebel::{CFEAT, GPU_ROW_BYTES, NSLOT, NTYPE, PILE_COUNTS, PUBFEAT};
use crate::search::{Cfr, Solver};
use crate::units::{write_card_features, CARD_FEATS};
use std::rc::Rc;

/// The byte format this module writes. Bump when an array changes shape or
/// meaning (docs/TREE.md "the version bumps when any of them changes shape or
/// meaning").
pub const JOB_VERSION: u32 = 6;
const MAGIC: u32 = 0x5743_4A36; // "WCJ6"

/// Runtime metadata that travels with a job (not part of the frozen tree
/// contract — it is per-request).
#[derive(Clone, Debug)]
pub struct PackedMeta {
    pub depth: usize,
    pub iters: usize,
    /// Keep the per-iterate average snapshots (generation) or not (evaluation).
    pub snapshots: bool,
    pub cfr: Cfr,
    /// Iterations the policy head's strategy is worth (0 = uniform start).
    pub warm: f32,
    /// The kept iteration numbers, in order (`Solver::snapshot_iters`).
    pub snap_iters: Vec<usize>,
    /// Exact v3 network shape. Admission uses it to account for every device
    /// scratch matrix before choosing an ordinary or exclusive lane.
    pub net_dims: Vec<usize>,
}

/// Every uploaded array of the contract. `rows` is the network batch size:
/// the non-terminal leaves, plus the decision rows the policy head reads
/// (warm start), in batch order — leaves first, so a leaf's row index is its
/// position in `leaf_rows`.
#[derive(Clone, Debug, Default)]
pub struct PackedTables {
    pub nodes: usize,
    pub children: usize,
    pub actions: usize,
    pub cells: usize,
    pub members: usize,
    pub draw_entries: usize,
    pub ncfg: usize,
    pub nleaf: usize,
    pub nterm: usize,
    pub rows: usize,
    pub n_inner: usize,
    pub leaf_configs: usize,
    pub nlevels: usize,
    pub ncells: usize,
    pub pubfeat: usize,
    pub reach_len: usize,
    // -- tree structure --
    pub node_kind: Vec<u8>,
    pub node_player: Vec<u8>,
    pub node_child_start: Vec<u32>,
    pub node_child: Vec<u32>,
    /// Offset of each node's `obs_start` segment into `obs_start` (one
    /// segment per node, `nch + 1` entries for `nch` public children).
    pub obs_off: Vec<u32>,
    pub obs_start: Vec<u32>,
    pub obs_act: Vec<u32>,
    pub obs_child: Vec<u32>,
    /// Per decision node, its first config row in `legal_off`
    /// (`u32::MAX` for leaves/chance nodes). `legal_off` is one global CSR
    /// boundary array; the other legal arrays have one entry per legal cell.
    pub legal_row_of: Vec<u32>,
    pub legal_off: Vec<u32>,
    pub legal_action: Vec<u32>,
    pub legal_child: Vec<u32>,
    pub legal_trans: Vec<u32>,
    /// Direct source-config row for each legal cell. This avoids recovering a
    /// row with a search in the hot sparse sweeps.
    pub cell_row: Vec<u32>,
    pub draw_off: Vec<u32>,
    pub draw_to: Vec<u32>,
    pub draw_p: Vec<f32>,
    /// Offset of each node's draw row-boundary segment into `draw_row_start`
    /// (one segment per chance node, `rows + 1` entries). The flat `draw_to`
    /// entries of a node split into rows through it.
    pub draw_row_off: Vec<u32>,
    pub draw_row_start: Vec<u32>,
    // -- config support --
    pub cfg_off: Vec<u32>,
    // -- arenas --
    pub reach_off: Vec<u32>,
    pub soff: Vec<u32>,
    // -- reverse (gather) transitions, for the GPU's forward sweep --
    /// Each non-root node's parent (`u32::MAX` for the root).
    pub node_parent: Vec<u32>,
    /// Per node: its first row in `rev_start` when its parent is a decision
    /// node (`u32::MAX` otherwise). A node's rows are its own me-configs.
    pub rev_row_of: Vec<u32>,
    pub rev_start: Vec<u32>,
    pub rev_src: Vec<u32>,
    pub rev_cell: Vec<u32>,
    /// The same, for chance parents: entries are (parent config, draw prob).
    pub rvd_row_of: Vec<u32>,
    pub rvd_start: Vec<u32>,
    pub rvd_src: Vec<u32>,
    pub rvd_p: Vec<f32>,
    // -- leaves --
    pub leaf_rows: Vec<u32>,
    /// Decision nodes in the network batch, after the leaves (warm start).
    pub inner_rows: Vec<u32>,
    pub term_leaves: Vec<u32>,
    pub terminal_utility: Vec<f32>,
    pub leaf_coff: Vec<u32>,
    pub leaf_cidx: Vec<u32>,
    /// Config spans for every possible walk exit: `leaf_rows` followed by
    /// `term_leaves`, two player spans per node.
    pub snap_coff: Vec<u32>,
    pub snapshot_configs: usize,
    /// Compact public rows, expanded to `PUBFEAT` by the GPU build kernels.
    pub leaf_raw: Vec<u8>,
    /// The solve-wide printed-card facts (`NTYPE * CARD_FEATS`).
    pub card_feat: Vec<f32>,
    // -- config table --
    pub cphi: Vec<f32>,
    // -- derived: BFS level order for the sweeps --
    pub bfs_order: Vec<u32>,
    pub level_start: Vec<u32>,
    // -- the draft's unit ids in player-major slot order --
    pub ids: Vec<u8>,
}

/// One solve job: the tree tables, the root beliefs, and the previous solve's
/// carried root vectors (Phase 2 must value each of them).
#[derive(Clone, Debug)]
pub struct PackedJob {
    pub meta: PackedMeta,
    pub tables: PackedTables,
    pub root: [Vec<f32>; 2],
    pub carried: Vec<[Vec<f32>; 2]>,
}

/// Local-index representation selected for a wave. Wave-global offsets stay
/// `u32`; narrow jobs store proven-local values as `u16`, while any job that
/// fails the proof takes the exact wide path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexWidth {
    Narrow,
    Wide,
}

/// Admission and scheduling cost carried by every packed solve. Counts are in
/// actual work units and bytes rather than a single job count, because roots
/// differ by orders of magnitude.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkVector {
    pub network_rows: usize,
    pub legal_cells: usize,
    pub reach_slots: usize,
    pub reverse_nonzeros: usize,
    pub table_bytes: usize,
    pub mutable_bytes: usize,
    pub carried_output_bytes: usize,
    pub levels: usize,
}

impl WorkVector {
    fn reserved_bytes(self) -> usize {
        let table_reservation = self
            .table_bytes
            .checked_next_power_of_two()
            .unwrap_or(self.table_bytes);
        self.mutable_bytes.saturating_add(table_reservation)
    }

    /// Jobs in this tail need an isolated one-job lane wave. `mutable_bytes`
    /// includes allocator power-of-two rounding, so this tests the reservation
    /// CUDA will actually attempt.
    pub fn requires_exclusive_route(self) -> bool {
        self.reserved_bytes() >= (4usize << 30)
    }

    /// A four-GiB contiguous arena needs extra headroom, but can still run
    /// beside three ordinary lanes after one helper lane is trimmed.
    pub fn requires_arena_guard_route(self) -> bool {
        self.mutable_bytes >= (4usize << 30) && !self.requires_card_exclusive_route()
    }

    /// This rarer tail needs at least six GiB by itself. Drain and trim the
    /// whole card around it rather than risking an allocator failure.
    pub fn requires_card_exclusive_route(self) -> bool {
        self.reserved_bytes() >= (6usize << 30)
    }
}

/// The compact public tree retained by a game actor while and after its GPU
/// solve runs. It contains only what the real game walk reads: public children,
/// action metadata, config-support identity, sparse policy rows, and leaf/draw
/// markers. Search states, transitions, reverse gathers, reaches, values, and
/// network scratch stay in `PackedJob` or die with the CPU builder.
#[derive(Clone)]
pub struct WalkTree {
    pub node_kind: Vec<u8>,
    pub node_player: Vec<u8>,
    pub node_child_off: Vec<u32>,
    pub node_child: Vec<u32>,
    pub node_action_off: Vec<u32>,
    pub actions: Vec<Action>,
    pub aslot: Vec<i8>,
    pub fdown: Vec<bool>,
    pub obs_child: Vec<u32>,
    pub legal_row_of: Vec<u32>,
    pub legal_off: Vec<u32>,
    pub legal_action: Vec<u32>,
    pub supports: Vec<[Rc<[Config]>; 2]>,
    pub draw_steps: Vec<u8>,
    pub soff: Vec<u32>,
}

impl WalkTree {
    pub fn from_solver(sv: &Solver<'_>) -> WalkTree {
        let nodes = sv.nodes.len();
        let mut w = WalkTree {
            node_kind: Vec::with_capacity(nodes),
            node_player: Vec::with_capacity(nodes),
            node_child_off: Vec::with_capacity(nodes + 1),
            node_child: Vec::new(),
            node_action_off: Vec::with_capacity(nodes + 1),
            actions: Vec::new(),
            aslot: Vec::new(),
            fdown: Vec::new(),
            obs_child: Vec::new(),
            legal_row_of: vec![u32::MAX; nodes],
            legal_off: vec![0],
            legal_action: Vec::new(),
            supports: Vec::with_capacity(nodes),
            draw_steps: Vec::with_capacity(nodes),
            soff: sv.soff.clone(),
        };
        for (i, n) in sv.nodes.iter().enumerate() {
            w.node_kind.push(if n.leaf {
                2
            } else if n.chance {
                1
            } else {
                0
            });
            w.node_player.push(n.player);
            w.node_child_off.push(w.node_child.len() as u32);
            w.node_child.extend(n.child.iter().map(|&x| x as u32));
            w.node_action_off.push(w.actions.len() as u32);
            w.actions.extend_from_slice(&n.acts);
            w.aslot.extend_from_slice(&n.aslot);
            w.fdown.extend_from_slice(&n.fdown);
            w.obs_child.extend(n.obs_child.iter().map(|&x| x as u32));
            if !n.leaf && !n.chance {
                w.legal_row_of[i] = (w.legal_off.len() - 1) as u32;
                let base = w.legal_action.len() as u32;
                w.legal_action.extend_from_slice(&n.legal_action);
                w.legal_off
                    .extend(n.legal_off.iter().skip(1).map(|&x| base + x));
            }
            w.supports.push(n.cfgs.clone());
            w.draw_steps.push(n.draw_steps);
        }
        w.node_child_off.push(w.node_child.len() as u32);
        w.node_action_off.push(w.actions.len() as u32);
        debug_assert_eq!(w.soff.len(), nodes + 1);
        w
    }

    #[inline]
    pub fn is_leaf(&self, node: usize) -> bool {
        self.node_kind[node] == 2
    }

    #[inline]
    pub fn is_chance(&self, node: usize) -> bool {
        self.node_kind[node] == 1
    }

    #[inline]
    pub fn children(&self, node: usize) -> &[u32] {
        &self.node_child[self.node_child_off[node] as usize..self.node_child_off[node + 1] as usize]
    }

    #[inline]
    pub fn action_range(&self, node: usize) -> std::ops::Range<usize> {
        self.node_action_off[node] as usize..self.node_action_off[node + 1] as usize
    }

    #[inline]
    pub fn legal_row(&self, node: usize, config: usize) -> std::ops::Range<usize> {
        let row = self.legal_row_of[node] as usize + config;
        self.legal_off[row] as usize..self.legal_off[row + 1] as usize
    }

    #[inline]
    pub fn child_for_action(&self, node: usize, action: usize) -> usize {
        let aa = self.node_action_off[node] as usize + action;
        let local = self.obs_child[aa] as usize;
        self.children(node)[local] as usize
    }

    /// Bytes owned by the actor-side walk, excluding shared `Rc` config
    /// supports. Used by the host byte-credit admission gate.
    pub fn owned_bytes(&self) -> usize {
        self.node_kind.len()
            + self.node_player.len()
            + 4 * self.node_child_off.len()
            + 4 * self.node_child.len()
            + 4 * self.node_action_off.len()
            + std::mem::size_of::<Action>() * self.actions.len()
            + self.aslot.len()
            + self.fdown.len()
            + 4 * self.obs_child.len()
            + 4 * self.legal_row_of.len()
            + 4 * self.legal_off.len()
            + 4 * self.legal_action.len()
            + self.draw_steps.len()
            + 4 * self.soff.len()
    }
}

// ---------------------------------------------------------------- building

impl PackedJob {
    /// Serialize a freshly built solver (before any iteration) into a job.
    ///
    /// `carried` are the probability vectors over the root support the next
    /// solve's Phase 2 must value: the previous solve's carried beliefs, or
    /// just the live belief for the first level.
    pub fn from_solver(sv: &Solver, carried: &[[Vec<f32>; 2]]) -> PackedJob {
        let _t = crate::timed!(SERIAL);
        let tables = PackedTables::from_solver(sv);
        let root = [sv.root_belief[0].p.clone(), sv.root_belief[1].p.clone()];
        PackedJob {
            meta: PackedMeta {
                depth: sv.cfg.depth,
                iters: sv.cfg.iters,
                snapshots: sv.cfg.snapshots,
                cfr: sv.cfg.cfr,
                warm: sv.cfg.warm,
                snap_iters: sv.snap_list.clone(),
                net_dims: sv.network_dims().to_vec(),
            },
            tables,
            root,
            carried: carried.to_vec(),
        }
    }

    /// Produce both builder outputs before the full solver is released.
    pub fn from_solver_with_walk(
        sv: &Solver<'_>,
        carried: &[[Vec<f32>; 2]],
    ) -> (PackedJob, WalkTree) {
        let walk = WalkTree::from_solver(sv);
        (PackedJob::from_solver(sv, carried), walk)
    }

    /// Whether every value selected for local 16-bit storage fits without
    /// truncation. `u16::MAX` is reserved for `NO_TRANS`; offsets remain wide.
    pub fn index_width(&self) -> IndexWidth {
        const MAX: u32 = u16::MAX as u32 - 1;
        let t = &self.tables;
        let local_counts_fit = t.cfg_off.windows(2).all(|w| w[1] - w[0] <= MAX);
        let trans_fit = t
            .legal_trans
            .iter()
            .all(|&x| x == crate::search::NO_TRANS || x <= MAX);
        if t.nodes <= MAX as usize
            && t.legal_action.iter().all(|&x| x <= MAX)
            && t.cell_row.iter().all(|&x| x <= MAX)
            && t.rev_src.iter().all(|&x| x <= MAX)
            && t.rvd_src.iter().all(|&x| x <= MAX)
            && t.leaf_cidx.iter().all(|&x| x <= MAX)
            && trans_fit
            && local_counts_fit
        {
            IndexWidth::Narrow
        } else {
            IndexWidth::Wide
        }
    }

    pub fn work(&self) -> WorkVector {
        let t = &self.tables;
        let reach = device_reach_slots(t);
        let vals = device_value_layout(t).map_or(usize::MAX, |(_, n)| n);
        let root_configs = (t.cfg_off[2] - t.cfg_off[0]) as usize;
        let snapshots = self.meta.snap_iters.len();
        let fp32_output = t
            .ncells
            .saturating_add(self.carried.len().saturating_mul(root_configs))
            .saturating_mul(std::mem::size_of::<f32>());
        let fp16_carry = snapshots
            .saturating_sub(1)
            .saturating_mul(t.snapshot_configs)
            .saturating_mul(std::mem::size_of::<u16>());
        WorkVector {
            network_rows: t.rows,
            legal_cells: t.ncells,
            reach_slots: reach,
            reverse_nonzeros: t.rev_src.len() + t.rvd_src.len(),
            table_bytes: t.owned_bytes().saturating_add(
                2usize
                    .saturating_mul(t.nodes)
                    .saturating_sub(t.reach_off.len())
                    .saturating_mul(std::mem::size_of::<u32>()),
            ) + 4 * (self.root[0].len() + self.root[1].len())
                + 4 * self.carried.len() * root_configs,
            // Exact one-job device arena, including network scratch and the
            // allocator's power-of-two growth policy.
            mutable_bytes: device_arena_bytes(self, vals, reach),
            carried_output_bytes: fp32_output.saturating_add(fp16_carry),
            levels: t.nlevels,
        }
    }
}

/// Exact one-job reservation for the contiguous mixed-width arena. Wave packing is
/// subadditive for its max-sized scratch blocks, so per-job admission is a
/// conservative bound for a multi-job wave. Keep this in lockstep with
/// `gpu::device::arena_layout`.
fn device_reach_slots(t: &PackedTables) -> usize {
    if t.nodes == 0 {
        return 0;
    }
    let nc =
        |node: usize, p: usize| (t.cfg_off[2 * node + p + 1] - t.cfg_off[2 * node + p]) as usize;
    let mut slots = nc(0, 0).saturating_add(nc(0, 1));
    for node in 1..t.nodes {
        let parent = t.node_parent[node] as usize;
        slots = slots.saturating_add(nc(node, t.node_player[parent] as usize));
    }
    slots
}

/// Values are evaluated for one traverser at a time, so the two players can
/// reuse the same arena span. Across a chance edge only the drawing player's
/// private configuration changes; the other player's value vector is exactly
/// its sole child's vector and aliases it instead of being copied.
pub(crate) fn device_value_layout(t: &PackedTables) -> Option<([Vec<u32>; 2], usize)> {
    let mut base: [Vec<u32>; 2] = std::array::from_fn(|_| vec![0; t.nodes]);
    let mut lengths = [0usize; 2];
    for p in 0..2 {
        let mut at = 0usize;
        let mut ready = vec![false; t.nodes];
        for &node_u in t.bfs_order.iter().rev() {
            let node = node_u as usize;
            if node >= t.nodes {
                return None;
            }
            if t.node_kind[node] == 1 && t.node_player[node] as usize != p {
                let lo = *t.node_child_start.get(node)? as usize;
                let hi = *t.node_child_start.get(node + 1)? as usize;
                if hi != lo + 1 {
                    return None;
                }
                let child = *t.node_child.get(lo)? as usize;
                if child >= t.nodes || !ready[child] {
                    return None;
                }
                base[p][node] = base[p][child];
            } else {
                base[p][node] = u32::try_from(at).ok()?;
                let n = (t.cfg_off[2 * node + p + 1] - t.cfg_off[2 * node + p]) as usize;
                at = at.checked_add(n)?;
            }
            ready[node] = true;
        }
        if ready.iter().any(|&x| !x) {
            return None;
        }
        lengths[p] = at;
    }
    Some((base, lengths[0].max(lengths[1])))
}

fn device_arena_bytes(job: &PackedJob, vals: usize, reach: usize) -> usize {
    let t = &job.tables;
    let Ok(l) = V3Layout::new(&job.meta.net_dims) else {
        return usize::MAX;
    };
    let (rows, cfgs, cells) = (t.rows, t.ncfg, t.ncells);
    let roots = job
        .carried
        .len()
        .saturating_mul(job.root[0].len().saturating_add(job.root[1].len()));
    let carry_snaps = if job.meta.snapshots {
        job.meta.snap_iters.len().saturating_sub(1)
    } else {
        0
    };
    let (pubw, _, cardw, slotw) = l.widths();
    let max_pub = pubw
        .into_iter()
        .chain([l.xdim(), l.head_in])
        .max()
        .unwrap_or(1);
    let slot_max = slotw
        .into_iter()
        .chain([l.hfeat(), l.dg])
        .max()
        .unwrap_or(1);
    let card_max = cardw.into_iter().max().unwrap_or(l.de);
    let mul = |a: usize, b: usize| a.saturating_mul(b);
    let bh = mul(rows, max_pub)
        .max(mul(mul(cfgs, NSLOT), slot_max))
        .max(mul(NTYPE, card_max))
        .max(mul(mul(rows, NTYPE), l.de))
        .max(mul(cfgs, l.dg));
    let bg = mul(NTYPE, CARD_FEATS)
        .max(mul(mul(rows, NTYPE), PILE_COUNTS))
        .max(mul(mul(rows, NTYPE), l.de).saturating_add(mul(NTYPE, l.de)))
        .max(mul(mul(cfgs, NSLOT), l.hfeat()))
        .max(mul(cfgs, l.rank + 1));
    let h_stride = std::iter::once(l.head_in)
        .chain(l.hmlp.iter().map(|x| x.o))
        .max()
        .unwrap_or(l.head_in);
    let fast_head = std::env::var_os("WARCHEST_GPU_PRECISE_GEMM").is_none() && l.hmlp.is_empty();
    let sizes = [
        reach,
        reach.max(vals),
        vals,
        cells,
        cells,
        cells,
        cells,
        mul(NTYPE, l.de),
        mul(cfgs, l.dg),
        mul(cfgs, l.rank + 1),
        if fast_head {
            mul(rows, l.head_in).div_ceil(2)
        } else {
            mul(rows, l.head_in)
        },
        if fast_head {
            mul(mul(rows, 2), l.dg).div_ceil(2)
        } else {
            mul(mul(rows, 2), l.dg)
        },
        if fast_head {
            mul(rows, l.head_in).div_ceil(2)
        } else {
            mul(rows, h_stride)
        },
        if fast_head {
            mul(rows, l.head_in).div_ceil(2)
        } else {
            mul(rows, h_stride)
        },
        mul(rows, l.rank),
        roots,
        mul(carry_snaps, t.snapshot_configs).div_ceil(2),
        mul(rows, l.xdim()),
        bh,
        bh,
        bg,
    ];
    let mut floats = 0usize;
    for n in sizes {
        floats = floats.saturating_add(31) & !31usize;
        floats = floats.saturating_add(n);
    }
    floats
        .checked_next_power_of_two()
        .unwrap_or(floats)
        .saturating_mul(std::mem::size_of::<f32>())
}

impl PackedTables {
    pub fn owned_bytes(&self) -> usize {
        let u8s =
            self.node_kind.len() + self.node_player.len() + self.leaf_raw.len() + self.ids.len();
        let u32s = self.node_child_start.len()
            + self.node_child.len()
            + self.obs_off.len()
            + self.obs_start.len()
            + self.obs_act.len()
            + self.obs_child.len()
            + self.legal_row_of.len()
            + self.legal_off.len()
            + self.legal_action.len()
            + self.legal_child.len()
            + self.legal_trans.len()
            + self.cell_row.len()
            + self.draw_off.len()
            + self.draw_to.len()
            + self.draw_row_off.len()
            + self.draw_row_start.len()
            + self.cfg_off.len()
            + self.reach_off.len()
            + self.soff.len()
            + self.node_parent.len()
            + self.rev_row_of.len()
            + self.rev_start.len()
            + self.rev_src.len()
            + self.rev_cell.len()
            + self.rvd_row_of.len()
            + self.rvd_start.len()
            + self.rvd_src.len()
            + self.leaf_rows.len()
            + self.inner_rows.len()
            + self.term_leaves.len()
            + self.leaf_coff.len()
            + self.leaf_cidx.len()
            + self.snap_coff.len()
            + self.bfs_order.len()
            + self.level_start.len();
        let f32s = self.draw_p.len()
            + self.rvd_p.len()
            + self.terminal_utility.len()
            + self.card_feat.len()
            + self.cphi.len();
        u8s + 4 * (u32s + f32s)
    }

    fn from_solver(sv: &Solver) -> PackedTables {
        let nodes = sv.nodes.len();
        let mut t = PackedTables {
            nodes,
            pubfeat: sv.pubfeat,
            ncfg: sv.ncfg,
            cphi: sv.cphi[..sv.ncfg * CFEAT].to_vec(),
            ..Default::default()
        };
        // Config support counts per node per player. Only the *leaf* configs
        // are interned into the feature table (`push_row` did that during the
        // build); inner supports travel as counts and local transitions,
        // which is all the sweeps read.
        let mut cfg_off = Vec::with_capacity(2 * nodes + 1);
        let mut cfg_at = 0u32;
        for i in 0..nodes {
            let n = &sv.nodes[i];
            cfg_off.push(cfg_at);
            cfg_at += n.cfgs[0].len() as u32;
            cfg_off.push(cfg_at);
            cfg_at += n.cfgs[1].len() as u32;
        }
        cfg_off.push(cfg_at);
        t.cfg_off = cfg_off;

        // Node arrays, flat CSRs, and the reverse (gather) transition tables.
        //
        // The reverse tables exist for the GPU's forward reach sweep: pushing
        // mass parent-to-child makes writers collide (several parent configs
        // reach the same child config), while gathering gives every output
        // exactly one writer and a fixed summation order. The entry order per
        // output is exactly the CPU's accumulation order — actions in
        // observation order, then parent configs ascending (decision), parent
        // configs ascending then draw outcomes (chance) — so the two sides
        // sum in the same sequence.
        let mut child_start = Vec::with_capacity(nodes + 1);
        let mut obs_off = Vec::with_capacity(nodes + 1);
        let mut draw_off = Vec::with_capacity(nodes + 1);
        let mut reach_off = Vec::with_capacity(nodes + 1);
        let mut soff = Vec::with_capacity(nodes + 1);
        let mut obs_start = Vec::new();
        let mut reach_at = 0u32;
        t.node_parent = vec![u32::MAX; nodes];
        t.rev_row_of = vec![u32::MAX; nodes];
        t.rvd_row_of = vec![u32::MAX; nodes];
        t.legal_row_of = vec![u32::MAX; nodes];
        t.legal_off.push(0);
        t.rev_start.push(0);
        t.rvd_start.push(0);
        // Scratch: per-target-config entry lists for the node being reversed.
        let mut gather: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut gather_p: Vec<Vec<(u32, f32)>> = Vec::new();
        for i in 0..nodes {
            let n = &sv.nodes[i];
            child_start.push(t.node_child.len() as u32);
            t.node_child.extend(n.child.iter().map(|&c| c as u32));
            for &c in &n.child {
                t.node_parent[c] = i as u32;
            }
            t.node_kind.push(if n.leaf {
                2
            } else if n.chance {
                1
            } else {
                0
            });
            t.node_player.push(n.player);
            obs_off.push(obs_start.len() as u32);
            obs_start.extend_from_slice(&n.obs_start);
            t.obs_act.extend_from_slice(&n.obs_act);
            t.obs_child.extend(n.obs_child.iter().map(|&c| c as u32));
            draw_off.push(t.draw_to.len() as u32);
            t.draw_to.extend_from_slice(&n.draw.to);
            t.draw_p.extend_from_slice(&n.draw.p);
            t.draw_row_off.push(t.draw_row_start.len() as u32);
            t.draw_row_start.extend_from_slice(&n.draw.start);
            if n.chance {
                debug_assert_eq!(n.draw.start.len(), n.draw.rows() + 1);
                // Reverse the draw CSR for the one public child: per child
                // config, the (parent config, probability) entries.
                let ch = n.child[0];
                let me = n.player as usize;
                let m = sv.nodes[ch].cfgs[me].len();
                gather_p.iter_mut().for_each(|v| v.clear());
                gather_p.resize(m.max(gather_p.len()), Vec::new());
                let nme = n.cfgs[me].len();
                for ci in 0..nme {
                    let (to, pr) = n.draw.row(ci);
                    for k in 0..to.len() {
                        gather_p[to[k] as usize].push((ci as u32, pr[k]));
                    }
                }
                t.rvd_row_of[ch] = (t.rvd_start.len() - 1) as u32;
                for tv in gather_p[..m].iter() {
                    for &(src, p) in tv {
                        t.rvd_src.push(src);
                        t.rvd_p.push(p);
                    }
                    t.rvd_start.push(t.rvd_src.len() as u32);
                }
            }
            reach_off.push(reach_at);
            let (c0, c1) = (n.cfgs[0].len() as u32, n.cfgs[1].len() as u32);
            reach_at += c0 + c1;
            soff.push(sv.soff[i]);
            if !n.leaf && !n.chance {
                let me = n.player as usize;
                t.legal_row_of[i] = (t.legal_off.len() - 1) as u32;
                let cell_base = t.legal_action.len() as u32;
                for &off in &n.legal_off[1..] {
                    t.legal_off.push(cell_base + off);
                }
                t.legal_action.extend_from_slice(&n.legal_action);
                t.legal_child.extend_from_slice(&n.legal_child);
                t.legal_trans.extend_from_slice(&n.legal_trans);
                t.cell_row.extend_from_slice(&n.cell_row);
                // Reverse the strategy transitions per public child: per
                // child config, the (parent config, strategy cell) entries.
                for ch_i in 0..n.child.len() {
                    let ch = n.child[ch_i];
                    let m = sv.nodes[ch].cfgs[me].len();
                    gather.iter_mut().for_each(|v| v.clear());
                    gather.resize(m.max(gather.len()), Vec::new());
                    let (s0, s1) = (n.obs_start[ch_i] as usize, n.obs_start[ch_i + 1] as usize);
                    for &au in &n.obs_act[s0..s1] {
                        let a = au as usize;
                        for &cell_u in
                            &n.action_cell[n.action_off[a] as usize..n.action_off[a + 1] as usize]
                        {
                            let cell = cell_u as usize;
                            let tr = n.legal_trans[cell];
                            if tr == crate::search::NO_TRANS {
                                continue;
                            }
                            let c = n.cell_row[cell] as usize;
                            gather[tr as usize].push((c as u32, sv.soff[i] + cell_u));
                        }
                    }
                    t.rev_row_of[ch] = (t.rev_start.len() - 1) as u32;
                    for tv in gather[..m].iter() {
                        for &(src, cell) in tv {
                            t.rev_src.push(src);
                            t.rev_cell.push(cell);
                        }
                        t.rev_start.push(t.rev_src.len() as u32);
                    }
                }
            }
        }
        child_start.push(t.node_child.len() as u32);
        obs_off.push(obs_start.len() as u32);
        draw_off.push(t.draw_to.len() as u32);
        t.draw_row_off.push(t.draw_row_start.len() as u32);
        reach_off.push(reach_at);
        soff.push(sv.soff[nodes]);
        t.node_child_start = child_start;
        t.obs_off = obs_off;
        t.obs_start = obs_start;
        t.draw_off = draw_off;
        t.reach_off = reach_off;
        t.soff = soff;
        t.cells = sv.ncells;
        t.ncells = sv.ncells;
        t.actions = t.obs_act.len();
        t.children = t.node_child.len();
        t.draw_entries = t.draw_to.len();
        t.reach_len = reach_at as usize;
        t.ids = sv.ids.to_vec();

        // Leaves.
        t.nleaf = sv.leaf_rows.len();
        t.nterm = sv.term_leaves.len();
        t.leaf_rows = sv.leaf_rows.iter().map(|&i| i as u32).collect();
        t.inner_rows = sv.inner_rows.iter().map(|&i| i as u32).collect();
        t.term_leaves = sv.term_leaves.iter().map(|&i| i as u32).collect();
        t.terminal_utility = sv
            .term_leaves
            .iter()
            .map(|&i| sv.nodes[i].s.utility(sv.nodes[i].player as usize))
            .collect();
        t.rows = sv.leaf_rows.len() + sv.inner_rows.len();
        t.n_inner = sv.inner_rows.len();
        // The batch sentinel is pushed by `ensure_leaf_batch`, which runs on
        // the first network query; a fresh solver has not queried yet, so the
        // serializer appends the sentinel to make the tables canonical.
        let mut leaf_coff = sv.leaf_coff.clone();
        if leaf_coff.len() == 2 * t.rows {
            leaf_coff.push(sv.leaf_cidx.len() as u32);
        }
        debug_assert_eq!(
            leaf_coff.len(),
            2 * t.rows + 1,
            "leaf_coff must be canonical"
        );
        t.leaf_coff = leaf_coff;
        t.leaf_cidx = sv.leaf_cidx.clone();
        t.leaf_configs = sv.leaf_cidx.len();
        let mut snap_at = 0u32;
        for &i in sv.leaf_rows.iter().chain(&sv.term_leaves) {
            for p in 0..2 {
                t.snap_coff.push(snap_at);
                snap_at += sv.nodes[i].cfgs[p].len() as u32;
            }
        }
        t.snap_coff.push(snap_at);
        t.snapshot_configs = snap_at as usize;
        let raw = t.rows * GPU_ROW_BYTES;
        t.leaf_raw = sv.gpu_rows[..raw].to_vec();
        t.card_feat.resize(NTYPE * CARD_FEATS, 0.0);
        for (slot, &id) in sv.ids.iter().enumerate() {
            write_card_features(
                id,
                &mut t.card_feat[slot * CARD_FEATS..(slot + 1) * CARD_FEATS],
            );
        }

        // BFS levels: node 0's children, then their children, ...
        let mut bfs: Vec<u32> = Vec::with_capacity(nodes);
        let mut level_start = vec![0u32];
        let mut frontier = vec![0u32];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for &i in &frontier {
                bfs.push(i);
                for &c in &t.node_child[t.node_child_start[i as usize] as usize
                    ..t.node_child_start[i as usize + 1] as usize]
                {
                    next.push(c);
                }
            }
            frontier = next;
            level_start.push(bfs.len() as u32);
        }
        t.bfs_order = bfs;
        t.nlevels = level_start.len() - 1;
        t.level_start = level_start;
        #[cfg(feature = "prof")]
        {
            use crate::prof::{
                add, CH_CHILD_CHANCE, CH_CHILD_DECISION, CH_CHILD_LEAF, CH_PARENT_CHANCE,
                CH_PARENT_DECISION, CH_PARENT_ROOT, NCFG, S_CELLS, S_DEC_REV_NNZ, S_DRAW_NNZ,
                S_DRAW_REV_NNZ, S_LEVELS, S_REACH, S_ROWS, S_SNAP_CFG,
            };
            add(&NCFG, t.ncfg as u64);
            add(&S_ROWS, t.rows as u64);
            add(&S_LEVELS, t.nlevels as u64);
            add(&S_REACH, t.reach_len as u64);
            add(&S_CELLS, t.ncells as u64);
            add(&S_DRAW_NNZ, t.draw_to.len() as u64);
            add(&S_DEC_REV_NNZ, t.rev_src.len() as u64);
            add(&S_DRAW_REV_NNZ, t.rvd_src.len() as u64);
            add(&S_SNAP_CFG, t.snapshot_configs as u64);
            for i in 0..nodes {
                if t.node_kind[i] != 1 {
                    continue;
                }
                let parent = t.node_parent[i];
                if parent == u32::MAX {
                    add(&CH_PARENT_ROOT, 1);
                } else if t.node_kind[parent as usize] == 1 {
                    add(&CH_PARENT_CHANCE, 1);
                } else {
                    add(&CH_PARENT_DECISION, 1);
                }
                let child = t.node_child[t.node_child_start[i] as usize] as usize;
                match t.node_kind[child] {
                    2 => add(&CH_CHILD_LEAF, 1),
                    1 => add(&CH_CHILD_CHANCE, 1),
                    _ => add(&CH_CHILD_DECISION, 1),
                }
            }
        }
        t
    }
}

// ---------------------------------------------------------------- bytes

struct W {
    b: Vec<u8>,
}

impl W {
    fn new() -> W {
        W { b: Vec::new() }
    }
    fn u32(&mut self, v: u32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.b.extend_from_slice(&v.to_le_bytes());
    }
    fn u8s(&mut self, v: &[u8]) {
        self.u32(v.len() as u32);
        self.b.extend_from_slice(v);
    }
    fn u32s(&mut self, v: &[u32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.u32(x);
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.f32(x);
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> R<'a> {
        R { b, at: 0 }
    }
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        if self.at + n > self.b.len() {
            return Err(format!("job truncated reading {what}"));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }
    fn f32(&mut self, what: &str) -> Result<f32, String> {
        let s = self.take(4, what)?;
        Ok(f32::from_le_bytes(s.try_into().unwrap()))
    }
    fn u8s(&mut self, what: &str) -> Result<Vec<u8>, String> {
        let n = self.u32(what)? as usize;
        Ok(self.take(n, what)?.to_vec())
    }
    fn u32s(&mut self, what: &str) -> Result<Vec<u32>, String> {
        let n = self.u32(what)? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.u32(what)?);
        }
        Ok(v)
    }
    fn f32s(&mut self, what: &str) -> Result<Vec<f32>, String> {
        let n = self.u32(what)? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.f32(what)?);
        }
        Ok(v)
    }
    fn done(&self) -> bool {
        self.at == self.b.len()
    }
}

/// One array section's count field, checked against the caller's expectation.
fn rd_check(got: usize, want: usize, what: &str) -> Result<(), String> {
    if got != want {
        return Err(format!("job {what}: count {got} != expected {want}"));
    }
    Ok(())
}

impl PackedJob {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = W::new();
        w.u32(MAGIC);
        w.u32(JOB_VERSION);
        let m = &self.meta;
        w.u32(m.depth as u32);
        w.u32(m.iters as u32);
        w.b.push(m.snapshots as u8);
        w.f32(m.cfr.alpha);
        w.f32(m.cfr.beta);
        w.f32(m.cfr.gamma);
        w.f32(m.cfr.predict);
        w.f32(m.warm);
        w.u32s(&m.snap_iters.iter().map(|&x| x as u32).collect::<Vec<_>>());
        w.u32s(&m.net_dims.iter().map(|&x| x as u32).collect::<Vec<_>>());
        let t = &self.tables;
        w.u32(t.nodes as u32);
        w.u32(t.ncfg as u32);
        w.u32(t.rows as u32);
        w.u32(t.pubfeat as u32);
        w.u32(t.ncells as u32);
        // tree
        w.u8s(&t.node_kind);
        w.u8s(&t.node_player);
        w.u32s(&t.node_child_start);
        w.u32s(&t.node_child);
        w.u32s(&t.obs_off);
        w.u32s(&t.obs_start);
        w.u32s(&t.obs_act);
        w.u32s(&t.obs_child);
        w.u32s(&t.legal_row_of);
        w.u32s(&t.legal_off);
        w.u32s(&t.legal_action);
        w.u32s(&t.legal_child);
        w.u32s(&t.legal_trans);
        w.u32s(&t.cell_row);
        w.u32s(&t.draw_off);
        w.u32s(&t.draw_to);
        w.f32s(&t.draw_p);
        w.u32s(&t.draw_row_off);
        w.u32s(&t.draw_row_start);
        w.u32s(&t.cfg_off);
        w.u32s(&t.reach_off);
        w.u32s(&t.soff);
        w.u32s(&t.node_parent);
        w.u32s(&t.rev_row_of);
        w.u32s(&t.rev_start);
        w.u32s(&t.rev_src);
        w.u32s(&t.rev_cell);
        w.u32s(&t.rvd_row_of);
        w.u32s(&t.rvd_start);
        w.u32s(&t.rvd_src);
        w.f32s(&t.rvd_p);
        // leaves
        w.u32s(&t.leaf_rows);
        w.u32s(&t.inner_rows);
        w.u32s(&t.term_leaves);
        w.f32s(&t.terminal_utility);
        w.u32s(&t.leaf_coff);
        w.u32s(&t.leaf_cidx);
        w.u32s(&t.snap_coff);
        w.u8s(&t.leaf_raw);
        w.f32s(&t.card_feat);
        // config table
        w.f32s(&t.cphi);
        // levels
        w.u32s(&t.bfs_order);
        w.u32s(&t.level_start);
        w.u8s(&t.ids);
        // beliefs
        w.f32s(&self.root[0]);
        w.f32s(&self.root[1]);
        w.u32(self.carried.len() as u32);
        for r in &self.carried {
            w.f32s(&r[0]);
            w.f32s(&r[1]);
        }
        w.b
    }

    pub fn from_bytes(b: &[u8]) -> Result<PackedJob, String> {
        let mut r = R::new(b);
        if r.u32("magic")? != MAGIC {
            return Err("job: bad magic".into());
        }
        if r.u32("version")? != JOB_VERSION {
            return Err(format!("job: unsupported version"));
        }
        let meta = PackedMeta {
            depth: r.u32("depth")? as usize,
            iters: r.u32("iters")? as usize,
            snapshots: r.take(1, "snapshots")?[0] != 0,
            cfr: Cfr {
                alpha: r.f32("alpha")?,
                beta: r.f32("beta")?,
                gamma: r.f32("gamma")?,
                predict: r.f32("predict")?,
            },
            warm: r.f32("warm")?,
            snap_iters: r.u32s("snap_iters")?.iter().map(|&x| x as usize).collect(),
            net_dims: r.u32s("net_dims")?.iter().map(|&x| x as usize).collect(),
        };
        let nodes = r.u32("nodes")? as usize;
        let ncfg = r.u32("ncfg")? as usize;
        let rows = r.u32("rows")? as usize;
        let pubfeat = r.u32("pubfeat")? as usize;
        let ncells = r.u32("ncells")? as usize;
        let mut t = PackedTables {
            nodes,
            ncfg,
            rows,
            pubfeat,
            ncells,
            ..Default::default()
        };
        t.node_kind = r.u8s("node_kind")?;
        t.node_player = r.u8s("node_player")?;
        t.node_child_start = r.u32s("node_child_start")?;
        t.node_child = r.u32s("node_child")?;
        t.obs_off = r.u32s("obs_off")?;
        t.obs_start = r.u32s("obs_start")?;
        t.obs_act = r.u32s("obs_act")?;
        t.obs_child = r.u32s("obs_child")?;
        t.legal_row_of = r.u32s("legal_row_of")?;
        t.legal_off = r.u32s("legal_off")?;
        t.legal_action = r.u32s("legal_action")?;
        t.legal_child = r.u32s("legal_child")?;
        t.legal_trans = r.u32s("legal_trans")?;
        t.cell_row = r.u32s("cell_row")?;
        t.draw_off = r.u32s("draw_off")?;
        t.draw_to = r.u32s("draw_to")?;
        t.draw_p = r.f32s("draw_p")?;
        t.draw_row_off = r.u32s("draw_row_off")?;
        t.draw_row_start = r.u32s("draw_row_start")?;
        t.cfg_off = r.u32s("cfg_off")?;
        t.reach_off = r.u32s("reach_off")?;
        t.soff = r.u32s("soff")?;
        t.node_parent = r.u32s("node_parent")?;
        t.rev_row_of = r.u32s("rev_row_of")?;
        t.rev_start = r.u32s("rev_start")?;
        t.rev_src = r.u32s("rev_src")?;
        t.rev_cell = r.u32s("rev_cell")?;
        t.rvd_row_of = r.u32s("rvd_row_of")?;
        t.rvd_start = r.u32s("rvd_start")?;
        t.rvd_src = r.u32s("rvd_src")?;
        t.rvd_p = r.f32s("rvd_p")?;
        t.leaf_rows = r.u32s("leaf_rows")?;
        t.inner_rows = r.u32s("inner_rows")?;
        t.term_leaves = r.u32s("term_leaves")?;
        t.terminal_utility = r.f32s("terminal_utility")?;
        t.leaf_coff = r.u32s("leaf_coff")?;
        t.leaf_cidx = r.u32s("leaf_cidx")?;
        t.snap_coff = r.u32s("snap_coff")?;
        t.leaf_raw = r.u8s("leaf_raw")?;
        t.card_feat = r.f32s("card_feat")?;
        t.cphi = r.f32s("cphi")?;
        t.bfs_order = r.u32s("bfs_order")?;
        t.level_start = r.u32s("level_start")?;
        t.ids = r.u8s("ids")?;
        // sanity checks
        rd_check(t.node_kind.len(), nodes, "node_kind")?;
        rd_check(t.node_player.len(), nodes, "node_player")?;
        rd_check(t.legal_row_of.len(), nodes, "legal_row_of")?;
        rd_check(t.node_child_start.len(), nodes + 1, "node_child_start")?;
        rd_check(t.obs_off.len(), nodes + 1, "obs_off")?;
        rd_check(t.draw_off.len(), nodes + 1, "draw_off")?;
        rd_check(t.draw_row_off.len(), nodes + 1, "draw_row_off")?;
        rd_check(t.reach_off.len(), nodes + 1, "reach_off")?;
        rd_check(t.soff.len(), nodes + 1, "soff")?;
        rd_check(t.cfg_off.len(), 2 * nodes + 1, "cfg_off")?;
        rd_check(t.cphi.len(), ncfg * CFEAT, "cphi")?;
        rd_check(pubfeat, PUBFEAT, "pubfeat")?;
        rd_check(t.leaf_raw.len(), rows * GPU_ROW_BYTES, "leaf_raw")?;
        rd_check(t.card_feat.len(), NTYPE * CARD_FEATS, "card_feat")?;
        rd_check(t.leaf_coff.len(), 2 * rows + 1, "leaf_coff")?;
        rd_check(t.node_parent.len(), nodes, "node_parent")?;
        rd_check(t.rev_row_of.len(), nodes, "rev_row_of")?;
        rd_check(t.rvd_row_of.len(), nodes, "rvd_row_of")?;
        rd_check(t.rev_src.len(), t.rev_cell.len(), "rev_cell")?;
        rd_check(t.rvd_src.len(), t.rvd_p.len(), "rvd_p")?;
        rd_check(t.legal_action.len(), ncells, "legal_action")?;
        rd_check(t.legal_child.len(), ncells, "legal_child")?;
        rd_check(t.legal_trans.len(), ncells, "legal_trans")?;
        rd_check(t.cell_row.len(), ncells, "cell_row")?;
        rd_check(
            *t.legal_off.last().unwrap_or(&0) as usize,
            ncells,
            "legal_off",
        )?;
        t.cells = t.legal_action.len();
        t.actions = t.obs_act.len();
        t.children = t.node_child.len();
        t.draw_entries = t.draw_to.len();
        t.nleaf = t.leaf_rows.len();
        t.nterm = t.term_leaves.len();
        rd_check(t.snap_coff.len(), 2 * (t.nleaf + t.nterm) + 1, "snap_coff")?;
        t.snapshot_configs = *t.snap_coff.last().unwrap_or(&0) as usize;
        t.n_inner = rows - t.nleaf;
        t.leaf_configs = t.leaf_cidx.len();
        t.nlevels = t.level_start.len() - 1;
        t.reach_len = *t.reach_off.last().unwrap_or(&0) as usize;
        let root = [r.f32s("root0")?, r.f32s("root1")?];
        let nroots = r.u32("nroots")? as usize;
        let mut carried = Vec::with_capacity(nroots);
        for _ in 0..nroots {
            carried.push([r.f32s("carried0")?, r.f32s("carried1")?]);
        }
        if !r.done() {
            return Err("job: trailing bytes".into());
        }
        Ok(PackedJob {
            meta,
            tables: t,
            root,
            carried,
        })
    }
}

#[cfg(test)]
impl PackedJob {
    /// The smallest well-formed job: a single terminal leaf with one config
    /// per player. Real trees come from `PackedJob::from_solver`; this exists for
    /// the layers that only need a shape to walk — the byte round trip and
    /// the device layout's alignment check.
    pub fn stub() -> PackedJob {
        PackedJob {
            meta: PackedMeta {
                depth: 2,
                iters: 4,
                snapshots: true,
                cfr: Cfr::DISCOUNTED,
                warm: 0.0,
                snap_iters: vec![0, 1, 2, 4],
                net_dims: vec![3, 32, 64, 64, 384, 1, 1, 64, 1, 384, 0, 0],
            },
            tables: PackedTables {
                nodes: 1,
                ncfg: 1,
                rows: 1,
                pubfeat: PUBFEAT,
                ncells: 0,
                node_kind: vec![2],
                node_player: vec![0],
                node_child_start: vec![0, 0],
                obs_off: vec![0, 0],
                obs_start: vec![0],
                obs_act: vec![],
                obs_child: vec![],
                legal_row_of: vec![u32::MAX],
                legal_off: vec![0],
                draw_off: vec![0, 0],
                draw_row_off: vec![0, 0],
                draw_row_start: vec![],
                cfg_off: vec![0, 1, 2],
                reach_off: vec![0, 2],
                reach_len: 2,
                soff: vec![0, 0],
                node_parent: vec![u32::MAX],
                rev_row_of: vec![u32::MAX],
                rev_start: vec![0],
                rvd_row_of: vec![u32::MAX],
                rvd_start: vec![0],
                leaf_rows: vec![0],
                term_leaves: vec![],
                terminal_utility: vec![],
                leaf_coff: vec![0, 1, 2],
                leaf_cidx: vec![0, 0],
                snap_coff: vec![0, 1, 2],
                snapshot_configs: 2,
                leaf_raw: vec![0; GPU_ROW_BYTES],
                card_feat: vec![0.0; NTYPE * CARD_FEATS],
                cphi: vec![0.0; CFEAT],
                bfs_order: vec![0],
                level_start: vec![0, 1],
                ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                ..Default::default()
            },
            root: [vec![1.0], vec![1.0]],
            carried: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU -> bytes -> CPU must be the identity, byte for byte.
    #[test]
    fn round_trip() {
        let job = PackedJob::stub();
        let bytes = job.to_bytes();
        let back = PackedJob::from_bytes(&bytes).expect("parse");
        assert_eq!(back.to_bytes(), bytes, "byte-identical round trip");
        assert_eq!(back.tables.cphi.len(), CFEAT);
    }

    #[test]
    fn narrow_indices_never_truncate() {
        let job = PackedJob::stub();
        assert_eq!(job.index_width(), IndexWidth::Narrow);
        let work = job.work();
        assert_eq!(work.network_rows, 1);
        assert!(work.table_bytes > 0);

        let mut wide = job.clone();
        wide.tables.legal_action.push(u16::MAX as u32);
        assert_eq!(wide.index_width(), IndexWidth::Wide);
    }

    #[test]
    fn whale_routes_match_card_memory_classes() {
        let ordinary = WorkVector {
            mutable_bytes: 3usize << 30,
            table_bytes: 1,
            ..Default::default()
        };
        assert!(!ordinary.requires_exclusive_route());
        assert!(!ordinary.requires_arena_guard_route());
        assert!(!ordinary.requires_card_exclusive_route());

        let lane_whale = WorkVector {
            mutable_bytes: 2usize << 30,
            table_bytes: (1usize << 30) + 1,
            ..Default::default()
        };
        assert!(lane_whale.requires_exclusive_route());
        assert!(!lane_whale.requires_arena_guard_route());
        assert!(!lane_whale.requires_card_exclusive_route());

        let arena_whale = WorkVector {
            mutable_bytes: 4usize << 30,
            table_bytes: 1,
            ..Default::default()
        };
        assert!(arena_whale.requires_exclusive_route());
        assert!(arena_whale.requires_arena_guard_route());
        assert!(!arena_whale.requires_card_exclusive_route());

        let card_whale = WorkVector {
            mutable_bytes: 4usize << 30,
            table_bytes: (1usize << 30) + 1,
            ..Default::default()
        };
        assert!(card_whale.requires_exclusive_route());
        assert!(!card_whale.requires_arena_guard_route());
        assert!(card_whale.requires_card_exclusive_route());
    }

    #[test]
    fn opponent_chance_values_alias_the_child() {
        let t = PackedTables {
            nodes: 2,
            node_kind: vec![1, 2],
            node_player: vec![0, 0],
            node_child_start: vec![0, 1, 1],
            node_child: vec![1],
            cfg_off: vec![0, 1, 2, 4, 5],
            bfs_order: vec![0, 1],
            ..Default::default()
        };
        let (base, len) = device_value_layout(&t).expect("value layout");
        assert_eq!(base[0], vec![2, 0]);
        assert_eq!(base[1], vec![0, 0]);
        assert_eq!(len, 3);
    }
}

#[cfg(test)]
mod gather_tests {
    use super::*;
    use crate::search::{Cfg, Nets};
    use crate::selfplay::{collect_roots, Agent, Collect, GameCfg};

    /// The reverse (gather) tables must reproduce the forward propagate
    /// exactly: replaying the solver's initial reach through them, level by
    /// level, must land on the reach the CPU solver computed. This is the
    /// arithmetic the GPU's forward sweep runs.
    #[test]
    fn gather_matches_forward() {
        let cfg = Cfg {
            depth: 2,
            iters: 4,
            snapshots: true,
            ..Default::default()
        };
        let nets = [Nets::default()];
        let gc = GameCfg {
            agents: [Agent::Rebel { cfg, slot: 0 }; 2],
            collect: Collect::Rebel,
            explore: 0.3,
            random_draft: true,
            eval_mix: 0.0,
            mc_mix: 0.0,
        };
        let mut checked = 0;
        for (s, bel) in collect_roots(10, 0x9E17, &nets, &gc, 6) {
            let ctx = crate::rebel::Ctx::new(&s);
            let sv = crate::search::Solver::new(&s, ctx, &nets[0], cfg, bel);
            if sv.capped() {
                continue;
            }
            let job = PackedJob::from_solver(&sv, &[]);
            let t = &job.tables;
            let nc =
                |i: usize, p: usize| (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize;
            let mut reach = vec![0.0f32; t.reach_len];
            // Root: both players' current beliefs, as the solver seeds them.
            let (r0, r1) = (nc(0, 0), nc(0, 1));
            reach[..r0].copy_from_slice(&job.root[0]);
            reach[r0..r0 + r1].copy_from_slice(&job.root[1]);
            for &ju in &t.bfs_order {
                let j = ju as usize;
                if j == 0 {
                    continue;
                }
                let p = t.node_parent[j] as usize;
                let me = t.node_player[p] as usize;
                let op = 1 - me;
                let at = |i: usize, pl: usize| {
                    t.reach_off[i] as usize + if pl == 1 { nc(i, 0) } else { 0 }
                };
                // Idle player's block passes through unchanged.
                for c in 0..nc(j, op) {
                    reach[at(j, op) + c] = reach[at(p, op) + c];
                }
                if t.rev_row_of[j] != u32::MAX {
                    let row0 = t.rev_row_of[j] as usize;
                    for c in 0..nc(j, me) {
                        let (lo, hi) = (
                            t.rev_start[row0 + c] as usize,
                            t.rev_start[row0 + c + 1] as usize,
                        );
                        let mut acc = 0.0f32;
                        for k in lo..hi {
                            acc += reach[at(p, me) + t.rev_src[k] as usize]
                                * sv.cur[t.rev_cell[k] as usize];
                        }
                        reach[at(j, me) + c] = acc;
                    }
                } else {
                    let row0 = t.rvd_row_of[j] as usize;
                    for c in 0..nc(j, me) {
                        let (lo, hi) = (
                            t.rvd_start[row0 + c] as usize,
                            t.rvd_start[row0 + c + 1] as usize,
                        );
                        let mut acc = 0.0f32;
                        for k in lo..hi {
                            acc += reach[at(p, me) + t.rvd_src[k] as usize] * t.rvd_p[k];
                        }
                        reach[at(j, me) + c] = acc;
                    }
                }
            }
            assert_eq!(reach.len(), sv.reach.len());
            for (i, (a, b)) in reach.iter().zip(&sv.reach).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-6 + 1e-5 * b.abs(),
                    "reach diverges at {i}: {a} vs {b}"
                );
            }
            checked += 1;
        }
        assert!(checked >= 4, "too few solves checked: {checked}");
    }
}
