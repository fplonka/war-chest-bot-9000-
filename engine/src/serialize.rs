//! The CPU-to-GPU tree contract, as bytes.
//!
//! One job = one solve: everything the GPU service needs to run the CFR
//! iterations of a solved subgame, plus the beliefs its Phase 2 must value.
//! The byte format is the TREE.md contract (`docs/TREE.md`, version 2) and
//! the same bytes serve three consumers: the GPU service (uploaded to the
//! device), the torch CFR specification (`train/cfr_spec.py`, the executable
//! oracle for the kernels), and the oracle tests (CPU -> bytes -> CPU).
//!
//! The solver's arenas (`regret`, `inst`, `cur`, `sum_strat`, `avg`, the
//! snapshots) are **not** uploaded: the service initialises them exactly as
//! `Solver::new` does (uniform current strategy, zero regrets, the
//! reach-weighted uniform strategy seed, snapshot 0 = the uniform average).
//! Only the initial `reach` (the uniform-strategy reach `new` propagates
//! before seeding) travels, because the walk's first belief queries read it.
//!
//! Layout: little-endian, `u32` counts prefix every array, `f32` floats,
//! `NONE = 255` where the solver uses it. Everything below mirrors
//! `docs/TREE.md`; see that file for the meaning of each array.

use crate::rebel::CFEAT;
use crate::search::{Cfr, Solver};

/// The byte format this module writes. Bump when an array changes shape or
/// meaning (docs/TREE.md "the version bumps when any of them changes shape or
/// meaning").
pub const JOB_VERSION: u32 = 3;
const MAGIC: u32 = 0x5743_4A33; // "WCJ3"

/// Runtime metadata that travels with a job (not part of the frozen tree
/// contract — it is per-request).
#[derive(Clone, Debug)]
pub struct JobMeta {
    pub depth: usize,
    pub iters: usize,
    /// Keep the per-iterate average snapshots (generation) or not (evaluation).
    pub snapshots: bool,
    pub cfr: Cfr,
    /// Iterations the policy head's strategy is worth (0 = uniform start).
    pub warm: f32,
    /// The kept iteration numbers, in order (`Solver::snapshot_iters`).
    pub snap_iters: Vec<usize>,
}

/// Every uploaded array of the contract. `rows` is the network batch size:
/// the non-terminal leaves, plus the decision rows the policy head reads
/// (warm start), in batch order — leaves first, so a leaf's row index is its
/// position in `leaf_rows`.
#[derive(Clone, Debug, Default)]
pub struct TreeTables {
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
    pub node_leaf: Vec<u8>,
    pub node_child_start: Vec<u32>,
    pub node_child: Vec<u32>,
    /// Offset of each node's `obs_start` segment into `obs_start` (one
    /// segment per node, `nch + 1` entries for `nch` public children).
    pub obs_off: Vec<u32>,
    pub obs_start: Vec<u32>,
    pub obs_act: Vec<u32>,
    pub obs_child: Vec<u32>,
    /// `legal`, bit-packed: cell `c * na + a` is bit `(c*na+a) & 7` of byte
    /// `(c*na+a) >> 3`, cells in `soff` order.
    pub legal_bits: Vec<u8>,
    pub trans: Vec<i32>,
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
    pub leaf_xpub: Vec<f32>,
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
pub struct Job {
    pub meta: JobMeta,
    pub tables: TreeTables,
    pub root: [Vec<f32>; 2],
    pub carried: Vec<[Vec<f32>; 2]>,
}

// ---------------------------------------------------------------- building

impl Job {
    /// Serialize a freshly built solver (before any iteration) into a job.
    ///
    /// `carried` are the probability vectors over the root support the next
    /// solve's Phase 2 must value: the previous solve's carried beliefs, or
    /// just the live belief for the first level.
    pub fn from_solver(sv: &Solver, carried: &[[Vec<f32>; 2]]) -> Job {
        let tables = TreeTables::from_solver(sv);
        let root = [
            sv.root_belief[0].p.clone(),
            sv.root_belief[1].p.clone(),
        ];
        Job {
            meta: JobMeta {
                depth: sv.cfg.depth,
                iters: sv.cfg.iters,
                snapshots: sv.cfg.snapshots,
                cfr: sv.cfg.cfr,
                warm: sv.cfg.warm,
                snap_iters: sv.snap_list.clone(),
            },
            tables,
            root,
            carried: carried.to_vec(),
        }
    }
}

impl TreeTables {
    fn from_solver(sv: &Solver) -> TreeTables {
        let nodes = sv.nodes.len();
        let mut t = TreeTables {
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
            t.node_leaf.push(n.leaf as u8);
            obs_off.push(obs_start.len() as u32);
            obs_start.extend_from_slice(&n.obs_start);
            t.obs_act.extend_from_slice(&n.obs_act);
            t.obs_child
                .extend(n.obs_child.iter().map(|&c| c as u32));
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
                let (na, me) = (n.na(), n.player as usize);
                let nc = n.cfgs[me].len();
                // legal bits + trans, in soff order.
                for c in 0..nc {
                    for a in 0..na {
                        let j = c * na + a;
                        let cell = sv.soff[i] as usize + j;
                        t.legal_bits.resize((cell >> 3) + 1, 0);
                        if n.legal[j] {
                            t.legal_bits[cell >> 3] |= 1 << (cell & 7);
                        }
                        t.trans.push(n.trans[j]);
                    }
                }
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
                        for c in 0..nc {
                            if !n.legal[c * na + a] {
                                continue;
                            }
                            let tr = n.trans[c * na + a];
                            if tr < 0 {
                                continue;
                            }
                            gather[tr as usize]
                                .push((c as u32, (sv.soff[i] as usize + c * na + a) as u32));
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
        debug_assert_eq!(leaf_coff.len(), 2 * t.rows + 1, "leaf_coff must be canonical");
        t.leaf_coff = leaf_coff;
        t.leaf_cidx = sv.leaf_cidx.clone();
        t.leaf_configs = sv.leaf_cidx.len();
        let pf = sv.pubfeat;
        t.leaf_xpub = sv.xpub[..t.rows * pf].to_vec();

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
    fn i32s(&mut self, v: &[i32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.b.extend_from_slice(&x.to_le_bytes());
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
    fn i32s(&mut self, what: &str) -> Result<Vec<i32>, String> {
        let n = self.u32(what)? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let s = self.take(4, what)?;
            v.push(i32::from_le_bytes(s.try_into().unwrap()));
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

impl Job {
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
        let t = &self.tables;
        w.u32(t.nodes as u32);
        w.u32(t.ncfg as u32);
        w.u32(t.rows as u32);
        w.u32(t.pubfeat as u32);
        w.u32(t.ncells as u32);
        // tree
        w.u8s(&t.node_kind);
        w.u8s(&t.node_player);
        w.u8s(&t.node_leaf);
        w.u32s(&t.node_child_start);
        w.u32s(&t.node_child);
        w.u32s(&t.obs_off);
        w.u32s(&t.obs_start);
        w.u32s(&t.obs_act);
        w.u32s(&t.obs_child);
        w.u8s(&t.legal_bits);
        w.i32s(&t.trans);
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
        w.f32s(&t.leaf_xpub);
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

    pub fn from_bytes(b: &[u8]) -> Result<Job, String> {
        let mut r = R::new(b);
        if r.u32("magic")? != MAGIC {
            return Err("job: bad magic".into());
        }
        if r.u32("version")? != JOB_VERSION {
            return Err(format!("job: unsupported version"));
        }
        let meta = JobMeta {
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
        };
        let nodes = r.u32("nodes")? as usize;
        let ncfg = r.u32("ncfg")? as usize;
        let rows = r.u32("rows")? as usize;
        let pubfeat = r.u32("pubfeat")? as usize;
        let ncells = r.u32("ncells")? as usize;
        let mut t = TreeTables {
            nodes,
            ncfg,
            rows,
            pubfeat,
            ncells,
            ..Default::default()
        };
        t.node_kind = r.u8s("node_kind")?;
        t.node_player = r.u8s("node_player")?;
        t.node_leaf = r.u8s("node_leaf")?;
        t.node_child_start = r.u32s("node_child_start")?;
        t.node_child = r.u32s("node_child")?;
        t.obs_off = r.u32s("obs_off")?;
        t.obs_start = r.u32s("obs_start")?;
        t.obs_act = r.u32s("obs_act")?;
        t.obs_child = r.u32s("obs_child")?;
        t.legal_bits = r.u8s("legal_bits")?;
        t.trans = r.i32s("trans")?;
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
        t.leaf_xpub = r.f32s("leaf_xpub")?;
        t.cphi = r.f32s("cphi")?;
        t.bfs_order = r.u32s("bfs_order")?;
        t.level_start = r.u32s("level_start")?;
        t.ids = r.u8s("ids")?;
        // sanity checks
        rd_check(t.node_kind.len(), nodes, "node_kind")?;
        rd_check(t.node_player.len(), nodes, "node_player")?;
        rd_check(t.node_leaf.len(), nodes, "node_leaf")?;
        rd_check(t.node_child_start.len(), nodes + 1, "node_child_start")?;
        rd_check(t.obs_off.len(), nodes + 1, "obs_off")?;
        rd_check(t.draw_off.len(), nodes + 1, "draw_off")?;
        rd_check(t.draw_row_off.len(), nodes + 1, "draw_row_off")?;
        rd_check(t.reach_off.len(), nodes + 1, "reach_off")?;
        rd_check(t.soff.len(), nodes + 1, "soff")?;
        rd_check(t.cfg_off.len(), 2 * nodes + 1, "cfg_off")?;
        rd_check(t.cphi.len(), ncfg * CFEAT, "cphi")?;
        rd_check(t.leaf_xpub.len(), rows * pubfeat, "leaf_xpub")?;
        rd_check(t.leaf_coff.len(), 2 * rows + 1, "leaf_coff")?;
        rd_check(t.node_parent.len(), nodes, "node_parent")?;
        rd_check(t.rev_row_of.len(), nodes, "rev_row_of")?;
        rd_check(t.rvd_row_of.len(), nodes, "rvd_row_of")?;
        rd_check(t.rev_src.len(), t.rev_cell.len(), "rev_cell")?;
        rd_check(t.rvd_src.len(), t.rvd_p.len(), "rvd_p")?;
        t.cells = t.trans.len();
        t.actions = t.obs_act.len();
        t.children = t.node_child.len();
        t.draw_entries = t.draw_to.len();
        t.nleaf = t.leaf_rows.len();
        t.nterm = t.term_leaves.len();
        t.n_inner = rows - t.nleaf;
        t.leaf_configs = t.leaf_cidx.len();
        t.nlevels = t.level_start.len() - 1;
        t.reach_len = *t.reach_off.last().unwrap_or(&0) as usize;
        let root = [
            r.f32s("root0")?,
            r.f32s("root1")?,
        ];
        let nroots = r.u32("nroots")? as usize;
        let mut carried = Vec::with_capacity(nroots);
        for _ in 0..nroots {
            carried.push([r.f32s("carried0")?, r.f32s("carried1")?]);
        }
        if !r.done() {
            return Err("job: trailing bytes".into());
        }
        Ok(Job { meta, tables: t, root, carried })
    }
}

#[cfg(test)]
impl Job {
    /// The smallest well-formed job: a single terminal leaf with one config
    /// per player. Real trees come from `Job::from_solver`; this exists for
    /// the layers that only need a shape to walk — the byte round trip and
    /// the device layout's alignment check.
    pub fn stub() -> Job {
        Job {
            meta: JobMeta {
                depth: 2,
                iters: 4,
                snapshots: true,
                cfr: Cfr::DISCOUNTED,
                warm: 0.0,
                snap_iters: vec![0, 1, 2, 4],
            },
            tables: TreeTables {
                nodes: 1,
                ncfg: 1,
                rows: 1,
                pubfeat: 8,
                ncells: 2,
                node_kind: vec![2],
                node_player: vec![0],
                node_leaf: vec![1],
                node_child_start: vec![0, 0],
                obs_off: vec![0, 0],
                obs_start: vec![0],
                obs_act: vec![],
                obs_child: vec![],
                legal_bits: vec![],
                trans: vec![],
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
                leaf_xpub: vec![0.0; 8],
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
        let job = Job::stub();
        let bytes = job.to_bytes();
        let back = Job::from_bytes(&bytes).expect("parse");
        assert_eq!(back.to_bytes(), bytes, "byte-identical round trip");
        assert_eq!(back.tables.cphi.len(), CFEAT);
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
        let cfg = Cfg { depth: 2, iters: 4, snapshots: true, ..Default::default() };
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
            let job = Job::from_solver(&sv, &[]);
            let t = &job.tables;
            let nc = |i: usize, p: usize| {
                (t.cfg_off[2 * i + p + 1] - t.cfg_off[2 * i + p]) as usize
            };
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
