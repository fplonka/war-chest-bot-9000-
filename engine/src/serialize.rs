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

use crate::rebel::{
    config_counts, write_action_feats, AFEAT, CFEAT, CCOUNTS, Config, NSLOT, NTYPE,
};
use crate::search::{Cfr, Solver};

/// The byte format this module writes. Bump when an array changes shape or
/// meaning (docs/TREE.md "the version bumps when any of them changes shape or
/// meaning").
pub const JOB_VERSION: u32 = 2;
const MAGIC: u32 = 0x5743_4A32; // "WCJ2"

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
    pub npsi_rows: usize,
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
    pub action_pays: Vec<i8>,
    pub action_fdown: Vec<u8>,
    pub draw_off: Vec<u32>,
    pub draw_to: Vec<u32>,
    pub draw_p: Vec<f32>,
    pub draw_steps: Vec<u8>,
    /// Offset of each node's draw row-boundary segment into `draw_row_start`
    /// (one segment per chance node, `rows + 1` entries). The flat `draw_to`
    /// entries of a node split into rows through it.
    pub draw_row_off: Vec<u32>,
    pub draw_row_start: Vec<u32>,
    // -- config support --
    pub cfg_off: Vec<u32>,
    pub cfg_id: Vec<u32>,
    pub cfg_hand: Vec<u8>,
    pub cfg_fd: Vec<u8>,
    pub cfg_pending: Vec<i8>,
    // -- arenas --
    pub reach_off: Vec<u32>,
    pub reach: Vec<f32>,
    pub soff: Vec<u32>,
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
    pub cmap_key: Vec<u64>,
    // -- derived: BFS level order for the sweeps --
    pub bfs_order: Vec<u32>,
    pub level_start: Vec<u32>,
    // -- action features (the policy head's input), per decision node --
    pub psi_off: Vec<u32>,
    pub psi: Vec<f32>,
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
            ..Default::default()
        };
        // The config table is extended with any support member the leaf batch
        // never interned, so every `cfg_id` is valid. The leaf batch's own
        // rows keep their indices; the extra rows are never read.
        let mut cphi = sv.cphi.clone();
        let mut cmap = sv.cmap.clone();
        let mut ncfg = sv.ncfg;
        let intern = |c: &Config, res: &[u8; NSLOT], p: usize,
                      cphi: &mut Vec<f32>, cmap: &mut std::collections::HashMap<u64, u32>,
                      ncfg: &mut usize| -> u32 {
            let mut cnt = [0u8; CCOUNTS];
            config_counts(c, res, &mut cnt);
            let mut key = p as u64;
            for x in cnt.iter() {
                key = (key << 4) | *x as u64;
            }
            if let Some(&i) = cmap.get(&key) {
                return i;
            }
            let i = *ncfg as u32;
            *ncfg += 1;
            let at = i as usize * CFEAT;
            cphi.resize(at + CFEAT, 0.0);
            for k in 0..CCOUNTS {
                cphi[at + k] = cnt[k] as f32 / 5.0;
            }
            cphi[at + CCOUNTS] = p as f32;
            cmap.insert(key, i);
            i
        };
        let mut cfg_off = Vec::with_capacity(2 * nodes + 1);
        let mut cfg_id = Vec::new();
        let mut cfg_hand = Vec::new();
        let mut cfg_fd = Vec::new();
        let mut cfg_pending = Vec::new();
        for i in 0..nodes {
            let n = &sv.nodes[i];
            let res = crate::rebel::reserve(&n.s, 0, &sv.ctx);
            for p in 0..2usize {
                cfg_off.push(cfg_id.len() as u32);
                let res_p = if p == 0 { res } else { crate::rebel::reserve(&n.s, 1, &sv.ctx) };
                for c in n.cfgs[p].iter() {
                    let id = intern(c, &res_p, p, &mut cphi, &mut cmap, &mut ncfg);
                    cfg_id.push(id);
                    for k in 0..NSLOT {
                        cfg_hand.push(c.hand[k]);
                        cfg_fd.push(c.fd[k]);
                    }
                    cfg_pending.push(c.pending_coin.map_or(-1, |k| k as i8));
                }
            }
        }
        cfg_off.push(cfg_id.len() as u32);
        t.members = cfg_id.len();
        t.cfg_off = cfg_off;
        t.cfg_id = cfg_id;
        t.cfg_hand = cfg_hand;
        t.cfg_fd = cfg_fd;
        t.cfg_pending = cfg_pending;
        t.ncfg = ncfg;
        t.cphi = cphi;
        // cmap_key in row order.
        let mut keys: Vec<(u32, u64)> = cmap.iter().map(|(&k, &i)| (i, k)).collect();
        keys.sort_unstable();
        t.cmap_key = keys.iter().map(|&(_, k)| k).collect();

        // Node arrays and flat CSRs.
        let mut child_start = Vec::with_capacity(nodes + 1);
        let mut obs_off = Vec::with_capacity(nodes + 1);
        let mut draw_off = Vec::with_capacity(nodes + 1);
        let mut reach_off = Vec::with_capacity(nodes + 1);
        let mut psi_off = Vec::with_capacity(nodes + 1);
        let mut soff = Vec::with_capacity(nodes + 1);
        let mut obs_start = Vec::new();
        let mut psi = Vec::new();
        for i in 0..nodes {
            let n = &sv.nodes[i];
            child_start.push(t.node_child.len() as u32);
            t.node_child.extend(n.child.iter().map(|&c| c as u32));
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
            t.obs_child.extend_from_slice(&n.obs_child.iter().map(|&c| c as u32).collect::<Vec<_>>());
            t.action_pays.extend_from_slice(&n.aslot);
            t.action_fdown.extend_from_slice(&n.fdown.iter().map(|&b| b as u8).collect::<Vec<_>>());
            draw_off.push(t.draw_to.len() as u32);
            t.draw_to.extend_from_slice(&n.draw.to);
            t.draw_p.extend_from_slice(&n.draw.p);
            t.draw_steps.push(n.draw_steps);
            t.draw_row_off.push(t.draw_row_start.len() as u32);
            t.draw_row_start.extend_from_slice(&n.draw.start);
            if n.chance {
                // Sanity: the flat CSR must cover the draw map's rows.
                debug_assert_eq!(n.draw.start.len(), n.draw.rows() + 1);
            }
            reach_off.push(sv.roff[i]);
            let (c0, c1) = (sv.nc[i][0] as usize, sv.nc[i][1] as usize);
            let at = sv.roff[i] as usize + c0;
            t.reach.extend_from_slice(&sv.reach[at - c0..at + c1]);
            soff.push(sv.soff[i]);
            if !n.leaf && !n.chance {
                let (na, me) = (n.na(), n.player as usize);
                let nc = n.nc(me);
                psi_off.push(psi.len() as u32 / AFEAT as u32);
                let mut row = vec![0.0f32; AFEAT];
                for a in 0..na {
                    write_action_feats(&n.acts[a], &sv.ctx, me, n.aslot[a], n.fdown[a], &mut row);
                    psi.extend_from_slice(&row);
                }
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
            } else {
                psi_off.push(psi.len() as u32 / AFEAT as u32);
            }
        }
        child_start.push(t.node_child.len() as u32);
        obs_off.push(obs_start.len() as u32);
        draw_off.push(t.draw_to.len() as u32);
        t.draw_row_off.push(t.draw_row_start.len() as u32);
        reach_off.push(sv.reach.len() as u32);
        psi_off.push(psi.len() as u32 / AFEAT as u32);
        soff.push(sv.soff[nodes]);
        t.node_child_start = child_start;
        t.obs_off = obs_off;
        t.obs_start = obs_start;
        t.draw_off = draw_off;
        t.reach_off = reach_off;
        t.psi_off = psi_off;
        t.soff = soff;
        t.cells = sv.ncells;
        t.ncells = sv.ncells;
        t.actions = t.obs_act.len();
        t.children = t.node_child.len();
        t.draw_entries = t.draw_to.len();
        t.reach_len = t.reach.len();
        t.npsi_rows = t.psi.len() / AFEAT;
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
    fn u64(&mut self, v: u64) {
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
    fn u64s(&mut self, v: &[u64]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.u64(x);
        }
    }
    fn i32s(&mut self, v: &[i32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.b.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn i8s(&mut self, v: &[i8]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.b.push(x as u8);
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
    fn u64(&mut self, what: &str) -> Result<u64, String> {
        let s = self.take(8, what)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
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
    fn u64s(&mut self, what: &str) -> Result<Vec<u64>, String> {
        let n = self.u32(what)? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.u64(what)?);
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
    fn i8s(&mut self, what: &str) -> Result<Vec<i8>, String> {
        let n = self.u32(what)? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(self.take(1, what)?[0] as i8);
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
        w.i8s(&t.action_pays);
        w.u8s(&t.action_fdown);
        w.u32s(&t.draw_off);
        w.u32s(&t.draw_to);
        w.f32s(&t.draw_p);
        w.u8s(&t.draw_steps);
        w.u32s(&t.draw_row_off);
        w.u32s(&t.draw_row_start);
        w.u32s(&t.cfg_off);
        w.u32s(&t.cfg_id);
        w.u8s(&t.cfg_hand);
        w.u8s(&t.cfg_fd);
        w.i8s(&t.cfg_pending);
        w.u32s(&t.reach_off);
        w.f32s(&t.reach);
        w.u32s(&t.soff);
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
        w.u64s(&t.cmap_key);
        // levels
        w.u32s(&t.bfs_order);
        w.u32s(&t.level_start);
        // action features
        w.u32s(&t.psi_off);
        w.f32s(&t.psi);
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
        t.action_pays = r.i8s("action_pays")?;
        t.action_fdown = r.u8s("action_fdown")?;
        t.draw_off = r.u32s("draw_off")?;
        t.draw_to = r.u32s("draw_to")?;
        t.draw_p = r.f32s("draw_p")?;
        t.draw_steps = r.u8s("draw_steps")?;
        t.draw_row_off = r.u32s("draw_row_off")?;
        t.draw_row_start = r.u32s("draw_row_start")?;
        t.cfg_off = r.u32s("cfg_off")?;
        t.cfg_id = r.u32s("cfg_id")?;
        t.cfg_hand = r.u8s("cfg_hand")?;
        t.cfg_fd = r.u8s("cfg_fd")?;
        t.cfg_pending = r.i8s("cfg_pending")?;
        t.reach_off = r.u32s("reach_off")?;
        t.reach = r.f32s("reach")?;
        t.soff = r.u32s("soff")?;
        t.leaf_rows = r.u32s("leaf_rows")?;
        t.inner_rows = r.u32s("inner_rows")?;
        t.term_leaves = r.u32s("term_leaves")?;
        t.terminal_utility = r.f32s("terminal_utility")?;
        t.leaf_coff = r.u32s("leaf_coff")?;
        t.leaf_cidx = r.u32s("leaf_cidx")?;
        t.leaf_xpub = r.f32s("leaf_xpub")?;
        t.cphi = r.f32s("cphi")?;
        t.cmap_key = r.u64s("cmap_key")?;
        t.bfs_order = r.u32s("bfs_order")?;
        t.level_start = r.u32s("level_start")?;
        t.psi_off = r.u32s("psi_off")?;
        t.psi = r.f32s("psi")?;
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
        rd_check(t.psi_off.len(), nodes + 1, "psi_off")?;
        rd_check(t.cfg_off.len(), 2 * nodes + 1, "cfg_off")?;
        rd_check(t.cphi.len(), ncfg * CFEAT, "cphi")?;
        rd_check(t.cmap_key.len(), ncfg, "cmap_key")?;
        rd_check(t.leaf_xpub.len(), rows * pubfeat, "leaf_xpub")?;
        rd_check(t.leaf_coff.len(), 2 * rows + 1, "leaf_coff")?;
        rd_check(t.psi.len() % AFEAT, 0, "psi")?;
        t.cells = t.trans.len();
        t.actions = t.obs_act.len();
        t.children = t.node_child.len();
        t.members = t.cfg_id.len();
        t.draw_entries = t.draw_to.len();
        t.nleaf = t.leaf_rows.len();
        t.nterm = t.term_leaves.len();
        t.n_inner = rows - t.nleaf;
        t.leaf_configs = t.leaf_cidx.len();
        t.nlevels = t.level_start.len() - 1;
        t.reach_len = t.reach.len();
        t.npsi_rows = t.psi.len() / AFEAT;
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
mod tests {
    use super::*;

    /// CPU -> bytes -> CPU must be the identity, byte for byte.
    #[test]
    fn round_trip() {
        // Build a tiny solver directly: the serializer only reads the tables,
        // so a hand-built Solver is overkill; instead we build one through the
        // real path (the tests below the engine have solvers; here we just
        // check the byte layer on a synthetic job).
        let job = Job {
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
                action_pays: vec![],
                action_fdown: vec![],
                draw_off: vec![0, 0],
                draw_row_off: vec![0, 0],
                draw_row_start: vec![],
                draw_steps: vec![0],
                cfg_off: vec![0, 0, 0],
                cfg_id: vec![],
                cfg_hand: vec![],
                cfg_fd: vec![],
                cfg_pending: vec![],
                reach_off: vec![0, 0],
                reach: vec![],
                soff: vec![0, 0],
                leaf_rows: vec![0],
                term_leaves: vec![],
                terminal_utility: vec![],
                leaf_coff: vec![0, 1, 2],
                leaf_cidx: vec![0, 0],
                leaf_xpub: vec![0.0; 8],
                cphi: vec![0.0; CFEAT],
                cmap_key: vec![42],
                bfs_order: vec![0],
                level_start: vec![0, 1],
                psi_off: vec![0, 0],
                psi: vec![],
                ids: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                ..Default::default()
            },
            root: [vec![1.0], vec![1.0]],
            carried: vec![],
        };
        let bytes = job.to_bytes();
        let back = Job::from_bytes(&bytes).expect("parse");
        assert_eq!(back.to_bytes(), bytes, "byte-identical round trip");
        assert_eq!(back.tables.cphi.len(), CFEAT);
    }
}
