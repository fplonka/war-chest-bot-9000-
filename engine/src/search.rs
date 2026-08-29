use crate::actions::Action;
use crate::board::NONE;
use crate::contract::{Call, Dst, Prime, QueryPick, Reply, Writes};
use crate::net::Net;
use crate::pbs::*;
use crate::rng::Rng;
use crate::state::{Cont, State};
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};
use std::sync::Arc;

pub const TRIES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Ent {
    Node = 0,
    Cell = 1,
    Reach = 2,
    Draw = 3,
    Row = 4,
    Board = 5,
    Config = 6,
    Cidx = 7,
}

impl Ent {
    pub const ALL: [Ent; 8] = [
        Ent::Node,
        Ent::Cell,
        Ent::Reach,
        Ent::Draw,
        Ent::Row,
        Ent::Board,
        Ent::Config,
        Ent::Cidx,
    ];
    pub const NAME: [&'static str; 8] = ["node", "cell", "reach", "draw", "row", "board", "config", "cidx"];
    pub fn name(self) -> &'static str {
        Self::NAME[self as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget(pub [usize; 8]);

impl Budget {
    pub fn for_s(s: u32) -> Budget {
        let k = |at512: usize| (at512 * s as usize / 512).max(1);
        let mut b = Budget(BUDGET_512.0.map(k));
        b.0[Ent::Config as usize] = b.0[Ent::Config as usize].max(2 * crate::pbs::MAX_CONFIG_SUPPORT);
        b
    }

    pub fn cap(&self, e: Ent) -> usize {
        self.0[e as usize]
    }

    pub fn host_slot_bytes(&self) -> usize {
        fn cap<T>(n: usize) -> usize {
            Vec::<T>::with_capacity(n).capacity() * std::mem::size_of::<T>()
        }
        cap::<TNode>(self.cap(Ent::Node))
            + cap::<u32>(self.cap(Ent::Cidx))
            + cap::<u32>(self.cap(Ent::Row)) * 4
            + cap::<f32>(self.cap(Ent::Cell))
            + cap::<u32>(self.cap(Ent::Cell)) * 6
            + cap::<u32>(self.cap(Ent::Reach))
            + cap::<u32>(self.cap(Ent::Draw)) * 3
            + cap::<f32>(self.cap(Ent::Board) * crate::pbs::PUBFEAT) * 2
            + cap::<f32>(self.cap(Ent::Config) * crate::pbs::CFEAT)
            + cap::<f32>(self.cap(Ent::Row) * crate::net::D)
            + cap::<f32>(self.cap(Ent::Config) * crate::net::D)
    }
}

impl Default for Budget {
    fn default() -> Budget {
        Budget::for_s(512)
    }
}


const BUDGET_512: Budget = Budget([16_595, 136_283, 346_018, 174_834, 10_090, 8_219, 921, 259_756]);

#[derive(Clone, Copy)]
pub struct Cfg {
    pub s: u32,
    pub c: f32,
    pub batch: usize,
    pub rounds: u8,
    pub cfr: Cfr,
    pub puct: f32,
    pub prior_temp: f32,
    pub budget: Budget,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            s: 512,
            c: 8.0,
            batch: 8,
            rounds: 0,
            cfr: Cfr::SOG,
            puct: 1.5,
            prior_temp: 1.0,
            budget: Budget::for_s(512),
        }
    }
}

impl Cfg {
    pub fn iters(&self) -> usize {
        if self.c <= 0.0 {
            return self.s as usize;
        }
        (self.s as f32 / self.c).ceil() as usize
    }

    fn expansions_at(&self, i: usize) -> usize {
        if self.c <= 0.0 {
            return 0;
        }
        let earned = |k: usize| (self.s as usize).min((k as f32 * self.c).floor() as usize);
        earned(i) - earned(i - 1)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Cfr {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub predict: f32,
}

impl Cfr {
    pub const LINEAR: Cfr = Cfr {
        alpha: 1.0,
        beta: 1.0,
        gamma: 1.0,
        predict: 0.0,
    };
    pub const DISCOUNTED: Cfr = Cfr {
        alpha: 1.5,
        beta: 0.0,
        gamma: 2.0,
        predict: 0.0,
    };

    pub const SOG: Cfr = Cfr {
        alpha: f32::INFINITY,
        beta: f32::NEG_INFINITY,
        gamma: 1.0,
        predict: 0.0,
    };

    pub const NAMED: [(&'static str, Cfr); 3] = [("linear", Cfr::LINEAR), ("dcfr", Cfr::DISCOUNTED), ("sog", Cfr::SOG)];

    pub fn named(name: &str) -> Option<Cfr> {
        Cfr::NAMED.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
    }
}

pub const STOP_NAMES: [&str; 11] = [
    "complete",
    "exhausted",
    "budget_node",
    "budget_cell",
    "budget_reach",
    "budget_draw",
    "budget_row",
    "budget_board",
    "budget_config",
    "budget_cidx",
    "other",
];

pub fn action_coin(a: &Action, s: &State) -> u8 {
    use Action::*;
    match *a {
        Deploy { unit, .. } | Bolster { unit, .. } => unit,
        ClaimInitiative { coin } | Recruit { coin, .. } | Pass { coin } | TacFootman { coin } => coin,
        TacRoyalGuard { .. } => ROYAL_COIN,
        TacEnsign { .. } => ENSIGN,
        TacMarshal { .. } => MARSHAL,
        Move { from, .. }
        | Control { from }
        | Attack { from, .. }
        | TacArcher { from, .. }
        | TacCavalryMove { from, .. }
        | TacCrossbow { from, .. }
        | TacLancer { from, .. }
        | TacLightCav { from, .. } => s.hex_type[from as usize],
        _ => NONE,
    }
}

pub fn node_actions(s: &State, player: u8, ctx: &Ctx, cfgs: &[Config]) -> (Vec<Action>, Vec<i8>, Vec<bool>) {
    let mut acts: Vec<Action> = Vec::new();
    let mut aslot: Vec<i8> = Vec::new();
    let mut fdown: Vec<bool> = Vec::new();
    let mut seen: Vec<u32> = Vec::new();
    let mut probe = *s;
    let forced = matches!(s.pending(), Cont::WarriorPriestPlay { .. });
    if matches!(s.pending(), Cont::MainPlay) || forced {
        let res = reserve(s, player, ctx);
        let playable: [bool; NSLOT] = std::array::from_fn(|k| {
            cfgs.is_empty()
                || cfgs.iter().any(|c| {
                    if forced {
                        c.inflight == Some(k as u8)
                    } else {
                        c.hand[k] > 0
                    }
                })
        });
        for k in 0..NSLOT {
            if res[k] == 0 || !playable[k] {
                continue;
            }
            let mut one = Config::default();
            if forced {
                one.inflight = Some(k as u8);
            } else {
                one.hand[k] = 1;
            }
            set_config(&mut probe, player, ctx, &one);
            for a in probe.legal_actions() {
                let key = a.encode();
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                let coin = action_coin(&a, &probe);
                let slot = if coin == NONE {
                    -1
                } else {
                    ctx.slot_of[player as usize][coin as usize]
                };
                if slot >= 0 && !playable[slot as usize] {
                    continue;
                }
                aslot.push(slot);
                fdown.push(is_facedown_play(&a));
                acts.push(a);
            }
        }
    } else {
        for a in s.legal_actions() {
            let key = a.encode();
            if !seen.contains(&key) {
                seen.push(key);
                acts.push(a);
                aslot.push(-1);
                fdown.push(false);
            }
        }
    }
    (acts, aslot, fdown)
}

pub struct TNode {
    pub state: State,
    pub parent: u32,
    pub nc: [u32; 2],
    pub roff: u32,
    pub voff: u32,
    pub soff: u32,
    pub row_of: u32,
    pub primed: bool,
    pub util: f32,
    pub player: u8,
    pub leaf: bool,
    pub expandable: bool,
    pub exhausted: bool,
    pub chance: bool,
    pub draw: DrawMap,
    pub draw_steps: u8,
    pub acts: Vec<Action>,
    pub aslot: Vec<i8>,
    pub fdown: Vec<bool>,
    pub obs_child: Vec<usize>,
    pub obs_start: Vec<u32>,
    pub obs_act: Vec<u32>,
    pub child: Vec<usize>,
    pub cfgs: [Arc<[Config]>; 2],
    pub legal_off: Vec<u32>,
    pub legal_action: Vec<u32>,
    pub legal_child: Vec<u32>,
    pub legal_trans: Vec<u32>,
    pub action_off: Vec<u32>,
    pub action_cell: Vec<u32>,
    pub cell_row: Vec<u32>,
}

pub const NO_TRANS: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Mark {
    nodes: usize,
    ncells: usize,
    ncfg: usize,
    ndraws: usize,
    leaf_rows: usize,
    nboards: usize,
    term_leaves: usize,
    leaf_coff: usize,
    leaf_cidx: usize,
    nlegal_off: usize,
    nrev_start: usize,
    nrvd_start: usize,
    ndraw_start: usize,
}

pub struct Solved {
    pub value: [Vec<f32>; 2],
    pub queries: Vec<(State, [Belief; 2])>,
    pub policy: Policy,
}

#[derive(Default, Clone)]
pub struct Policy {
    pub acts: Vec<[u8; ACT_BYTES]>,
    pub off: Vec<u32>,
    pub act: Vec<u16>,
    pub p: Vec<f32>,
}

pub const ACT_BYTES: usize = 6;

pub(crate) fn action_desc(a: &Action, player: u8, ctx: &Ctx, slot: i8) -> [u8; ACT_BYTES] {
    let coin = |k: i8| {
        if k < 0 {
            NONE
        } else {
            player * NSLOT as u8 + k as u8
        }
    };
    let recruited = a.recruited();
    let rslot = if recruited == NONE {
        -1
    } else {
        ctx.slot_of[player as usize][recruited as usize]
    };
    let h = a.hexes();
    [a.kind() as u8, coin(slot), coin(rslot), h[0], h[1], h[2]]
}

impl TNode {
    #[inline]
    pub fn na(&self) -> usize {
        self.acts.len()
    }
    #[inline]
    pub fn legal_row(&self, c: usize) -> std::ops::Range<usize> {
        self.legal_off[c] as usize..self.legal_off[c + 1] as usize
    }
}

thread_local! {
    static CONFIG_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

thread_local! {
    static NODE_POOL: std::cell::RefCell<Vec<Vec<TNode>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn pool_budget(cfg: &Cfg) -> usize {
    32 * cfg.s as usize
}

fn take_nodes() -> Vec<TNode> {
    NODE_POOL.with(|b| b.borrow_mut().pop().unwrap_or_default())
}

fn give_nodes(mut v: Vec<TNode>, budget: usize) {
    if v.capacity() == 0 || v.capacity() > budget {
        return;
    }
    v.clear();
    NODE_POOL.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() < 2 {
            b.push(v);
        }
    });
}

#[derive(Default, Clone, Copy)]
pub(crate) struct KeyHash;

#[derive(Default)]
pub(crate) struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, n: u64) {
        let x = (n ^ self.0).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = x ^ (x >> 29);
    }
}

impl std::hash::BuildHasher for KeyHash {
    type Hasher = KeyHasher;
    fn build_hasher(&self) -> KeyHasher {
        KeyHasher(0)
    }
}

fn row_key(row: &[u8]) -> u64 {
    let mut h = 0u64;
    for &x in row {
        h = (x as u64 ^ h).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    h
}

pub(crate) fn group_by<I>(keys: I, n_keys: usize) -> (Vec<u32>, Vec<u32>)
where
    I: Iterator<Item = usize> + Clone,
{
    let mut start = vec![0u32; n_keys + 1];
    for k in keys.clone() {
        start[k + 1] += 1;
    }
    for k in 0..n_keys {
        start[k + 1] += start[k];
    }
    let mut fill = start.clone();
    let mut order = vec![0u32; start[n_keys] as usize];
    for (i, k) in keys.enumerate() {
        order[fill[k] as usize] = i as u32;
        fill[k] += 1;
    }
    (start, order)
}

fn sample_indices(rng: &mut Rng, n: usize, k: usize) -> Vec<usize> {
    debug_assert!(k <= n);
    let mut out = Vec::with_capacity(k);
    for j in n - k..n {
        let pick = rng.below(j + 1);
        out.push(if out.contains(&pick) { j } else { pick });
    }
    out
}

fn take_config_buf() -> Vec<f32> {
    CONFIG_POOL.with(|pool| pool.borrow_mut().pop().unwrap_or_default())
}

fn give_config_buf(v: Vec<f32>) {
    if v.capacity() == 0 {
        return;
    }
    CONFIG_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 2 {
            pool.push(v);
        }
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Phase {
    #[default]
    Fresh,
    Iterating,
    Reading,
    Done,
}

pub enum Step {
    Calls(Vec<Call>),
    Done(Option<Solved>),
}

#[derive(Default)]
pub struct Solver {
    pub(crate) ctx: Ctx,
    net: Arc<Net>,
    rng: Rng,
    slot: usize,
    collect: Option<usize>,
    at: usize,
    expansions: u32,
    phase: Phase,
    queries: Vec<(State, [Belief; 2])>,
    query_seen: usize,
    query_nodes: Vec<usize>,
    pub(crate) cfg: Cfg,
    pub nodes: Vec<TNode>,
    resealed: Vec<u32>,
    pub root_belief: [Belief; 2],
    pub cur: Vec<f32>,
    pub avg: Vec<f32>,
    pub grown: Vec<u32>,
    pub(crate) avg_touched: [bool; 2],
    pub ncells: usize,
    pub nreach: usize,
    pub nvals: usize,
    pub ndraws: usize,
    nlegal_off: usize,
    nrev_start: usize,
    nrvd_start: usize,
    ndraw_start: usize,
    budget_hit: u8,
    wants_prior: Vec<u32>,
    pub(crate) steps: [usize; 2],

    pub leaf_rows: Vec<usize>,
    pub(crate) term_leaves: Vec<usize>,
    pub(crate) leaf_cidx: Vec<u32>,
    pub(crate) leaf_coff: Vec<u32>,
    pub(crate) cphi: Vec<f32>,
    pub(crate) cplayer: Vec<u8>,
    pub(crate) cmap: std::collections::HashMap<u64, u32, KeyHash>,
    pub ncfg: usize,
    batch_rows: usize,
    batch_boards: usize,
    batch_cfgs: usize,
    pub cards: Vec<f32>,
    pub(crate) board_of: Vec<u32>,
    bmap: std::collections::HashMap<u64, u32, KeyHash>,
    pub nboards: usize,
    pub(crate) packed: Vec<u8>,
    abandon: bool,
    draw_scratch: DrawScratch,
    cell_order: Vec<(u64, u32)>,

    contract: Arc<crate::contract::Contract>,
    sent_from: usize,
    rewrite: Vec<u32>,
    resent: Vec<u32>,
    sent_cells: usize,
    seed: u64,
    sent: crate::contract::Sent,
}

const _: () = {
    const fn is_send<T: Send>() {}
    is_send::<Solver>();
};

impl Drop for Solver {
    fn drop(&mut self) {
        give_config_buf(std::mem::take(&mut self.cphi));
        give_nodes(std::mem::take(&mut self.nodes), pool_budget(&self.cfg));
    }
}

impl Solver {
    pub fn new(root: &State, ctx: Ctx, net: Arc<Net>, cfg: Cfg, belief: [Belief; 2], rng: Rng) -> Solver {
        let root_configs: usize = belief.iter().map(Belief::len).sum();
        assert!(
            root_configs <= 2 * crate::pbs::MAX_CONFIG_SUPPORT,
            "root has {root_configs} configs, above the game's support bound"
        );
        assert!(
            root_configs <= cfg.budget.cap(Ent::Config),
            "device slot holds {} configs but this root needs {root_configs}",
            cfg.budget.cap(Ent::Config)
        );
        let cfgs: [Arc<[Config]>; 2] = [belief[0].cfg.as_slice().into(), belief[1].cfg.as_slice().into()];
        let mut sv = Solver::default();
        sv.ctx = ctx;
        sv.net = net;
        sv.rng = rng;
        sv.cfg = cfg;
        sv.root_belief = belief;
        sv.nodes = take_nodes();
        sv.cphi = take_config_buf();
        sv.cmap = std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash);
        sv.bmap = std::collections::HashMap::with_capacity_and_hasher(1024, KeyHash);
        sv.nodes.reserve(640);
        sv.cur.reserve(640);
        sv.seed = Rng::new(sv.rng.next_u64()).0;
        let root = sv.push_node(crate::contract::NO_ROW, root.clone(), cfgs);
        sv.expand(root);
        sv
    }

    pub fn pin(&mut self, slot: usize) {
        self.slot = slot;
    }

    pub fn collect(&mut self, queries: usize) {
        self.collect = Some(queries);
    }

    fn with_rng<T>(&mut self, f: impl FnOnce(&mut Self, &mut Rng) -> T) -> T {
        let mut rng = std::mem::replace(&mut self.rng, Rng(1));
        let out = f(self, &mut rng);
        self.rng = rng;
        out
    }

    fn plan_query_events(&mut self, events: usize) -> Vec<usize> {
        let keep = self.collect.unwrap_or(0);
        if keep == 0 || events == 0 {
            self.query_seen += events;
            return Vec::new();
        }
        self.with_rng(|sv, rng| {
            let total = sv.query_seen + events;
            sv.query_seen = total;
            if total <= keep {
                return (0..events).collect();
            }

            let mut new_left = events;
            let mut all_left = total;
            let mut new_count = 0;
            for _ in 0..keep {
                if rng.below(all_left) < new_left {
                    new_count += 1;
                    new_left -= 1;
                }
                all_left -= 1;
            }
            let old_count = keep - new_count;
            let old = sample_indices(rng, sv.queries.len(), old_count);
            sv.queries = std::mem::take(&mut sv.queries)
                .into_iter()
                .enumerate()
                .filter_map(|(i, row)| old.contains(&i).then_some(row))
                .collect();
            sample_indices(rng, events, new_count)
        })
    }

    fn leaf_query_rows(&self, from: usize) -> Vec<usize> {
        self.leaf_rows[from..]
            .iter()
            .copied()
            .filter(|&node| self.nodes[node].leaf)
            .collect()
    }

    fn absorb_queries(&mut self, reach: &[f32]) {
        let mut cut = 0;
        for node in std::mem::take(&mut self.query_nodes) {
            let beliefs = std::array::from_fn(|p| {
                let n = self.nodes[node].nc[p] as usize;
                let mut w = vec![0.0; n];
                normalize_weights(&reach[cut..cut + n], &mut w);
                cut += n;
                Belief {
                    cfg: self.nodes[node].cfgs[p].to_vec(),
                    p: w,
                }
            });
            self.queries.push((self.nodes[node].state, beliefs));
        }
        assert_eq!(cut, reach.len(), "query reach reply has a trailing tail");
    }

    fn push_node(&mut self, parent: u32, s: State, cfgs: [Arc<[Config]>; 2]) -> usize {
        let player = s.to_act();
        let terminal = s.is_terminal();
        let coin = !terminal && s.is_valued();
        let (c0, c1) = (cfgs[0].len(), cfgs[1].len());
        let next_row = if terminal {
            self.leaf_rows.len().max(self.term_leaves.len() + 1)
        } else if coin {
            (self.leaf_rows.len() + 1).max(self.term_leaves.len())
        } else {
            self.leaf_rows.len().max(self.term_leaves.len())
        };
        let next_cell = if parent == crate::contract::NO_ROW {
            self.ncells
        } else {
            self.ncells.max(self.nodes.len())
        };
        if !self.reserve(Ent::Node, self.nodes.len() + 1)
            || !self.reserve(Ent::Reach, self.nreach + c0 + c1)
            || !self.reserve(Ent::Reach, self.nvals + c0.max(c1))
            || !self.reserve(Ent::Cell, next_cell)
            || !self.reserve(Ent::Row, next_row)
            || (parent == crate::contract::NO_ROW && !self.reserve(Ent::Config, (c0 + c1).div_ceil(2)))
        {
            return parent as usize;
        }
        let id = self.nodes.len();
        self.nodes.push(TNode {
            state: s,
            parent,
            nc: [c0 as u32, c1 as u32],
            roff: self.nreach as u32,
            voff: self.nvals as u32,
            soff: self.ncells as u32,
            row_of: u32::MAX,
            primed: false,
            util: if terminal { s.utility(player as usize) } else { 0.0 },
            player,
            leaf: true,
            expandable: !terminal,
            exhausted: false,
            chance: false,
            draw: DrawMap::default(),
            draw_steps: 0,
            acts: Vec::new(),
            aslot: Vec::new(),
            fdown: Vec::new(),
            obs_child: Vec::new(),
            obs_start: Vec::new(),
            obs_act: Vec::new(),
            child: Vec::new(),
            cfgs: cfgs.clone(),
            legal_off: Vec::new(),
            legal_action: Vec::new(),
            legal_child: Vec::new(),
            legal_trans: Vec::new(),
            action_off: Vec::new(),
            action_cell: Vec::new(),
            cell_row: Vec::new(),
        });
        self.nreach += c0 + c1;
        self.nvals += c0.max(c1);
        if terminal {
            self.term_leaves.push(id);
        } else if coin {
            self.nodes[id].row_of = (self.leaf_coff.len() / 2) as u32;
            self.push_row(&s, &cfgs);
            if !self.abandon {
                self.leaf_rows.push(id);
            }
        }
        id
    }

    fn push_child(&mut self, parent: usize, s: State, cfgs: [Arc<[Config]>; 2]) -> usize {
        if self.abandon {
            return parent;
        }
        let stop = s.is_terminal() || s.is_valued();
        let ch = self.push_node(parent as u32, s, cfgs);
        if self.abandon {
            return ch;
        }
        if !stop {
            self.grow(ch);
        }
        ch
    }

    fn expand(&mut self, id: usize) {
        debug_assert!(
            self.nodes[id].leaf && self.nodes[id].expandable,
            "growth turns an expandable leaf into a decision node, and nothing else"
        );
        let fresh = self.nodes.len();
        self.grow(id);
        if self.abandon {
            self.abandon = false;
            self.nodes[id].expandable = false;
        }
        if !self.nodes[id].leaf {
            self.expansions += 1;
        }
        self.seal(id, fresh);
    }

    fn seal(&mut self, id: usize, fresh: usize) {
        for i in (fresh..self.nodes.len()).rev() {
            self.set_exhausted(i);
        }
        let mut at = id;
        loop {
            if !self.set_exhausted(at) {
                return;
            }
            match self.nodes[at].parent {
                crate::contract::NO_ROW => return,
                p => at = p as usize,
            }
        }
    }

    fn set_exhausted(&mut self, i: usize) -> bool {
        let n = &self.nodes[i];
        let e = if n.leaf {
            !n.expandable
        } else {
            n.child.iter().all(|&c| self.nodes[c].exhausted)
        };
        if e != self.nodes[i].exhausted {
            self.nodes[i].exhausted = e;
            self.resealed.push(i as u32);
        }
        e
    }

    pub(crate) fn take_resealed(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.resealed)
    }

    pub fn budget_hit(&self) -> bool {
        self.budget_hit != 0
    }

    pub fn hit_mask(&self) -> u8 {
        self.budget_hit
    }

    pub fn used(&self, e: Ent) -> usize {
        match e {
            Ent::Node => self.nodes.len(),
            Ent::Cell => self.ncells.max(self.nodes.len().saturating_sub(1)),
            Ent::Reach => self.nreach.max(self.nvals).max(self.reach_aux()),
            Ent::Draw => self.ndraws,
            Ent::Row => self.leaf_rows.len().max(self.term_leaves.len()),
            Ent::Board => self.nboards,
            Ent::Config => self.ncfg.max(self.rootb_len()),
            Ent::Cidx => self.leaf_cidx.len(),
        }
    }

    fn reach_aux(&self) -> usize {
        self.nlegal_off
            .max(self.nrev_start)
            .max(self.nrvd_start)
            .max(self.ndraw_start)
    }

    fn rootb_len(&self) -> usize {
        match self.nodes.first() {
            Some(n) => (n.nc[0] as usize + n.nc[1] as usize).div_ceil(2),
            None => 0,
        }
    }

    pub fn stop_reason(&self) -> u32 {
        if self.budget_hit != 0 {
            debug_assert!(self.budget_hit.is_power_of_two());
            return 2 + self.budget_hit.trailing_zeros();
        }
        if self.expansions >= self.cfg.s {
            0
        } else if self.nodes[0].exhausted {
            1
        } else {
            (STOP_NAMES.len() - 1) as u32
        }
    }

    pub fn counts(&self) -> [u32; 9] {
        let mut out = [0; 9];
        for (i, e) in Ent::ALL.into_iter().enumerate() {
            out[i] = self.used(e) as u32;
        }
        out[8] = self.stop_reason();
        out
    }

    fn reserve(&mut self, e: Ent, n: usize) -> bool {
        if n > self.cfg.budget.cap(e) {
            self.abandon = true;
            self.budget_hit |= 1 << (e as u8);
            false
        } else {
            true
        }
    }

    fn mark(&self) -> Mark {
        Mark {
            nodes: self.nodes.len(),
            ncells: self.ncells,
            ncfg: self.ncfg,
            ndraws: self.ndraws,
            leaf_rows: self.leaf_rows.len(),
            nboards: self.nboards,
            term_leaves: self.term_leaves.len(),
            leaf_coff: self.leaf_coff.len(),
            leaf_cidx: self.leaf_cidx.len(),
            nlegal_off: self.nlegal_off,
            nrev_start: self.nrev_start,
            nrvd_start: self.nrvd_start,
            ndraw_start: self.ndraw_start,
        }
    }

    fn rewind(&mut self, id: usize, m: Mark) {
        if let Some(n) = self.nodes.get(m.nodes) {
            self.nreach = n.roff as usize;
            self.nvals = n.voff as usize;
        }
        self.nodes.truncate(m.nodes);
        self.resealed.retain(|&i| (i as usize) < m.nodes);
        self.wants_prior.retain(|&i| (i as usize) < m.nodes);
        self.leaf_rows.truncate(m.leaf_rows);
        self.board_of.truncate(m.leaf_rows);
        self.nboards = m.nboards;
        self.bmap.retain(|_, &mut b| (b as usize) < m.nboards);
        self.term_leaves.truncate(m.term_leaves);
        self.leaf_coff.truncate(m.leaf_coff);
        self.leaf_cidx.truncate(m.leaf_cidx);
        self.cur.truncate(m.ncells);
        self.ncells = m.ncells;
        self.ndraws = m.ndraws;
        self.cplayer.truncate(m.ncfg);
        self.cmap.retain(|_, &mut i| (i as usize) < m.ncfg);
        self.ncfg = m.ncfg;
        self.nlegal_off = m.nlegal_off;
        self.nrev_start = m.nrev_start;
        self.nrvd_start = m.nrvd_start;
        self.ndraw_start = m.ndraw_start;
        self.grown.retain(|&g| (g as usize) < m.nodes && g != id as u32);
        let n = &mut self.nodes[id];
        n.leaf = true;
        n.chance = false;
        n.draw = DrawMap::default();
        n.draw_steps = 0;
        n.acts.clear();
        n.aslot.clear();
        n.fdown.clear();
        n.obs_child.clear();
        n.obs_start.clear();
        n.obs_act.clear();
        n.child.clear();
        n.legal_off.clear();
        n.legal_action.clear();
        n.legal_child.clear();
        n.legal_trans.clear();
        n.action_off.clear();
        n.action_cell.clear();
        n.cell_row.clear();
    }

    fn alloc_cells(&mut self, id: usize) {
        let cells = self.nodes[id].legal_action.len();
        self.nodes[id].soff = self.ncells as u32;
        self.ncells += cells;
        let n = &self.nodes[id];
        let nc = n.nc[n.player as usize] as usize;
        let mut u = vec![0.0f32; cells];
        for c in 0..nc {
            let row = n.legal_row(c);
            let k = row.len() as f32;
            for cell in row {
                u[cell] = 1.0 / k;
            }
        }
        self.cur.extend_from_slice(&u);
    }

    fn grow(&mut self, id: usize) {
        debug_assert!(self.nodes[id].leaf, "only a leaf can be grown");
        if self.nodes[id].row_of != u32::MAX {
            self.wants_prior.push(id as u32);
        }
        if self.abandon {
            return;
        }
        let mark = self.mark();
        let s = self.nodes[id].state;
        debug_assert!(!s.is_terminal(), "a terminal has nothing to grow");
        let cfgs = self.nodes[id].cfgs.clone();
        self.nodes[id].leaf = false;
        let player = s.to_act();
        let draw_pass = matches!(s.pending(), Cont::Draw { .. } | Cont::WarriorPriestDraw { .. });
        if draw_pass {
            let me = player as usize;
            let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
            let mut cs = s;
            set_config(&mut cs, player, &self.ctx, &cfgs[me][0]);
            let mut support: Vec<Config> = Vec::new();
            let mut draw = DrawMap::default();
            let res = reserve(&cs, player, &self.ctx);
            let fu = faceup_counts(&cs, player, &self.ctx);
            let mut steps = 0u8;
            loop {
                let unit = cs.first_drawable(player).unwrap_or(crate::board::NONE);
                cs.apply_inplace(Action::DrawCoin { unit });
                steps += 1;
                if !(matches!(cs.pending(), Cont::Draw { .. }) && cs.to_act() == player) {
                    break;
                }
            }
            if wp {
                debug_assert_eq!(steps, 1, "a WP draw is a single forced draw");
                self.draw_scratch
                    .transition(&cfgs[me], &res, &fu, &mut support, &mut draw, true);
            } else {
                self.draw_scratch
                    .run(&cfgs[me], &res, &fu, steps, &mut support, &mut draw);
            }
            let mut cc = cfgs;
            cc[me] = support.as_slice().into();
            let ch = self.push_child(id, cs, cc);
            if self.abandon {
                self.rewind(id, mark);
                return;
            }
            if !wp && cs.round > self.nodes[0].state.round + self.cfg.rounds as u16 {
                for n in &mut self.nodes[mark.nodes..] {
                    n.expandable = false;
                }
            }
            let extra_draw_start = draw.rows() + 1;
            let extra_rvd = self.nodes[ch].nc[me] as usize + 1;
            let n = &mut self.nodes[id];
            n.chance = true;
            n.child = vec![ch];
            if !self.reserve(Ent::Draw, self.ndraws + draw.len())
                || !self.reserve(Ent::Reach, self.nrvd_start + extra_rvd)
                || !self.reserve(Ent::Reach, self.ndraw_start + extra_draw_start)
            {
                self.rewind(id, mark);
                return;
            }
            self.ndraws += draw.len();
            self.ndraw_start += extra_draw_start;
            self.nrvd_start += extra_rvd;
            let n = &mut self.nodes[id];
            n.draw = draw;
            n.draw_steps = steps;
            self.grown.push(id as u32);
            return;
        }

        let me = player as usize;
        let mine = cfgs[me].clone();
        let nc = mine.len();
        let (acts, aslot, fdown) = node_actions(&s, player, &self.ctx, &mine);
        let na = acts.len();
        debug_assert!(na > 0, "a decision node must offer a reachable action");
        let mut legal_off = Vec::with_capacity(nc + 1);
        let mut legal_action = Vec::new();
        let mut legal_child = Vec::new();
        let mut legal_trans = Vec::new();
        let mut cell_row = Vec::new();
        legal_off.push(0);
        for (ci, c) in mine.iter().enumerate() {
            for a in 0..na {
                if action_legal(c, aslot[a]) {
                    legal_action.push(a as u32);
                    legal_child.push(0);
                    legal_trans.push(NO_TRANS);
                    cell_row.push(ci as u32);
                }
            }
            legal_off.push(legal_action.len() as u32);
        }
        let mut obs_keys: Vec<u32> = Vec::new();
        let mut obs_child = vec![0usize; na];
        for a in 0..na {
            let k = obs_key(&acts[a]);
            obs_child[a] = match obs_keys.iter().position(|&x| x == k) {
                Some(i) => i,
                None => {
                    obs_keys.push(k);
                    obs_keys.len() - 1
                }
            };
        }
        let nch = obs_keys.len();
        let (obs_start, obs_act) = group_by(obs_child.iter().copied(), nch);
        for (cell, &au) in legal_action.iter().enumerate() {
            legal_child[cell] = obs_child[au as usize] as u32;
        }
        let (action_off, action_cell) = group_by(legal_action.iter().map(|&a| a as usize), na);
        let mut child_cfgs: Vec<Vec<Config>> = vec![Vec::new(); nch];
        let mut ent = std::mem::take(&mut self.cell_order);
        for ch in 0..nch {
            ent.clear();
            for &au in &obs_act[obs_start[ch] as usize..obs_start[ch + 1] as usize] {
                let a = au as usize;
                for &cell_u in &action_cell[action_off[a] as usize..action_off[a + 1] as usize] {
                    let cell = cell_u as usize;
                    if legal_child[cell] as usize != ch {
                        continue;
                    }
                    let ci = cell_row[cell] as usize;
                    if let Some(n) = advance_config(&mine[ci], aslot[a], fdown[a]) {
                        ent.push((n.key(), cell_u));
                    }
                }
            }
            ent.sort_unstable_by_key(|&(key, cell)| (key, cell));
            let sup = &mut child_cfgs[ch];
            let mut prev = u64::MAX;
            for &(k, cell_u) in ent.iter() {
                let cell = cell_u as usize;
                if k != prev {
                    prev = k;
                    let ci = cell_row[cell] as usize;
                    let a = legal_action[cell] as usize;
                    sup.push(advance_config(&mine[ci], aslot[a], fdown[a]).unwrap());
                }
                legal_trans[cell] = (sup.len() - 1) as u32;
            }
        }
        self.cell_order = ent;

        let mut child = Vec::with_capacity(nch);
        for ch in 0..nch {
            let a = obs_act[obs_start[ch] as usize] as usize;
            let rep = *mine
                .iter()
                .find(|c| action_legal(c, aslot[a]))
                .expect("a kept action is playable by some config in the support");
            let mut cs = s.clone();
            set_config(&mut cs, player, &self.ctx, &rep);
            cs.apply_inplace(acts[a]);
            let mut cc = cfgs.clone();
            cc[me] = std::mem::take(&mut child_cfgs[ch]).into();
            child.push(self.push_child(id, cs, cc));
            if self.abandon {
                self.rewind(id, mark);
                return;
            }
        }

        let extra_legal = legal_off.len();
        let extra_rev: usize = child.iter().map(|&c| self.nodes[c].nc[me] as usize + 1).sum();
        let extra_cells = legal_action.len();
        let n = &mut self.nodes[id];
        n.acts = acts;
        n.aslot = aslot;
        n.fdown = fdown;
        n.obs_child = obs_child;
        n.obs_start = obs_start;
        n.obs_act = obs_act;
        n.child = child;
        for c in &mut legal_child {
            *c = n.child[*c as usize] as u32;
        }
        n.legal_off = legal_off;
        n.legal_action = legal_action;
        n.legal_child = legal_child;
        n.legal_trans = legal_trans;
        n.action_off = action_off;
        n.action_cell = action_cell;
        n.cell_row = cell_row;
        if !self.reserve(
            Ent::Cell,
            (self.ncells + extra_cells).max(self.nodes.len().saturating_sub(1)),
        ) || !self.reserve(Ent::Reach, self.nlegal_off + extra_legal)
            || !self.reserve(Ent::Reach, self.nrev_start + extra_rev)
        {
            self.rewind(id, mark);
            return;
        }
        self.nlegal_off += extra_legal;
        self.nrev_start += extra_rev;
        self.alloc_cells(id);
        self.grown.push(id as u32);
    }

    fn encode(&mut self, s: &State) -> u32 {
        let at = self.nboards * ROW_BYTES;
        if self.packed.len() < at + ROW_BYTES {
            self.packed.resize(at + 128 * ROW_BYTES, 0);
        }
        pack_row(s, &self.ctx, &mut self.packed[at..at + ROW_BYTES]);
        let key = row_key(&self.packed[at..at + ROW_BYTES]);
        if let Some(&b) = self.bmap.get(&key) {
            let old = b as usize * ROW_BYTES;
            if self.packed[old..old + ROW_BYTES] == self.packed[at..at + ROW_BYTES] {
                return b;
            }
        }
        if !self.reserve(Ent::Board, self.nboards + 1) {
            return 0;
        }
        let b = self.nboards as u32;
        self.bmap.insert(key, b);
        self.nboards += 1;
        b
    }

    fn push_row(&mut self, s: &State, cfgs: &[Arc<[Config]>; 2]) {
        debug_assert!(s.is_valued(), "a network row must be a valued decision");
        if !self.reserve(Ent::Row, (self.leaf_rows.len() + 1).max(self.term_leaves.len())) {
            return;
        }
        let board = self.encode(s);
        self.board_of.push(board);
        for p in 0..2 {
            let res = reserve(s, p as u8, &self.ctx);
            self.leaf_coff.push(self.leaf_cidx.len() as u32);
            for c in cfgs[p].iter() {
                if !self.reserve(Ent::Cidx, self.leaf_cidx.len() + 1) {
                    break;
                }
                let idx = self.intern_config(c, &res, p);
                self.leaf_cidx.push(idx);
            }
        }
    }

    fn intern_config(&mut self, c: &Config, res: &[u8; NSLOT], p: usize) -> u32 {
        let mut cnt = [0u8; CCOUNTS];
        config_counts(c, res, &mut cnt);
        let mut key = p as u64;
        for x in cnt.iter() {
            debug_assert!(*x < 16, "count over the key width");
            key = (key << 4) | *x as u64;
        }
        if let Some(&i) = self.cmap.get(&key) {
            return i;
        }
        if !self.reserve(Ent::Config, self.ncfg + 1) {
            return 0;
        }
        let i = self.ncfg as u32;
        self.ncfg += 1;
        let at = i as usize * CFEAT;
        if self.cphi.len() < at + CFEAT {
            self.cphi.resize(at + 64 * CFEAT, 0.0);
        }
        for k in 0..CCOUNTS {
            self.cphi[at + k] = cnt[k] as f32 / CNORM;
        }
        self.cplayer.push(p as u8);
        self.cmap.insert(key, i);
        i
    }

    fn growth_calls(&mut self) -> Vec<Call> {
        let rows = self.leaf_rows.len();
        if rows == self.batch_rows && self.ncfg == self.batch_cfgs {
            return Vec::new();
        }
        if self.net.is_empty() {
            self.batch_rows = rows;
            self.batch_cfgs = self.ncfg;
            return Vec::new();
        }
        let mut calls = Vec::with_capacity(2);
        let fresh_rows = rows - self.batch_rows;
        let fresh_cfgs = self.ncfg - self.batch_cfgs;
        if self.cards.is_empty() && (fresh_rows > 0 || fresh_cfgs > 0) {
            let (me, other) = (self.ctx.slots[0], self.ctx.slots[1]);
            self.net.cards(&[me, other, other, me].concat(), &mut self.cards);
        }
        if fresh_rows > 0 {
            let at = self.batch_boards * ROW_BYTES;
            let end = self.nboards * ROW_BYTES;
            let q0 = 2 * self.batch_rows;
            let cs = self.leaf_coff[q0] as usize;
            let mut coff: Vec<u32> = self.leaf_coff[q0..].iter().map(|x| x - cs as u32).collect();
            coff.push((self.leaf_cidx.len() - cs) as u32);
            calls.push(Call::Trunk {
                solve: self.slot,
                at: self.batch_rows,
                queries: fresh_rows,
                board_of: self.board_of[self.batch_rows..].to_vec(),
                boards_at: self.batch_boards,
                boards: self.nboards - self.batch_boards,
                packed: self.packed[at..end].to_vec(),
                cards: self.cards.clone(),
                cidx: self.leaf_cidx[cs..].to_vec(),
                coff,
            });
        }
        if fresh_cfgs > 0 {
            calls.push(Call::Configs {
                solve: self.slot,
                at: self.batch_cfgs,
                phi: self.cphi[self.batch_cfgs * CFEAT..self.ncfg * CFEAT].to_vec(),
                owner: self.cplayer[self.batch_cfgs..].iter().map(|&p| p as u32).collect(),
                cards: self.cards.clone(),
                n: fresh_cfgs,
            });
        }
        calls
    }

    fn absorb(&mut self) {
        if self.leaf_rows.len() > self.batch_rows {
            self.batch_rows = self.leaf_rows.len();
            self.batch_boards = self.nboards;
        }
        self.batch_cfgs = self.batch_cfgs.max(self.ncfg);
    }

    fn expansions_at(&self, i: usize) -> usize {
        if self.budget_hit() || self.nodes[0].exhausted {
            0
        } else {
            self.cfg.expansions_at(i)
        }
    }

    pub fn advance(&mut self, replies: &[Reply]) -> Step {
        match self.phase {
            Phase::Fresh => {}
            Phase::Iterating => {
                self.absorb();
                let last = replies.last().expect("a round answers every call it was given");
                self.absorb_queries(&last.c);
                for &leaf in &last.leaves.clone() {
                    if leaf == crate::contract::NO_ROW {
                        continue;
                    }
                    if self.budget_hit() {
                        break;
                    }
                    self.expand(leaf as usize);
                }
            }
            Phase::Reading => {
                self.absorb();
                self.phase = Phase::Done;
                let last = replies.last().expect("a round answers every call it was given");
                return Step::Done(self.read_back(last));
            }
            Phase::Done => unreachable!("a finished solve is not advanced again"),
        }
        if self.at < self.cfg.iters() {
            self.phase = Phase::Iterating;
            Step::Calls(self.iterate_round())
        } else {
            self.phase = Phase::Reading;
            Step::Calls(self.read_round())
        }
    }

    fn round_shape(&self) -> (usize, usize) {
        let (iters, at) = (self.cfg.iters(), self.at);
        let want = self.expansions_at(at + 1);
        let done = (at + 1..=iters)
            .take_while(|&k| self.expansions_at(k) == want)
            .count()
            .min(self.cfg.batch.max(1));
        (done, want)
    }

    fn iterate_round(&mut self) -> Vec<Call> {
        let (done, expand) = self.round_shape();
        let mut calls = self.opening_calls();
        let query_rows = self.leaf_query_rows(0);
        let rows = query_rows.len();
        let selected = self.plan_query_events(done * rows);
        self.query_nodes = selected.iter().map(|&e| query_rows[e % rows]).collect();
        let query = selected
            .into_iter()
            .map(|e| {
                let node = query_rows[e % rows];
                QueryPick {
                    iter: (e / rows) as u32,
                    reach: self.nodes[node].roff,
                    len: self.nodes[node].nc[0] + self.nodes[node].nc[1],
                }
            })
            .collect();
        calls.push(Call::Iterate {
            solve: self.slot,
            step: self.steps[0],
            iters: done,
            expand,
            query,
            cfr: self.cfg.cfr,
            puct: self.cfg.puct,
        });
        self.steps = [self.steps[0] + done, self.steps[1] + done];
        self.avg_touched = [true; 2];
        self.at += done;
        calls
    }

    fn opening_calls(&mut self) -> Vec<Call> {
        let mut calls = self.growth_calls();
        calls.push(self.tree_call());
        calls
    }

    fn read_round(&mut self) -> Vec<Call> {
        let mut calls = self.opening_calls();
        let nvals = self.nvals as u32;
        let vals_at = match self.collect {
            None => [(0, 0); 2],
            Some(_) => [(self.nodes[0].voff, self.nodes[0].nc[0]), (nvals + self.nodes[0].voff, self.nodes[0].nc[1])],
        };
        let (at, cells) = self.root_cells();
        calls.push(Call::Read {
            solve: self.slot,
            touched: self.avg_touched,
            vals_at,
            policy_at: (at as u32, cells as u32),
        });
        calls
    }

    fn read_back(&mut self, r: &Reply) -> Option<Solved> {
        let (at, cells) = self.root_cells();
        self.avg = vec![0.0; at + cells];
        self.avg[at..at + cells].copy_from_slice(&r.b);
        self.collect?;
        let n0 = self.nodes[0].nc[0] as usize;
        let value = [r.a[..n0].to_vec(), r.a[n0..].to_vec()];
        let policy = self.root_policy();
        let queries = std::mem::take(&mut self.queries);
        Some(Solved { value, queries, policy })
    }

    fn root_cells(&self) -> (usize, usize) {
        let n = &self.nodes[0];
        if n.leaf || n.chance {
            (0, 0)
        } else {
            (n.soff as usize, n.legal_action.len())
        }
    }

    fn tree_call(&mut self) -> Call {
        self.contract_extend();
        let sent = self.sent_cells;
        self.sent_cells = self.ncells;
        let first = self.steps[0] == 0 && self.sent_from == 0;
        if first {
            self.sent = Default::default();
        }
        let mut w = Writes::default();
        let resent = std::mem::take(&mut self.resent);
        self.contract
            .write_into(&mut w, &mut self.sent, self.sent_from, &self.rewrite, &resent);
        w.u32s(
            Dst::LeafNode,
            0,
            &self.leaf_rows.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        );
        w.u32s(
            Dst::Term,
            0,
            &self.term_leaves.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        );
        if first {
            let b = [&self.root_belief[0].p[..], &self.root_belief[1].p[..]].concat();
            w.f32s(Dst::Rootb, 0, &b);
        }
        w.f32s_both(Dst::Cur, Dst::Prior, sent, &self.cur[sent..]);
        let (prime, acts, cells) = self.prime();
        let call = Call::Tree {
            solve: self.slot,
            writes: w,
            fresh: first,
            ncells: self.ncells,
            nreach: self.nreach,
            nvals: self.nvals,
            levels: self.contract.level_start.clone(),
            nterm: self.term_leaves.len(),
            seed: first.then_some(self.seed),
            prime,
            acts,
            cells,
            prior_temp: self.cfg.prior_temp,
        };
        self.sent_from = self.nodes.len();
        call
    }

    fn contract_extend(&mut self) {
        let grown = std::mem::take(&mut self.grown);
        let resealed = self.take_resealed();
        self.sent_from = self.contract.built;
        self.rewrite.clear();
        self.rewrite
            .extend(grown.iter().copied().filter(|&g| (g as usize) < self.sent_from));
        let mut c = std::mem::take(&mut self.contract);
        Arc::make_mut(&mut c).extend(self, &grown, &resealed);
        self.contract = c;
        self.resent = resealed;
    }

    fn ready_for_prior(&mut self) -> Vec<usize> {
        let mut want: Vec<usize> = Vec::new();
        let mut queue = std::mem::take(&mut self.wants_prior);
        queue.retain(|&i| {
            let i = i as usize;
            if self.nodes[i].primed || self.nodes[i].leaf || self.nodes[i].chance {
                return false;
            }
            if self.nodes[i].row_of == u32::MAX {
                return false;
            }
            if (self.nodes[i].row_of as usize) < self.batch_rows {
                want.push(i);
                return false;
            }
            true
        });
        self.wants_prior = queue;
        want
    }

    pub(crate) fn prime(&mut self) -> (Vec<Prime>, Vec<u32>, Vec<u32>) {
        let want = self.ready_for_prior();
        let (mut prime, mut acts, mut cells) = (Vec::new(), Vec::new(), Vec::new());
        for i in want {
            let n = &self.nodes[i];
            prime.push(Prime {
                node: i as u32,
                row: n.row_of,
                at: (acts.len() / ACT_BYTES) as u32,
                na: n.na() as u32,
                cell_at: cells.len() as u32,
                nc: n.nc[n.player as usize],
            });
            for a in 0..n.na() {
                acts.extend(action_desc(&n.acts[a], n.player, &self.ctx, n.aslot[a]).map(u32::from));
            }
            cells.extend_from_slice(&n.legal_action);
            self.nodes[i].primed = true;
        }
        (prime, acts, cells)
    }

    pub fn root_policy(&self) -> Policy {
        let n = &self.nodes[0];
        if n.leaf || n.chance || self.avg.is_empty() {
            return Policy::default();
        }
        let me = n.player as usize;
        let mut out = Policy {
            acts: (0..n.na())
                .map(|a| action_desc(&n.acts[a], n.player, &self.ctx, n.aslot[a]))
                .collect(),
            ..Default::default()
        };
        let so = self.nodes[0].soff as usize;
        out.off.push(0);
        for c in 0..n.nc[me] as usize {
            for cell in n.legal_row(c) {
                out.act.push(n.legal_action[cell] as u16);
                out.p.push(self.avg[so + cell]);
            }
            out.off.push(out.act.len() as u32);
        }
        out
    }

    pub(crate) fn root_strategy(&self, config: usize) -> &[f32] {
        let row = self.nodes[0].legal_row(config);
        assert!(!self.avg.is_empty(), "the solve must finish before its average is read");
        let so = self.nodes[0].soff as usize;
        &self.avg[so + row.start..so + row.end]
    }
}