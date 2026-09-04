use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::state::{State, CONT_CAP, PENDING_KINDS, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_HAND, Z_INFLIGHT, Z_SUPPLY};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

pub const NSLOT: usize = 5;
pub const HAND_CAP: usize = 3;
pub const MAX_CONFIG_SUPPORT: usize = 4_628;

pub const CCOUNTS: usize = 3 * NSLOT;
pub const CFEAT: usize = CCOUNTS;
pub const CNORM: f32 = 5.0;

#[derive(Clone, Copy, Default)]
pub struct Ctx {
    pub slots: [[u8; NSLOT]; 2],
    pub slot_of: [[i8; N_UNITS]; 2],
}

impl Ctx {
    pub fn new(s: &State) -> Ctx {
        let mut slots = [[0u8; NSLOT]; 2];
        let mut slot_of = [[-1i8; N_UNITS]; 2];
        for p in 0..2usize {
            let mut n = 0;
            for u in 0..N_UNITS {
                if s.total_coins(p as u8, u) > 0 {
                    assert!(n < NSLOT, "a player may own at most {} coin types", NSLOT);
                    slots[p][n] = u as u8;
                    slot_of[p][u] = n as i8;
                    n += 1;
                }
            }
            assert_eq!(n, NSLOT, "expected exactly {} coin types per player", NSLOT);
        }
        Ctx { slots, slot_of }
    }

    pub fn mirrored(self) -> Ctx {
        Ctx {
            slots: [self.slots[1], self.slots[0]],
            slot_of: [self.slot_of[1], self.slot_of[0]],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct Config {
    pub hand: [u8; NSLOT],
    pub fd: [u8; NSLOT],
    pub inflight: Option<u8>,
}

impl Config {
    #[inline]
    pub fn hand_size(&self) -> u8 {
        self.hand.iter().sum()
    }
    #[inline]
    pub fn fd_size(&self) -> u8 {
        self.fd.iter().sum()
    }
    #[inline]
    pub fn bag(&self, reserve: &[u8; NSLOT]) -> [u8; NSLOT] {
        let mut b = [0u8; NSLOT];
        for k in 0..NSLOT {
            b[k] = reserve[k].saturating_sub(self.hand[k] + self.fd[k] + u8::from(self.inflight == Some(k as u8)));
        }
        b
    }
    #[inline]
    pub fn key(&self) -> u64 {
        let mut k = 0u64;
        for i in 0..NSLOT {
            debug_assert!(self.hand[i] < 4, "hand slot over the key width");
            k = (k << 2) | self.hand[i] as u64;
        }
        for i in 0..NSLOT {
            debug_assert!(self.fd[i] < 32, "face-down slot over the key width");
            k = (k << 5) | self.fd[i] as u64;
        }
        debug_assert!(self.inflight.map_or(0, |p| p as u64 + 1) < 8);
        k = (k << 3) | self.inflight.map_or(0, |p| p as u64 + 1);
        k
    }
}

#[inline]
pub fn config_counts(c: &Config, reserve: &[u8; NSLOT], out: &mut [u8; CCOUNTS]) {
    for k in 0..NSLOT {
        let inflight = u8::from(c.inflight == Some(k as u8));
        out[k] = c.hand[k];
        out[NSLOT + k] = c.fd[k];
        out[2 * NSLOT + k] = reserve[k].saturating_sub(c.hand[k] + c.fd[k] + inflight);
    }
}

#[inline]
pub fn write_config_feats(c: &Config, reserve: &[u8; NSLOT], out: &mut [f32]) {
    debug_assert_eq!(out.len(), CFEAT);
    let mut cnt = [0u8; CCOUNTS];
    config_counts(c, reserve, &mut cnt);
    for k in 0..CCOUNTS {
        out[k] = cnt[k] as f32 / CNORM;
    }
}

pub fn reserve(s: &State, p: u8, ctx: &Ctx) -> [u8; NSLOT] {
    let mut out = [0u8; NSLOT];
    for k in 0..NSLOT {
        let u = ctx.slots[p as usize][k] as usize;
        out[k] = s.zones[p as usize][Z_BAG][u]
            + s.zones[p as usize][Z_HAND][u]
            + s.zones[p as usize][Z_FACEDOWN][u]
            + s.zones[p as usize][Z_INFLIGHT][u];
    }
    out
}

pub fn true_config(s: &State, p: u8, ctx: &Ctx) -> Config {
    let mut c = Config::default();
    for k in 0..NSLOT {
        let u = ctx.slots[p as usize][k] as usize;
        c.hand[k] = s.zones[p as usize][Z_HAND][u];
        c.fd[k] = s.zones[p as usize][Z_FACEDOWN][u];
        if s.zones[p as usize][Z_INFLIGHT][u] > 0 {
            c.inflight = Some(k as u8);
        }
    }
    c
}

pub fn set_config(s: &mut State, p: u8, ctx: &Ctx, c: &Config) {
    debug_assert_eq!(
        u8::from(c.inflight.is_some()),
        s.zones[p as usize][Z_INFLIGHT].iter().copied().sum::<u8>(),
        "in-flight size is public"
    );
    for k in 0..NSLOT {
        let u = ctx.slots[p as usize][k] as usize;
        let inflight = u8::from(c.inflight == Some(k as u8));
        let res = s.zones[p as usize][Z_BAG][u]
            + s.zones[p as usize][Z_HAND][u]
            + s.zones[p as usize][Z_FACEDOWN][u]
            + s.zones[p as usize][Z_INFLIGHT][u];
        debug_assert!(c.hand[k] + c.fd[k] + inflight <= res, "config must fit in the reserve");
        s.zones[p as usize][Z_HAND][u] = c.hand[k];
        s.zones[p as usize][Z_FACEDOWN][u] = c.fd[k];
        s.zones[p as usize][Z_INFLIGHT][u] = inflight;
        s.zones[p as usize][Z_BAG][u] = res - c.hand[k] - c.fd[k] - inflight;
    }
}

pub fn uniform_belief(s: &State, ctx: &Ctx, p: u8) -> Belief {
    let truth = true_config(s, p, ctx);
    let cfg = enumerate_configs(
        &reserve(s, p, ctx),
        truth.hand_size(),
        truth.fd_size(),
        truth.inflight.is_some(),
    );
    let w = 1.0 / cfg.len().max(1) as f32;
    Belief {
        p: vec![w; cfg.len()],
        cfg,
    }
}

fn spread(room: &[u8; NSLOT], left: u8, out: &mut impl FnMut(&[u8; NSLOT])) {
    fn rec(room: &[u8; NSLOT], bin: &mut [u8; NSLOT], k: usize, left: u8, out: &mut impl FnMut(&[u8; NSLOT])) {
        if k == NSLOT - 1 {
            if left <= room[k] {
                bin[k] = left;
                out(bin);
                bin[k] = 0;
            }
            return;
        }
        for t in 0..=left.min(room[k]) {
            bin[k] = t;
            rec(room, bin, k + 1, left - t, out);
        }
        bin[k] = 0;
    }
    rec(room, &mut [0u8; NSLOT], 0, left, out);
}

pub fn enumerate_configs(reserve: &[u8; NSLOT], hand_size: u8, fd_size: u8, inflight: bool) -> Vec<Config> {
    let mut out = Vec::new();
    spread(reserve, hand_size, &mut |hand| {
        let free = std::array::from_fn(|k| reserve[k] - hand[k]);
        spread(&free, fd_size, &mut |fd| {
            let (hand, fd) = (*hand, *fd);
            if !inflight {
                out.push(Config { hand, fd, inflight: None });
                return;
            }
            for s in 0..NSLOT {
                if hand[s] + fd[s] < reserve[s] {
                    out.push(Config { hand, fd, inflight: Some(s as u8) });
                }
            }
        });
    });
    out
}

#[derive(Clone, Debug, Default)]
pub struct Belief {
    pub cfg: Vec<Config>,
    pub p: Vec<f32>,
}

pub(crate) const SMOOTH: f32 = 1e-30;

impl Belief {
    pub fn point(c: Config) -> Belief {
        Belief {
            cfg: vec![c],
            p: vec![1.0],
        }
    }

    pub fn len(&self) -> usize {
        self.cfg.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cfg.is_empty()
    }

    pub fn from_pairs(mut pairs: Vec<(Config, f32)>) -> Belief {
        pairs.sort_unstable_by_key(|a| a.0);
        let mut cfg: Vec<Config> = Vec::with_capacity(pairs.len());
        let mut p: Vec<f32> = Vec::with_capacity(pairs.len());
        for (c, w) in pairs {
            if w < 0.0 {
                continue;
            }
            if cfg.last() == Some(&c) {
                *p.last_mut().unwrap() += w;
            } else {
                cfg.push(c);
                p.push(w);
            }
        }
        let mut b = Belief { cfg, p };
        b.normalize();
        b
    }

    pub fn normalize(&mut self) {
        let s: f32 = self.p.iter().sum();
        if s > SMOOTH {
            for v in self.p.iter_mut() {
                *v /= s;
            }
        } else {
            let n = self.p.len().max(1) as f32;
            for v in self.p.iter_mut() {
                *v = 1.0 / n;
            }
        }
    }

    pub fn index_of(&self, c: &Config) -> Option<usize> {
        self.cfg.binary_search(c).ok()
    }
}

#[inline]
pub fn action_legal(c: &Config, slot: i8) -> bool {
    match c.inflight {
        Some(p) => slot == p as i8,
        None => slot < 0 || c.hand[slot as usize] > 0,
    }
}

#[inline]
pub fn advance_config(c: &Config, slot: i8, facedown: bool) -> Option<Config> {
    if let Some(k) = c.inflight {
        if slot != k as i8 {
            return None;
        }
        let mut n = *c;
        n.inflight = None;
        if facedown {
            n.fd[k as usize] += 1;
        }
        return Some(n);
    }
    if slot < 0 {
        return Some(*c);
    }
    let k = slot as usize;
    if c.hand[k] == 0 {
        return None;
    }
    let mut n = *c;
    n.hand[k] -= 1;
    if facedown {
        n.fd[k] += 1;
    }
    Some(n)
}

fn draw_children(
    c: &Config,
    reserve: &[u8; NSLOT],
    faceup: &[u8; NSLOT],
    set_pending: bool,
    mut emit: impl FnMut(Config, f32),
) {
    let (src, base) = draw_source(c, reserve, faceup);
    let total: u32 = src.iter().map(|&x| x as u32).sum();
    if total == 0 {
        emit(*c, 1.0);
        return;
    }
    let inv = 1.0 / total as f32;
    for k in 0..NSLOT {
        if src[k] == 0 {
            continue;
        }
        let mut n = base;
        if set_pending {
            debug_assert!(n.inflight.is_none(), "a coin is already in flight");
            n.inflight = Some(k as u8);
            emit(n, src[k] as f32 * inv);
        } else {
            n.hand[k] += 1;
            if n.hand_size() as usize <= HAND_CAP {
                emit(n, src[k] as f32 * inv);
            }
        }
    }
}

pub fn belief_after_draw(b: &Belief, reserve: &[u8; NSLOT], faceup: &[u8; NSLOT], set_pending: bool) -> Belief {
    let mut pairs: Vec<(Config, f32)> = Vec::with_capacity(b.len() * NSLOT);
    for (c, w) in b.cfg.iter().zip(b.p.iter()) {
        draw_children(c, reserve, faceup, set_pending, |n, p| pairs.push((n, *w * p)));
    }
    Belief::from_pairs(pairs)
}

#[derive(Clone, Debug, Default)]
pub struct DrawMap {
    pub start: Vec<u32>,
    pub to: Vec<u32>,
    pub p: Vec<f32>,
}

impl DrawMap {
    #[inline]
    pub fn row(&self, ci: usize) -> (&[u32], &[f32]) {
        let (a, b) = (self.start[ci] as usize, self.start[ci + 1] as usize);
        (&self.to[a..b], &self.p[a..b])
    }
    pub fn rows(&self) -> usize {
        self.start.len().saturating_sub(1)
    }
    pub fn len(&self) -> usize {
        self.to.len()
    }
    pub fn is_empty(&self) -> bool {
        self.to.is_empty()
    }
}

pub const IDX_BITS: u32 = 24;
pub const IDX_MASK: u64 = (1 << IDX_BITS) - 1;

#[derive(Default)]
pub struct DrawScratch {
    kid: Vec<Config>,
    prob: Vec<f32>,
    order: Vec<u64>,
    acc: Vec<f32>,
    hit: Vec<bool>,
    touched: Vec<u32>,
}

impl DrawScratch {
    pub fn transition(
        &mut self,
        cfg: &[Config],
        reserve: &[u8; NSLOT],
        faceup: &[u8; NSLOT],
        support: &mut Vec<Config>,
        map: &mut DrawMap,
        set_pending: bool,
    ) {
        self.kid.clear();
        self.prob.clear();
        map.start.clear();
        map.start.push(0);
        let (kid, prob) = (&mut self.kid, &mut self.prob);
        for c in cfg.iter() {
            draw_children(c, reserve, faceup, set_pending, |n, p| {
                kid.push(n);
                prob.push(p);
            });
            map.start.push(kid.len() as u32);
        }
        self.pack(support, map);
    }

    pub fn run(
        &mut self,
        cfg: &[Config],
        reserve: &[u8; NSLOT],
        faceup: &[u8; NSLOT],
        k: u8,
        support: &mut Vec<Config>,
        map: &mut DrawMap,
    ) {
        self.kid.clear();
        self.prob.clear();
        map.start.clear();
        map.start.push(0);
        let k = k as u32;
        for c in cfg.iter() {
            let bag = c.bag(reserve);
            let b: u32 = bag.iter().map(|&x| x as u32).sum();
            if b >= k {
                self.deal(&bag, k, *c);
            } else {
                let mut base = *c;
                base.fd = [0; NSLOT];
                for s in 0..NSLOT {
                    base.hand[s] += bag[s];
                }
                let refill: [u8; NSLOT] = std::array::from_fn(|s| faceup[s] + c.fd[s]);
                let total: u32 = refill.iter().map(|&x| x as u32).sum();
                let left = k - b;
                if total > left {
                    self.deal(&refill, left, base);
                } else {
                    for s in 0..NSLOT {
                        base.hand[s] += refill[s];
                    }
                    self.kid.push(base);
                    self.prob.push(1.0);
                }
            }
            map.start.push(self.kid.len() as u32);
        }
        self.pack(support, map);
    }

    fn deal(&mut self, src: &[u8; NSLOT], k: u32, base: Config) {
        fn choose(n: u32, k: u32) -> f64 {
            (0..k).map(|i| (n - i) as f64 / (i + 1) as f64).product()
        }
        fn rec(
            src: &[u8; NSLOT],
            slot: usize,
            left: u32,
            num: f64,
            c: Config,
            denom: f64,
            kid: &mut Vec<Config>,
            prob: &mut Vec<f32>,
        ) {
            if slot == NSLOT {
                if left == 0 {
                    debug_assert!(c.hand_size() as usize <= HAND_CAP);
                    kid.push(c);
                    prob.push((num / denom) as f32);
                }
                return;
            }
            for x in 0..=left.min(src[slot] as u32) {
                let mut c2 = c;
                c2.hand[slot] += x as u8;
                rec(
                    src,
                    slot + 1,
                    left - x,
                    num * choose(src[slot] as u32, x),
                    c2,
                    denom,
                    kid,
                    prob,
                );
            }
        }
        let total: u32 = src.iter().map(|&x| x as u32).sum();
        rec(src, 0, k, 1.0, base, choose(total, k), &mut self.kid, &mut self.prob);
    }

    fn pack(&mut self, support: &mut Vec<Config>, map: &mut DrawMap) {
        assert!(self.kid.len() < 1 << IDX_BITS, "draw fan-out over the index width");
        self.order.clear();
        self.order.extend(
            self.kid
                .iter()
                .enumerate()
                .map(|(i, c)| (c.key() << IDX_BITS) | i as u64),
        );
        self.order.sort_unstable();
        support.clear();
        map.to.clear();
        map.to.resize(self.kid.len(), 0);
        map.p.clear();
        map.p.extend_from_slice(&self.prob);
        let mut prev = u64::MAX;
        for &packed in self.order.iter() {
            let (k, i) = (packed >> IDX_BITS, (packed & IDX_MASK) as usize);
            if k != prev {
                prev = k;
                support.push(self.kid[i]);
            }
            map.to[i] = (support.len() - 1) as u32;
        }
    }

    pub fn compose(&mut self, a: &DrawMap, b: &DrawMap, n_child: usize, out: &mut DrawMap) {
        if self.acc.len() < n_child {
            self.acc.resize(n_child, 0.0);
            self.hit.resize(n_child, false);
        }
        let rows = a.rows();
        out.start.clear();
        out.start.push(0);
        out.to.clear();
        out.p.clear();
        for i in 0..rows {
            self.touched.clear();
            let (mid, pm) = a.row(i);
            for k in 0..mid.len() {
                let (to, pt) = b.row(mid[k] as usize);
                for j in 0..to.len() {
                    let t = to[j] as usize;
                    if !self.hit[t] {
                        self.hit[t] = true;
                        self.touched.push(to[j]);
                    }
                    self.acc[t] += pm[k] * pt[j];
                }
            }
            self.touched.sort_unstable();
            for &t in &self.touched {
                out.to.push(t);
                out.p.push(self.acc[t as usize]);
                self.acc[t as usize] = 0.0;
                self.hit[t as usize] = false;
            }
            out.start.push(out.to.len() as u32);
        }
    }
}

fn draw_source(c: &Config, reserve: &[u8; NSLOT], faceup: &[u8; NSLOT]) -> ([u8; NSLOT], Config) {
    let bag = c.bag(reserve);
    if bag.iter().any(|&x| x > 0) {
        return (bag, *c);
    }
    let mut refill = [0u8; NSLOT];
    for k in 0..NSLOT {
        refill[k] = faceup[k] + c.fd[k];
    }
    let mut base = *c;
    base.fd = [0; NSLOT];
    (refill, base)
}

pub fn faceup_counts(s: &State, p: u8, ctx: &Ctx) -> [u8; NSLOT] {
    let mut out = [0u8; NSLOT];
    for k in 0..NSLOT {
        out[k] = s.zones[p as usize][Z_FACEUP][ctx.slots[p as usize][k] as usize];
    }
    out
}

#[inline]
pub fn is_facedown_play(a: &Action) -> bool {
    matches!(
        a,
        Action::Pass { .. } | Action::ClaimInitiative { .. } | Action::Recruit { .. }
    )
}

pub fn obs_key(a: &Action) -> u32 {
    let code = a.encode();
    match a {
        Action::Pass { .. } | Action::ClaimInitiative { .. } => code & 63,
        Action::Recruit { unit, .. } => (code & 63) | ((*unit as u32) << 12),
        _ => code,
    }
}

pub const MAX_COINS: f32 = 4.0 * 5.0 + 1.0;

pub const NTYPE: usize = 2 * NSLOT;

pub const HEX_FACTS: usize = 2 + 1 + 2 + 1 + CONT_CAP;
pub const HEX_CH: usize = HEX_FACTS + NTYPE;
pub const HEX_BLOCK: usize = N_HEXES * HEX_CH;
pub const PILE_COUNTS: usize = 4;
pub const PLAYER_SCALARS: usize = 6;
pub const GLOBAL_SCALARS: usize = 3 + CONT_CAP * PENDING_KINDS;

pub const OFF_PILES: usize = HEX_BLOCK;
pub const OFF_CARDS: usize = OFF_PILES + NTYPE * PILE_COUNTS;
pub const OFF_LOOSE: usize = OFF_CARDS + NTYPE * CARD_FEATS;
pub const LOOSE: usize = 2 * PLAYER_SCALARS + GLOBAL_SCALARS;

pub const ROW_IDS: usize = 0;
pub const ROW_HEX_OWNER: usize = ROW_IDS + 2 * NSLOT;
pub const ROW_HEX_SLOT: usize = ROW_HEX_OWNER + N_HEXES;
pub const ROW_HEX_HEIGHT: usize = ROW_HEX_SLOT + N_HEXES;
pub const ROW_HEX_MARKER: usize = ROW_HEX_HEIGHT + N_HEXES;
pub const ROW_PILES: usize = ROW_HEX_MARKER + N_HEXES;
pub const ROW_HAND_SIZE: usize = ROW_PILES + 2 * NSLOT * PILE_COUNTS;
pub const ROW_FD_SIZE: usize = ROW_HAND_SIZE + 2;
pub const ROW_BAG_SIZE: usize = ROW_FD_SIZE + 2;
pub const ROW_INITIATIVE: usize = ROW_BAG_SIZE + 2;
pub const ROW_INIT_MOVED: usize = ROW_INITIATIVE + 1;
pub const ROW_TO_ACT: usize = ROW_INIT_MOVED + 1;
pub const ROW_WP: usize = ROW_TO_ACT + 1;
pub const ROW_STACK_KIND: usize = ROW_WP + 1;
pub const ROW_STACK_OWED: usize = ROW_STACK_KIND + CONT_CAP;
pub const ROW_BYTES: usize = ROW_STACK_OWED + CONT_CAP * 8;

pub fn rules_table_hash() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    let bd = board();
    for &l in bd.is_location.iter() {
        mix(l as u64);
    }
    for u in 0..N_UNITS {
        let mut f = [0.0f32; CARD_FEATS];
        write_card_features(u as u8, &mut f);
        for x in f {
            mix(x.to_bits() as u64);
        }
    }
    h
}

pub const PUBFEAT: usize = OFF_LOOSE + LOOSE;

pub fn normalize_weights(w: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), out.len());
    let tot: f32 = w.iter().sum();
    let (scale, flat) = if tot > SMOOTH {
        (1.0 / tot, 0.0)
    } else {
        (0.0, 1.0 / w.len().max(1) as f32)
    };
    for (o, x) in out.iter_mut().zip(w.iter()) {
        *o = *x * scale + flat;
    }
}

fn stack_bits(s: &State) -> ([u8; CONT_CAP], [u64; CONT_CAP]) {
    let mut kinds = [0xffu8; CONT_CAP];
    let mut owed = [0u64; CONT_CAP];
    for (i, c) in s.stack().into_iter().enumerate() {
        if let Some(c) = c {
            kinds[i] = c.tag();
            owed[i] = c.owed_hexes().0;
        }
    }
    (kinds, owed)
}

fn stack_from_row(row: &[u8]) -> ([u8; CONT_CAP], [u64; CONT_CAP]) {
    let mut kinds = [0u8; CONT_CAP];
    let mut owed = [0u64; CONT_CAP];
    kinds.copy_from_slice(&row[ROW_STACK_KIND..ROW_STACK_KIND + CONT_CAP]);
    for d in 0..CONT_CAP {
        let o = ROW_STACK_OWED + d * 8;
        owed[d] = u64::from_le_bytes(row[o..o + 8].try_into().unwrap());
    }
    (kinds, owed)
}

pub fn pack_row(s: &State, ctx: &Ctx, out: &mut [u8]) {
    debug_assert_eq!(out.len(), ROW_BYTES);
    for t in 0..2 * NSLOT {
        out[ROW_IDS + t] = ctx.slots[t / NSLOT][t % NSLOT];
    }
    for h in 0..N_HEXES {
        out[ROW_HEX_OWNER + h] = s.hex_owner[h];
        out[ROW_HEX_SLOT + h] = if s.hex_owner[h] == NONE {
            NONE
        } else {
            ctx.slot_of[s.hex_owner[h] as usize][s.hex_type[h] as usize] as u8
        };
        out[ROW_HEX_HEIGHT + h] = s.hex_height[h];
        out[ROW_HEX_MARKER + h] = s.loc_marker[h];
    }
    for p in 0..2usize {
        let res = reserve(s, p as u8, ctx);
        for k in 0..NSLOT {
            let u = ctx.slots[p][k] as usize;
            let at = ROW_PILES + (p * NSLOT + k) * PILE_COUNTS;
            out[at] = res[k];
            out[at + 1] = s.zones[p][Z_FACEUP][u];
            out[at + 2] = s.zones[p][Z_SUPPLY][u];
            out[at + 3] = s.zones[p][Z_ELIM][u];
        }
        out[ROW_HAND_SIZE + p] = s.hand_size(p as u8);
        out[ROW_FD_SIZE + p] = s.zones[p][Z_FACEDOWN].iter().sum();
        out[ROW_BAG_SIZE + p] = s.bag_size(p as u8);
    }
    out[ROW_INITIATIVE] = s.initiative;
    out[ROW_INIT_MOVED] = s.initiative_moved as u8;
    out[ROW_TO_ACT] = s.to_act();
    out[ROW_WP] = s.wp_v2_triggered as u8;
    let (kinds, owed) = stack_bits(s);
    out[ROW_STACK_KIND..ROW_STACK_KIND + CONT_CAP].copy_from_slice(&kinds);
    for d in 0..CONT_CAP {
        let o = ROW_STACK_OWED + d * 8;
        out[o..o + 8].copy_from_slice(&owed[d].to_le_bytes());
    }
}

pub fn mirror_row(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), ROW_BYTES);
    debug_assert_eq!(dst.len(), ROW_BYTES);
    let flip = |v: u8| match v {
        0 => 1,
        1 => 0,
        other => other,
    };
    dst.copy_from_slice(src);
    for h in 0..N_HEXES {
        let m = crate::state::mirror_hex(h);
        dst[ROW_HEX_OWNER + h] = flip(src[ROW_HEX_OWNER + m]);
        dst[ROW_HEX_SLOT + h] = src[ROW_HEX_SLOT + m];
        dst[ROW_HEX_HEIGHT + h] = src[ROW_HEX_HEIGHT + m];
        dst[ROW_HEX_MARKER + h] = flip(src[ROW_HEX_MARKER + m]);
    }
    for k in 0..NSLOT {
        dst[ROW_IDS + k] = src[ROW_IDS + NSLOT + k];
        dst[ROW_IDS + NSLOT + k] = src[ROW_IDS + k];
    }
    let piles = NSLOT * PILE_COUNTS;
    for k in 0..piles {
        dst[ROW_PILES + k] = src[ROW_PILES + piles + k];
        dst[ROW_PILES + piles + k] = src[ROW_PILES + k];
    }
    for at in [ROW_HAND_SIZE, ROW_FD_SIZE, ROW_BAG_SIZE] {
        dst[at] = src[at + 1];
        dst[at + 1] = src[at];
    }
    dst[ROW_INITIATIVE] = flip(src[ROW_INITIATIVE]);
    dst[ROW_TO_ACT] = flip(src[ROW_TO_ACT]);
    for d in 0..CONT_CAP {
        let at = ROW_STACK_OWED + d * 8;
        dst[at..at + 8].fill(0);
        for h in 0..N_HEXES {
            if src[at + h / 8] >> (h % 8) & 1 == 1 {
                let m = crate::state::mirror_hex(h);
                dst[at + m / 8] |= 1 << (m % 8);
            }
        }
    }
}

pub fn expand_row(row: &[u8], out: &mut [f32]) {
    debug_assert_eq!(row.len(), ROW_BYTES);
    let hex = |at: usize| -> &[u8; N_HEXES] { row[at..at + N_HEXES].try_into().unwrap() };
    let (hex_owner, hex_slot) = (hex(ROW_HEX_OWNER), hex(ROW_HEX_SLOT));
    let (hex_height, hex_marker) = (hex(ROW_HEX_HEIGHT), hex(ROW_HEX_MARKER));
    let piles = &row[ROW_PILES..ROW_PILES + 2 * NSLOT * PILE_COUNTS];
    let ids = &row[ROW_IDS..ROW_IDS + 2 * NSLOT];
    let hand_size: &[u8] = &row[ROW_HAND_SIZE..ROW_HAND_SIZE + 2];
    let fd_size: &[u8] = &row[ROW_FD_SIZE..ROW_FD_SIZE + 2];
    let bag_size: &[u8] = &row[ROW_BAG_SIZE..ROW_BAG_SIZE + 2];
    let mut markers_hand = [6u8; 2];
    for h in 0..N_HEXES {
        if hex_marker[h] != NONE {
            markers_hand[hex_marker[h] as usize] -= 1;
        }
    }
    let initiative = if row[ROW_INITIATIVE] > 1 { 0 } else { row[ROW_INITIATIVE] };
    let initiative_moved = row[ROW_INIT_MOVED] != 0;
    let to_act = row[ROW_TO_ACT];
    let wp_v2_triggered = row[ROW_WP] != 0;
    let (kinds, owed) = stack_from_row(row);
    let (kinds, owed) = (&kinds, &owed);
    debug_assert_eq!(out.len(), PUBFEAT);
    debug_assert_eq!(piles.len(), 2 * NSLOT * PILE_COUNTS);
    debug_assert_eq!(ids.len(), 2 * NSLOT);
    out.fill(0.0);
    let bd = board();

    let mut i = 0;
    for h in 0..N_HEXES {
        let owner = hex_owner[h];
        if owner != NONE {
            out[i + owner as usize] = 1.0;
            out[i + 2] = hex_height[h] as f32 / 5.0;
            if hex_slot[h] != NONE {
                out[i + HEX_FACTS + owner as usize * NSLOT + hex_slot[h] as usize] = 1.0;
            }
        }
        if hex_marker[h] != NONE {
            out[i + 3 + hex_marker[h] as usize] = 1.0;
        }
        out[i + 5] = bd.is_location[h] as u8 as f32;
        for d in 0..CONT_CAP {
            if owed[d] & (1u64 << h) != 0 {
                out[i + 6 + d] = 1.0;
            }
        }
        i += HEX_CH;
    }
    debug_assert_eq!(i, OFF_PILES);

    for t in 0..2 * NSLOT {
        let at = t * PILE_COUNTS;
        out[i] = piles[at] as f32 / 5.0;
        out[i + 1] = piles[at + 1] as f32 / 5.0;
        out[i + 2] = piles[at + 2] as f32 / 5.0;
        out[i + 3] = piles[at + 3] as f32 / 5.0;
        i += PILE_COUNTS;
    }
    debug_assert_eq!(i, OFF_CARDS);

    for t in 0..2 * NSLOT {
        write_card_features(ids[t], &mut out[i..i + CARD_FEATS]);
        i += CARD_FEATS;
    }
    debug_assert_eq!(i, OFF_LOOSE);

    for p in 0..2usize {
        out[i] = markers_hand[p] as f32 / 6.0;
        out[i + 1] = (6 - markers_hand[p]) as f32 / 6.0;
        out[i + 2] = hand_size[p] as f32 / 3.0;
        out[i + 3] = fd_size[p] as f32 / MAX_COINS;
        out[i + 4] = bag_size[p] as f32 / MAX_COINS;
        out[i + 5] = (initiative == p as u8) as u8 as f32;
        i += PLAYER_SCALARS;
    }

    out[i] = wp_v2_triggered as u8 as f32;
    out[i + 1] = initiative_moved as u8 as f32;
    out[i + 2] = (to_act == 0) as u8 as f32;
    for d in 0..CONT_CAP {
        let k = kinds[d];
        if (k as usize) < PENDING_KINDS {
            out[i + 3 + d * PENDING_KINDS + k as usize] = 1.0;
        }
    }
    i += GLOBAL_SCALARS;
    debug_assert_eq!(i, PUBFEAT);
}

#[cfg(test)]
mod draw_tests {
    use super::*;

    fn composed(cfg: &[Config], res0: &[u8; NSLOT], fu0: &[u8; NSLOT], k: u8) -> (Vec<Config>, DrawMap) {
        let mut sc = DrawScratch::default();
        let (mut res, mut fu) = (*res0, *fu0);
        let mut cur = cfg.to_vec();
        let mut next = Vec::new();
        let (mut draw, mut step, mut acc) = (DrawMap::default(), DrawMap::default(), DrawMap::default());
        for j in 0..k {
            let empty = cur[0].bag(&res).iter().all(|&x| x == 0);
            sc.transition(&cur, &res, &fu, &mut next, &mut step, false);
            if j == 0 {
                std::mem::swap(&mut draw, &mut step);
            } else {
                sc.compose(&draw, &step, next.len(), &mut acc);
                std::mem::swap(&mut draw, &mut acc);
            }
            std::mem::swap(&mut cur, &mut next);
            if empty {
                for s in 0..NSLOT {
                    res[s] += fu[s];
                    fu[s] = 0;
                }
            }
        }
        (cur, draw)
    }

    fn assert_same(cfg: &[Config], res: &[u8; NSLOT], fu: &[u8; NSLOT], k: u8, what: &str) {
        let (want_sup, want) = composed(cfg, res, fu, k);
        let mut sc = DrawScratch::default();
        let (mut sup, mut map) = (Vec::new(), DrawMap::default());
        sc.run(cfg, res, fu, k, &mut sup, &mut map);
        assert_eq!(sup, want_sup, "{what}: support");
        assert_eq!(map.rows(), want.rows(), "{what}: rows");
        for i in 0..map.rows() {
            let (gt, gp) = map.row(i);
            let (wt, wp) = want.row(i);
            let mut got: Vec<(u32, f32)> = gt.iter().copied().zip(gp.iter().copied()).collect();
            let mut wnt: Vec<(u32, f32)> = wt.iter().copied().zip(wp.iter().copied()).collect();
            got.sort_by_key(|e| e.0);
            wnt.sort_by_key(|e| e.0);
            assert_eq!(
                got.iter().map(|e| e.0).collect::<Vec<_>>(),
                wnt.iter().map(|e| e.0).collect::<Vec<_>>(),
                "{what}: row {i} children"
            );
            for (g, w) in got.iter().zip(&wnt) {
                assert!((g.1 - w.1).abs() < 1e-5, "{what}: row {i} prob {} vs {}", g.1, w.1);
            }
            let tot: f32 = gp.iter().sum();
            assert!((tot - 1.0).abs() < 1e-5, "{what}: row {i} sums to {tot}");
        }
    }

    fn spread(rng: &mut crate::rng::Rng, total: u8, cap: &[u8; NSLOT], used: &[u8; NSLOT]) -> Option<[u8; NSLOT]> {
        let mut out = [0u8; NSLOT];
        'outer: for _ in 0..total {
            for _ in 0..32 {
                let s = (rng.next_u64() % NSLOT as u64) as usize;
                if used[s] + out[s] < cap[s] {
                    out[s] += 1;
                    continue 'outer;
                }
            }
            return None;
        }
        Some(out)
    }

    #[test]
    fn run_matches_composition() {
        let c = |hand: [u8; NSLOT], fd: [u8; NSLOT]| Config {
            hand,
            fd,
            inflight: None,
        };

        assert_same(
            &[c([0; NSLOT], [1, 1, 0, 0, 0]), c([0; NSLOT], [0, 0, 1, 1, 0])],
            &[5, 4, 3, 2, 5],
            &[1, 0, 2, 0, 0],
            3,
            "plenty",
        );
        assert_same(
            &[
                c([0; NSLOT], [2, 1, 0, 0, 0]),
                c([0; NSLOT], [2, 0, 1, 0, 0]),
                c([0; NSLOT], [1, 1, 1, 0, 0]),
            ],
            &[2, 1, 1, 0, 0],
            &[1, 2, 0, 0, 0],
            3,
            "refill",
        );
        assert_same(
            &[c([1, 0, 0, 0, 0], [0; NSLOT])],
            &[2, 0, 0, 0, 0],
            &[0; NSLOT],
            2,
            "empty refill",
        );
        assert_same(
            &[c([0; NSLOT], [0; NSLOT])],
            &[0; NSLOT],
            &[2, 0, 0, 0, 0],
            3,
            "shortage",
        );
        assert_same(
            &[c([0; NSLOT], [0; NSLOT])],
            &[1, 1, 0, 0, 0],
            &[1, 0, 0, 0, 0],
            3,
            "exact dry",
        );

        let mut rng = crate::rng::Rng::new(0xD12A);
        let mut done = 0;
        while done < 300 {
            let mut res = [0u8; NSLOT];
            for s in 0..NSLOT {
                res[s] = (rng.next_u64() % 4) as u8;
            }
            let mut fu = [0u8; NSLOT];
            for s in 0..NSLOT {
                fu[s] = (rng.next_u64() % 2) as u8;
            }
            let ht = (rng.next_u64() % 3) as u8;
            let fdt = (rng.next_u64() % 4) as u8;
            let k = 3 - ht;
            let mut cfgs = Vec::new();
            for _ in 0..6 {
                let Some(hand) = spread(&mut rng, ht, &res, &[0; NSLOT]) else {
                    continue;
                };
                let Some(fd) = spread(&mut rng, fdt, &res, &hand) else {
                    continue;
                };
                cfgs.push(c(hand, fd));
            }
            if cfgs.is_empty() {
                continue;
            }
            cfgs.sort_by_key(|x| x.key());
            cfgs.dedup();
            assert_same(&cfgs, &res, &fu, k, "fuzz");
            done += 1;
        }
    }
}

#[cfg(test)]
mod enumerate_tests {
    use super::*;
    use crate::state::{Cont, State, BLACK, WHITE, Z_BAG, Z_INFLIGHT};
    use crate::units::{CROSSBOWMAN, PIKEMAN, ROYAL_COIN, SWORDSMAN, WARRIOR_PRIEST};

    #[test]
    fn set_config_accepts_enumerated_inflight_configs() {
        let mut s = State::blank(WHITE);
        for u in [WARRIOR_PRIEST, SWORDSMAN, PIKEMAN, CROSSBOWMAN, ROYAL_COIN] {
            s.add_zone(WHITE, Z_BAG, u, 2);
            s.add_zone(BLACK, Z_BAG, u, 2);
        }
        s.add_zone(WHITE, Z_INFLIGHT, WARRIOR_PRIEST, 1);
        s.pending = Cont::WarriorPriestPlay { player: WHITE };
        let ctx = Ctx::new(&s);
        let res = reserve(&s, WHITE, &ctx);
        let truth = true_config(&s, WHITE, &ctx);
        assert!(truth.inflight.is_some());
        let all = enumerate_configs(&res, truth.hand_size(), truth.fd_size(), true);
        assert!(!all.is_empty());
        assert!(all.iter().all(|c| c.inflight.is_some()));
        for c in &all {
            let mut w = s;
            set_config(&mut w, WHITE, &ctx, c);
        }
    }
}
