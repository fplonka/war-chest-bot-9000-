//! Public belief states (PBS) for War Chest: the factored-observation view of
//! the engine, the belief representation, and the tensor encoding the value
//! network consumes.
//!
//! # What is actually private
//!
//! Everything a player owns is public *in aggregate*: the draft fixes 4 unit
//! types plus the Royal Coin (so at most `NSLOT = 5` distinct coin types per
//! player), and the board, supply, face-up discards and eliminated coins are
//! all open. Two things are hidden:
//!
//!   * the **hand** — which coins were drawn this round;
//!   * the **identities of face-down discards** — the coin spent on Pass,
//!     Claim Initiative, or a Recruit payment is never revealed.
//!
//! So a player's private state is the pair `(hand, facedown)`, and the bag is
//! *derived*:
//!
//!   `bag = reserve - hand - facedown`,  `reserve = bag + hand + facedown`
//!
//! where `reserve` is public (total owned minus board, face-up discard, supply
//! and eliminated). Hand size, bag size and face-down count are public too —
//! you can see how many coins someone is holding.
//!
//! A `Config` is that pair. The reachable set is small — measured over 41k
//! positions of random play: median 8, mean 34, p99 385 — so CFR enumerates
//! information states exactly, with no particle approximation.
//!
//! Beliefs are per player and independent (separate bags, no shared hidden
//! resource), so a PBS factorises as `(public state, belief_0, belief_1)`.
//!
//! # What the network is a function of
//!
//! The value of a leaf is a *counterfactual value per information state*, so it
//! is indexed by the same object beliefs and regrets are indexed by: the config.
//! Two encodings come out of here and they have different jobs.
//!
//!   * `write_public_features` — the public state. Identical for every config,
//!     which is what lets one public tree carry all of them.
//!   * `write_config_feats` — one config's exact counts: hand, face-down and the
//!     derived bag, plus whose config it is. This is the *argument* of the value
//!     function, and the belief encoding is a weighted sum of the same vectors,
//!     so both sides of the network see the same description of a config.
//!
//! Nothing here summarises a config into anything coarser. An earlier build
//! keyed values by hand alone and averaged the face-down composition into a
//! marginal; that made the strategy measurable with respect to the hand, so the
//! agent could not act on coins it had buried itself — a restriction of the
//! strategy space, i.e. a different game. `docs/REBEL.md` records what it cost.

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::state::{Cont, State, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_HAND, Z_SUPPLY};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

/// Distinct coin types a player can own: 4 drafted units + the Royal Coin.
pub const NSLOT: usize = 5;
/// Hand size cap.
pub const HAND_CAP: usize = 3;

// ------------------------------------------------------------ config features

/// Raw counts describing one config: hand, face-down, bag, in slot order.
pub const CCOUNTS: usize = 3 * NSLOT;
/// The config vector the network reads: `CCOUNTS` counts plus a flag saying
/// whose config it is. Player 0's and player 1's slots mean different units, and
/// the value asked for is "this player's counterfactual value", so the seat has
/// to be part of the argument.
pub const CFEAT: usize = CCOUNTS + 1;
/// Divisor for every count. A player owns at most 5 coins of any one type, so
/// every count lands in `[0, 1]` and one constant serves all three zones.
pub const CNORM: f32 = 5.0;

// --------------------------------------------------------------- game context

/// Per-game constants: which unit index each of a player's 5 coin slots is.
#[derive(Clone, Copy)]
pub struct Ctx {
    pub slots: [[u8; NSLOT]; 2],
    /// Reverse map, unit index -> slot, or -1.
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
}

// ------------------------------------------------------------------- configs

/// A player's private state: what is in hand, and what they have discarded face
/// down since the last reshuffle. The bag follows from the public reserve.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct Config {
    pub hand: [u8; NSLOT],
    pub fd: [u8; NSLOT],
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
    /// `bag = reserve - hand - facedown`.
    #[inline]
    pub fn bag(&self, reserve: &[u8; NSLOT]) -> [u8; NSLOT] {
        let mut b = [0u8; NSLOT];
        for k in 0..NSLOT {
            b[k] = reserve[k].saturating_sub(self.hand[k] + self.fd[k]);
        }
        b
    }
    /// A single integer that orders configs exactly as `Ord` does: hand
    /// lexicographically, then face-down. Sorting and deduping child supports
    /// is the hot part of building a subgame, and comparing one `u64` beats
    /// walking two five-byte arrays — but only if the key is computed once per
    /// element rather than once per comparison, which is why this is a method
    /// and not an `Ord` impl.
    /// 2 bits per hand slot (a hand holds at most `HAND_CAP` = 3 coins in
    /// total, so no slot exceeds 3) and 5 per face-down slot (bounded by the
    /// largest supply in the game), for 35 bits — which leaves room to pack an
    /// element index alongside it in one `u64`.
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
        k
    }
}

/// One config as raw counts: hand, then face-down, then the derived bag.
///
/// The bag is `reserve - hand - facedown` and so carries no information the
/// first two do not — but it is the quantity that decides what you draw next
/// round, and handing it over saves the network from having to learn a
/// subtraction it can only do if it also finds the reserve in the public block.
#[inline]
pub fn config_counts(c: &Config, reserve: &[u8; NSLOT], out: &mut [u8; CCOUNTS]) {
    for k in 0..NSLOT {
        out[k] = c.hand[k];
        out[NSLOT + k] = c.fd[k];
        out[2 * NSLOT + k] = reserve[k].saturating_sub(c.hand[k] + c.fd[k]);
    }
}

/// The config vector the network reads. `player` is the seat the config belongs
/// to — the value asked for is that player's counterfactual value.
#[inline]
pub fn write_config_feats(c: &Config, reserve: &[u8; NSLOT], player: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), CFEAT);
    let mut cnt = [0u8; CCOUNTS];
    config_counts(c, reserve, &mut cnt);
    for k in 0..CCOUNTS {
        out[k] = cnt[k] as f32 / CNORM;
    }
    out[CCOUNTS] = player as f32;
}

/// `reserve[k] = bag[k] + hand[k] + facedown[k]` — public, and invariant to how
/// the unseen coins split. Every action either moves a coin *within* the
/// reserve (a face-down play) or out of it in full view (deploy, bolster, any
/// maneuver, the recruited coin), so the reserve is the same for every config.
pub fn reserve(s: &State, p: u8, ctx: &Ctx) -> [u8; NSLOT] {
    let mut out = [0u8; NSLOT];
    for k in 0..NSLOT {
        let u = ctx.slots[p as usize][k] as usize;
        out[k] = s.zones[p as usize][Z_BAG][u]
            + s.zones[p as usize][Z_HAND][u]
            + s.zones[p as usize][Z_FACEDOWN][u];
    }
    out
}

/// The true config of `p` in a world state.
pub fn true_config(s: &State, p: u8, ctx: &Ctx) -> Config {
    let mut c = Config::default();
    for k in 0..NSLOT {
        let u = ctx.slots[p as usize][k] as usize;
        c.hand[k] = s.zones[p as usize][Z_HAND][u];
        c.fd[k] = s.zones[p as usize][Z_FACEDOWN][u];
    }
    c
}

/// Instantiate a world: overwrite `p`'s hand and face-down discard with `c`,
/// putting the rest of the reserve back in the bag. The public projection is
/// unchanged, which is what lets one public tree carry every config.
pub fn set_config(s: &mut State, p: u8, ctx: &Ctx, c: &Config) {
    for k in 0..NSLOT {
        let u = ctx.slots[p as usize][k] as usize;
        let res = s.zones[p as usize][Z_BAG][u]
            + s.zones[p as usize][Z_HAND][u]
            + s.zones[p as usize][Z_FACEDOWN][u];
        debug_assert!(c.hand[k] + c.fd[k] <= res, "config must fit in the reserve");
        s.zones[p as usize][Z_HAND][u] = c.hand[k];
        s.zones[p as usize][Z_FACEDOWN][u] = c.fd[k];
        s.zones[p as usize][Z_BAG][u] = res.saturating_sub(c.hand[k] + c.fd[k]);
    }
}

/// Every config consistent with the public counts. Self-play tracks the
/// reachable support forward instead; this exists for the brute-force belief
/// test and for sizing experiments.
pub fn enumerate_configs(reserve: &[u8; NSLOT], hand_size: u8, fd_size: u8) -> Vec<Config> {
    fn rec_fd(
        res: &[u8; NSLOT],
        hand: &[u8; NSLOT],
        fd: &mut [u8; NSLOT],
        k: usize,
        left: u8,
        out: &mut Vec<Config>,
    ) {
        if k == NSLOT - 1 {
            if left + hand[k] <= res[k] {
                fd[k] = left;
                out.push(Config { hand: *hand, fd: *fd });
                fd[k] = 0;
            }
            return;
        }
        for t in 0..=left.min(res[k].saturating_sub(hand[k])) {
            fd[k] = t;
            rec_fd(res, hand, fd, k + 1, left - t, out);
        }
        fd[k] = 0;
    }
    fn rec_hand(
        res: &[u8; NSLOT],
        hand: &mut [u8; NSLOT],
        k: usize,
        left: u8,
        fd_size: u8,
        out: &mut Vec<Config>,
    ) {
        if k == NSLOT - 1 {
            if left <= res[k] {
                hand[k] = left;
                let mut fd = [0u8; NSLOT];
                rec_fd(res, hand, &mut fd, 0, fd_size, out);
                hand[k] = 0;
            }
            return;
        }
        for t in 0..=left.min(res[k]) {
            hand[k] = t;
            rec_hand(res, hand, k + 1, left - t, fd_size, out);
        }
        hand[k] = 0;
    }
    let mut out = Vec::new();
    let mut hand = [0u8; NSLOT];
    rec_hand(reserve, &mut hand, 0, hand_size, fd_size, &mut out);
    out
}

// ------------------------------------------------------------------- beliefs

/// A distribution over one player's configs: explicit support plus weights,
/// kept sorted and merged so beliefs over the same support line up by index.
#[derive(Clone, Debug, Default)]
pub struct Belief {
    pub cfg: Vec<Config>,
    pub p: Vec<f32>,
}

/// Mirrors `kReachSmoothingEps` in the reference: stops an underflowed belief
/// from turning into NaN downstream.
const SMOOTH: f32 = 1e-30;

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

    /// Build from (config, weight) pairs: sort, merge duplicates, normalise.
    pub fn from_pairs(mut pairs: Vec<(Config, f32)>) -> Belief {
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut cfg: Vec<Config> = Vec::with_capacity(pairs.len());
        let mut p: Vec<f32> = Vec::with_capacity(pairs.len());
        for (c, w) in pairs {
            // Support is reachability, not weight. The subgame tree keeps every
            // reachable config and the walk indexes its strategy rows by
            // position in this list, so a weight that underflowed to zero must
            // not delete a row.
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

/// How a coin play moves a config. `slot` is the coin spent (-1 for the
/// micro-decisions that spend nothing); `facedown` says whether it goes to the
/// face-down discard (Pass / Claim Initiative / Recruit payment) or leaves the
/// reserve in full view.
#[inline]
pub fn advance_config(c: &Config, slot: i8, facedown: bool) -> Option<Config> {
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

/// Belief update through a round-start draw. The drawn coin is private, so the
/// public tree does not branch: the belief is convolved with each config's own
/// draw distribution. A config whose bag is empty reshuffles first, which folds
/// its face-down discards into the bag and therefore *erases* them.
pub fn belief_after_draw(b: &Belief, reserve: &[u8; NSLOT], faceup: &[u8; NSLOT]) -> Belief {
    let mut pairs: Vec<(Config, f32)> = Vec::with_capacity(b.len() * NSLOT);
    for (c, w) in b.cfg.iter().zip(b.p.iter()) {
        let (src, base) = draw_source(c, reserve, faceup);
        let total: u32 = src.iter().map(|&x| x as u32).sum();
        if total == 0 {
            pairs.push((*c, *w));
            continue;
        }
        for k in 0..NSLOT {
            if src[k] == 0 {
                continue;
            }
            let mut n = base;
            n.hand[k] += 1;
            if n.hand_size() as usize <= HAND_CAP {
                pairs.push((n, *w * src[k] as f32 / total as f32));
            }
        }
    }
    Belief::from_pairs(pairs)
}

/// A draw's chance transition, CSR by parent config: parent `ci` becomes child
/// `to[k]` with probability `p[k]` for `k` in `start[ci]..start[ci + 1]`.
///
/// Sparse because a config can draw at most `NSLOT` distinct coin types while
/// the child support is the union over the whole belief — the dense matrix this
/// replaces was ~30x30 with at most 5 non-zeros per row, and the subgame walks
/// it twice per CFR iteration.
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
}

/// Reusable working memory for building draw transitions.
///
/// A subgame builds one transition per draw and composes it over the run, and
/// the buffers involved are all small — which is exactly why the allocator
/// traffic mattered more than the arithmetic. One scratch lives on the solver
/// and is reused for the whole tree.
/// Bits reserved for the element index when a config key and an index are
/// packed into one `u64`. `Config::key` occupies 35, so this leaves nine spare
/// — and the largest thing sorted this way, a decision node's
/// `config x action` grid, runs to a few tens of thousands.
pub const IDX_BITS: u32 = 24;
pub const IDX_MASK: u64 = (1 << IDX_BITS) - 1;

#[derive(Default)]
pub struct DrawScratch {
    kid: Vec<Config>,
    prob: Vec<f32>,
    /// `key << IDX_BITS | index into kid`, sorted to build the child support.
    /// One `u64` rather than a `(u64, u32)` pair: a config key needs 50 bits
    /// and the index at most `IDX_BITS`, so packing them halves the width the
    /// sort moves around.
    order: Vec<u64>,
    acc: Vec<f32>,
    hit: Vec<bool>,
    touched: Vec<u32>,
}

impl DrawScratch {
    /// The chance transition of one draw: for each config in `cfg`, which
    /// configs it can become (one per drawable coin type, reshuffling first if
    /// its bag is empty) and with what probability. Writes the sorted-deduped
    /// child support and the transition — the per-config chance factor the
    /// subgame tree's reach accounting needs, kept separate from both players'
    /// strategies.
    pub fn transition(
        &mut self,
        cfg: &[Config],
        reserve: &[u8; NSLOT],
        faceup: &[u8; NSLOT],
        support: &mut Vec<Config>,
        map: &mut DrawMap,
    ) {
        let tg = crate::timed!(DGEN);
        self.kid.clear();
        self.prob.clear();
        map.start.clear();
        map.start.push(0);
        for c in cfg.iter() {
            let (src, base) = draw_source(c, reserve, faceup);
            let total: u32 = src.iter().map(|&x| x as u32).sum();
            if total == 0 {
                self.kid.push(*c);
                self.prob.push(1.0);
            } else {
                let inv = 1.0 / total as f32;
                for k in 0..NSLOT {
                    if src[k] == 0 {
                        continue;
                    }
                    let mut n = base;
                    n.hand[k] += 1;
                    if n.hand_size() as usize <= HAND_CAP {
                        self.kid.push(n);
                        self.prob.push(src[k] as f32 * inv);
                    }
                }
            }
            map.start.push(self.kid.len() as u32);
        }
        // Sort once by integer key, then read the support and the child index
        // of every row off that single ordering — no per-row binary search.
        drop(tg);
        let _ts = crate::timed!(DSORT);
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

    /// `a` then `b`, as one transition into `out`. A round start queues up to
    /// three consecutive draws for the same player and none of them branches
    /// the public tree, so composing them collapses the run into one node —
    /// and with it the states, reach vectors and value vectors of the rest.
    pub fn compose(&mut self, a: &DrawMap, b: &DrawMap, n_child: usize, out: &mut DrawMap) {
        let _t = crate::timed!(DCOMP);
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
            // Kept sorted so the child's rows stay in support order.
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

/// One draw's transition, allocating. For tests and one-off callers; the
/// subgame builder goes through `DrawScratch` so it can reuse its buffers.
pub fn draw_transition(
    cfg: &[Config],
    reserve: &[u8; NSLOT],
    faceup: &[u8; NSLOT],
) -> (Vec<Config>, DrawMap) {
    let (mut support, mut map) = (Vec::new(), DrawMap::default());
    DrawScratch::default().transition(cfg, reserve, faceup, &mut support, &mut map);
    (support, map)
}

/// What a config draws from, and what it looks like after any reshuffle:
/// normally the bag, but when the bag is empty the whole discard pile is
/// shuffled in and the face-down component is forgotten.
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

/// Public face-up discard composition, in slot order.
pub fn faceup_counts(s: &State, p: u8, ctx: &Ctx) -> [u8; NSLOT] {
    let mut out = [0u8; NSLOT];
    for k in 0..NSLOT {
        out[k] = s.zones[p as usize][Z_FACEUP][ctx.slots[p as usize][k] as usize];
    }
    out
}

// ------------------------------------ public observation of a private action

/// Does this action put the spent coin face down, hiding its identity?
#[inline]
pub fn is_facedown_play(a: &Action) -> bool {
    matches!(
        a,
        Action::Pass { .. } | Action::ClaimInitiative { .. } | Action::Recruit { .. }
    )
}

/// What both players see. Face-down plays announce the *event* but not the
/// coin, so several private actions collapse onto one public child of the
/// public tree; a Recruit reveals the unit taken but not the coin that paid.
pub fn obs_key(a: &Action) -> u32 {
    let code = a.encode();
    match a {
        Action::Pass { .. } | Action::ClaimInitiative { .. } => code & 63,
        Action::Recruit { unit, .. } => (code & 63) | ((*unit as u32) << 12),
        _ => code,
    }
}

// ------------------------------------------------------------------ features

/// The most coins a player can ever own: four drafted types at the largest
/// supply in the game (5 each) plus one Royal Coin. Every per-player coin count
/// -- bag, hand, face-up, face-down -- is bounded by this, so it is the correct
/// divisor for those features rather than an estimate.
const MAX_COINS: f32 = 4.0 * 5.0 + 1.0;

pub const PEND_KINDS: usize = 12;

/// Channels in the per-hex block: owner (2), stack height, the owner's slot
/// one-hot (`NSLOT`), the location marker's owner (2), is-location, and the
/// pending-maneuver mask.
///
/// The block is laid out hex-major and comes first in the encoding, so a
/// convolutional trunk can read it as a `[N_HEXES, HEX_CH]` image directly.
///
/// Every channel here is a raw fact about the position. A precomputed
/// distance-to-nearest-unit map was tried and removed: it is a *derived*
/// summary — the same quantity `eval_static`'s coverage term uses — so it
/// imports the handcrafted bot's opinion into the encoding, and it exists only
/// to paper over the limited reach of a shallow convolution. If the network
/// needs board-wide context, the honest fix is in the architecture (pooling),
/// not a hand-built feature.
pub const HEX_CH: usize = 2 + 1 + NSLOT + 2 + 1 + 1;
/// Size of the per-hex block.
pub const HEX_BLOCK: usize = N_HEXES * HEX_CH;
/// Per player: reserve, face-up, supply and eliminated counts per coin slot.
pub const ZONE_FEATS: usize = 4 * NSLOT;
pub const PLAYER_SCALARS: usize = 8;
pub const GLOBAL_SCALARS: usize = 5;
/// Slot one-hot for the coin a Footman-V2 instant deploy is holding. Public:
/// a Recruit reveals which unit was taken.
pub const PEND_SLOT: usize = NSLOT;

/// Offset of the per-player zone counts (just past the hex block).
pub const OFF_ZONES: usize = HEX_BLOCK;
/// Offset of the per-slot unit identity one-hots.
pub const OFF_IDENT: usize = OFF_ZONES + 2 * ZONE_FEATS;
/// Offset of the per-slot card-property blocks.
pub const OFF_CARDS: usize = OFF_IDENT + 2 * NSLOT * N_UNITS;
/// Offset of the per-player scalars.
pub const OFF_PLAYER: usize = OFF_CARDS + 2 * NSLOT * CARD_FEATS;
/// Offset of the global scalars.
pub const OFF_GLOBAL: usize = OFF_PLAYER + 2 * PLAYER_SCALARS;

/// Width of the public encoding. It is the whole of it: beliefs no longer ride
/// along as a fixed-width tail, because a belief is a distribution over configs
/// and the network reads it as a weighted sum of config vectors instead.
pub const PUBFEAT: usize = OFF_GLOBAL + GLOBAL_SCALARS + PEND_KINDS + PEND_SLOT;

/// Round divisor. Measured on the starter draft (`examples/featstats.rs`):
/// rounds reach 81 under random play and 121 under one-ply greedy, because a
/// drained reserve makes rounds short in coin plays and therefore numerous.
/// The previous divisor of 40 left this feature pinned at 1.0 for most of
/// essentially every game.
const MAX_ROUND: f32 = 128.0;

fn pending_kind(s: &State) -> usize {
    match s.pending() {
        Cont::Draw { .. } => 0,
        Cont::MainPlay => 1,
        Cont::RoyalGuardChoice { .. } => 2,
        Cont::SwordsmanMove { .. } => 3,
        Cont::BerserkerChain { .. } => 4,
        Cont::FootmanManeuver { .. } => 5,
        Cont::CavalryAttack { .. } => 6,
        Cont::MercenaryManeuver { .. } => 7,
        Cont::FootmanInstantDeploy { .. } => 8,
        Cont::WarriorPriestDraw { .. } => 9,
        Cont::WarriorPriestPlay { .. } => 10,
        Cont::_AttackPost { .. } => 11,
    }
}

/// Belief weights, normalised the way the network consumes them. `w` is the
/// caller's *unnormalised* reach; normalisation matches `Belief::normalize`,
/// including its fallback to uniform when the weights have underflowed.
///
/// The network turns this into `sum_c beta(c) * z(config_feats(c))` with a
/// learned `z`, so the belief reaches it as a weighted set of exact configs
/// rather than as a summary statistic of them.
pub fn normalize_weights(w: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), out.len());
    // One reciprocal for the whole support rather than a divide per config:
    // this runs once per leaf per player per CFR iteration over a support that
    // averages ~35 configs, and f32 division does not pipeline.
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

/// The public encoding. Fixed for a given state, so a solve computes it once
/// per leaf. Reads only public information: bag, hand and face-down discards
/// appear solely through their public sum (the reserve) and their public sizes.
pub fn write_public_features(s: &State, ctx: &Ctx, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PUBFEAT);
    out.fill(0.0);
    let bd = board();

    // Which board unit, if any, still owes a maneuver at this decision node.
    // `pending_kind` below says *what kind* of trigger is open; without this it
    // does not say *whose*, and the Footman tactic can owe two at once.
    //
    // Only the hex-valued payloads are encoded. `WarriorPriestPlay { coin }`
    // carries a coin drawn privately, so encoding it would leak — which is the
    // same reason `docs/REBEL.md` keeps the Warrior Priest out of the draft
    // pool. `FootmanInstantDeploy`'s coin is public (a Recruit reveals the unit
    // taken) and is encoded with the globals instead.
    let mut pending_hexes = crate::state::HexSet(0);
    let mark = |h: u8, set: &mut crate::state::HexSet| {
        if (h as usize) < N_HEXES {
            set.insert(h);
        }
    };
    match *s.pending() {
        Cont::SwordsmanMove { hex }
        | Cont::BerserkerChain { hex, .. }
        | Cont::CavalryAttack { hex }
        | Cont::MercenaryManeuver { hex } => mark(hex, &mut pending_hexes),
        Cont::FootmanManeuver { hexes } => {
            for h in hexes.iter() {
                mark(h, &mut pending_hexes);
            }
        }
        Cont::RoyalGuardChoice { rg_hex, .. } | Cont::WarriorPriestDraw { rg_hex, .. } => {
            mark(rg_hex, &mut pending_hexes)
        }
        Cont::_AttackPost { atk_hex } => mark(atk_hex, &mut pending_hexes),
        _ => {}
    }

    let mut i = 0;
    for h in 0..N_HEXES {
        let owner = s.hex_owner[h];
        if owner != NONE {
            out[i + owner as usize] = 1.0;
            // Divisor is the largest coin count on any card. Bolstering has no
            // height limit (RULES.md section 5) and heights of 4 and 5 are
            // ~20% of occupied-hex observations under random play, so the
            // previous /3 collapsed them onto the same value.
            out[i + 2] = s.hex_height[h] as f32 / 5.0;
            let k = ctx.slot_of[owner as usize][s.hex_type[h] as usize];
            if k >= 0 {
                out[i + 3 + k as usize] = 1.0;
            }
        }
        if s.loc_marker[h] != NONE {
            out[i + 3 + NSLOT + s.loc_marker[h] as usize] = 1.0;
        }
        out[i + 5 + NSLOT] = bd.is_location[h] as u8 as f32;
        out[i + 6 + NSLOT] = ((pending_hexes.0 >> h) & 1) as f32;
        i += HEX_CH;
    }
    debug_assert_eq!(i, OFF_ZONES);

    for p in 0..2usize {
        let res = reserve(s, p as u8, ctx);
        for k in 0..NSLOT {
            let u = ctx.slots[p][k] as usize;
            out[i + k] = res[k] as f32 / 5.0;
            out[i + NSLOT + k] = s.zones[p][Z_FACEUP][u] as f32 / 5.0;
            out[i + 2 * NSLOT + k] = s.zones[p][Z_SUPPLY][u] as f32 / 5.0;
            out[i + 3 * NSLOT + k] = s.zones[p][Z_ELIM][u] as f32 / 5.0;
        }
        i += ZONE_FEATS;
    }
    debug_assert_eq!(i, OFF_IDENT);

    for p in 0..2usize {
        for k in 0..NSLOT {
            out[i + ctx.slots[p][k] as usize] = 1.0;
            i += N_UNITS;
        }
    }
    debug_assert_eq!(i, OFF_CARDS);

    // What each drafted card actually does. Under a fixed draft these are
    // constant across games and carry no information, but they are what lets a
    // draft the network has never seen be encoded at all, so they are the
    // prerequisite for `--random-draft`.
    for p in 0..2usize {
        for k in 0..NSLOT {
            write_card_features(ctx.slots[p][k], &mut out[i..i + CARD_FEATS]);
            i += CARD_FEATS;
        }
    }
    debug_assert_eq!(i, OFF_PLAYER);

    for p in 0..2usize {
        let fd: u8 = s.zones[p][Z_FACEDOWN].iter().sum();
        out[i] = s.markers_hand[p] as f32 / 6.0;
        out[i + 1] = s.markers_on_board(p as u8) as f32 / 6.0;
        out[i + 2] = s.hand_size(p as u8) as f32 / 3.0;
        // Divisors are the true maxima, not estimates. A player's coins are
        // bounded by the whole reserve, and both of these saturated under the
        // previous /10 and /12: the face-down count reaches 14 and the bag
        // reaches 18, so the two most dynamic scalars in the encoding were
        // being clipped in exactly the late-game states that matter most.
        out[i + 3] = fd as f32 / MAX_COINS;
        out[i + 4] = s.bag_size(p as u8) as f32 / MAX_COINS;
        out[i + 5] = s.turns_taken[p] as f32 / 3.0;
        out[i + 6] = (s.initiative == p as u8) as u8 as f32;
        out[i + 7] = (s.first_player == p as u8) as u8 as f32;
        i += PLAYER_SCALARS;
    }
    debug_assert_eq!(i, OFF_GLOBAL);

    out[i] = (s.round as f32 / MAX_ROUND).min(1.0);
    // plies_remaining: PBS values near the horizon are not well defined without it.
    let cap = crate::state::MAX_MAIN_PLAYS;
    out[i + 1] = (cap - s.main_plays.min(cap)) as f32 / cap as f32;
    out[i + 2] = s.initiative_moved as u8 as f32;
    out[i + 3] = (s.active == 0) as u8 as f32;
    out[i + 4] = (s.to_act() == 0) as u8 as f32;
    i += GLOBAL_SCALARS;
    out[i + pending_kind(s)] = 1.0;
    i += PEND_KINDS;
    // The coin a Footman-V2 instant deploy is holding. Public, unlike the
    // Warrior Priest's drawn coin (see the pending-mask note above).
    if let Cont::FootmanInstantDeploy { coin } = *s.pending() {
        let k = ctx.slot_of[s.to_act() as usize][coin as usize];
        if k >= 0 {
            out[i + k as usize] = 1.0;
        }
    }
    i += PEND_SLOT;
    debug_assert_eq!(i, PUBFEAT);
}
