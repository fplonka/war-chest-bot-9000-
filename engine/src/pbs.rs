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
//! So a player's private state is the triple `(hand, facedown, inflight)`,
//! where `inflight` is the Warrior Priest's drawn coin waiting for its forced
//! play — at most one coin, and empty at every other node. The bag is
//! *derived*:
//!
//!   `bag = reserve - hand - facedown - inflight`
//!
//! where `reserve` is public (total owned minus board, face-up discard, supply
//! and eliminated). Hand size, bag size and face-down count are public too —
//! you can see how many coins someone is holding.
//!
//! A `Config` is that triple (`inflight` is `None` at every MainPlay
//! boundary). The reachable set is small — measured over 120k
//! positions of random play with the full draft pool: median 22, mean 57,
//! p99 567 — so CFR enumerates information states exactly, with no particle
//! approximation.
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
//! strategy space, i.e. a different game.

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::state::{
    State, CONT_CAP, PENDING_KINDS, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_HAND, Z_INFLIGHT,
    Z_SUPPLY,
};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

/// Distinct coin types a player can own: 4 drafted units + the Royal Coin.
pub const NSLOT: usize = 5;
/// Hand size cap.
pub const HAND_CAP: usize = 3;
/// Largest private support one player can have. It is attained at reserve
/// `(5, 5, 5, 5, 1)`, hand size three and nine face-down coins.
pub const MAX_CONFIG_SUPPORT: usize = 4_628;

// ------------------------------------------------------------ config features

/// Raw counts describing one config: hand, face-down, bag, in slot order.
pub const CCOUNTS: usize = 3 * NSLOT;
/// The config vector is only hand, face-down, and derived bag counts. The
/// canonical query supplies the player's unit table.
pub const CFEAT: usize = CCOUNTS;
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

    pub fn mirrored(self) -> Ctx {
        Ctx {
            slots: [self.slots[1], self.slots[0]],
            slot_of: [self.slot_of[1], self.slot_of[0]],
        }
    }
}

// ------------------------------------------------------------------- configs

/// A player's private state: what is in hand, and what they have discarded face
/// down since the last reshuffle. The bag follows from the public reserve.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub struct Config {
    pub hand: [u8; NSLOT],
    pub fd: [u8; NSLOT],
    /// The Warrior Priest's drawn coin, waiting for the forced play that must
    /// spend it (`Z_INFLIGHT` in the world state). Empty at every MainPlay
    /// boundary. Never a replay or network feature: the forced play is walked
    /// through, so no training row carries an in-flight coin.
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
    /// `bag = reserve - hand - facedown - inflight`.
    #[inline]
    pub fn bag(&self, reserve: &[u8; NSLOT]) -> [u8; NSLOT] {
        let mut b = [0u8; NSLOT];
        for k in 0..NSLOT {
            b[k] = reserve[k].saturating_sub(
                self.hand[k] + self.fd[k] + u8::from(self.inflight == Some(k as u8)),
            );
        }
        b
    }
    /// A single integer that orders configs exactly as `Ord` does: hand
    /// lexicographically, then face-down, then the in-flight coin. Sorting and
    /// deduping child supports is the hot part of building a subgame, and
    /// comparing one `u64` beats walking the arrays — but only if the key is
    /// computed once per element rather than once per comparison, which is why
    /// this is a method and not an `Ord` impl.
    /// 2 bits per hand slot (a hand holds at most `HAND_CAP` = 3 coins in
    /// total, so no slot exceeds 3 — `key`'s debug_assert is the overflow
    /// test), 5 per face-down slot (bounded by the largest supply in the
    /// game) and 3 for the in-flight coin (`None` = 0, `Some(k)` = k + 1), for
    /// 38 bits — which leaves room to pack an element index alongside it in
    /// one `u64` (`config_key_packing_has_headroom` pins the headroom).
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

/// One config as raw counts: hand, then face-down, then the derived bag.
///
/// The bag is `reserve - hand - facedown` and so carries no information the
/// first two do not — but it is the quantity that decides what you draw next
/// round, and handing it over saves the network from having to learn a
/// subtraction it can only do if it also finds the reserve in the public block.
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

/// `reserve[k] = bag[k] + hand[k] + facedown[k] + inflight[k]` — public, and
/// invariant to how the unseen coins split. Every action either moves a coin
/// *within* the reserve (a face-down play) or out of it in full view (deploy,
/// bolster, any maneuver, the recruited coin), so the reserve is the same for
/// every config.
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

/// The true config of `p` in a world state: a straight read of three private
/// zones. The continuation stack is not consulted.
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

/// Instantiate a world: overwrite `p`'s hand, face-down discard and in-flight
/// zone with `c`, putting the rest of the reserve back in the bag. The public
/// projection is unchanged, which is what lets one public tree carry every
/// config.
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
        debug_assert!(
            c.hand[k] + c.fd[k] + inflight <= res,
            "config must fit in the reserve"
        );
        s.zones[p as usize][Z_HAND][u] = c.hand[k];
        s.zones[p as usize][Z_FACEDOWN][u] = c.fd[k];
        s.zones[p as usize][Z_INFLIGHT][u] = inflight;
        s.zones[p as usize][Z_BAG][u] = res - c.hand[k] - c.fd[k] - inflight;
    }
}

/// Every config consistent with the public counts. Self-play tracks the
/// reachable support forward instead; this exists for the brute-force belief
/// test and for sizing experiments.
/// Every config consistent with the public counts: hand size, face-down size,
/// and whether a forced-play coin is in flight (the identity of that coin is
/// private; its count is not). Self-play tracks the reachable support forward
/// instead; this exists for the brute-force belief test and for sizing.
pub fn enumerate_configs(
    reserve: &[u8; NSLOT],
    hand_size: u8,
    fd_size: u8,
    inflight: bool,
) -> Vec<Config> {
    fn rec_fd(
        res: &[u8; NSLOT],
        hand: &[u8; NSLOT],
        fd: &mut [u8; NSLOT],
        k: usize,
        left: u8,
        inflight: bool,
        out: &mut Vec<Config>,
    ) {
        if k == NSLOT - 1 {
            if left + hand[k] <= res[k] {
                fd[k] = left;
                if inflight {
                    for s in 0..NSLOT {
                        if hand[s] + fd[s] < res[s] {
                            out.push(Config {
                                hand: *hand,
                                fd: *fd,
                                inflight: Some(s as u8),
                            });
                        }
                    }
                } else {
                    out.push(Config {
                        hand: *hand,
                        fd: *fd,
                        inflight: None,
                    });
                }
                fd[k] = 0;
            }
            return;
        }
        for t in 0..=left.min(res[k].saturating_sub(hand[k])) {
            fd[k] = t;
            rec_fd(res, hand, fd, k + 1, left - t, inflight, out);
        }
        fd[k] = 0;
    }
    fn rec_hand(
        res: &[u8; NSLOT],
        hand: &mut [u8; NSLOT],
        k: usize,
        left: u8,
        fd_size: u8,
        inflight: bool,
        out: &mut Vec<Config>,
    ) {
        if k == NSLOT - 1 {
            if left <= res[k] {
                hand[k] = left;
                let mut fd = [0u8; NSLOT];
                rec_fd(res, hand, &mut fd, 0, fd_size, inflight, out);
                hand[k] = 0;
            }
            return;
        }
        for t in 0..=left.min(res[k]) {
            hand[k] = t;
            rec_hand(res, hand, k + 1, left - t, fd_size, inflight, out);
        }
        hand[k] = 0;
    }
    let mut out = Vec::new();
    let mut hand = [0u8; NSLOT];
    rec_hand(
        reserve, &mut hand, 0, hand_size, fd_size, inflight, &mut out,
    );
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
            // Support is reachability, not weight. Tree strategy rows index
            // this list, so an underflowed zero must not delete a row.
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

/// Is `slot` a legal coin for this config to spend right now? `slot` is the
/// coin slot the action spends (-1 for the micro-decisions that spend
/// nothing). An in-flight coin allows only that slot; otherwise any coin the
/// config holds.
#[inline]
pub fn action_legal(c: &Config, slot: i8) -> bool {
    match c.inflight {
        Some(p) => slot == p as i8,
        None => slot < 0 || c.hand[slot as usize] > 0,
    }
}

/// How a coin play moves a config. `slot` is the coin spent (-1 for the
/// micro-decisions that spend nothing); `facedown` says whether it goes to the
/// face-down discard (Pass / Claim Initiative / Recruit payment) or leaves the
/// reserve in full view.
///
/// A Warrior Priest forced play spends the in-flight coin and clears it; that
/// coin is not in the hand.
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

/// Belief update through a draw. The drawn coin is private, so the public tree
/// does not branch: the belief is convolved with each config's own draw
/// distribution. A config whose bag is empty reshuffles first, which folds its
/// face-down discards into the bag and therefore *erases* them.
///
/// `set_pending` marks a Warrior Priest draw: the drawn coin waits in flight
/// for the forced play, so it does not join the hand.
pub fn belief_after_draw(
    b: &Belief,
    reserve: &[u8; NSLOT],
    faceup: &[u8; NSLOT],
    set_pending: bool,
) -> Belief {
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
            if set_pending {
                debug_assert!(n.inflight.is_none(), "a coin is already in flight");
                n.inflight = Some(k as u8);
                pairs.push((n, *w * src[k] as f32 / total as f32));
            } else {
                n.hand[k] += 1;
                if n.hand_size() as usize <= HAND_CAP {
                    pairs.push((n, *w * src[k] as f32 / total as f32));
                }
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
    /// Host bytes this transition holds.
    pub fn bytes(&self) -> usize {
        (self.start.capacity() + self.to.capacity() + self.p.capacity()) * 4
    }

    #[inline]
    pub fn row(&self, ci: usize) -> (&[u32], &[f32]) {
        let (a, b) = (self.start[ci] as usize, self.start[ci + 1] as usize);
        (&self.to[a..b], &self.p[a..b])
    }
    pub fn rows(&self) -> usize {
        self.start.len().saturating_sub(1)
    }
    /// Entries: one per (parent config, child config) the draw can move mass
    /// between. Both directions of the transition are this long.
    pub fn len(&self) -> usize {
        self.to.len()
    }
    pub fn is_empty(&self) -> bool {
        self.to.is_empty()
    }
}

/// Reusable working memory for building draw transitions.
///
/// A subgame builds one transition per draw and composes it over the run, and
/// the buffers involved are all small — which is exactly why the allocator
/// traffic mattered more than the arithmetic. One scratch lives on the solver
/// and is reused for the whole tree.
/// Bits reserved for the element index when a config key and an index are
/// packed into one `u64`. `Config::key` occupies 38, so this leaves two spare
/// — and the largest thing sorted this way, a decision node's
/// `config x action` grid, runs to a few tens of thousands.
pub const IDX_BITS: u32 = 24;
pub const IDX_MASK: u64 = (1 << IDX_BITS) - 1;

#[derive(Default)]
pub struct DrawScratch {
    kid: Vec<Config>,
    prob: Vec<f32>,
    /// `key << IDX_BITS | index into kid`, sorted to build the child support.
    /// One `u64` rather than a `(u64, u32)` pair: a config key needs 38 bits
    /// and the index at most `IDX_BITS`, so packing them halves the width the
    /// sort moves around.
    order: Vec<u64>,
    acc: Vec<f32>,
    hit: Vec<bool>,
    touched: Vec<u32>,
}

impl DrawScratch {
    /// Host bytes this scratch holds.
    pub fn bytes(&self) -> usize {
        self.kid.capacity() * std::mem::size_of::<Config>()
            + (self.prob.capacity() + self.acc.capacity()) * 4
            + self.order.capacity() * 8
            + self.hit.capacity()
            + self.touched.capacity() * 4
    }

    /// The chance transition of one draw: for each config in `cfg`, which
    /// configs it can become (one per drawable coin type, reshuffling first if
    /// its bag is empty) and with what probability. Writes the sorted-deduped
    /// child support and the transition — the per-config chance factor the
    /// subgame tree's reach accounting needs, kept separate from both players'
    /// strategies.
    ///
    /// `set_pending` marks a Warrior Priest draw: the drawn coin waits in
    /// flight for the forced play, so it does not join the hand.
    pub fn transition(
        &mut self,
        cfg: &[Config],
        reserve: &[u8; NSLOT],
        faceup: &[u8; NSLOT],
        support: &mut Vec<Config>,
        map: &mut DrawMap,
        set_pending: bool,
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
                    if set_pending {
                        debug_assert!(n.inflight.is_none(), "a coin is already in flight");
                        n.inflight = Some(k as u8);
                        self.kid.push(n);
                        self.prob.push(src[k] as f32 * inv);
                    } else {
                        n.hand[k] += 1;
                        if n.hand_size() as usize <= HAND_CAP {
                            self.kid.push(n);
                            self.prob.push(src[k] as f32 * inv);
                        }
                    }
                }
            }
            map.start.push(self.kid.len() as u32);
        }
        drop(tg);
        self.pack(support, map);
    }

    /// A whole run of `k` round-start draws as one transition.
    ///
    /// Drawing `k` coins without replacement is multivariate hypergeometric:
    /// `P(x) = prod C(bag_i, x_i) / C(B, k)`. The per-draw composition this
    /// replaces built the same distribution one draw at a time over supports
    /// that grew ~5x per step, and was the largest non-network CPU cost.
    ///
    /// The refill rule (RULES.md §4.1) splits a run in two: when the bag
    /// holds fewer than `k` coins, the whole bag is drawn (deterministic),
    /// the discard pile refills it (face-down forgotten), and the remainder
    /// is hypergeometric over the refilled source. Warrior Priest draws stay
    /// on `transition`: they are single draws that set `inflight`.
    pub fn run(
        &mut self,
        cfg: &[Config],
        reserve: &[u8; NSLOT],
        faceup: &[u8; NSLOT],
        k: u8,
        support: &mut Vec<Config>,
        map: &mut DrawMap,
    ) {
        let tg = crate::timed!(DGEN);
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
                // The whole bag is drawn. Then the discards refill it.
                let mut base = *c;
                for s in 0..NSLOT {
                    base.hand[s] += bag[s];
                }
                let mut r = [0u8; NSLOT];
                for s in 0..NSLOT {
                    r[s] = faceup[s] + c.fd[s];
                }
                let rt: u32 = r.iter().map(|&x| x as u32).sum();
                let left = k - b;
                if rt == 0 {
                    // Nothing to refill from: the remaining draws are no-ops
                    // and the face-down pile never moved (`transition` keeps
                    // the config unchanged in this case too).
                    self.kid.push(base);
                    self.prob.push(1.0);
                } else {
                    base.fd = [0; NSLOT];
                    if rt <= left {
                        // Shortage: the refilled bag is drawn dry as well.
                        for s in 0..NSLOT {
                            base.hand[s] += r[s];
                        }
                        self.kid.push(base);
                        self.prob.push(1.0);
                    } else {
                        self.deal(&r, left, base);
                    }
                }
            }
            map.start.push(self.kid.len() as u32);
        }
        drop(tg);
        self.pack(support, map);
    }

    /// Every way to draw `k` coins from `src`, with hypergeometric weight,
    /// appended to `kid`/`prob`. Children enumerate in slot order, which is
    /// deterministic; `pack` sorts them by key anyway.
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
        rec(
            src,
            0,
            k,
            1.0,
            base,
            choose(total, k),
            &mut self.kid,
            &mut self.prob,
        );
    }

    /// Sort the emitted children once by integer key, then read the support
    /// and the child index of every row off that single ordering — no per-row
    /// binary search. Shared by `transition` and `run`.
    fn pack(&mut self, support: &mut Vec<Config>, map: &mut DrawMap) {
        let _ts = crate::timed!(DSORT);
        assert!(
            self.kid.len() < 1 << IDX_BITS,
            "draw fan-out over the index width"
        );
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
pub const MAX_COINS: f32 = 4.0 * 5.0 + 1.0;

/// The coin types in play: each player's `NSLOT` slots, player-major, so type
/// `p * NSLOT + k` is player `p`'s slot `k`. Everywhere the encoding refers to a
/// card — on a hex, in a pile, in a holding, paying for an action — it refers to
/// an index into this list, and the network turns that index into the card's
/// embedding. Nothing anywhere names a *unit*, which is what lets a draft the
/// network has never seen be read at all.
pub const NTYPE: usize = 2 * NSLOT;

/// Raw per-hex facts: occupant owner (2), stack height, the location marker's
/// owner (2), is-location, then one owed-bit per continuation-stack level.
///
/// Every one is a raw fact about the position. A precomputed
/// distance-to-nearest-unit map was tried and removed: it is a *derived*
/// summary of how far each side is from the locations it still needs — so it
/// imports the handcrafted bot's opinion into the encoding.
pub const HEX_FACTS: usize = 2 + 1 + 2 + 1 + CONT_CAP;
/// Per hex: the raw facts, then a one-hot of the occupant's coin type.
pub const HEX_CH: usize = HEX_FACTS + NTYPE;
pub const HEX_BLOCK: usize = N_HEXES * HEX_CH;
/// Per coin type: reserve, face-up discard, supply, eliminated.
pub const PILE_COUNTS: usize = 4;
/// Per player: markers in hand, markers on board, hand size, face-down count,
/// bag count, holds initiative. Round, turns-taken and first-player are
/// dropped with the format freeze: at a normal-turn boundary none of them
/// affects the future (the next round's first player is the initiative
/// holder, and the horizon is carried by `plies_remaining`).
pub const PLAYER_SCALARS: usize = 6;
/// Shared: plies remaining, initiative moved, the player to act, then a
/// kind one-hot for every continuation-stack level.
pub const GLOBAL_SCALARS: usize = 3 + CONT_CAP * PENDING_KINDS;

pub const OFF_PILES: usize = HEX_BLOCK;
/// The rulebook facts of each coin type in play — the card describer's input.
pub const OFF_CARDS: usize = OFF_PILES + NTYPE * PILE_COUNTS;
/// The scalars that belong to no hex and no card.
pub const OFF_LOOSE: usize = OFF_CARDS + NTYPE * CARD_FEATS;
pub const LOOSE: usize = 2 * PLAYER_SCALARS + GLOBAL_SCALARS;

// --------------------------------------------------------------- replay rows
//
// The frozen compact row format contains raw small integers, never floats of a
// learned encoding. The network input is rebuilt from it when a batch is made
// (`expand_row`). Board
// coordinates, location hexes and card facts are constants and are not stored
// per row. A dump pins the rules tables once in its header.

/// Packed size of one row in bytes. Layout (offsets are `ROW_*`):
///
/// ```text
/// 0   u8   unit_ids[2 * NSLOT]      player-major, slot order
/// 10  u8   hex_owner[37]            NONE sentinel
/// 47  u8   hex_unit_slot[37]        NONE sentinel
/// 84  u8   hex_height[37]
/// 121 u8   hex_marker_owner[37]     NONE sentinel
/// 158 u8   pile_counts[2][5][4]     reserve, face-up, supply, eliminated
/// 198 u8   public_sizes[3][2]       hand, face-down, bag; player-minor
/// 204 u8   initiative_holder
/// 205 u8   initiative_moved
/// 206 u8   player_to_act
/// 207 u16  main_plays_remaining
/// 209 u8   stack kind[CONT_CAP]     0xFF = empty slot
/// 225 u64  stack owed[CONT_CAP]     hex bitmask per level
/// ```

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
pub const ROW_PLIES: usize = ROW_TO_ACT + 1;
pub const ROW_STACK_KIND: usize = ROW_PLIES + 2;
pub const ROW_STACK_OWED: usize = ROW_STACK_KIND + CONT_CAP;
pub const ROW_BYTES: usize = ROW_STACK_OWED + CONT_CAP * 8;

/// Hash of the rules a referee needs: the board's location map and the unit
/// card-fact table. Encoding layout is the bot's own; `Dump.check` pins that
/// with `ROW_BYTES` / `PUBFEAT` / `CCOUNTS`.
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

/// Width of the public encoding.
///
/// This is what a *replay row* stores, and it is deliberately not what the
/// trunk consumes: the card embedding is a learned function, so a stored row
/// that contained it would hold whichever weights were live when the row was
/// written, go stale as training moved them, and pass no gradient back to the
/// describer. So the row keeps raw facts and one-hots, and the network does the
/// lookup itself — `Net::cards` builds the coin-type tokens once and
/// `Net::board` gathers them onto the hexes the one-hots name.
pub const PUBFEAT: usize = OFF_LOOSE + LOOSE;

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

/// The public encoding. Fixed for a given state, so a solve computes it once
/// per leaf. Reads only public information: bag, hand and face-down discards
/// appear solely through their public sum (the reserve) and their public
/// sizes.
///
/// Two producers feed this one layout: `write_public_features` from a live
/// `State` (the solver's leaf batch) and `expand_row` from a stored replay
/// row (the training batches). Both go through `write_public_features_raw`,
/// so the training side and the inference side cannot drift.
pub fn write_public_features(s: &State, ctx: &Ctx, out: &mut [f32]) {
    debug_assert_eq!(out.len(), PUBFEAT);
    let mut hex_owner = [NONE; N_HEXES];
    let mut hex_slot = [NONE; N_HEXES];
    let mut hex_height = [0u8; N_HEXES];
    let mut hex_marker = [NONE; N_HEXES];
    for h in 0..N_HEXES {
        hex_owner[h] = s.hex_owner[h];
        hex_slot[h] = if s.hex_owner[h] == NONE {
            NONE
        } else {
            ctx.slot_of[s.hex_owner[h] as usize][s.hex_type[h] as usize] as u8
        };
        hex_height[h] = s.hex_height[h];
        hex_marker[h] = s.loc_marker[h];
    }
    let mut piles = [0u8; 2 * NSLOT * PILE_COUNTS];
    for p in 0..2usize {
        let res = reserve(s, p as u8, ctx);
        for k in 0..NSLOT {
            let u = ctx.slots[p][k] as usize;
            let at = (p * NSLOT + k) * PILE_COUNTS;
            piles[at] = res[k];
            piles[at + 1] = s.zones[p][Z_FACEUP][u];
            piles[at + 2] = s.zones[p][Z_SUPPLY][u];
            piles[at + 3] = s.zones[p][Z_ELIM][u];
        }
    }
    let mut ids = [0u8; 2 * NSLOT];
    for t in 0..2 * NSLOT {
        ids[t] = ctx.slots[t / NSLOT][t % NSLOT];
    }
    let mut markers_hand = [0u8; 2];
    let mut hand_size = [0u8; 2];
    let mut fd_size = [0u8; 2];
    let mut bag_size = [0u8; 2];
    for p in 0..2usize {
        markers_hand[p] = s.markers_hand[p];
        hand_size[p] = s.hand_size(p as u8);
        fd_size[p] = s.zones[p][Z_FACEDOWN].iter().sum();
        bag_size[p] = s.bag_size(p as u8);
    }
    let (kinds, owed) = stack_bits(s);
    write_public_features_raw(
        &hex_owner,
        &hex_slot,
        &hex_height,
        &hex_marker,
        &piles,
        &ids,
        &markers_hand,
        &hand_size,
        &fd_size,
        &bag_size,
        s.initiative,
        s.initiative_moved,
        s.to_act(),
        (crate::state::MAX_MAIN_PLAYS - s.main_plays.min(crate::state::MAX_MAIN_PLAYS)) as u16,
        &kinds,
        &owed,
        out,
    );
}

/// The shared core of the public encoding: raw fields in, `PUBFEAT` floats
/// out. Called by `write_public_features` (from a `State`) and `expand_row`
/// (from a stored replay row).
#[allow(clippy::too_many_arguments)]
pub fn write_public_features_raw(
    hex_owner: &[u8; N_HEXES],
    hex_slot: &[u8; N_HEXES],
    hex_height: &[u8; N_HEXES],
    hex_marker: &[u8; N_HEXES],
    piles: &[u8],
    ids: &[u8],
    markers_hand: &[u8],
    hand_size: &[u8],
    fd_size: &[u8],
    bag_size: &[u8],
    initiative: u8,
    initiative_moved: bool,
    to_act: u8,
    plies_remaining: u16,
    kinds: &[u8; CONT_CAP],
    owed: &[u64; CONT_CAP],
    out: &mut [f32],
) {
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
            // Divisor is the largest coin count on any card. Bolstering has no
            // height limit (RULES.md section 5) and heights of 4 and 5 are
            // ~20% of occupied-hex observations under random play, so the
            // previous /3 collapsed them onto the same value.
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

    // The piles, per coin type rather than per player-and-slot: the same
    // counts, indexed the way everything else in the encoding indexes a card.
    for t in 0..2 * NSLOT {
        let at = t * PILE_COUNTS;
        out[i] = piles[at] as f32 / 5.0;
        out[i + 1] = piles[at + 1] as f32 / 5.0;
        out[i + 2] = piles[at + 2] as f32 / 5.0;
        out[i + 3] = piles[at + 3] as f32 / 5.0;
        i += PILE_COUNTS;
    }
    debug_assert_eq!(i, OFF_CARDS);

    // What each card in play actually does. The describer reads these; every
    // other reference to a card is a one-hot into this table.
    for t in 0..2 * NSLOT {
        write_card_features(ids[t], &mut out[i..i + CARD_FEATS]);
        i += CARD_FEATS;
    }
    debug_assert_eq!(i, OFF_LOOSE);

    for p in 0..2usize {
        out[i] = markers_hand[p] as f32 / 6.0;
        out[i + 1] = (6 - markers_hand[p]) as f32 / 6.0;
        out[i + 2] = hand_size[p] as f32 / 3.0;
        // Divisors are the true maxima, not estimates. A player's coins are
        // bounded by the whole reserve, and both of these saturated under the
        // previous /10 and /12: the face-down count reaches 14 and the bag
        // reaches 18, so the two most dynamic scalars in the encoding were
        // being clipped in exactly the late-game states that matter most.
        out[i + 3] = fd_size[p] as f32 / MAX_COINS;
        out[i + 4] = bag_size[p] as f32 / MAX_COINS;
        out[i + 5] = (initiative == p as u8) as u8 as f32;
        i += PLAYER_SCALARS;
    }

    // plies_remaining: PBS values near the horizon are not well defined
    // without it. The round number, turns taken and the first player are
    // dropped with the format freeze: none of them affects the future at a
    // normal-turn boundary.
    out[i] = plies_remaining as f32 / crate::state::MAX_MAIN_PLAYS as f32;
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

/// Pack one replay row from a live state.
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
    let plies = crate::state::MAX_MAIN_PLAYS - s.main_plays.min(crate::state::MAX_MAIN_PLAYS);
    out[ROW_PLIES..ROW_PLIES + 2].copy_from_slice(&plies.to_le_bytes());
    let (kinds, owed) = stack_bits(s);
    out[ROW_STACK_KIND..ROW_STACK_KIND + CONT_CAP].copy_from_slice(&kinds);
    for d in 0..CONT_CAP {
        let o = ROW_STACK_OWED + d * 8;
        out[o..o + 8].copy_from_slice(&owed[d].to_le_bytes());
    }
}


/// Expand a stored replay row into the public encoding, in place.
pub fn expand_row(row: &[u8], out: &mut [f32]) {
    debug_assert_eq!(row.len(), ROW_BYTES);
    let mut hex_owner = [NONE; N_HEXES];
    let mut hex_slot = [NONE; N_HEXES];
    let mut hex_height = [0u8; N_HEXES];
    let mut hex_marker = [NONE; N_HEXES];
    hex_owner.copy_from_slice(&row[ROW_HEX_OWNER..ROW_HEX_OWNER + N_HEXES]);
    hex_slot.copy_from_slice(&row[ROW_HEX_SLOT..ROW_HEX_SLOT + N_HEXES]);
    hex_height.copy_from_slice(&row[ROW_HEX_HEIGHT..ROW_HEX_HEIGHT + N_HEXES]);
    hex_marker.copy_from_slice(&row[ROW_HEX_MARKER..ROW_HEX_MARKER + N_HEXES]);
    let mut markers_hand = [0u8; 2];
    for h in 0..N_HEXES {
        if hex_marker[h] != NONE {
            markers_hand[hex_marker[h] as usize] += 1;
        }
    }
    for p in 0..2usize {
        markers_hand[p] = 6 - markers_hand[p];
    }
    let mut initiative = row[ROW_INITIATIVE];
    if initiative > 1 {
        initiative = 0; // defensive; a packed row is engine-written
    }
    let (kinds, owed) = stack_from_row(row);
    write_public_features_raw(
        &hex_owner,
        &hex_slot,
        &hex_height,
        &hex_marker,
        &row[ROW_PILES..ROW_PILES + 2 * NSLOT * PILE_COUNTS],
        &row[ROW_IDS..ROW_IDS + 2 * NSLOT],
        &markers_hand,
        row[ROW_HAND_SIZE..ROW_HAND_SIZE + 2].try_into().unwrap(),
        row[ROW_FD_SIZE..ROW_FD_SIZE + 2].try_into().unwrap(),
        row[ROW_BAG_SIZE..ROW_BAG_SIZE + 2].try_into().unwrap(),
        initiative,
        row[ROW_INIT_MOVED] != 0,
        row[ROW_TO_ACT],
        u16::from_le_bytes([row[ROW_PLIES], row[ROW_PLIES + 1]]),
        &kinds,
        &owed,
        out,
    );
}

#[cfg(test)]
mod draw_tests {
    use super::*;

    /// The per-draw composition `run` replaced, kept as the test reference.
    /// `(res, fu)` evolve the way the state's do: a step that finds the bag
    /// empty refills from the discards, after which the face-up pile is empty
    /// and the reserve has grown by it.
    fn composed(
        cfg: &[Config],
        res0: &[u8; NSLOT],
        fu0: &[u8; NSLOT],
        k: u8,
    ) -> (Vec<Config>, DrawMap) {
        let mut sc = DrawScratch::default();
        let (mut res, mut fu) = (*res0, *fu0);
        let mut cur = cfg.to_vec();
        let mut next = Vec::new();
        let (mut draw, mut step, mut acc) =
            (DrawMap::default(), DrawMap::default(), DrawMap::default());
        for j in 0..k {
            // Bag size is public, so config 0 speaks for the whole support.
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
                assert!(
                    (g.1 - w.1).abs() < 1e-5,
                    "{what}: row {i} prob {} vs {}",
                    g.1,
                    w.1
                );
            }
            let tot: f32 = gp.iter().sum();
            assert!((tot - 1.0).abs() < 1e-5, "{what}: row {i} sums to {tot}");
        }
    }

    /// Distribute `total` over the slots without exceeding `cap[s] - used[s]`.
    fn spread(
        rng: &mut crate::rng::Rng,
        total: u8,
        cap: &[u8; NSLOT],
        used: &[u8; NSLOT],
    ) -> Option<[u8; NSLOT]> {
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
        // Hand-built cases for each branch, then fuzz.
        let c = |hand: [u8; NSLOT], fd: [u8; NSLOT]| Config {
            hand,
            fd,
            inflight: None,
        };

        // Plenty in the bag: pure hypergeometric.
        assert_same(
            &[
                c([0; NSLOT], [1, 1, 0, 0, 0]),
                c([0; NSLOT], [0, 0, 1, 1, 0]),
            ],
            &[5, 4, 3, 2, 5],
            &[1, 0, 2, 0, 0],
            3,
            "plenty",
        );
        // Mid-run refill: one coin in the bag, three to draw. Every config
        // fits under the reserve per slot and shares the public totals.
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
        // Empty refill: nothing face-up, nothing face-down.
        assert_same(
            &[c([1, 0, 0, 0, 0], [0; NSLOT])],
            &[2, 0, 0, 0, 0],
            &[0; NSLOT],
            2,
            "empty refill",
        );
        // Shortage: an empty bag, and a refill too small for the run.
        assert_same(
            &[c([0; NSLOT], [0; NSLOT])],
            &[0; NSLOT],
            &[2, 0, 0, 0, 0],
            3,
            "shortage",
        );
        // Refill drawn exactly dry.
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
            // A support whose members share the public totals.
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
    fn config_support_has_a_fixed_maximum() {
        assert_eq!(
            enumerate_configs(&[5, 5, 5, 5, 1], 3, 9, false).len(),
            MAX_CONFIG_SUPPORT
        );
    }

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
