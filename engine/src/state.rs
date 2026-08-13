//! Flat game state and the decision-node continuation model.

use crate::board::{board, NONE, N_HEXES};
use crate::units::{def, index_of_id, N_UNITS, ROYAL_COIN};

pub const N_PLAYERS: usize = 2;
pub const HAND_MAX: u8 = 3;
pub const MARKERS_TOTAL: u8 = 6;
pub const WIN_MARKERS: u8 = 6;
/// The training/evaluation game is finite (ReBeL's theory needs that). A coin
/// play that reaches this count resolves completely, then the game is
/// adjudicated before another top-level coin play begins.
pub const MAX_MAIN_PLAYS: u16 = 256;
/// Default value of one control marker of lead when the horizon is reached.
///
/// A flat zero at the horizon is a trap: under early, near-random play almost
/// no game ends by placing all six markers, so every target is zero and `V = 0`
/// becomes a self-consistent fixed point with no gradient toward winning.
/// Scoring the marker differential instead keeps the payoff zero-sum and
/// strictly inside +/-1, and induces a curriculum (take locations -> deny
/// locations -> race to six).
///
/// It is a change to the terminal payoff of the game being solved, so it is
/// annealed to zero as soon as horizon games become rare — see
/// `set_cap_marker_value`. At zero the payoff is the real game's: a timeout is
/// a draw. Evaluation always runs at zero.
pub const CAP_MARKER_VALUE_DEFAULT: f32 = 0.15;

/// 0.15f32 in IEEE-754 bits (`AtomicU32` cannot be initialised from a float).
static CAP_MARKER_VALUE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0x3E19_999A);

/// The current horizon payoff per marker of lead.
#[inline]
pub fn cap_marker_value() -> f32 {
    f32::from_bits(CAP_MARKER_VALUE.load(std::sync::atomic::Ordering::Relaxed))
}

/// Set the horizon payoff. The trainer anneals this toward 0 once the fraction
/// of games reaching the horizon falls, so the distortion is temporary.
pub fn set_cap_marker_value(v: f32) {
    CAP_MARKER_VALUE.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

// Zones (indices into State::zones[player][zone][unit]).
pub const Z_BAG: usize = 0;
pub const Z_HAND: usize = 1;
pub const Z_FACEUP: usize = 2;
pub const Z_FACEDOWN: usize = 3;
pub const Z_SUPPLY: usize = 4;
pub const Z_ELIM: usize = 5;
/// The Warrior Priest's drawn coin, between the draw and the forced play that
/// must spend it. A one-coin private zone: its size is public (the pending
/// node says whether a forced play is owed), its identity is not.
pub const Z_INFLIGHT: usize = 6;
pub const N_ZONES: usize = 7;

pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;

/// A queued micro-decision or forced follow-up. The stack is LIFO: `pending`
/// (below) is the item currently being decided; `conts` holds items to resolve
/// after it. Every variant that requires a player choice becomes a decision
/// node; forced steps are executed inline by `apply` without a node.
/// A set of board hexes as a bitmask (`N_HEXES <= 64`), iterating in ascending
/// hex order.
///
/// The point is that it is inline: it is what makes `Cont`, and therefore
/// `State`, `Copy`. Cloning a state used to allocate — twice, for the `conts`
/// vector and for this list — and a depth-2 subgame clones one per node, per
/// child probe, and per greedy rollout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct HexSet(pub u64);

impl HexSet {
    #[inline]
    pub fn insert(&mut self, h: u8) {
        debug_assert!((h as usize) < 64);
        self.0 |= 1u64 << h;
    }
    /// Removing a non-hex (`NONE`) is a no-op, which is what the Footman
    /// tactic's "drop whoever just acted" wants when nobody did.
    #[inline]
    pub fn remove(&mut self, h: u8) {
        if h < 64 {
            self.0 &= !(1u64 << h);
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.0.count_ones() as usize
    }
    #[inline]
    pub fn iter(self) -> HexSetIter {
        HexSetIter(self.0)
    }
    pub fn to_vec(self) -> Vec<u8> {
        self.iter().collect()
    }
}

pub struct HexSetIter(u64);

impl Iterator for HexSetIter {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            return None;
        }
        let h = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(h)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cont {
    /// A chance node: `player` draws one coin from their bag (refill applies).
    /// Each round-start draw is one such node; a "draw to 3" is a run of them.
    Draw { player: u8 },
    /// Active player's normal coin play for their turn.
    MainPlay,
    /// Defender chooses whether to soak an attack on a Royal Guard from supply.
    /// Set up mid-attack; on resolution the coin removal completes, then the
    /// queued attacker post-triggers run.
    RoyalGuardChoice { defender: u8, rg_hex: u8 },
    /// Optional Swordsman free one-step move after it attacked (from `hex`).
    SwordsmanMove { hex: u8 },
    /// Optional Berserker chain: maneuver the Berserker at `hex` again by
    /// discarding a bolstered coin. `v2` restricts to attack/move only.
    BerserkerChain { hex: u8, v2: bool },
    /// Footman tactic: `hexes` are the Footman units still owed a maneuver.
    /// The player picks which of them maneuvers next (order is a free choice,
    /// verified against server replays); each acts at most once.
    FootmanManeuver { hexes: HexSet },
    /// Cavalry tactic second step: the unit at `hex` must attack if able (the
    /// "move, then attack" of the Cavalry card). Skipped if no target.
    CavalryAttack { hex: u8 },
    /// Optional Mercenary free maneuver (unit at `hex`) after recruiting it.
    MercenaryManeuver { hex: u8 },
    /// Optional Footman V2 instant deploy of the just-recruited coin (`coin` is
    /// the recruited footman unit index, sitting in the face-up discard until
    /// the deploy actually happens — verified against server snapshots).
    FootmanInstantDeploy { coin: u8 },
    /// Warrior Priest draw (chance node) then forced play of the drawn coin.
    /// `rg_hex` != NONE marks the interrupted-attack case: the WP attacked a
    /// Royal Guard whose defender still owes a soak choice. The draw resolves
    /// BEFORE that choice (server-verified) but the forced play comes after
    /// it, so the draw's apply re-installs the RoyalGuardChoice node.
    WarriorPriestDraw { player: u8, rg_hex: u8 },
    /// Warrior Priest forced play of the coin just drawn (any action using that
    /// coin type; pass always legal). The coin itself is in `Z_INFLIGHT`, which
    /// is where the play pays from — this node names no private information.
    WarriorPriestPlay { player: u8 },
    /// Internal bookkeeping: after a deferred (RoyalGuard) attack resolves,
    /// queue the attacker's post-triggers. Never a decision node; consumed by
    /// advance() the instant it surfaces.
    _AttackPost { atk_hex: u8 },
}

/// The LIFO continuation stack, inline.
///
/// A round start queues at most six draws plus the main play, and a coin play's
/// follow-ups never went deeper than that over millions of random playouts —
/// but the cap is asserted rather than assumed, and `tests/invariants.rs`
/// exercises it.
pub const CONT_CAP: usize = 16;

#[derive(Clone, Copy, Debug, Eq)]
pub struct ContStack {
    n: u8,
    v: [Cont; CONT_CAP],
}

impl Default for ContStack {
    fn default() -> ContStack {
        ContStack {
            n: 0,
            v: [Cont::MainPlay; CONT_CAP],
        }
    }
}

impl PartialEq for ContStack {
    fn eq(&self, o: &ContStack) -> bool {
        self.n == o.n && self.v[..self.n as usize] == o.v[..o.n as usize]
    }
}

impl ContStack {
    #[inline]
    pub fn push(&mut self, c: Cont) {
        assert!(
            (self.n as usize) < CONT_CAP,
            "continuation stack overflow: raise CONT_CAP"
        );
        self.v[self.n as usize] = c;
        self.n += 1;
    }
    #[inline]
    pub fn pop(&mut self) -> Option<Cont> {
        if self.n == 0 {
            return None;
        }
        self.n -= 1;
        Some(self.v[self.n as usize])
    }
    #[inline]
    pub fn clear(&mut self) {
        self.n = 0;
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.n as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// Continuations in resolution order (the stack's top first).
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Cont> {
        self.v[..self.n as usize].iter().rev()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    // ---- board occupancy (flat, indexed by hex) ----
    pub hex_type: [u8; N_HEXES],  // unit index or NONE
    pub hex_owner: [u8; N_HEXES], // unit owner WHITE/BLACK, or NONE if empty
    pub hex_height: [u8; N_HEXES],
    /// Control-marker owner for location hexes; NONE if uncontrolled or the hex
    /// is not a location. Independent of whether a unit stands on the hex.
    pub loc_marker: [u8; N_HEXES],

    // ---- per-player zone multisets ----
    pub zones: [[[u8; N_UNITS]; N_ZONES]; N_PLAYERS],

    // ---- markers / initiative / turn ----
    pub markers_hand: [u8; N_PLAYERS],
    pub initiative: u8,         // holder
    pub initiative_moved: bool, // moved this round already
    pub round: u16,
    pub first_player: u8,             // who acts first this round
    pub active: u8,                   // whose main turn it is
    pub turns_taken: [u8; N_PLAYERS], // coin plays this round (0..3)
    pub main_plays: u16,              // fully started top-level coin plays

    // ---- winner ----
    pub winner: u8, // NONE = ongoing, WHITE/BLACK = decided
    pub adjudicated_draw: bool,

    // ---- decision context ----
    pub pending: Cont,
    pub conts: ContStack,
    /// Warrior Priest V2: whether V2 itself already triggered this turn (reset
    /// at end_turn). V1 has no cap, and a V1 trigger must not block V2.
    pub wp_v2_triggered: bool,
    /// Transient flag: set when a maneuver's resolution installed a mid-attack
    /// decision node (defender Royal Guard choice), so the caller must NOT
    /// advance past it. Reset whenever a new decision node is set.
    pub interrupt: bool,
}

impl State {
    /// Build the initial state from a draft.
    /// `white_units`/`black_units`: 4 unitTypeIds each. `first_player`: WHITE/BLACK.
    pub fn from_draft(white_units: &[u16], black_units: &[u16], first_player: u8) -> State {
        assert!(first_player == WHITE || first_player == BLACK);
        let mut s = State {
            hex_type: [NONE; N_HEXES],
            hex_owner: [NONE; N_HEXES],
            hex_height: [0; N_HEXES],
            loc_marker: [NONE; N_HEXES],
            zones: [[[0; N_UNITS]; N_ZONES]; N_PLAYERS],
            markers_hand: [0; N_PLAYERS],
            initiative: first_player,
            initiative_moved: false,
            round: 1,
            first_player,
            active: first_player,
            turns_taken: [0; N_PLAYERS],
            main_plays: 0,
            winner: NONE,
            adjudicated_draw: false,
            pending: Cont::MainPlay, // replaced below
            conts: ContStack::default(),
            wp_v2_triggered: false,
            interrupt: false,
        };

        for (p, units) in [(WHITE, white_units), (BLACK, black_units)] {
            assert_eq!(units.len(), 4, "each player drafts exactly 4 unit types");
            // 2 coins of each drafted type + 1 Royal Coin into the bag; the
            // rest of each type's coins into supply.
            for &id in units {
                let u = index_of_id(id).expect("drafted unit must be a known unitTypeId");
                let total = def(u).coins;
                s.zones[p as usize][Z_BAG][u as usize] += 2;
                s.zones[p as usize][Z_SUPPLY][u as usize] += total - 2;
            }
            // Royal Coin: 1 into the bag, that player owns exactly one.
            s.zones[p as usize][Z_BAG][ROYAL_COIN as usize] += 1;
            s.markers_hand[p as usize] = MARKERS_TOTAL - 2; // 2 on the board at setup
        }

        // Place starting control markers on each player's two start locations.
        let b = board();
        for (p, locs) in [(WHITE, [0usize, 1usize]), (BLACK, [2usize, 3usize])] {
            for li in locs {
                let h = b.location_hexes[li] as usize;
                s.loc_marker[h] = p; // control marker; no unit stands here
            }
        }

        // Kick off round 1: both players draw to 3, initiative holder first.
        s.start_round_draws();
        s
    }

    /// Queue the round-start draws and set the first draw as pending.
    /// The continuation stack `conts` is LIFO; we push in reverse so items pop
    /// in intended order: first player draws, then other, then MainPlay.
    pub fn start_round_draws(&mut self) {
        self.turns_taken = [0, 0];
        self.initiative_moved = false;
        let first = self.first_player;
        let other = 1 - first;
        // A player draws up to 3, but no more coins than they actually have
        // available (bag + discard pile, which refills the bag mid-draw).
        let cap_draws = |s: &State, p: u8| -> u8 {
            let need = HAND_MAX.saturating_sub(s.hand_size(p));
            let mut avail = 0u8;
            for u in 0..N_UNITS {
                avail += s.zones[p as usize][Z_BAG][u]
                    + s.zones[p as usize][Z_FACEUP][u]
                    + s.zones[p as usize][Z_FACEDOWN][u];
            }
            need.min(avail)
        };
        let need_first = cap_draws(self, first);
        let need_other = cap_draws(self, other);

        // Neither player can draw and neither holds a coin: every coin is on
        // the board, so no play is possible and the game cannot continue. Left
        // alone this produces a non-terminal state with zero legal actions --
        // reachable in a long game with heavy bolstering, and a panic rather
        // than a hang, because it is `begin_main_turn` that guards the ply cap
        // and this path bypasses it.
        if need_first == 0
            && need_other == 0
            && self.hand_size(first) == 0
            && self.hand_size(other) == 0
        {
            self.adjudicated_draw = true;
            return;
        }

        // Forward order is `need_first` draws, then `need_other`, then the main
        // play; the stack is LIFO, so push in reverse and take the first item
        // as pending.
        self.conts.clear();
        self.conts.push(Cont::MainPlay);
        for _ in 0..need_other {
            self.conts.push(Cont::Draw { player: other });
        }
        for _ in 1..need_first {
            self.conts.push(Cont::Draw { player: first });
        }
        self.pending = if need_first > 0 {
            Cont::Draw { player: first }
        } else if need_other > 0 {
            // No draws for the first player: the other's first draw leads, and
            // the stack must not keep a copy of it.
            self.conts.pop();
            Cont::Draw { player: other }
        } else {
            self.conts.pop();
            Cont::MainPlay
        };
    }

    /// Push a follow-up onto the LIFO continuation stack (resolved after the
    /// current pending item).
    #[inline]
    pub fn push_cont(&mut self, c: Cont) {
        self.conts.push(c);
    }

    /// Interrupt: make `c` the current decision. The current pending item is
    /// assumed already consumed by the in-progress apply, so we just overwrite.
    #[inline]
    pub fn set_pending(&mut self, c: Cont) {
        self.pending = c;
    }

    #[inline]
    pub fn hand_size(&self, p: u8) -> u8 {
        self.zones[p as usize][Z_HAND].iter().copied().sum()
    }

    #[inline]
    pub fn bag_size(&self, p: u8) -> u8 {
        self.zones[p as usize][Z_BAG].iter().copied().sum()
    }

    #[inline]
    pub fn markers_on_board(&self, p: u8) -> u8 {
        let mut n = 0u8;
        for h in 0..N_HEXES {
            if self.loc_marker[h] == p {
                n += 1;
            }
        }
        n
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.winner != NONE || self.adjudicated_draw
    }

    #[inline]
    pub fn winner(&self) -> Option<u8> {
        if self.winner == NONE {
            None
        } else {
            Some(self.winner)
        }
    }

    #[inline]
    pub fn utility(&self, player: usize) -> f32 {
        assert!(player < N_PLAYERS);
        assert!(self.is_terminal(), "utility is defined only at terminals");
        match self.winner() {
            Some(winner) if winner as usize == player => 1.0,
            Some(_) => -1.0,
            // Horizon: score the marker differential (see cap_marker_value).
            None => {
                let me = self.markers_on_board(player as u8) as f32;
                let them = self.markers_on_board(1 - player as u8) as f32;
                cap_marker_value() * (me - them)
            }
        }
    }

    /// Whose decision it is right now (the player to act, including chance,
    /// where the "player" is whoever the draw belongs to).
    pub fn to_act(&self) -> u8 {
        match &self.pending {
            Cont::Draw { player, .. } => *player,
            Cont::WarriorPriestDraw { player, .. } => *player,
            Cont::WarriorPriestPlay { player } => *player,
            Cont::RoyalGuardChoice { defender, .. } => *defender,
            _ => self.active,
        }
    }

    /// Is the current decision a chance node (a draw)?
    pub fn is_chance(&self) -> bool {
        matches!(
            self.pending,
            Cont::Draw { .. } | Cont::WarriorPriestDraw { .. }
        )
    }

    /// Coins physically in a stack on the board at `hex` (unit's coin count).
    #[inline]
    pub fn stack_height(&self, hex: usize) -> u8 {
        self.hex_height[hex]
    }

    /// Total coins of a type a player currently holds across all zones + board.
    /// Used by invariant checks.
    pub fn total_coins(&self, p: u8, unit: usize) -> u8 {
        let mut n = 0u8;
        for z in 0..N_ZONES {
            n += self.zones[p as usize][z][unit];
        }
        for h in 0..N_HEXES {
            if self.hex_owner[h] == p && self.hex_type[h] as usize == unit {
                n += self.hex_height[h];
            }
        }
        n
    }
}

// ------------------------------------------------------------- test builders
// Public constructors used by the scenario tests to build hand-crafted
// positions. They bypass the normal setup, so they leave coin conservation to
// the caller; not intended for gameplay.

impl State {
    /// A blank state: empty board, empty zones, no markers in hand, WHITE to
    /// act on a MainPlay. Round 1. Callers place units/markers/hand coins.
    pub fn blank(active: u8) -> State {
        State {
            hex_type: [NONE; N_HEXES],
            hex_owner: [NONE; N_HEXES],
            hex_height: [0; N_HEXES],
            loc_marker: [NONE; N_HEXES],
            zones: [[[0; N_UNITS]; N_ZONES]; N_PLAYERS],
            markers_hand: [0; N_PLAYERS],
            initiative: active,
            initiative_moved: false,
            round: 1,
            first_player: active,
            active,
            turns_taken: [0; N_PLAYERS],
            main_plays: 0,
            winner: NONE,
            adjudicated_draw: false,
            pending: Cont::MainPlay,
            conts: ContStack::default(),
            wp_v2_triggered: false,
            interrupt: false,
        }
    }

    /// Place (or overwrite) a unit stack on the board for tests.
    pub fn set_unit(&mut self, hex: usize, owner: u8, unit: u8, height: u8) {
        self.hex_type[hex] = unit;
        self.hex_owner[hex] = owner;
        self.hex_height[hex] = height;
    }

    /// Set a location's control marker for tests.
    pub fn set_marker(&mut self, hex: usize, owner: u8) {
        self.loc_marker[hex] = owner;
    }

    /// Add `n` coins of `unit` to a zone for `p`.
    pub fn add_zone(&mut self, p: u8, zone: usize, unit: u8, n: u8) {
        self.zones[p as usize][zone][unit as usize] += n;
    }

    /// Directly set the markers a player still holds in hand.
    pub fn set_markers_hand(&mut self, p: u8, n: u8) {
        self.markers_hand[p as usize] = n;
    }

    /// Set initiative holder + whether it has moved this round (tests).
    pub fn set_initiative(&mut self, holder: u8, moved: bool) {
        self.initiative = holder;
        self.initiative_moved = moved;
    }

    /// Read the current pending decision (tests/replay hooks).
    pub fn pending(&self) -> &Cont {
        &self.pending
    }
}

#[cfg(test)]
mod horizon_tests {
    use super::*;
    use crate::actions::Action;

    #[test]
    fn final_main_play_resolves_then_adjudicates_zero_utility_draw() {
        let mut state = State::blank(WHITE);
        state.main_plays = MAX_MAIN_PLAYS - 1;
        state.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);

        let terminal = state.apply(Action::Pass { coin: ROYAL_COIN });

        assert!(terminal.is_terminal());
        assert!(terminal.adjudicated_draw);
        assert_eq!(terminal.main_plays, MAX_MAIN_PLAYS);
        assert_eq!(terminal.winner(), None);
        assert_eq!(terminal.utility(WHITE as usize), 0.0);
        assert_eq!(terminal.utility(BLACK as usize), 0.0);
        assert_eq!(
            terminal.zones[WHITE as usize][Z_FACEDOWN][ROYAL_COIN as usize],
            1
        );
    }

    #[test]
    fn winner_utility_is_zero_sum() {
        let mut state = State::blank(WHITE);
        state.winner = BLACK;
        assert_eq!(state.utility(WHITE as usize), -1.0);
        assert_eq!(state.utility(BLACK as usize), 1.0);
    }
}
