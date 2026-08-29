use crate::board::{board, NONE, N_HEXES};
use crate::units::{def, index_of_id, N_UNITS, ROYAL_COIN};

pub const N_PLAYERS: usize = 2;
pub const HAND_MAX: u8 = 3;
pub const MARKERS_TOTAL: u8 = 6;
pub const WIN_MARKERS: u8 = 6;
pub const MAX_MAIN_PLAYS: u16 = 256;
pub const Z_BAG: usize = 0;
pub const Z_HAND: usize = 1;
pub const Z_FACEUP: usize = 2;
pub const Z_FACEDOWN: usize = 3;
pub const Z_SUPPLY: usize = 4;
pub const Z_ELIM: usize = 5;
pub const Z_INFLIGHT: usize = 6;
pub const N_ZONES: usize = 7;

pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct HexSet(pub u64);

impl HexSet {
    #[inline]
    pub fn insert(&mut self, h: u8) {
        debug_assert!((h as usize) < 64);
        self.0 |= 1u64 << h;
    }
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
    Draw { player: u8 },
    MainPlay,
    RoyalGuardChoice { defender: u8, rg_hex: u8 },
    SwordsmanMove { hex: u8 },
    BerserkerChain { hex: u8, v2: bool },
    FootmanManeuver { hexes: HexSet },
    CavalryAttack { hex: u8 },
    MercenaryManeuver { hex: u8 },
    FootmanInstantDeploy { coin: u8 },
    WarriorPriestDraw { player: u8, rg_hex: u8 },
    WarriorPriestPlay { player: u8 },
    _AttackPost { atk_hex: u8 },
}

pub const PENDING_KINDS: usize = 12;

impl Cont {
    pub fn tag(self) -> u8 {
        match self {
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

    pub fn owed_hexes(self) -> HexSet {
        let one = |h: u8| {
            let mut s = HexSet::default();
            if (h as usize) < N_HEXES {
                s.insert(h);
            }
            s
        };
        match self {
            Cont::RoyalGuardChoice { rg_hex, .. } => one(rg_hex),
            Cont::SwordsmanMove { hex }
            | Cont::BerserkerChain { hex, .. }
            | Cont::CavalryAttack { hex }
            | Cont::MercenaryManeuver { hex } => one(hex),
            Cont::FootmanManeuver { hexes } => hexes,
            Cont::WarriorPriestDraw { rg_hex, .. } => one(rg_hex),
            Cont::_AttackPost { atk_hex } => one(atk_hex),
            _ => HexSet::default(),
        }
    }

    fn mirrored(self) -> Cont {
        let flip = |p: u8| if p == NONE { NONE } else { 1 - p };
        let hex = |h: u8| {
            if (h as usize) < N_HEXES {
                mirror_hex(h as usize) as u8
            } else {
                h
            }
        };
        match self {
            Cont::Draw { player } => Cont::Draw { player: flip(player) },
            Cont::MainPlay => Cont::MainPlay,
            Cont::RoyalGuardChoice { defender, rg_hex } => Cont::RoyalGuardChoice {
                defender: flip(defender),
                rg_hex: hex(rg_hex),
            },
            Cont::SwordsmanMove { hex: h } => Cont::SwordsmanMove { hex: hex(h) },
            Cont::BerserkerChain { hex: h, v2 } => Cont::BerserkerChain { hex: hex(h), v2 },
            Cont::FootmanManeuver { hexes } => {
                let mut out = HexSet::default();
                for h in hexes.iter() {
                    out.insert(hex(h));
                }
                Cont::FootmanManeuver { hexes: out }
            }
            Cont::CavalryAttack { hex: h } => Cont::CavalryAttack { hex: hex(h) },
            Cont::MercenaryManeuver { hex: h } => Cont::MercenaryManeuver { hex: hex(h) },
            Cont::FootmanInstantDeploy { coin } => Cont::FootmanInstantDeploy { coin },
            Cont::WarriorPriestDraw { player, rg_hex } => Cont::WarriorPriestDraw {
                player: flip(player),
                rg_hex: hex(rg_hex),
            },
            Cont::WarriorPriestPlay { player } => Cont::WarriorPriestPlay { player: flip(player) },
            Cont::_AttackPost { atk_hex } => Cont::_AttackPost { atk_hex: hex(atk_hex) },
        }
    }
}

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
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Cont> {
        self.v[..self.n as usize].iter().rev()
    }

    fn mirrored(self) -> ContStack {
        let mut o = self;
        for i in 0..o.n as usize {
            o.v[i] = o.v[i].mirrored();
        }
        o
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub hex_type: [u8; N_HEXES],
    pub hex_owner: [u8; N_HEXES],
    pub hex_height: [u8; N_HEXES],
    pub loc_marker: [u8; N_HEXES],

    pub zones: [[[u8; N_UNITS]; N_ZONES]; N_PLAYERS],

    pub markers_hand: [u8; N_PLAYERS],
    pub initiative: u8,
    pub initiative_moved: bool,
    pub round: u16,
    pub first_player: u8,
    pub active: u8,
    pub turns_taken: [u8; N_PLAYERS],
    pub main_plays: u16,

    pub winner: u8,
    pub adjudicated_draw: bool,

    pub pending: Cont,
    pub conts: ContStack,
    pub wp_v2_triggered: bool,
    pub interrupt: bool,
}

impl State {
    pub fn from_draft(white_units: &[u16], black_units: &[u16], first_player: u8) -> State {
        assert!(first_player == WHITE || first_player == BLACK);
        let mut s = State::blank(first_player);

        for (p, units) in [(WHITE, white_units), (BLACK, black_units)] {
            assert_eq!(units.len(), 4, "each player drafts exactly 4 unit types");
            for &id in units {
                let u = index_of_id(id).expect("drafted unit must be a known unitTypeId");
                let total = def(u).coins;
                s.zones[p as usize][Z_BAG][u as usize] += 2;
                s.zones[p as usize][Z_SUPPLY][u as usize] += total - 2;
            }
            s.zones[p as usize][Z_BAG][ROYAL_COIN as usize] += 1;
            s.markers_hand[p as usize] = MARKERS_TOTAL - 2;
        }

        let b = board();
        for (p, locs) in [(WHITE, [0usize, 1usize]), (BLACK, [2usize, 3usize])] {
            for li in locs {
                let h = b.location_hexes[li] as usize;
                s.loc_marker[h] = p;
            }
        }

        s.start_round_draws();
        s
    }

    pub fn start_round_draws(&mut self) {
        self.turns_taken = [0, 0];
        self.initiative_moved = false;
        let first = self.first_player;
        let other = 1 - first;
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

        if need_first == 0 && need_other == 0 && self.hand_size(first) == 0 && self.hand_size(other) == 0 {
            self.adjudicated_draw = true;
            return;
        }

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
            self.conts.pop();
            Cont::Draw { player: other }
        } else {
            self.conts.pop();
            Cont::MainPlay
        };
    }

    #[inline]
    pub fn push_cont(&mut self, c: Cont) {
        self.conts.push(c);
    }

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
            None => 0.0,
        }
    }

    pub fn to_act(&self) -> u8 {
        match &self.pending {
            Cont::Draw { player, .. } => *player,
            Cont::WarriorPriestDraw { player, .. } => *player,
            Cont::WarriorPriestPlay { player } => *player,
            Cont::RoyalGuardChoice { defender, .. } => *defender,
            _ => self.active,
        }
    }

    pub fn is_chance(&self) -> bool {
        matches!(self.pending, Cont::Draw { .. } | Cont::WarriorPriestDraw { .. })
    }

    pub fn is_valued(&self) -> bool {
        !self.is_terminal() && !self.is_chance() && !matches!(self.pending, Cont::WarriorPriestPlay { .. })
    }

    #[inline]

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

    pub fn set_unit(&mut self, hex: u8, owner: u8, unit: u8, height: u8) {
        let hex = hex as usize;
        self.hex_type[hex] = unit;
        self.hex_owner[hex] = owner;
        self.hex_height[hex] = height;
    }

    pub fn set_marker(&mut self, hex: u8, owner: u8) {
        self.loc_marker[hex as usize] = owner;
    }

    pub fn add_zone(&mut self, p: u8, zone: usize, unit: u8, n: u8) {
        self.zones[p as usize][zone][unit as usize] += n;
    }

    pub fn set_markers_hand(&mut self, p: u8, n: u8) {
        self.markers_hand[p as usize] = n;
    }

    pub fn set_initiative(&mut self, holder: u8, moved: bool) {
        self.initiative = holder;
        self.initiative_moved = moved;
    }

    pub fn pending(&self) -> &Cont {
        &self.pending
    }

    pub fn stack(&self) -> [Option<Cont>; CONT_CAP] {
        let mut a = [None; CONT_CAP];
        a[0] = Some(self.pending);
        for (i, c) in self.conts.iter().enumerate() {
            if 1 + i < CONT_CAP {
                a[1 + i] = Some(*c);
            }
        }
        a
    }

    pub fn mirror(&self) -> State {
        let flip = |p: u8| if p == NONE { NONE } else { 1 - p };
        let mut m = *self;
        for h in 0..N_HEXES {
            let k = mirror_hex(h);
            m.hex_type[k] = self.hex_type[h];
            m.hex_height[k] = self.hex_height[h];
            m.hex_owner[k] = flip(self.hex_owner[h]);
            m.loc_marker[k] = flip(self.loc_marker[h]);
        }
        m.zones.swap(0, 1);
        m.markers_hand.swap(0, 1);
        m.turns_taken.swap(0, 1);
        m.initiative = flip(self.initiative);
        m.first_player = flip(self.first_player);
        m.active = flip(self.active);
        m.winner = flip(self.winner);
        m.pending = self.pending.mirrored();
        m.conts = self.conts.mirrored();
        m
    }
}


pub fn mirror_hex(h: usize) -> usize {
    let bd = board();
    let (x, y) = bd.coord[h];
    (0..N_HEXES)
        .find(|&k| bd.coord[k] == (6 - x, 6 - y))
        .expect("the rotation stays on the board")
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
        assert_eq!(terminal.zones[WHITE as usize][Z_FACEDOWN][ROYAL_COIN as usize], 1);
    }
}