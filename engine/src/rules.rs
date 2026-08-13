//! Legality + apply. One module because the two share a lot of geometry and
//! attack helpers, and the hot path benefits from inlining across them.

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::state::*;
use crate::units::*;

// ------------------------------------------------------------------ helpers

#[inline]
fn other(p: u8) -> u8 {
    1 - p
}

#[inline]
fn occupied(s: &State, hex: usize) -> bool {
    s.hex_type[hex] != NONE
}

#[inline]
fn is_enemy_unit(s: &State, hex: usize, me: u8) -> bool {
    s.hex_type[hex] != NONE && s.hex_owner[hex] == other(me)
}

#[inline]
fn is_friendly_unit(s: &State, hex: usize, me: u8) -> bool {
    s.hex_type[hex] != NONE && s.hex_owner[hex] == me
}

/// Does `me` control location `hex`? (marker present and it is a location)
#[inline]
fn controls(s: &State, hex: usize, me: u8) -> bool {
    board().is_location[hex] && s.loc_marker[hex] == me
}

/// Knight immunity: an attack on `target_hex` is allowed only if the attacker
/// stack is bolstered (>=2), unless the target is not a Knight. Pikeman reflex
/// bypasses this (handled separately; not routed through here).
#[inline]
fn attack_allowed(s: &State, target_hex: usize, attacker_hex: usize) -> bool {
    attack_allowed_h(s, target_hex, s.hex_height[attacker_hex])
}

/// Knight immunity against an attacker of EFFECTIVE height `h` (the Berserker
/// chain pays its coin before the chained maneuver, so its chained attacks use
/// height-after-payment; 0 takes across the census for the h==2 chain attack
/// on a Knight confirm the cost-first reading).
#[inline]
fn attack_allowed_h(s: &State, target_hex: usize, h: u8) -> bool {
    let t = s.hex_type[target_hex];
    if t != NONE && def(t).knight {
        return h >= 2;
    }
    true
}

/// True if the straight two-space line from `from` to `target` passes through
/// an EMPTY intermediate hex (Crossbowman rule). `from`/`target` exactly 2 apart.
fn clear_two_line(s: &State, from: usize, target: usize) -> bool {
    let mid = board().between[from][target];
    mid != NONE && !occupied(s, mid as usize)
}

// ----------------------------------------------------------- zone primitives

impl State {
    #[inline]
    fn zmove(&mut self, p: u8, from: usize, to: usize, unit: u8) {
        self.zones[p as usize][from][unit as usize] -= 1;
        self.zones[p as usize][to][unit as usize] += 1;
    }

    /// Remove one coin from a unit stack on the board. If the unit's last coin
    /// is removed, the unit leaves the board. `eliminate` sends the coin to the
    /// eliminated zone; otherwise it goes face-up to discard (Berserker chain).
    fn remove_stack_coin(&mut self, hex: usize, eliminate: bool) {
        let p = self.hex_owner[hex];
        let unit = self.hex_type[hex];
        debug_assert!(unit != NONE);
        let z = if eliminate { Z_ELIM } else { Z_FACEUP };
        self.zones[p as usize][z][unit as usize] += 1;
        self.hex_height[hex] -= 1;
        if self.hex_height[hex] == 0 {
            self.hex_type[hex] = NONE;
            self.hex_owner[hex] = NONE;
        }
    }

    /// Move a whole stack from `from` to `to` (to must be empty).
    fn move_stack(&mut self, from: usize, to: usize) {
        debug_assert!(self.hex_type[to] == NONE);
        self.hex_type[to] = self.hex_type[from];
        self.hex_owner[to] = self.hex_owner[from];
        self.hex_height[to] = self.hex_height[from];
        self.hex_type[from] = NONE;
        self.hex_owner[from] = NONE;
        self.hex_height[from] = 0;
    }

    /// Place a control marker for `p` on location `hex`, returning any displaced
    /// enemy marker to that enemy's hand. Runs the win check.
    fn place_marker(&mut self, hex: usize, p: u8) {
        let prev = self.loc_marker[hex];
        if prev != NONE && prev != p {
            self.markers_hand[prev as usize] += 1;
        }
        if prev != p {
            debug_assert!(self.markers_hand[p as usize] > 0);
            self.markers_hand[p as usize] -= 1;
        }
        self.loc_marker[hex] = p;
        if self.markers_on_board(p) >= WIN_MARKERS {
            self.winner = p;
        }
    }
}

// ------------------------------------------------------------- attack engine

/// Outcome plumbing for an attack whose target coin-removal may require a
/// defender Royal Guard choice. We resolve the attacker-side effects (Pikeman
/// reflex) immediately, then either remove the target coin now or defer to a
/// defender decision node, and finally queue the attacker's post-triggers.
impl State {
    /// Queue a unit's optional post-maneuver triggers (Berserker chain after any
    /// maneuver; Swordsman move after an attack) if the unit still stands at
    /// `hex`. `was_attack`/`was_move` classify the maneuver just performed.
    /// Per the FAQ these fire for granted maneuvers too.
    fn queue_maneuver_post(&mut self, hex: usize, was_attack: bool, was_move: bool) {
        if self.hex_type[hex] == NONE {
            return; // unit died (e.g. to Pikeman reflex): no post-trigger.
        }
        let unit = self.hex_type[hex];
        let d = def(unit);
        // Berserker V1: after any maneuver. Berserker V2: only after attack/move.
        let bers = if d.berserker_v1 {
            true
        } else if d.berserker_v2 {
            was_attack || was_move
        } else {
            false
        };
        if bers {
            // The extra maneuver needs a bolstered coin to spend; if the stack
            // has only 1 coin the chain has no legal option but STOP, which the
            // decision node still offers. Only queue when >=2 coins remain.
            if self.hex_height[hex] >= 2 {
                self.push_cont(Cont::BerserkerChain {
                    hex: hex as u8,
                    v2: d.berserker_v2,
                });
            }
        }
        if d.swordsman && was_attack {
            self.push_cont(Cont::SwordsmanMove { hex: hex as u8 });
        }
    }

    /// Back-compat shim used by resolve_attack.
    fn queue_attack_post(&mut self, atk_hex: usize) {
        self.queue_maneuver_post(atk_hex, true, false);
    }

    /// Resolve an attack from `atk_hex` on `tgt_hex`. Applies the Pikeman reflex
    /// to the attacker, then handles the target coin removal (deferring to a
    /// defender RG choice if applicable), then queues attacker post-triggers.
    fn resolve_attack(&mut self, atk_hex: usize, tgt_hex: usize) {
        let b = board();
        let adjacent = b.dist[atk_hex][tgt_hex] == 1;
        let tgt_unit = self.hex_type[tgt_hex];
        let defender = self.hex_owner[tgt_hex];

        // 1. Pikeman reflex: adjacent attacker loses one coin (eliminated),
        //    simultaneous, ignores Knight immunity. Computed from pre-attack
        //    state so both single-coin units can mutually die.
        let pikeman_reflex = tgt_unit != NONE && def(tgt_unit).pikeman && adjacent;

        // 2. Target coin removal. Royal Guard defender may soak from supply.
        let rg_soak_available = tgt_unit != NONE
            && def(tgt_unit).royal_guard
            && self.zones[defender as usize][Z_SUPPLY][ROYAL_GUARD as usize] > 0;

        if pikeman_reflex {
            // remove one coin from the attacker (eliminated).
            self.remove_stack_coin(atk_hex, true);
        }

        if rg_soak_available {
            // Defender decision node. The attacker post-triggers run after the
            // choice resolves; stash a marker cont so we know the attacker hex.
            self.push_cont(Cont::_AttackPost {
                atk_hex: atk_hex as u8,
            });
            self.set_pending(Cont::RoyalGuardChoice {
                defender,
                rg_hex: tgt_hex as u8,
            });
            self.interrupt = true; // caller must not advance past this node.
            return;
        }

        // No defender choice: remove the target coin from its stack now.
        if tgt_unit != NONE {
            self.remove_stack_coin(tgt_hex, true);
        }
        self.queue_attack_post(atk_hex);
    }
}

// ---------------------------------------------------- continuation advancing

impl State {
    /// Finish a maneuver: advance to the next decision UNLESS the maneuver
    /// installed a mid-attack interrupt (defender Royal Guard choice), in which
    /// case that node is already pending and we must stop here.
    fn finish(&mut self) {
        if self.interrupt {
            self.interrupt = false;
            return;
        }
        self.advance();
    }

    /// After an action fully resolves, set up the next decision. Pops the LIFO
    /// stack, executes inline (non-decision) conts, and drives turn/round flow.
    fn advance(&mut self) {
        loop {
            if self.winner != NONE {
                return;
            }
            match self.conts.pop() {
                Some(Cont::_AttackPost { atk_hex }) => {
                    self.queue_attack_post(atk_hex as usize);
                    // queue_attack_post pushes further conts; loop to pick one.
                    continue;
                }
                Some(c) => {
                    // Mandatory nodes with no legal maneuver are auto-skipped
                    // (Footman "perform one maneuver ... if able"; Cavalry
                    // "move, then attack" only when a target exists). Optional
                    // nodes always carry a decline action, so never need this.
                    match &c {
                        Cont::FootmanManeuver { .. } => {
                            // Mandatory-if-able: if no remaining footman has a
                            // legal maneuver, the node is skipped entirely.
                            self.pending = c;
                            if self.legal_actions().is_empty() {
                                continue;
                            }
                            return;
                        }
                        Cont::CavalryAttack { .. } => {
                            self.pending = c;
                            if self.legal_actions().is_empty() {
                                continue;
                            }
                            return;
                        }
                        Cont::MainPlay => {
                            // Route through begin_main_turn to honor shortages.
                            self.begin_main_turn();
                            return;
                        }
                        _ => {
                            self.pending = c;
                            return;
                        }
                    }
                }
                None => {
                    // Stack exhausted: the active player's turn is over.
                    self.end_turn();
                    return;
                }
            }
        }
    }

    /// The active player's coin play (and all its triggers) has fully resolved.
    /// Advance to the next player's turn, or to the next round.
    fn end_turn(&mut self) {
        self.turns_taken[self.active as usize] += 1;
        self.wp_v2_triggered = false;
        // Alternate to the opponent (begin_main_turn corrects for shortages).
        self.active = other(self.active);
        self.begin_main_turn();
    }

    /// Set up the next MainPlay. Picks the player who still has coins; if
    /// neither does, the round is over and round-start draws are queued. Handles
    /// the shortage case ("if one player runs out, the other continues alone").
    pub fn begin_main_turn(&mut self) {
        if self.main_plays >= crate::state::MAX_MAIN_PLAYS {
            self.adjudicated_draw = true;
            return;
        }
        let a = self.active;
        let o = other(a);
        if self.hand_size(a) > 0 {
            self.pending = Cont::MainPlay;
        } else if self.hand_size(o) > 0 {
            self.active = o;
            self.pending = Cont::MainPlay;
        } else {
            // Round over. Initiative holder acts first next round.
            self.round += 1;
            self.first_player = self.initiative;
            self.active = self.initiative;
            self.start_round_draws();
        }
    }
}

// -------------------------------------------------- maneuver effect primitives

impl State {
    /// Warrior Priest post-trigger after it ATTACKS or CONTROLS: draw a coin and
    /// forcibly play it. V2 caps at one trigger per turn, and only V2's own
    /// trigger counts against that cap — a V1 trigger must not block V2.
    fn queue_wp_post(&mut self, hex: usize) {
        if !self.wp_trigger_ready(hex) {
            return;
        }
        if def(self.hex_type[hex]).warrior_priest_v2 {
            self.wp_v2_triggered = true;
        }
        let p = self.hex_owner[hex];
        self.push_cont(Cont::WarriorPriestDraw {
            player: p,
            rg_hex: NONE,
        });
    }

    /// Would the unit at `hex` trigger its Warrior Priest draw right now?
    fn wp_trigger_ready(&self, hex: usize) -> bool {
        if self.hex_type[hex] == NONE {
            return false; // unit died (e.g. reflex): no trigger.
        }
        let d = def(self.hex_type[hex]);
        if !d.warrior_priest {
            return false;
        }
        if d.warrior_priest_v2 && self.wp_v2_triggered {
            return false; // V2 once-per-turn cap.
        }
        true
    }

    /// Apply a MOVE effect (whole stack one/many steps already validated) then
    /// queue post-maneuver triggers (Berserker).
    fn do_move(&mut self, from: usize, to: usize) {
        self.move_stack(from, to);
        self.queue_maneuver_post(to, false, true);
    }

    /// Apply a CONTROL effect: place marker on the location the unit occupies,
    /// then queue Warrior Priest and Berserker post-triggers.
    fn do_control(&mut self, hex: usize) {
        let p = self.hex_owner[hex];
        self.place_marker(hex, p);
        // Warrior Priest (control) then Berserker (any maneuver). Order: WP
        // first so its forced play resolves before the chain (arbitrary but
        // fixed; the only unit with both would be a contradiction).
        self.queue_maneuver_post(hex, false, false);
        self.queue_wp_post(hex);
    }

    /// Apply an ATTACK effect via the attack engine, then Warrior Priest post.
    /// Berserker/Swordsman are queued inside resolve_attack. When multiple of
    /// the attacker's attributes trigger, resolution order among them is a free
    /// choice per the FAQ; we pick a fixed order (WP resolves after the queued
    /// Berserker/Swordsman conts, since it is pushed last onto the LIFO stack).
    fn do_attack(&mut self, from: usize, target: usize) {
        self.resolve_attack(from, target);
        if self.interrupt {
            // The attack was deferred to a defender Royal Guard soak choice.
            // If the attacker's Warrior Priest draw is due, the DRAW resolves
            // BEFORE the defender's choice (server-verified), while the forced
            // play still comes after it: swap the WP draw in as the pending
            // node; its apply re-installs the RoyalGuardChoice.
            if self.wp_trigger_ready(from) {
                if let Cont::RoyalGuardChoice { rg_hex, .. } = self.pending {
                    if def(self.hex_type[from]).warrior_priest_v2 {
                        self.wp_v2_triggered = true;
                    }
                    let p = self.hex_owner[from];
                    self.set_pending(Cont::WarriorPriestDraw { player: p, rg_hex });
                }
            }
        } else {
            self.queue_wp_post(from);
        }
    }
}

// -------------------------------------------------- shared move/attack targets

impl State {
    /// All empty hexes one step from `from` (single-step move targets).
    fn adj_empty(&self, from: usize, out: &mut Vec<u8>) {
        let b = board();
        for d in 0..6 {
            let n = b.neighbors[from][d];
            if n != NONE && !occupied(self, n as usize) {
                out.push(n);
            }
        }
    }

    /// Normal attack targets from `from`: adjacent enemy units the attacker may
    /// legally hit (Knight immunity respected). `eff_h` is the attacker's
    /// effective stack height (reduced by 1 inside a Berserker chain).
    fn normal_attack_targets(&self, from: usize, eff_h: u8, out: &mut Vec<u8>) {
        let b = board();
        let me = self.hex_owner[from];
        for d in 0..6 {
            let n = b.neighbors[from][d];
            if n != NONE
                && is_enemy_unit(self, n as usize, me)
                && attack_allowed_h(self, n as usize, eff_h)
            {
                out.push(n);
            }
        }
    }
}

// ---------------------------------------------------------- draw / bag helpers

impl State {
    /// Refill the bag from the whole discard pile (face-up + facedown) by moving
    /// those coins into the bag. Deterministic here (order is irrelevant: draws
    /// are modeled as chance nodes over the multiset).
    fn refill_bag(&mut self, p: u8) {
        for u in 0..N_UNITS {
            let up = self.zones[p as usize][Z_FACEUP][u];
            let dn = self.zones[p as usize][Z_FACEDOWN][u];
            self.zones[p as usize][Z_BAG][u] += up + dn;
            self.zones[p as usize][Z_FACEUP][u] = 0;
            self.zones[p as usize][Z_FACEDOWN][u] = 0;
        }
    }

    /// The distinct coin types currently drawable by `p`, with multiplicities,
    /// accounting for a refill if the bag is empty. Returns pairs (unit, count).
    /// If both bag and discards are empty, returns empty (no draw possible).
    fn drawable(&self, p: u8) -> Vec<(u8, u8)> {
        let mut bag_empty = true;
        for u in 0..N_UNITS {
            if self.zones[p as usize][Z_BAG][u] > 0 {
                bag_empty = false;
                break;
            }
        }
        let mut out = Vec::new();
        if !bag_empty {
            for u in 0..N_UNITS {
                let c = self.zones[p as usize][Z_BAG][u];
                if c > 0 {
                    out.push((u as u8, c));
                }
            }
        } else {
            // effective bag after refill = bag + faceup + facedown discards.
            for u in 0..N_UNITS {
                let c = self.zones[p as usize][Z_BAG][u]
                    + self.zones[p as usize][Z_FACEUP][u]
                    + self.zones[p as usize][Z_FACEDOWN][u];
                if c > 0 {
                    out.push((u as u8, c));
                }
            }
        }
        out
    }

    /// The first coin `drawable` would list, without building the list.
    ///
    /// A subgame's draw run applies one coin per step and only ever reads the
    /// first: which coin is drawn changes nothing public, so any legal
    /// `DrawCoin` produces the same child. Asking through `legal_actions` cost
    /// two heap allocations and two passes over the unit table per step, about
    /// a thousand times per solve.
    pub(crate) fn first_drawable(&self, p: u8) -> Option<u8> {
        let z = &self.zones[p as usize];
        for u in 0..N_UNITS {
            if z[Z_BAG][u] > 0 {
                return Some(u as u8);
            }
        }
        // Effective bag after the refill, exactly as `drawable` computes it.
        for u in 0..N_UNITS {
            if z[Z_BAG][u] + z[Z_FACEUP][u] + z[Z_FACEDOWN][u] > 0 {
                return Some(u as u8);
            }
        }
        None
    }

    /// Which private zone the current coin play pays from: the Warrior Priest's
    /// drawn coin while a forced play is owed, the hand otherwise.
    #[inline]
    fn pay_zone(&self) -> usize {
        match self.pending {
            Cont::WarriorPriestPlay { .. } => Z_INFLIGHT,
            _ => Z_HAND,
        }
    }

    /// Draw `unit` for `p`: refill first if the bag is empty, then move one
    /// coin from the bag into `to`.
    fn do_draw(&mut self, p: u8, unit: u8, to: usize) {
        let mut bag_empty = true;
        for u in 0..N_UNITS {
            if self.zones[p as usize][Z_BAG][u] > 0 {
                bag_empty = false;
                break;
            }
        }
        if bag_empty {
            self.refill_bag(p);
        }
        debug_assert!(self.zones[p as usize][Z_BAG][unit as usize] > 0);
        self.zmove(p, Z_BAG, to, unit);
    }
}

// --------------------------------------------------- maneuver action listing

impl State {
    /// List the basic maneuvers (Move/Control/Attack) available to the friendly
    /// unit at `hex`, plus its Tactic-driven attacks that are ALSO normal
    /// maneuvers. `variant` selects which action variants to emit so the same
    /// geometry serves main-play, footman, berserker, merc, and cavalry-follow.
    fn list_basic_maneuvers(&self, hex: usize, variant: ManVariant, out: &mut Vec<Action>) {
        let unit = self.hex_type[hex];
        if unit == NONE {
            return;
        }
        let me = self.hex_owner[hex];
        let d = def(unit);

        // MOVE (one step to empty). Excluded for the "attack-only" contexts.
        if variant.allows_move() {
            let mut tos = Vec::new();
            self.adj_empty(hex, &mut tos);
            for to in tos {
                out.push(variant.mk_move(hex as u8, to));
            }
        }
        // CONTROL (stand on a location you don't already control). Requires an
        // unplaced control marker in hand.
        if variant.allows_control()
            && self.markers_hand[me as usize] > 0
            && controls_target(self, hex, me)
        {
            out.push(variant.mk_control(hex as u8));
        }
        // ATTACK (normal). Archer/Lancer cannot normal-attack. In a Berserker
        // chain the cost coin is discarded BEFORE the chained maneuver, so the
        // Knight-immunity check uses the post-payment height.
        if variant.allows_attack() && !d.no_normal_attack {
            let eff_h = self.hex_height[hex] - variant.chain_cost();
            let mut tgts = Vec::new();
            self.normal_attack_targets(hex, eff_h, &mut tgts);
            for t in tgts {
                out.push(variant.mk_attack(hex as u8, t));
            }
        }
    }
}

/// True if the unit at `hex` may CONTROL: it stands on a location `me` does not
/// already control.
fn controls_target(s: &State, hex: usize, me: u8) -> bool {
    board().is_location[hex] && s.loc_marker[hex] != me
}

/// Which maneuver variants a context permits and how to encode them.
#[derive(Clone, Copy)]
enum ManVariant {
    Main,        // main-play Move/Control/Attack
    Footman,     // Footman-tactic maneuvers
    Berserker,   // Berserker chain (move/control/attack)
    BerserkerV2, // move/attack only
    Merc,        // Mercenary free maneuver
    CavalryAtk,  // attack only (cavalry follow-up)
}

impl ManVariant {
    /// Stack coins spent BEFORE the maneuver resolves (Berserker chain cost).
    fn chain_cost(self) -> u8 {
        match self {
            ManVariant::Berserker | ManVariant::BerserkerV2 => 1,
            _ => 0,
        }
    }
    fn allows_move(self) -> bool {
        matches!(
            self,
            ManVariant::Main
                | ManVariant::Footman
                | ManVariant::Berserker
                | ManVariant::BerserkerV2
                | ManVariant::Merc
        )
    }
    fn allows_control(self) -> bool {
        matches!(
            self,
            ManVariant::Main | ManVariant::Footman | ManVariant::Berserker | ManVariant::Merc
        )
    }
    fn allows_attack(self) -> bool {
        matches!(
            self,
            ManVariant::Main
                | ManVariant::Footman
                | ManVariant::Berserker
                | ManVariant::BerserkerV2
                | ManVariant::Merc
                | ManVariant::CavalryAtk
        )
    }
    fn mk_move(self, from: u8, to: u8) -> Action {
        match self {
            ManVariant::Footman => Action::FootMove { from, to },
            ManVariant::Berserker | ManVariant::BerserkerV2 => Action::BerserkMove { from, to },
            ManVariant::Merc => Action::MercMove { from, to },
            _ => Action::Move { from, to },
        }
    }
    fn mk_control(self, from: u8) -> Action {
        match self {
            ManVariant::Footman => Action::FootControl { from },
            ManVariant::Berserker => Action::BerserkControl { from },
            ManVariant::Merc => Action::MercControl { from },
            _ => Action::Control { from },
        }
    }
    fn mk_attack(self, from: u8, target: u8) -> Action {
        match self {
            ManVariant::Footman => Action::FootAttack { from, target },
            ManVariant::Berserker | ManVariant::BerserkerV2 => {
                Action::BerserkAttack { from, target }
            }
            ManVariant::Merc => Action::MercAttack { from, target },
            ManVariant::CavalryAtk => Action::TacCavalryAttack { from, target },
            _ => Action::Attack { from, target },
        }
    }
}

// ------------------------------------------------------ unit-location helpers

impl State {
    /// Hexes where player `p` has a deployed unit of type `unit`.
    fn hexes_of(&self, p: u8, unit: u8) -> Vec<usize> {
        let mut v = Vec::new();
        for h in 0..N_HEXES {
            if self.hex_owner[h] == p && self.hex_type[h] == unit {
                v.push(h);
            }
        }
        v
    }

    /// Does `p` have any deployed unit of type `unit`?
    fn has_deployed(&self, p: u8, unit: u8) -> bool {
        for h in 0..N_HEXES {
            if self.hex_owner[h] == p && self.hex_type[h] == unit {
                return true;
            }
        }
        false
    }

    /// Empty control locations `p` may deploy onto (controls + empty).
    fn deploy_locations(&self, p: u8, out: &mut Vec<u8>) {
        for h in 0..N_HEXES {
            if controls(self, h, p) && !occupied(self, h) {
                out.push(h as u8);
            }
        }
    }

    /// Empty hexes adjacent to any friendly unit (Scout deploy).
    fn scout_deploy_hexes(&self, p: u8, out: &mut Vec<u8>) {
        let b = board();
        for h in 0..N_HEXES {
            if occupied(self, h) {
                continue;
            }
            let mut ok = false;
            for d in 0..6 {
                let n = b.neighbors[h][d];
                if n != NONE && is_friendly_unit(self, n as usize, p) {
                    ok = true;
                    break;
                }
            }
            if ok {
                out.push(h as u8);
            }
        }
    }
}

// ------------------------------------------------------------- tactic listing

impl State {
    /// List all legal Tactic actions for the friendly unit at `hex`.
    /// Requires the caller to have already checked the unit type has a tactic.
    fn list_tactics(&self, hex: usize, out: &mut Vec<Action>) {
        let unit = self.hex_type[hex];
        let me = self.hex_owner[hex];
        let b = board();
        match def(unit).tactic {
            Tactic::None => {}
            Tactic::Footman => {
                // One tactic action per footman TYPE; it maneuvers every footman
                // unit of that type. `hex` is a deployed footman of `unit`, so
                // the tactic is available. dedup collapses duplicates.
                out.push(Action::TacFootman { coin: unit });
            }
            Tactic::Archer => {
                for t in 0..N_HEXES {
                    if b.dist[hex][t] == 2
                        && is_enemy_unit(self, t, me)
                        && attack_allowed(self, t, hex)
                    {
                        out.push(Action::TacArcher {
                            from: hex as u8,
                            target: t as u8,
                        });
                    }
                }
            }
            Tactic::Cavalry => {
                // Move one step to empty; the follow-up attack is a later node.
                //
                // The destination must have an attackable enemy. The FAQ
                // (rules PDF p.17, LANCER) requires a legally attackable target
                // to exist at the moment the tactic is chosen, and the Cavalry
                // card has the same move-then-attack shape. Without this the
                // tactic degenerates into a plain Move whenever no target
                // follows -- `advance` silently drops the queued CavalryAttack
                // -- producing an action byte-identical to `Move{from,to}` and
                // corrupting every uniform-over-actions distribution in the
                // search and self-play loop.
                //
                // Evaluated on the pre-move board, which is exact here: the
                // move only vacates `hex` (our own unit, never an enemy target)
                // and fills `to` (empty, likewise never a target).
                let h = self.hex_height[hex];
                let mut tos = Vec::new();
                self.adj_empty(hex, &mut tos);
                for to in tos {
                    let has_target = (0..6).any(|d| {
                        let n = b.neighbors[to as usize][d];
                        n != NONE
                            && is_enemy_unit(self, n as usize, me)
                            && attack_allowed_h(self, n as usize, h)
                    });
                    if has_target {
                        out.push(Action::TacCavalryMove {
                            from: hex as u8,
                            to,
                        });
                    }
                }
            }
            Tactic::Crossbowman => {
                for t in 0..N_HEXES {
                    if b.dist[hex][t] == 2
                        && is_enemy_unit(self, t, me)
                        && clear_two_line(self, hex, t)
                        && attack_allowed(self, t, hex)
                    {
                        out.push(Action::TacCrossbow {
                            from: hex as u8,
                            target: t as u8,
                        });
                    }
                }
            }
            Tactic::Ensign => {
                // friendly unit within 2 of the ensign makes a one-step move
                // ending within 2 of the ensign. NOT the ensign itself: 0 of
                // 8,999 "(Ensign)" moves in the census move the Ensign, and a
                // self-move would be a pure transposition of the plain Move.
                for u in 0..N_HEXES {
                    if u == hex || !is_friendly_unit(self, u, me) || b.dist[hex][u] > 2 {
                        continue;
                    }
                    for d in 0..6 {
                        let to = b.neighbors[u][d];
                        if to == NONE {
                            continue;
                        }
                        let to = to as usize;
                        // moving the ensign itself: distance measured from the
                        // ensign's ORIGINAL hex per card ("within 2 of the
                        // Ensign"); the ensign moves, so use its start hex.
                        if !occupied(self, to) && b.dist[hex][to] <= 2 {
                            out.push(Action::TacEnsign {
                                from: u as u8,
                                to: to as u8,
                            });
                        }
                    }
                }
            }
            Tactic::Lancer => {
                let bolstered = self.hex_height[hex] >= 2;
                for d in 0..6 {
                    // step 1
                    let s1 = b.step[hex][d];
                    if s1 == NONE || occupied(self, s1 as usize) {
                        continue;
                    }
                    // option A: move 1, attack the hex beyond s1.
                    let beyond1 = b.step[s1 as usize][d];
                    if beyond1 != NONE && is_enemy_unit(self, beyond1 as usize, me) {
                        let knight = def(self.hex_type[beyond1 as usize]).knight;
                        if !knight || bolstered {
                            out.push(Action::TacLancer {
                                from: hex as u8,
                                to: s1,
                                target: beyond1,
                            });
                        }
                    }
                    // option B: move 2 (s2 empty), attack the hex beyond s2.
                    let s2 = b.step[s1 as usize][d];
                    if s2 != NONE && !occupied(self, s2 as usize) {
                        let beyond2 = b.step[s2 as usize][d];
                        if beyond2 != NONE && is_enemy_unit(self, beyond2 as usize, me) {
                            let knight = def(self.hex_type[beyond2 as usize]).knight;
                            if !knight || bolstered {
                                out.push(Action::TacLancer {
                                    from: hex as u8,
                                    to: s2,
                                    target: beyond2,
                                });
                            }
                        }
                    }
                }
            }
            Tactic::LightCavalry => {
                // exactly 2 steps, each into an empty hex; may turn.
                for d1 in 0..6 {
                    let m = b.neighbors[hex][d1];
                    if m == NONE || occupied(self, m as usize) {
                        continue;
                    }
                    for d2 in 0..6 {
                        let to = b.neighbors[m as usize][d2];
                        if to == NONE || to as usize == hex || occupied(self, to as usize) {
                            continue;
                        }
                        // Two axial steps with a 120-degree turn land adjacent
                        // to the start, where the plain Move produces an
                        // identical successor state. Legal, but redundant, and
                        // duplicate actions skew every uniform distribution
                        // over the action set.
                        if b.dist[hex][to as usize] != 2 {
                            continue;
                        }
                        out.push(Action::TacLightCav {
                            from: hex as u8,
                            to,
                        });
                    }
                }
            }
            Tactic::Marshal => {
                // NOT the Marshal itself: 0 of 3,490 "(Marshal)" attacks in
                // the census are by the Marshal, and a self-grant would be a
                // pure transposition of the plain Attack.
                for u in 0..N_HEXES {
                    if u == hex || !is_friendly_unit(self, u, me) || b.dist[hex][u] > 2 {
                        continue;
                    }
                    if def(self.hex_type[u]).no_normal_attack {
                        continue; // never grants Archer/Lancer tactic-attacks.
                    }
                    let mut tgts = Vec::new();
                    self.normal_attack_targets(u, self.hex_height[u], &mut tgts);
                    for t in tgts {
                        out.push(Action::TacMarshal {
                            unit_hex: u as u8,
                            target: t,
                        });
                    }
                }
            }
            Tactic::RoyalGuard => {
                // needs a Royal Coin to pay with; move up to 2 through empty
                // hexes, ending on an empty location you control. At a Warrior
                // Priest forced play that is the drawn coin, not the hand —
                // which is why a drawn RG coin does not offer this tactic.
                if self.zones[me as usize][self.pay_zone()][ROYAL_COIN as usize] == 0 {
                    return;
                }
                // reachable in 1 or 2 steps through empty hexes.
                for d1 in 0..6 {
                    let m = b.neighbors[hex][d1];
                    if m == NONE || occupied(self, m as usize) {
                        continue;
                    }
                    // 1 step:
                    if controls(self, m as usize, me) {
                        out.push(Action::TacRoyalGuard {
                            from: hex as u8,
                            to: m,
                        });
                    }
                    // 2 steps:
                    for d2 in 0..6 {
                        let to = b.neighbors[m as usize][d2];
                        if to == NONE || to as usize == hex || occupied(self, to as usize) {
                            continue;
                        }
                        if controls(self, to as usize, me) {
                            out.push(Action::TacRoyalGuard {
                                from: hex as u8,
                                to,
                            });
                        }
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------- main-play legal set

impl State {
    /// All coin plays available to `p` from the paying zone. On a normal turn
    /// that is the hand; at a Warrior Priest forced play it is the single drawn
    /// coin, which is the whole of that rule.
    fn list_main_play(&self, p: u8, out: &mut Vec<Action>) {
        // For each distinct coin type available, list its legal plays. Facedown
        // plays (claim/recruit/pass) spend that specific coin.
        let claim_ok = self.initiative != p && !self.initiative_moved;
        let pay = self.pay_zone();
        for u in 0..N_UNITS {
            if self.zones[p as usize][pay][u] == 0 {
                continue;
            }
            let unit = u as u8;
            let d = def(unit);

            // ---- facedown plays (any coin, including Royal Coin) ----
            if claim_ok {
                out.push(Action::ClaimInitiative { coin: unit });
            }
            out.push(Action::Pass { coin: unit });
            for r in 0..N_UNITS {
                if self.zones[p as usize][Z_SUPPLY][r] > 0 {
                    out.push(Action::Recruit {
                        coin: unit,
                        unit: r as u8,
                    });
                }
            }

            if d.is_royal_coin {
                // Royal Coin: facedown actions, plus it is the coin that PAYS
                // for the Royal Guard's tactic — the tactic must be offered
                // whenever the Royal Coin is in hand and a Royal Guard unit is
                // deployed, even with no RG coin in hand (verified vs replays).
                for h in self.hexes_of(p, ROYAL_GUARD) {
                    self.list_tactics(h, out);
                }
                continue;
            }

            // ---- deploy ----
            if !self.has_deployed(p, unit) || d.two_footmen {
                // capacity: normal 1, footman 2.
                let cap = if d.two_footmen { 2 } else { 1 };
                if (self.hexes_of(p, unit).len() as u8) < cap {
                    let mut locs = Vec::new();
                    self.deploy_locations(p, &mut locs);
                    for h in locs {
                        out.push(Action::Deploy { unit, hex: h });
                    }
                    if d.scout {
                        let mut sc = Vec::new();
                        self.scout_deploy_hexes(p, &mut sc);
                        for h in sc {
                            out.push(Action::Deploy { unit, hex: h });
                        }
                    }
                }
            }
            // ---- bolster (onto a matching deployed unit) ----
            for h in self.hexes_of(p, unit) {
                out.push(Action::Bolster { unit, hex: h as u8 });
            }
            // ---- face-up maneuvers + tactic, per deployed unit of this type ----
            for h in self.hexes_of(p, unit) {
                self.list_basic_maneuvers(h, ManVariant::Main, out);
                self.list_tactics(h, out);
            }
        }
        dedup(out);
    }
}

/// De-duplicate an action list (some generators can emit the same action twice
/// via different geometric paths, e.g. Light Cavalry / Royal Guard).
fn dedup(out: &mut Vec<Action>) {
    let mut seen = std::collections::HashSet::new();
    out.retain(|a| seen.insert(a.encode()));
}

// ================================================================ public API

impl State {
    /// All legal actions for whoever is to act (including chance draws).
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut out = Vec::new();
        if self.winner != NONE {
            return out;
        }
        match self.pending {
            Cont::Draw { player } => {
                for (u, mult) in self.drawable(player) {
                    // Multiplicity: emit once per distinct type; the count is the
                    // draw weight (documented via legal_action_weights).
                    let _ = mult;
                    out.push(Action::DrawCoin { unit: u });
                }
            }
            Cont::WarriorPriestDraw { player, .. } => {
                for (u, _mult) in self.drawable(player) {
                    out.push(Action::DrawCoin { unit: u });
                }
                if out.is_empty() {
                    // No coin to draw (bag+discards empty): WP trigger fizzles.
                    // Represented as a single no-op draw of NONE.
                    out.push(Action::DrawCoin { unit: NONE });
                }
            }
            Cont::MainPlay => {
                self.list_main_play(self.active, &mut out);
            }
            Cont::WarriorPriestPlay { player } => {
                self.list_main_play(player, &mut out);
            }
            Cont::RoyalGuardChoice { .. } => {
                out.push(Action::RGSoakSupply);
                out.push(Action::RGSoakStack);
            }
            Cont::SwordsmanMove { hex } => {
                let mut tos = Vec::new();
                self.adj_empty(hex as usize, &mut tos);
                for to in tos {
                    out.push(Action::SwordsmanMove { from: hex, to });
                }
                out.push(Action::SwordsmanDecline);
            }
            Cont::BerserkerChain { hex, v2 } => {
                let variant = if v2 {
                    ManVariant::BerserkerV2
                } else {
                    ManVariant::Berserker
                };
                // The chain costs a bolstered coin; it is only reachable with
                // stack >= 2, so all listed maneuvers are payable.
                self.list_basic_maneuvers(hex as usize, variant, &mut out);
                self.list_berserker_tactics(hex as usize, v2, &mut out);
                out.push(Action::BerserkStop);
            }
            Cont::MercenaryManeuver { hex } => {
                self.list_basic_maneuvers(hex as usize, ManVariant::Merc, &mut out);
                self.list_merc_tactics(hex as usize, &mut out);
                out.push(Action::MercDecline);
            }
            Cont::FootmanManeuver { hexes } => {
                // The player chooses which remaining footman maneuvers next
                // (order is free; verified against replays).
                for h in hexes.iter() {
                    self.list_basic_maneuvers(h as usize, ManVariant::Footman, &mut out);
                }
                // Mandatory if any option exists; advance() skips it otherwise.
            }
            Cont::CavalryAttack { hex } => {
                self.list_basic_maneuvers(hex as usize, ManVariant::CavalryAtk, &mut out);
                // Mandatory attack if a target exists; else skipped by advance().
            }
            Cont::FootmanInstantDeploy { coin } => {
                // deploy the just-recruited Footman coin (in hand now). Normal
                // deploy legality; the two-footman cap still applies.
                let p = self.active;
                if (self.hexes_of(p, coin).len() as u8) < 2 {
                    let mut locs = Vec::new();
                    self.deploy_locations(p, &mut locs);
                    for h in locs {
                        out.push(Action::FootmanInstantDeploy { hex: h });
                    }
                }
                out.push(Action::FootmanInstantDecline);
            }
            Cont::_AttackPost { .. } => {
                // Never a decision node; advance() consumes it. Defensive: empty.
            }
        }
        dedup(&mut out);
        out
    }

    // Tactic listings restricted to the chaining/free contexts. Berserker/Merc
    // extra maneuvers may themselves invoke the unit's tactic (a tactic is a
    // maneuver). We reuse list_tactics but re-encode via the context variant is
    // not needed: a chained/free tactic resolves identically to a normal tactic
    // and we simply reuse the ordinary tactic actions.
    fn list_berserker_tactics(&self, hex: usize, _v2: bool, out: &mut Vec<Action>) {
        // AMBIGUITY: whether a Berserker "extra maneuver" may itself be the
        // unit's Tactic. The Berserker has no Tactic of its own, and no unit is
        // both a Berserker and a tactic-bearer, so this never applies. Left
        // empty intentionally.
        let _ = (hex, out);
    }
    fn list_merc_tactics(&self, hex: usize, out: &mut Vec<Action>) {
        // Mercenary has no Tactic; a free maneuver is move/control/attack only.
        let _ = (hex, out);
    }
}

// ------------------------------------------------------------------ apply

impl State {
    /// Apply an action, returning the resulting state (immutable-style step).
    pub fn apply(&self, action: Action) -> State {
        let mut s = self.clone();
        s.apply_inplace(action);
        s
    }

    /// In-place apply. Assumes `action` is legal for the current pending node.
    pub fn apply_inplace(&mut self, action: Action) {
        use Action::*;
        match self.pending {
            // -------- chance: round-start draw --------
            Cont::Draw { player } => {
                if let DrawCoin { unit } = action {
                    self.do_draw(player, unit, Z_HAND);
                }
                self.finish();
                return;
            }
            // -------- chance: Warrior Priest draw --------
            Cont::WarriorPriestDraw { player, rg_hex } => {
                if let DrawCoin { unit } = action {
                    let rg_choice = |s: &State| -> Option<Cont> {
                        if rg_hex == NONE {
                            None
                        } else {
                            Some(Cont::RoyalGuardChoice {
                                defender: s.hex_owner[rg_hex as usize],
                                rg_hex,
                            })
                        }
                    };
                    if unit == NONE {
                        // fizzle: nothing to draw; skip the forced play too.
                        match rg_choice(self) {
                            Some(c) => self.set_pending(c),
                            None => self.finish(),
                        }
                    } else {
                        // The drawn coin waits in `Z_INFLIGHT`, which is what
                        // the forced play pays from.
                        self.do_draw(player, unit, Z_INFLIGHT);
                        match rg_choice(self) {
                            Some(c) => {
                                // interrupted-attack case: the defender's soak
                                // choice resolves next; the forced play after.
                                self.push_cont(Cont::WarriorPriestPlay { player });
                                self.set_pending(c);
                            }
                            None => self.set_pending(Cont::WarriorPriestPlay { player }),
                        }
                    }
                }
                return;
            }
            _ => {}
        }

        // For the remaining pending kinds, dispatch on the action itself.
        let actor = self.to_act();
        match self.pending {
            Cont::MainPlay => {
                self.main_plays += 1;
                self.apply_main(self.active, action);
            }
            Cont::WarriorPriestPlay { player } => {
                // Forced play resolves like a main play (it can itself trigger a
                // fresh WP draw via a Warrior-Priest attack/control).
                self.apply_main(player, action);
                return; // apply_main advances.
            }
            Cont::RoyalGuardChoice { defender, rg_hex } => {
                match action {
                    RGSoakSupply => {
                        // remove one Royal Guard coin from the defender's supply.
                        self.zones[defender as usize][Z_SUPPLY][ROYAL_GUARD as usize] -= 1;
                        self.zones[defender as usize][Z_ELIM][ROYAL_GUARD as usize] += 1;
                    }
                    RGSoakStack => {
                        self.remove_stack_coin(rg_hex as usize, true);
                    }
                    _ => {}
                }
                self.finish();
                return;
            }
            Cont::SwordsmanMove { hex } => {
                match action {
                    SwordsmanMove { from, to } => {
                        debug_assert_eq!(from, hex);
                        self.do_move(from as usize, to as usize);
                    }
                    SwordsmanDecline => {}
                    _ => {}
                }
                self.finish();
                return;
            }
            Cont::BerserkerChain { hex, .. } => {
                self.apply_berserker(hex, action);
                return;
            }
            Cont::MercenaryManeuver { hex } => {
                match action {
                    MercMove { from, to } => self.do_move(from as usize, to as usize),
                    MercControl { from } => self.do_control(from as usize),
                    MercAttack { from, target } => self.do_attack(from as usize, target as usize),
                    MercDecline => {}
                    _ => {}
                }
                let _ = hex;
                self.finish();
                return;
            }
            Cont::FootmanManeuver { hexes } => {
                self.apply_footman(hexes, action);
                return;
            }
            Cont::CavalryAttack { hex } => {
                if let TacCavalryAttack { from, target } = action {
                    debug_assert_eq!(from, hex);
                    self.do_attack(from as usize, target as usize);
                }
                self.finish();
                return;
            }
            Cont::FootmanInstantDeploy { coin } => {
                match action {
                    FootmanInstantDeploy { hex } => {
                        // deploy the just-recruited coin out of the discard.
                        self.zones[self.active as usize][Z_FACEUP][coin as usize] -= 1;
                        self.place_unit(self.active, coin, hex as usize);
                    }
                    FootmanInstantDecline => {}
                    _ => {}
                }
                self.finish();
                return;
            }
            _ => {}
        }
        let _ = actor;
    }
}

// ------------------------------------------------------- apply sub-dispatchers

impl State {
    /// Put a fresh unit coin from hand onto the board (deploy). Sets stack=1.
    fn place_unit(&mut self, p: u8, unit: u8, hex: usize) {
        debug_assert!(self.hex_type[hex] == NONE);
        self.hex_type[hex] = unit;
        self.hex_owner[hex] = p;
        self.hex_height[hex] = 1;
    }

    /// A normal (main-play or Warrior-Priest forced) coin play by `p`.
    fn apply_main(&mut self, p: u8, action: Action) {
        use Action::*;
        // Where the played coin comes from: the hand, or the Warrior Priest's
        // drawn coin at a forced play. Read before any effect changes `pending`.
        let pay = self.pay_zone();
        match action {
            Deploy { unit, hex } => {
                self.zones[p as usize][pay][unit as usize] -= 1;
                self.place_unit(p, unit, hex as usize);
                self.finish();
            }
            Bolster { unit, hex } => {
                self.zones[p as usize][pay][unit as usize] -= 1;
                self.hex_height[hex as usize] += 1;
                self.finish();
            }
            ClaimInitiative { coin } => {
                self.zmove(p, pay, Z_FACEDOWN, coin);
                self.initiative = p;
                self.initiative_moved = true;
                self.finish();
            }
            Pass { coin } => {
                self.zmove(p, pay, Z_FACEDOWN, coin);
                self.finish();
            }
            Recruit { coin, unit } => {
                self.zmove(p, pay, Z_FACEDOWN, coin); // spent coin: facedown
                                                         // recruited coin: supply -> faceup discard (public).
                self.zones[p as usize][Z_SUPPLY][unit as usize] -= 1;
                self.zones[p as usize][Z_FACEUP][unit as usize] += 1;
                // Attribute triggers on recruit:
                let d = def(unit);
                if d.mercenary && self.has_deployed(p, unit) {
                    // free maneuver with the (first) deployed Mercenary.
                    let hexes = self.hexes_of(p, unit);
                    let h = hexes[0];
                    self.push_cont(Cont::MercenaryManeuver { hex: h as u8 });
                }
                if d.footman_v2 && self.has_deployed(p, unit) {
                    // may immediately deploy the recruited coin. The coin stays
                    // in the (face-up) discard until the deploy actually happens
                    // (verified from server snapshots).
                    self.push_cont(Cont::FootmanInstantDeploy { coin: unit });
                }
                self.finish();
            }
            Move { from, to } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            Control { from } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_control(from as usize);
                self.finish();
            }
            Attack { from, target } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_attack(from as usize, target as usize);
                self.finish();
            }
            // ----- tactics -----
            TacArcher { from, target } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_attack(from as usize, target as usize);
                self.finish();
            }
            TacCrossbow { from, target } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_attack(from as usize, target as usize);
                self.finish();
            }
            TacCavalryMove { from, to } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_move(from as usize, to as usize);
                // then a mandatory-if-able attack from the new hex.
                self.push_cont(Cont::CavalryAttack { hex: to });
                self.finish();
            }
            TacEnsign { from, to } => {
                // spend the Ensign coin; the mover is the unit at `from`.
                // find the ensign hex to spend its coin: the ensign is the unit
                // whose tactic this is. We must spend an Ensign coin from the
                // ensign UNIT, but the action only carries the moved unit. The
                // ensign is the deployed Ensign of p; spend from its stack via
                // the face-up discard of an Ensign coin (a maneuver spends the
                // played coin, which is the Ensign's own coin from hand).
                // The played coin is the Ensign hand coin: discard it face-up.
                self.zmove(p, pay, Z_FACEUP, ENSIGN);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            TacLancer { from, to, target } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.move_stack(from as usize, to as usize);
                // Lancer attribute: none. attack from the new hex.
                self.do_attack(to as usize, target as usize);
                self.finish();
            }
            TacLightCav { from, to } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            TacMarshal { unit_hex, target } => {
                self.zmove(p, pay, Z_FACEUP, MARSHAL);
                self.do_attack(unit_hex as usize, target as usize);
                self.finish();
            }
            TacRoyalGuard { from, to } => {
                // discard the Royal Coin (not an RG coin) as the played coin.
                // The tactic discard is FACE-UP (verified from server replays;
                // only facedown *actions* hide the Royal Coin).
                self.zmove(p, pay, Z_FACEUP, ROYAL_COIN);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            TacFootman { coin } => {
                // spend the played footman coin (face-up), then maneuver each
                // friendly Footman unit on the board, player-chosen order.
                // BOTH versions count: a Footman coin maneuvers Footman V2
                // units and vice versa (verified from server replays).
                self.zmove(p, pay, Z_FACEUP, coin);
                let mut hexes = crate::state::HexSet::default();
                for u in [FOOTMAN, FOOTMAN_V2] {
                    for &h in self.hexes_of(p, u).iter() {
                        hexes.insert(h as u8);
                    }
                }
                if !hexes.is_empty() {
                    self.push_cont(Cont::FootmanManeuver { hexes });
                }
                self.finish();
            }
            _ => {
                // Any action not valid here is ignored (caller guarantees
                // legality); advance to avoid a stuck state.
                self.finish();
            }
        }
    }

    /// Spend the played coin of a maneuver: one coin of the unit-type at `hex`
    /// goes from hand to the FACE-UP discard.
    fn spend_maneuver_coin(&mut self, p: u8, hex: usize, pay: usize) {
        let unit = self.hex_type[hex];
        debug_assert!(unit != NONE);
        self.zmove(p, pay, Z_FACEUP, unit);
    }
}

// ------------------------------------------------- berserker / footman apply

impl State {
    /// Resolve a Berserker chain decision at `hex`.
    fn apply_berserker(&mut self, hex: u8, action: Action) {
        use Action::*;
        let h = hex as usize;
        match action {
            BerserkStop => {
                self.finish();
                return;
            }
            _ => {}
        }
        // Pay the chain cost: discard one coin from the stack FACE-UP (recycles,
        // not eliminated). The stack must keep >= 1 coin (guaranteed: only
        // offered when height >= 2). AMBIGUITY: the cost coin is discarded
        // BEFORE the chained maneuver, so a chained attack uses the reduced
        // stack height for Knight-immunity purposes. (Most conservative reading
        // of "maneuver again BY discarding a bolstered coin".)
        let unit = self.hex_type[h];
        debug_assert!(self.hex_height[h] >= 2);
        self.zones[self.hex_owner[h] as usize][Z_FACEUP][unit as usize] += 1;
        self.hex_height[h] -= 1;

        match action {
            BerserkMove { from, to } => {
                debug_assert_eq!(from, hex);
                self.do_move(from as usize, to as usize);
            }
            BerserkControl { from } => {
                self.do_control(from as usize);
            }
            BerserkAttack { from, target } => {
                self.do_attack(from as usize, target as usize);
            }
            _ => {}
        }
        self.finish();
    }

    /// Resolve one Footman-tactic maneuver (the acting footman is the action's
    /// `from` hex); the other remaining footmen stay owed a maneuver.
    fn apply_footman(&mut self, hexes: crate::state::HexSet, action: Action) {
        use Action::*;
        let from = match action {
            FootMove { from, .. } | FootControl { from } | FootAttack { from, .. } => from,
            _ => NONE,
        };
        match action {
            FootMove { from, to } => self.do_move(from as usize, to as usize),
            FootControl { from } => self.do_control(from as usize),
            FootAttack { from, target } => self.do_attack(from as usize, target as usize),
            _ => {}
        }
        // Queue the remaining footmen as a fresh decision node (re-checked for
        // legality by advance()); post-triggers from this maneuver resolve first.
        let mut rest = hexes;
        rest.remove(from);
        if !rest.is_empty() {
            self.push_cont(Cont::FootmanManeuver { hexes: rest });
        }
        self.finish();
    }
}

// ---- debug helper: describe the pending decision (for diagnostics) ----
impl State {
    pub fn pending_debug(&self) -> String {
        format!(
            "{:?} active={} round={} hands=[{},{}]",
            self.pending,
            self.active,
            self.round,
            self.hand_size(0),
            self.hand_size(1)
        )
    }
}
