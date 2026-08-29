use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES};
use crate::state::*;
use crate::units::*;

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

#[inline]
fn controls(s: &State, hex: usize, me: u8) -> bool {
    board().is_location[hex] && s.loc_marker[hex] == me
}

#[inline]
fn attack_allowed_h(s: &State, target_hex: usize, h: u8) -> bool {
    let t = s.hex_type[target_hex];
    if t != NONE && def(t).knight {
        return h >= 2;
    }
    true
}

fn clear_two_line(s: &State, from: usize, target: usize) -> bool {
    let mid = board().between[from][target];
    mid != NONE && !occupied(s, mid as usize)
}

impl State {
    #[inline]
    fn zmove(&mut self, p: u8, from: usize, to: usize, unit: u8) {
        self.zones[p as usize][from][unit as usize] -= 1;
        self.zones[p as usize][to][unit as usize] += 1;
    }

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

    fn move_stack(&mut self, from: usize, to: usize) {
        debug_assert!(self.hex_type[to] == NONE);
        self.hex_type[to] = self.hex_type[from];
        self.hex_owner[to] = self.hex_owner[from];
        self.hex_height[to] = self.hex_height[from];
        self.hex_type[from] = NONE;
        self.hex_owner[from] = NONE;
        self.hex_height[from] = 0;
    }

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
    fn queue_maneuver_post(&mut self, hex: usize, was_attack: bool, was_move: bool) {
        if self.hex_type[hex] == NONE {
            return;
        }
        let unit = self.hex_type[hex];
        let d = def(unit);
        let bers = if d.berserker_v1 {
            true
        } else if d.berserker_v2 {
            was_attack || was_move
        } else {
            false
        };
        if bers {
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

    fn resolve_attack(&mut self, atk_hex: usize, tgt_hex: usize) {
        let b = board();
        let adjacent = b.dist[atk_hex][tgt_hex] == 1;
        let tgt_unit = self.hex_type[tgt_hex];
        let defender = self.hex_owner[tgt_hex];

        let pikeman_reflex = tgt_unit != NONE && def(tgt_unit).pikeman && adjacent;

        let rg_soak_available = tgt_unit != NONE
            && def(tgt_unit).royal_guard
            && self.zones[defender as usize][Z_SUPPLY][ROYAL_GUARD as usize] > 0;

        if pikeman_reflex {
            self.remove_stack_coin(atk_hex, true);
        }

        if rg_soak_available {
            self.push_cont(Cont::_AttackPost { atk_hex: atk_hex as u8 });
            self.set_pending(Cont::RoyalGuardChoice {
                defender,
                rg_hex: tgt_hex as u8,
            });
            self.interrupt = true;
            return;
        }

        if tgt_unit != NONE {
            self.remove_stack_coin(tgt_hex, true);
        }
        self.queue_maneuver_post(atk_hex, true, false);
    }
    fn finish(&mut self) {
        if self.interrupt {
            self.interrupt = false;
            return;
        }
        self.advance();
    }

    fn advance(&mut self) {
        loop {
            if self.winner != NONE {
                return;
            }
            match self.conts.pop() {
                Some(Cont::_AttackPost { atk_hex }) => {
                    self.queue_maneuver_post(atk_hex as usize, true, false);
                    continue;
                }
                Some(c) => {
                    self.pending = c;
                    if matches!(c, Cont::MainPlay) {
                        self.begin_main_turn();
                    } else if matches!(c, Cont::FootmanManeuver { .. } | Cont::CavalryAttack { .. })
                        && self.legal_actions().is_empty()
                    {
                        continue;
                    }
                    return;
                }
                None => {
                    self.end_turn();
                    return;
                }
            }
        }
    }

    fn end_turn(&mut self) {
        self.turns_taken[self.active as usize] += 1;
        self.wp_v2_triggered = false;
        self.active = other(self.active);
        self.begin_main_turn();
    }

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
            self.round += 1;
            self.first_player = self.initiative;
            self.active = self.initiative;
            self.start_round_draws();
        }
    }
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

    fn wp_trigger_ready(&self, hex: usize) -> bool {
        if self.hex_type[hex] == NONE {
            return false;
        }
        let d = def(self.hex_type[hex]);
        if !d.warrior_priest {
            return false;
        }
        if d.warrior_priest_v2 && self.wp_v2_triggered {
            return false;
        }
        true
    }

    fn do_move(&mut self, from: usize, to: usize) {
        self.move_stack(from, to);
        self.queue_maneuver_post(to, false, true);
    }

    fn do_control(&mut self, hex: usize) {
        let p = self.hex_owner[hex];
        self.place_marker(hex, p);
        self.queue_maneuver_post(hex, false, false);
        self.queue_wp_post(hex);
    }

    fn do_attack(&mut self, from: usize, target: usize) {
        self.resolve_attack(from, target);
        if self.interrupt {
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
    fn adj_empty(&self, from: usize, out: &mut Vec<u8>) {
        let b = board();
        for d in 0..6 {
            let n = b.neighbors[from][d];
            if n != NONE && !occupied(self, n as usize) {
                out.push(n);
            }
        }
    }

    fn normal_attack_targets(&self, from: usize, eff_h: u8, out: &mut Vec<u8>) {
        let b = board();
        let me = self.hex_owner[from];
        for d in 0..6 {
            let n = b.neighbors[from][d];
            if n != NONE && is_enemy_unit(self, n as usize, me) && attack_allowed_h(self, n as usize, eff_h) {
                out.push(n);
            }
        }
    }
    fn refill_bag(&mut self, p: u8) {
        for u in 0..N_UNITS {
            let up = self.zones[p as usize][Z_FACEUP][u];
            let dn = self.zones[p as usize][Z_FACEDOWN][u];
            self.zones[p as usize][Z_BAG][u] += up + dn;
            self.zones[p as usize][Z_FACEUP][u] = 0;
            self.zones[p as usize][Z_FACEDOWN][u] = 0;
        }
    }

    pub(crate) fn draw_pool(&self, p: u8) -> [u8; N_UNITS] {
        let z = &self.zones[p as usize];
        if z[Z_BAG].iter().any(|&c| c > 0) {
            return z[Z_BAG];
        }
        std::array::from_fn(|u| z[Z_BAG][u] + z[Z_FACEUP][u] + z[Z_FACEDOWN][u])
    }

    pub(crate) fn first_drawable(&self, p: u8) -> Option<u8> {
        self.draw_pool(p).iter().position(|&c| c > 0).map(|u| u as u8)
    }

    #[inline]
    fn pay_zone(&self) -> usize {
        match self.pending {
            Cont::WarriorPriestPlay { .. } => Z_INFLIGHT,
            _ => Z_HAND,
        }
    }

    fn do_draw(&mut self, p: u8, unit: u8, to: usize) {
        if self.zones[p as usize][Z_BAG].iter().all(|&c| c == 0) {
            self.refill_bag(p);
        }
        debug_assert!(self.zones[p as usize][Z_BAG][unit as usize] > 0);
        self.zmove(p, Z_BAG, to, unit);
    }
    fn list_basic_maneuvers(&self, hex: usize, variant: ManVariant, out: &mut Vec<Action>) {
        let unit = self.hex_type[hex];
        if unit == NONE {
            return;
        }
        let me = self.hex_owner[hex];
        let d = def(unit);

        if let Some(mk) = variant.step {
            let mut tos = Vec::new();
            self.adj_empty(hex, &mut tos);
            out.extend(tos.into_iter().map(|to| mk(hex as u8, to)));
        }
        if let Some(mk) = variant.control {
            if self.markers_hand[me as usize] > 0 && controls_target(self, hex, me) {
                out.push(mk(hex as u8));
            }
        }
        if let Some(mk) = variant.attack {
            if !d.no_normal_attack {
                let eff_h = self.hex_height[hex] - variant.chain_cost;
                let mut tgts = Vec::new();
                self.normal_attack_targets(hex, eff_h, &mut tgts);
                out.extend(tgts.into_iter().map(|t| mk(hex as u8, t)));
            }
        }
    }
    fn hexes_of(&self, p: u8, unit: u8) -> Vec<usize> {
        let mut v = Vec::new();
        for h in 0..N_HEXES {
            if self.hex_owner[h] == p && self.hex_type[h] == unit {
                v.push(h);
            }
        }
        v
    }

    fn has_deployed(&self, p: u8, unit: u8) -> bool {
        for h in 0..N_HEXES {
            if self.hex_owner[h] == p && self.hex_type[h] == unit {
                return true;
            }
        }
        false
    }

    fn deploy_locations(&self, p: u8, out: &mut Vec<u8>) {
        for h in 0..N_HEXES {
            if controls(self, h, p) && !occupied(self, h) {
                out.push(h as u8);
            }
        }
    }

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
    fn list_tactics(&self, hex: usize, out: &mut Vec<Action>) {
        let unit = self.hex_type[hex];
        let me = self.hex_owner[hex];
        let b = board();
        match def(unit).tactic {
            Tactic::None => {}
            Tactic::Footman => {
                out.push(Action::TacFootman { coin: unit });
            }
            Tactic::Archer => {
                for t in 0..N_HEXES {
                    if b.dist[hex][t] == 2 && is_enemy_unit(self, t, me) && attack_allowed_h(self, t, self.hex_height[hex]) {
                        out.push(Action::TacArcher {
                            from: hex as u8,
                            target: t as u8,
                        });
                    }
                }
            }
            Tactic::Cavalry => {
                let h = self.hex_height[hex];
                let mut tos = Vec::new();
                self.adj_empty(hex, &mut tos);
                for to in tos {
                    let has_target = (0..6).any(|d| {
                        let n = b.neighbors[to as usize][d];
                        n != NONE && is_enemy_unit(self, n as usize, me) && attack_allowed_h(self, n as usize, h)
                    });
                    if has_target {
                        out.push(Action::TacCavalryMove { from: hex as u8, to });
                    }
                }
            }
            Tactic::Crossbowman => {
                for t in 0..N_HEXES {
                    if b.dist[hex][t] == 2
                        && is_enemy_unit(self, t, me)
                        && clear_two_line(self, hex, t)
                        && attack_allowed_h(self, t, self.hex_height[hex])
                    {
                        out.push(Action::TacCrossbow {
                            from: hex as u8,
                            target: t as u8,
                        });
                    }
                }
            }
            Tactic::Ensign => {
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
                    let s1 = b.neighbors[hex][d];
                    if s1 == NONE || occupied(self, s1 as usize) {
                        continue;
                    }
                    let beyond1 = b.neighbors[s1 as usize][d];
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
                    let s2 = b.neighbors[s1 as usize][d];
                    if s2 != NONE && !occupied(self, s2 as usize) {
                        let beyond2 = b.neighbors[s2 as usize][d];
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
                        if b.dist[hex][to as usize] != 2 {
                            continue;
                        }
                        out.push(Action::TacLightCav { from: hex as u8, to });
                    }
                }
            }
            Tactic::Marshal => {
                for u in 0..N_HEXES {
                    if u == hex || !is_friendly_unit(self, u, me) || b.dist[hex][u] > 2 {
                        continue;
                    }
                    if def(self.hex_type[u]).no_normal_attack {
                        continue;
                    }
                    let mut tgts = Vec::new();
                    self.normal_attack_targets(u, self.hex_height[u], &mut tgts);
                    for t in tgts {
                        out.push(Action::TacMarshal {
                            from: u as u8,
                            target: t,
                        });
                    }
                }
            }
            Tactic::RoyalGuard => {
                if self.zones[me as usize][self.pay_zone()][ROYAL_COIN as usize] == 0 {
                    return;
                }
                for d1 in 0..6 {
                    let m = b.neighbors[hex][d1];
                    if m == NONE || occupied(self, m as usize) {
                        continue;
                    }
                    if controls(self, m as usize, me) {
                        out.push(Action::TacRoyalGuard { from: hex as u8, to: m });
                    }
                    for d2 in 0..6 {
                        let to = b.neighbors[m as usize][d2];
                        if to == NONE || to as usize == hex || occupied(self, to as usize) {
                            continue;
                        }
                        if controls(self, to as usize, me) {
                            out.push(Action::TacRoyalGuard { from: hex as u8, to });
                        }
                    }
                }
            }
        }
    }
    fn list_main_play(&self, p: u8, out: &mut Vec<Action>) {
        let claim_ok = self.initiative != p && !self.initiative_moved;
        let pay = self.pay_zone();
        for u in 0..N_UNITS {
            if self.zones[p as usize][pay][u] == 0 {
                continue;
            }
            let unit = u as u8;
            let d = def(unit);

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
                for h in self.hexes_of(p, ROYAL_GUARD) {
                    self.list_tactics(h, out);
                }
                continue;
            }

            if !self.has_deployed(p, unit) || d.two_footmen {
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
            for h in self.hexes_of(p, unit) {
                out.push(Action::Bolster { unit, hex: h as u8 });
            }
            for h in self.hexes_of(p, unit) {
                self.list_basic_maneuvers(h, ManVariant::MAIN, out);
                self.list_tactics(h, out);
            }
        }
        dedup(&mut *out);
    }
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut out = Vec::new();
        self.legal_actions_into(&mut out);
        out
    }

    pub fn legal_actions_into(&self, out: &mut Vec<Action>) {
        out.clear();
        if self.winner != NONE {
            return;
        }
        match self.pending {
            Cont::Draw { player } | Cont::WarriorPriestDraw { player, .. } => {
                for (u, &c) in self.draw_pool(player).iter().enumerate() {
                    if c > 0 {
                        out.push(Action::DrawCoin { unit: u as u8 });
                    }
                }
                if out.is_empty() && matches!(self.pending, Cont::WarriorPriestDraw { .. }) {
                    out.push(Action::DrawCoin { unit: NONE });
                }
            }
            Cont::MainPlay => {
                self.list_main_play(self.active, out);
            }
            Cont::WarriorPriestPlay { player } => {
                self.list_main_play(player, out);
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
                    ManVariant::BERSERKER_V2
                } else {
                    ManVariant::BERSERKER
                };
                self.list_basic_maneuvers(hex as usize, variant, out);
                out.push(Action::BerserkStop);
            }
            Cont::MercenaryManeuver { hex } => {
                self.list_basic_maneuvers(hex as usize, ManVariant::MERC, out);
                out.push(Action::MercDecline);
            }
            Cont::FootmanManeuver { hexes } => {
                for h in hexes.iter() {
                    self.list_basic_maneuvers(h as usize, ManVariant::FOOTMAN, out);
                }
            }
            Cont::CavalryAttack { hex } => {
                self.list_basic_maneuvers(hex as usize, ManVariant::CAVALRY_ATTACK, out);
            }
            Cont::FootmanInstantDeploy { coin } => {
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
            Cont::_AttackPost { .. } => {}
        }
        dedup(out);
    }

    pub fn apply(&self, action: Action) -> State {
        let mut s = self.clone();
        s.apply_inplace(action);
        s
    }

    pub fn apply_inplace(&mut self, action: Action) {
        use Action::*;
        match self.pending {
            Cont::Draw { player } => {
                if let DrawCoin { unit } = action {
                    self.do_draw(player, unit, Z_HAND);
                }
                self.finish();
                return;
            }
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
                        match rg_choice(self) {
                            Some(c) => self.set_pending(c),
                            None => self.finish(),
                        }
                    } else {
                        self.do_draw(player, unit, Z_INFLIGHT);
                        match rg_choice(self) {
                            Some(c) => {
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

        match self.pending {
            Cont::MainPlay => {
                self.main_plays += 1;
                self.apply_main(self.active, action);
            }
            Cont::WarriorPriestPlay { player } => {
                self.apply_main(player, action);
                return;
            }
            Cont::RoyalGuardChoice { defender, rg_hex } => {
                match action {
                    RGSoakSupply => {
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
            Cont::MercenaryManeuver { .. } => {
                match action {
                    MercMove { from, to } => self.do_move(from as usize, to as usize),
                    MercControl { from } => self.do_control(from as usize),
                    MercAttack { from, target } => self.do_attack(from as usize, target as usize),
                    MercDecline => {}
                    _ => {}
                }
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
    }
    fn place_unit(&mut self, p: u8, unit: u8, hex: usize) {
        debug_assert!(self.hex_type[hex] == NONE);
        self.hex_type[hex] = unit;
        self.hex_owner[hex] = p;
        self.hex_height[hex] = 1;
    }

    fn apply_main(&mut self, p: u8, action: Action) {
        use Action::*;
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
                self.zmove(p, pay, Z_FACEDOWN, coin);
                self.zones[p as usize][Z_SUPPLY][unit as usize] -= 1;
                self.zones[p as usize][Z_FACEUP][unit as usize] += 1;
                let d = def(unit);
                if d.mercenary && self.has_deployed(p, unit) {
                    let hexes = self.hexes_of(p, unit);
                    let h = hexes[0];
                    self.push_cont(Cont::MercenaryManeuver { hex: h as u8 });
                }
                if d.footman_v2 && self.has_deployed(p, unit) {
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
                self.push_cont(Cont::CavalryAttack { hex: to });
                self.finish();
            }
            TacEnsign { from, to } => {
                self.zmove(p, pay, Z_FACEUP, ENSIGN);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            TacLancer { from, to, target } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.move_stack(from as usize, to as usize);
                self.do_attack(to as usize, target as usize);
                self.finish();
            }
            TacLightCav { from, to } => {
                self.spend_maneuver_coin(p, from as usize, pay);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            TacMarshal { from, target } => {
                self.zmove(p, pay, Z_FACEUP, MARSHAL);
                self.do_attack(from as usize, target as usize);
                self.finish();
            }
            TacRoyalGuard { from, to } => {
                self.zmove(p, pay, Z_FACEUP, ROYAL_COIN);
                self.do_move(from as usize, to as usize);
                self.finish();
            }
            TacFootman { coin } => {
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
                self.finish();
            }
        }
    }

    fn spend_maneuver_coin(&mut self, p: u8, hex: usize, pay: usize) {
        let unit = self.hex_type[hex];
        debug_assert!(unit != NONE);
        self.zmove(p, pay, Z_FACEUP, unit);
    }
    fn apply_berserker(&mut self, hex: u8, action: Action) {
        use Action::*;
        let h = hex as usize;
        if action == BerserkStop {
            self.finish();
            return;
        }
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
        let mut rest = hexes;
        rest.remove(from);
        if !rest.is_empty() {
            self.push_cont(Cont::FootmanManeuver { hexes: rest });
        }
        self.finish();
    }
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

fn controls_target(s: &State, hex: usize, me: u8) -> bool {
    board().is_location[hex] && s.loc_marker[hex] != me
}

#[derive(Clone, Copy)]
struct ManVariant {
    step: Option<fn(u8, u8) -> Action>,
    control: Option<fn(u8) -> Action>,
    attack: Option<fn(u8, u8) -> Action>,
    chain_cost: u8,
}

impl ManVariant {
    const MAIN: ManVariant = ManVariant {
        step: Some(|from, to| Action::Move { from, to }),
        control: Some(|from| Action::Control { from }),
        attack: Some(|from, target| Action::Attack { from, target }),
        chain_cost: 0,
    };
    const FOOTMAN: ManVariant = ManVariant {
        step: Some(|from, to| Action::FootMove { from, to }),
        control: Some(|from| Action::FootControl { from }),
        attack: Some(|from, target| Action::FootAttack { from, target }),
        chain_cost: 0,
    };
    const BERSERKER: ManVariant = ManVariant {
        step: Some(|from, to| Action::BerserkMove { from, to }),
        control: Some(|from| Action::BerserkControl { from }),
        attack: Some(|from, target| Action::BerserkAttack { from, target }),
        chain_cost: 1,
    };
    const BERSERKER_V2: ManVariant = ManVariant {
        control: None,
        ..ManVariant::BERSERKER
    };
    const MERC: ManVariant = ManVariant {
        step: Some(|from, to| Action::MercMove { from, to }),
        control: Some(|from| Action::MercControl { from }),
        attack: Some(|from, target| Action::MercAttack { from, target }),
        chain_cost: 0,
    };
    const CAVALRY_ATTACK: ManVariant = ManVariant {
        step: None,
        control: None,
        attack: Some(|from, target| Action::TacCavalryAttack { from, target }),
        chain_cost: 0,
    };
}

fn dedup(out: &mut Vec<Action>) {
    let mut seen = std::collections::HashSet::new();
    out.retain(|a| seen.insert(a.encode()));
}