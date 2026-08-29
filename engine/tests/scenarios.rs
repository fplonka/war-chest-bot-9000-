use warchest::actions::Action;
use warchest::board::board;
use warchest::state::*;
use warchest::units::*;

const CENTER: u8 = 18;
const E1: u8 = 19;
const E2: u8 = 20;
const E3: u8 = 21;
const W1: u8 = 17;
const LOC_53: u8 = 20;
const LOC_11: u8 = 11;
const LOC_25: u8 = 25;

fn has(acts: &[Action], a: Action) -> bool {
    acts.iter().any(|x| x.encode() == a.encode())
}

fn count_kind(acts: &[Action], pred: impl Fn(&Action) -> bool) -> usize {
    acts.iter().filter(|a| pred(a)).count()
}

fn one_unit(unit: u8, hex: u8) -> State {
    let mut s = State::blank(WHITE);
    s.set_unit(hex, WHITE, unit, 1);
    s.add_zone(WHITE, Z_HAND, unit, 1);
    s
}

#[test]
fn archer_cannot_normal_attack_but_tactic_hits_at_two() {
    let mut s = one_unit(ARCHER, CENTER);
    s.set_unit(E1, BLACK, KNIGHT, 1);
    s.set_unit(E2, BLACK, SWORDSMAN, 1);
    let acts = s.legal_actions();
    assert_eq!(count_kind(&acts, |a| matches!(a, Action::Attack { .. })), 0);
    assert!(has(&acts, Action::TacArcher { from: CENTER, target: E2 }));
    assert!(!has(&acts, Action::TacArcher { from: CENTER, target: E1 }));
}

#[test]
fn crossbowman_straight_line_blocked_by_intervening_unit() {
    let mut s = one_unit(CROSSBOWMAN, W1);
    s.set_unit(E1, BLACK, SWORDSMAN, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacCrossbow { from: W1, target: E1 }));
    s.set_unit(CENTER, BLACK, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(!has(&acts, Action::TacCrossbow { from: W1, target: E1 }));
    assert!(has(&acts, Action::Attack { from: W1, target: CENTER }));
}

#[test]
fn knight_immune_to_unbolstered_and_hit_by_bolstered() {
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, KNIGHT, 1);
    let acts = s.legal_actions();
    assert!(!has(&acts, Action::Attack { from: CENTER, target: E1 }));
    s.set_unit(CENTER, WHITE, SWORDSMAN, 2);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::Attack { from: CENTER, target: E1 }));
}

#[test]
fn knight_immunity_applies_to_tactic_attacks() {
    let mut s = one_unit(ARCHER, CENTER);
    s.set_unit(E2, BLACK, KNIGHT, 1);
    let acts = s.legal_actions();
    assert!(!has(&acts, Action::TacArcher { from: CENTER, target: E2 }));
    s.set_unit(CENTER, WHITE, ARCHER, 2);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacArcher { from: CENTER, target: E2 }));
}

#[test]
fn pikeman_reflex_mutual_death_and_ignores_knight() {
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, PIKEMAN, 1);
    let s2 = s.apply(Action::Attack { from: CENTER, target: E1, });
    assert_eq!(
        s2.hex_type[CENTER as usize],
        warchest::board::NONE,
        "attacker should die to reflex"
    );
    assert_eq!(
        s2.hex_type[E1 as usize],
        warchest::board::NONE,
        "pikeman should die to the attack"
    );
    assert_eq!(s2.zones[WHITE as usize][Z_ELIM][SWORDSMAN as usize], 1);
    assert_eq!(s2.zones[WHITE as usize][Z_FACEUP][SWORDSMAN as usize], 1);
    assert_eq!(s2.zones[BLACK as usize][Z_ELIM][PIKEMAN as usize], 1);
}

#[test]
fn royal_guard_defender_supply_choice() {
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, ROYAL_GUARD, 1);
    s.add_zone(BLACK, Z_SUPPLY, ROYAL_GUARD, 1);
    let s2 = s.apply(Action::Attack { from: CENTER, target: E1, });
    assert_eq!(s2.to_act(), BLACK, "defender chooses RG soak");
    let choices = s2.legal_actions();
    assert!(has(&choices, Action::RGSoakSupply));
    assert!(has(&choices, Action::RGSoakStack));
    let s3 = s2.apply(Action::RGSoakSupply);
    assert_eq!(
        s3.hex_type[E1 as usize], ROYAL_GUARD,
        "RG survives when soaked from supply"
    );
    assert_eq!(s3.zones[BLACK as usize][Z_SUPPLY][ROYAL_GUARD as usize], 0);
    assert_eq!(s3.zones[BLACK as usize][Z_ELIM][ROYAL_GUARD as usize], 1);
    let s3b = s2.apply(Action::RGSoakStack);
    assert_eq!(
        s3b.hex_type[E1 as usize],
        warchest::board::NONE,
        "RG dies when soaked from stack"
    );
    assert_eq!(s3b.zones[BLACK as usize][Z_SUPPLY][ROYAL_GUARD as usize], 1);
}

#[test]
fn royal_guard_tactic_needs_royal_coin_and_ends_on_controlled_location() {
    let mut s = one_unit(ROYAL_GUARD, CENTER);
    s.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);
    s.set_marker(LOC_53, WHITE);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacRoyalGuard { from: CENTER, to: LOC_53 }));
    let mut s2 = one_unit(ROYAL_GUARD, CENTER);
    s2.set_marker(LOC_53, WHITE);
    let acts2 = s2.legal_actions();
    assert_eq!(count_kind(&acts2, |a| matches!(a, Action::TacRoyalGuard { .. })), 0);
    let mut s3 = one_unit(ROYAL_GUARD, CENTER);
    s3.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);
    let acts3 = s3.legal_actions();
    assert_eq!(count_kind(&acts3, |a| matches!(a, Action::TacRoyalGuard { .. })), 0);
}

#[test]
fn scout_deploy_adjacent_to_friendly() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, SCOUT, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::Deploy { unit: SCOUT, hex: E1 }));
    assert!(!has(&acts, Action::Deploy { unit: SCOUT, hex: E3 }));
}

#[test]
fn swordsman_optional_move_after_attack() {
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, FOOTMAN, 2);
    let s2 = s.apply(Action::Attack { from: CENTER, target: E1, });
    match s2.pending() {
        Cont::SwordsmanMove { hex } => assert_eq!(*hex, CENTER),
        other => panic!("expected SwordsmanMove pending, got {:?}", other),
    }
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::SwordsmanDecline));
    assert!(has(&acts, Action::SwordsmanMove { from: CENTER, to: W1 }));
}

#[test]
fn berserker_v1_chain_via_any_maneuver_coin_to_faceup() {
    let mut s = one_unit(BERSERKER, CENTER);
    s.set_unit(CENTER, WHITE, BERSERKER, 3);
    let s2 = s.apply(Action::Move { from: CENTER, to: W1 });
    match s2.pending() {
        Cont::BerserkerChain { hex, v2 } => {
            assert_eq!(*hex, W1);
            assert!(!*v2);
        }
        other => panic!("expected BerserkerChain, got {:?}", other),
    }
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::BerserkStop));
    let s3 = s2.apply(Action::BerserkMove { from: W1, to: CENTER });
    assert_eq!(
        s3.zones[WHITE as usize][Z_FACEUP][BERSERKER as usize], 2,
        "chain coin recycles face-up, not eliminated"
    );
    assert_eq!(s3.zones[WHITE as usize][Z_ELIM][BERSERKER as usize], 0, "chain coin is NOT eliminated");
    assert_eq!(s3.hex_height[CENTER as usize], 2);
}

#[test]
fn berserker_cannot_remove_final_coin() {
    let s = one_unit(BERSERKER, CENTER);
    let s2 = s.apply(Action::Move { from: CENTER, to: W1 });
    assert!(!matches!(s2.pending(), Cont::BerserkerChain { .. }));
}

#[test]
fn berserker_v2_only_attack_or_move_no_control() {
    let mut s = State::blank(WHITE);
    s.set_unit(LOC_11, WHITE, BERSERKER_V2, 2);
    s.add_zone(WHITE, Z_HAND, BERSERKER_V2, 1);
    s.set_markers_hand(WHITE, 4);
    let s2 = s.apply(Action::Control { from: LOC_11 });
    assert!(
        !matches!(s2.pending(), Cont::BerserkerChain { .. }),
        "V2 must not chain after control"
    );
    let mut s3 = State::blank(WHITE);
    s3.set_unit(CENTER, WHITE, BERSERKER_V2, 2);
    s3.add_zone(WHITE, Z_HAND, BERSERKER_V2, 1);
    s3.set_marker(LOC_25, BLACK);
    let s4 = s3.apply(Action::Move { from: CENTER, to: LOC_25, });
    match s4.pending() {
        Cont::BerserkerChain { v2, .. } => assert!(*v2),
        other => panic!("expected V2 chain, got {:?}", other),
    }
    let acts = s4.legal_actions();
    assert_eq!(count_kind(&acts, |a| matches!(a, Action::BerserkControl { .. })), 0);
}

#[test]
fn lancer_must_attack_and_knight_immunity_in_line() {
    let mut s = one_unit(LANCER, W1);
    s.set_unit(E1, BLACK, SWORDSMAN, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacLancer { from: W1, to: CENTER, target: E1 }));
    assert_eq!(count_kind(&acts, |a| matches!(a, Action::Attack { .. })), 0);
    let mut s2 = one_unit(LANCER, W1);
    s2.set_unit(E1, BLACK, KNIGHT, 1);
    let acts2 = s2.legal_actions();
    assert_eq!(count_kind(&acts2, |a| matches!(a, Action::TacLancer { .. })), 0);
    s2.set_unit(W1, WHITE, LANCER, 2);
    let acts3 = s2.legal_actions();
    assert!(has(&acts3, Action::TacLancer { from: W1, to: CENTER, target: E1 }));
}

#[test]
fn cavalry_move_then_attack() {
    let mut s = one_unit(CAVALRY, W1);
    s.set_unit(E1, BLACK, FOOTMAN, 2);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacCavalryMove { from: W1, to: CENTER }));
    let s2 = s.apply(Action::TacCavalryMove { from: W1, to: CENTER });
    match s2.pending() {
        Cont::CavalryAttack { hex } => assert_eq!(*hex, CENTER),
        other => panic!("expected CavalryAttack, got {:?}", other),
    }
    let acts2 = s2.legal_actions();
    assert!(has(&acts2, Action::TacCavalryAttack { from: CENTER, target: E1 }));
}

#[test]
fn light_cavalry_moves_exactly_two_through_empty() {
    let mut s = one_unit(LIGHT_CAVALRY, W1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacLightCav { from: W1, to: E1 }));
    s.set_unit(CENTER, BLACK, FOOTMAN, 1);
    let acts2 = s.legal_actions();
    assert!(!has(&acts2, Action::TacLightCav { from: W1, to: E1 }));
}

#[test]
fn ensign_moves_friendly_within_two() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, ENSIGN, 1);
    s.add_zone(WHITE, Z_HAND, ENSIGN, 1);
    s.set_unit(W1, WHITE, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(count_kind(&acts, |a| matches!(a, Action::TacEnsign { from, .. } if *from == W1)) > 0);
}

#[test]
fn marshal_grants_normal_attack_not_archer() {
    const NB: u8 = 24;
    assert_eq!(board().dist[W1 as usize][NB as usize], 1);
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, MARSHAL, 1);
    s.add_zone(WHITE, Z_HAND, MARSHAL, 1);
    s.set_unit(W1, WHITE, ARCHER, 1);
    s.set_unit(NB, BLACK, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert_eq!(count_kind(
            &acts,
            |a| matches!(a, Action::TacMarshal { from, .. } if *from == W1)
        ), 0);
    let mut s2 = State::blank(WHITE);
    s2.set_unit(CENTER, WHITE, MARSHAL, 1);
    s2.add_zone(WHITE, Z_HAND, MARSHAL, 1);
    s2.set_unit(W1, WHITE, SWORDSMAN, 1);
    s2.set_unit(NB, BLACK, FOOTMAN, 1);
    let acts2 = s2.legal_actions();
    assert!(has(&acts2, Action::TacMarshal { from: W1, target: NB }));
}

#[test]
fn footman_two_units_and_tactic_two_maneuvers() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN, 1);
    s.set_unit(E1, BLACK, FOOTMAN, 1);
    s.set_unit(W1, WHITE, FOOTMAN, 1);
    s.add_zone(WHITE, Z_HAND, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacFootman { coin: FOOTMAN }));
    let s2 = s.apply(Action::TacFootman { coin: FOOTMAN });
    assert!(matches!(s2.pending(), Cont::FootmanManeuver { .. }));
}

#[test]
fn footman_deploy_two_allowed_others_capped_at_one() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN, 1);
    s.set_marker(LOC_11, WHITE);
    s.set_marker(LOC_25, WHITE);
    s.add_zone(WHITE, Z_HAND, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(count_kind(&acts, |a| matches!(a, Action::Deploy { unit: FOOTMAN, .. })) >= 1);
    let mut s2 = State::blank(WHITE);
    s2.set_unit(CENTER, WHITE, KNIGHT, 1);
    s2.set_marker(LOC_11, WHITE);
    s2.add_zone(WHITE, Z_HAND, KNIGHT, 1);
    let acts2 = s2.legal_actions();
    assert_eq!(count_kind(&acts2, |a| matches!(a, Action::Deploy { unit: KNIGHT, .. })), 0);
}

#[test]
fn mercenary_free_maneuver_only_when_deployed() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, MERCENARY, 1);
    s.add_zone(WHITE, Z_HAND, MERCENARY, 1);
    s.add_zone(WHITE, Z_SUPPLY, MERCENARY, 1);
    let s2 = s.apply(Action::Recruit { coin: MERCENARY, unit: MERCENARY, });
    assert!(matches!(s2.pending(), Cont::MercenaryManeuver { .. }));
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::MercDecline));
    let mut s3 = State::blank(WHITE);
    s3.add_zone(WHITE, Z_HAND, MERCENARY, 1);
    s3.add_zone(WHITE, Z_SUPPLY, MERCENARY, 1);
    let s4 = s3.apply(Action::Recruit { coin: MERCENARY, unit: MERCENARY, });
    assert!(!matches!(s4.pending(), Cont::MercenaryManeuver { .. }));
}

#[test]
fn footman_v2_recruit_instant_deploy() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN_V2, 1);
    s.set_marker(LOC_11, WHITE);
    s.add_zone(WHITE, Z_HAND, FOOTMAN_V2, 1);
    s.add_zone(WHITE, Z_SUPPLY, FOOTMAN_V2, 1);
    let s2 = s.apply(Action::Recruit { coin: FOOTMAN_V2, unit: FOOTMAN_V2, });
    assert!(matches!(s2.pending(), Cont::FootmanInstantDeploy { .. }));
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::FootmanInstantDeploy { hex: LOC_11 }));
    assert!(has(&acts, Action::FootmanInstantDecline));
    let s3 = s2.apply(Action::FootmanInstantDeploy { hex: LOC_11 });
    assert_eq!(s3.hex_type[LOC_11 as usize], FOOTMAN_V2);
}

#[test]
fn warrior_priest_draws_and_forces_use() {
    let mut s = State::blank(WHITE);
    s.set_unit(LOC_11, WHITE, WARRIOR_PRIEST, 1);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    s.add_zone(WHITE, Z_BAG, ARCHER, 1);
    s.set_markers_hand(WHITE, 4);
    let s2 = s.apply(Action::Control { from: LOC_11 });
    match s2.pending() {
        Cont::WarriorPriestDraw { player, .. } => assert_eq!(*player, WHITE),
        other => panic!("expected WarriorPriestDraw, got {:?}", other),
    }
    let draws = s2.legal_actions();
    assert!(has(&draws, Action::DrawCoin { unit: ARCHER }));
    let s3 = s2.apply(Action::DrawCoin { unit: ARCHER });
    assert!(matches!(s3.pending(), Cont::WarriorPriestPlay { .. }));
    assert_eq!(s3.zones[WHITE as usize][Z_INFLIGHT][ARCHER as usize], 1);
    let plays = s3.legal_actions();
    assert!(has(&plays, Action::Pass { coin: ARCHER }));
}

#[test]
fn warrior_priest_v2_once_per_turn() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, WARRIOR_PRIEST_V2, 1);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST_V2, 1);
    s.set_unit(E1, BLACK, FOOTMAN, 3);
    s.add_zone(WHITE, Z_BAG, WARRIOR_PRIEST_V2, 1);
    let s2 = s.apply(Action::Attack { from: CENTER, target: E1, });
    assert!(matches!(s2.pending(), Cont::WarriorPriestDraw { .. }));
    let s3 = s2.apply(Action::DrawCoin { unit: WARRIOR_PRIEST_V2, });
    let s4 = s3.apply(Action::Attack { from: CENTER, target: E1, });
    assert!(
        !matches!(s4.pending(), Cont::WarriorPriestDraw { .. }),
        "V2 caps at one trigger per turn"
    );
}

#[test]
fn warrior_priest_v1_does_not_block_v2() {
    let mut s = State::blank(WHITE);
    s.set_unit(LOC_11, WHITE, WARRIOR_PRIEST, 1);
    s.set_unit(CENTER, WHITE, WARRIOR_PRIEST_V2, 1);
    s.set_unit(E1, BLACK, FOOTMAN, 3);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    s.add_zone(WHITE, Z_BAG, WARRIOR_PRIEST_V2, 1);
    let s2 = s.apply(Action::Attack { from: LOC_11, target: E1, });
    assert!(matches!(s2.pending(), Cont::WarriorPriestDraw { .. }));
    let s3 = s2.apply(Action::DrawCoin { unit: WARRIOR_PRIEST_V2, });
    let s4 = s3.apply(Action::Attack { from: CENTER, target: E1, });
    assert!(
        matches!(s4.pending(), Cont::WarriorPriestDraw { .. }),
        "V1's trigger must not block V2's"
    );
}

#[test]
fn warrior_priest_forced_play_of_rg_coin_cannot_use_rg_tactic() {
    let mut s = State::blank(WHITE);
    s.set_unit(W1, WHITE, WARRIOR_PRIEST, 1);
    s.set_unit(CENTER, WHITE, ROYAL_GUARD, 1);
    s.set_marker(LOC_25, WHITE);
    s.set_unit(E1, BLACK, FOOTMAN, 3);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    s.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);
    s.add_zone(WHITE, Z_BAG, ROYAL_GUARD, 1);
    let s2 = s.apply(Action::Attack { from: W1, target: E1 });
    let s3 = s2.apply(Action::DrawCoin { unit: ROYAL_GUARD });
    let plays = s3.legal_actions();
    assert!(has(&plays, Action::Pass { coin: ROYAL_GUARD }));
    assert!(!has(&plays, Action::TacRoyalGuard { from: CENTER, to: LOC_25 }));
    let mut t = State::blank(WHITE);
    t.set_unit(W1, WHITE, WARRIOR_PRIEST, 1);
    t.set_unit(CENTER, WHITE, ROYAL_GUARD, 1);
    t.set_marker(LOC_25, WHITE);
    t.set_unit(E1, BLACK, FOOTMAN, 3);
    t.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    t.add_zone(WHITE, Z_BAG, ROYAL_COIN, 1);
    let t2 = t.apply(Action::Attack { from: W1, target: E1 });
    let t3 = t2.apply(Action::DrawCoin { unit: ROYAL_COIN });
    let tplays = t3.legal_actions();
    assert!(has(&tplays, Action::TacRoyalGuard { from: CENTER, to: LOC_25 }));
}

#[test]
fn initiative_claim_rules() {
    let mut s = State::blank(WHITE);
    s.add_zone(WHITE, Z_HAND, ARCHER, 1);
    s.set_initiative(BLACK, false);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::ClaimInitiative { coin: ARCHER }));
    s.set_initiative(WHITE, false);
    let acts2 = s.legal_actions();
    assert_eq!(count_kind(&acts2, |a| matches!(a, Action::ClaimInitiative { .. })), 0);
    s.set_initiative(BLACK, true);
    let acts3 = s.legal_actions();
    assert_eq!(count_kind(&acts3, |a| matches!(a, Action::ClaimInitiative { .. })), 0);
}

#[test]
fn control_returns_opponent_marker_and_wins_at_six() {
    let mut s = State::blank(WHITE);
    s.set_marker(1, WHITE);
    s.set_marker(4, WHITE);
    s.set_marker(8, WHITE);
    s.set_marker(11, WHITE);
    s.set_marker(16, WHITE);
    s.set_markers_hand(WHITE, 1);
    s.set_marker(LOC_25, BLACK);
    s.set_markers_hand(BLACK, 5);
    s.set_unit(LOC_25, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, KNIGHT, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::Control { from: LOC_25 }));
    let s2 = s.apply(Action::Control { from: LOC_25 });
    assert_eq!(s2.winner(), Some(WHITE), "6th marker wins");
    assert_eq!(s2.markers_hand[BLACK as usize], 6);
}

#[test]
fn control_illegal_on_already_controlled_location() {
    let mut s = State::blank(WHITE);
    s.set_marker(LOC_11, WHITE);
    s.set_unit(LOC_11, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, KNIGHT, 1);
    let acts = s.legal_actions();
    assert_eq!(count_kind(&acts, |a| matches!(a, Action::Control { .. })), 0);
}

#[test]
fn deploy_only_on_empty_controlled_locations() {
    let mut s = State::blank(WHITE);
    s.set_marker(LOC_11, WHITE);
    s.set_marker(LOC_25, WHITE);
    s.set_unit(LOC_25, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, ARCHER, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::Deploy { unit: ARCHER, hex: LOC_11 }));
    assert!(!has(&acts, Action::Deploy { unit: ARCHER, hex: LOC_25 }));
    assert!(!has(&acts, Action::Deploy { unit: ARCHER, hex: LOC_53 }));
}

#[test]
fn move_only_into_empty_hexes() {
    let mut s = one_unit(KNIGHT, CENTER);
    s.set_unit(E1, BLACK, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(!has(&acts, Action::Move { from: CENTER, to: E1 }));
    assert!(has(&acts, Action::Move { from: CENTER, to: W1 }));
}

#[test]
fn action_encode_decode_roundtrip() {
    let samples = [
        Action::Deploy { unit: ARCHER, hex: 5 },
        Action::Bolster { unit: KNIGHT, hex: 18 },
        Action::ClaimInitiative { coin: ROYAL_COIN },
        Action::Recruit { coin: ARCHER, unit: MARSHAL, },
        Action::Pass { coin: ROYAL_COIN },
        Action::Move { from: 18, to: 19 },
        Action::Control { from: 11 },
        Action::Attack { from: 18, target: 19 },
        Action::TacArcher { from: 18, target: 20 },
        Action::TacCavalryMove { from: 18, to: 19 },
        Action::TacCavalryAttack { from: 19, target: 20 },
        Action::TacCrossbow { from: 17, target: 19 },
        Action::TacEnsign { from: 18, to: 19 },
        Action::TacLancer { from: 17, to: 18, target: 19, },
        Action::TacLightCav { from: 17, to: 19 },
        Action::TacMarshal { from: 17, target: 24, },
        Action::TacRoyalGuard { from: 18, to: 20 },
        Action::TacFootman { coin: FOOTMAN },
        Action::FootMove { from: 18, to: 19 },
        Action::FootControl { from: 11 },
        Action::FootAttack { from: 18, target: 19 },
        Action::DrawCoin { unit: ARCHER },
        Action::DrawCoin { unit: warchest::board::NONE, },
        Action::RGSoakSupply,
        Action::RGSoakStack,
        Action::SwordsmanMove { from: 18, to: 19 },
        Action::SwordsmanDecline,
        Action::BerserkMove { from: 18, to: 19 },
        Action::BerserkControl { from: 11 },
        Action::BerserkAttack { from: 18, target: 19 },
        Action::BerserkStop,
        Action::MercMove { from: 18, to: 19 },
        Action::MercControl { from: 11 },
        Action::MercAttack { from: 18, target: 19 },
        Action::MercDecline,
        Action::FootmanInstantDeploy { hex: 11 },
        Action::FootmanInstantDecline,
    ];
    for a in samples {
        let code = a.encode();
        let back = Action::decode(code).expect("decode");
        assert_eq!(a.encode(), back.encode(), "roundtrip mismatch for {a:?}");
        assert_eq!(a, back);
    }
}

#[test]
fn royal_guard_tactic_with_only_royal_coin_in_hand_discards_it_faceup() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, ROYAL_GUARD, 1);
    s.set_marker(LOC_11, WHITE);
    s.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacRoyalGuard { from: CENTER, to: LOC_11 }));
    let s2 = s.apply(Action::TacRoyalGuard { from: CENTER, to: LOC_11, });
    assert_eq!(
        s2.zones[WHITE as usize][Z_FACEUP][ROYAL_COIN as usize], 1,
        "Royal Coin goes to the FACE-UP discard"
    );
    assert_eq!(s2.zones[WHITE as usize][Z_FACEDOWN][ROYAL_COIN as usize], 0);
    assert_eq!(s2.hex_type[LOC_11 as usize], ROYAL_GUARD);
}

#[test]
fn footman_v2_recruit_coin_stays_in_discard_until_deployed() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN_V2, 1);
    s.set_marker(LOC_11, WHITE);
    s.add_zone(WHITE, Z_HAND, FOOTMAN_V2, 1);
    s.add_zone(WHITE, Z_SUPPLY, FOOTMAN_V2, 1);
    let s2 = s.apply(Action::Recruit { coin: FOOTMAN_V2, unit: FOOTMAN_V2, });
    assert!(matches!(s2.pending(), Cont::FootmanInstantDeploy { .. }));
    assert_eq!(s2.zones[WHITE as usize][Z_FACEUP][FOOTMAN_V2 as usize], 1);
    assert_eq!(s2.zones[WHITE as usize][Z_HAND][FOOTMAN_V2 as usize], 0);
    let s3 = s2.apply(Action::FootmanInstantDeploy { hex: LOC_11 });
    assert_eq!(s3.hex_type[LOC_11 as usize], FOOTMAN_V2);
    assert_eq!(s3.zones[WHITE as usize][Z_FACEUP][FOOTMAN_V2 as usize], 0);
    let s4 = s2.apply(Action::FootmanInstantDecline);
    assert_eq!(s4.zones[WHITE as usize][Z_FACEUP][FOOTMAN_V2 as usize], 1);
}

#[test]
fn footman_tactic_covers_both_versions_in_any_order() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN, 1);
    s.set_unit(W1, WHITE, FOOTMAN_V2, 1);
    s.add_zone(WHITE, Z_HAND, FOOTMAN, 1);
    let s2 = s.apply(Action::TacFootman { coin: FOOTMAN });
    match s2.pending() {
        Cont::FootmanManeuver { hexes } => {
            assert_eq!(
                hexes.to_vec().as_slice(),
                &[W1, CENTER],
                "both footman versions owe a maneuver"
            );
        }
        other => panic!("expected FootmanManeuver, got {:?}", other),
    }
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::FootMove { from: CENTER, to: E1 }));
    assert!(has(&acts, Action::FootMove { from: W1, to: 16 }));
    let s3 = s2.apply(Action::FootMove { from: W1, to: 16 });
    match s3.pending() {
        Cont::FootmanManeuver { hexes } => assert_eq!(hexes.to_vec().as_slice(), &[CENTER]),
        other => panic!("expected FootmanManeuver for the remaining footman, got {:?}", other),
    }
}

#[test]
fn berserker_chain_attack_uses_post_payment_height_for_knight_immunity() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, BERSERKER, 2);
    s.set_unit(19, BLACK, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, BERSERKER, 1);
    let s2 = s.apply(Action::Move { from: CENTER, to: 12 });
    assert!(matches!(s2.pending(), Cont::BerserkerChain { .. }));
    let acts = s2.legal_actions();
    assert!(!has(&acts, Action::BerserkAttack { from: 12, target: 19 }));
    let mut t = State::blank(WHITE);
    t.set_unit(CENTER, WHITE, BERSERKER, 3);
    t.set_unit(19, BLACK, KNIGHT, 1);
    t.add_zone(WHITE, Z_HAND, BERSERKER, 1);
    let t2 = t.apply(Action::Move { from: CENTER, to: 12 });
    assert!(has(&t2.legal_actions(), Action::BerserkAttack { from: 12, target: 19 }));
}

#[test]
fn warrior_priest_draw_resolves_before_rg_soak_choice() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, WARRIOR_PRIEST, 1);
    s.set_unit(E1, BLACK, ROYAL_GUARD, 1);
    s.add_zone(BLACK, Z_SUPPLY, ROYAL_GUARD, 1);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    s.add_zone(WHITE, Z_BAG, ARCHER, 1);
    let s2 = s.apply(Action::Attack { from: CENTER, target: E1, });
    match s2.pending() {
        Cont::WarriorPriestDraw { player, rg_hex } => {
            assert_eq!(*player, WHITE);
            assert_eq!(*rg_hex, E1);
        }
        other => panic!("expected WarriorPriestDraw before the soak, got {:?}", other),
    }
    let s3 = s2.apply(Action::DrawCoin { unit: ARCHER });
    assert!(
        matches!(s3.pending(), Cont::RoyalGuardChoice { .. }),
        "defender's soak choice comes after the draw"
    );
    let s4 = s3.apply(Action::RGSoakSupply);
    assert_eq!(s4.zones[BLACK as usize][Z_SUPPLY][ROYAL_GUARD as usize], 0);
    assert_eq!(s4.zones[BLACK as usize][Z_ELIM][ROYAL_GUARD as usize], 1);
    assert_eq!(s4.hex_height[E1 as usize], 1, "stack untouched after a supply soak");
    assert!(
        matches!(s4.pending(), Cont::WarriorPriestPlay { .. }),
        "the forced play comes after the soak, got {:?}",
        s4.pending()
    );
    assert_eq!(s4.zones[WHITE as usize][Z_INFLIGHT][ARCHER as usize], 1);
}

#[test]
fn ensign_cannot_move_itself() {
    let mut s = one_unit(ENSIGN, CENTER);
    s.set_unit(W1, WHITE, SWORDSMAN, 1);
    let acts = s.legal_actions();
    assert_eq!(count_kind(
            &acts,
            |a| matches!(a, Action::TacEnsign { from, .. } if *from == CENTER)
        ), 0);
    assert!(count_kind(&acts, |a| matches!(a, Action::TacEnsign { from, .. } if *from == W1)) > 0);
}

#[test]
fn marshal_cannot_grant_itself_an_attack() {
    let mut s = one_unit(MARSHAL, CENTER);
    s.set_unit(E1, BLACK, SWORDSMAN, 1);
    s.set_unit(W1, WHITE, PIKEMAN, 1);
    s.set_unit(16, BLACK, SCOUT, 1);
    let acts = s.legal_actions();
    assert_eq!(count_kind(
            &acts,
            |a| matches!(a, Action::TacMarshal { from, .. } if *from == CENTER)
        ), 0);
    assert!(has(&acts, Action::TacMarshal { from: W1, target: 16 }));
    assert!(has(&acts, Action::Attack { from: CENTER, target: E1 }));
}
