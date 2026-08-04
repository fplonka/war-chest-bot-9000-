//! Scenario tests: hand-built positions asserting exact legal-action sets and
//! post-states for every tactic, attribute, and FAQ ruling in RULES.md.

use warchest::actions::Action;
use warchest::board::board;
use warchest::state::*;
use warchest::units::*;

// -- geometry cheatsheet (from examples/coords) --
// center (3,3)=18; neighbors: (4,3)=19 (2,3)=17 (3,4)=25 (3,2)=11 (4,2)=12 (2,4)=24
// straight line along +x from 17: 17(2,3)->18(3,3)->19(4,3)->20(5,3 loc)->21(6,3)
// locations: (4,0)=1 (6,1)=8 (0,5)=28 (2,6)=35 (2,1)=4 (3,2)=11 (5,3)=20 (1,3)=16 (3,4)=25 (4,5)=32

const CENTER: usize = 18; // (3,3)
const E1: usize = 19; // (4,3)
const E2: usize = 20; // (5,3) location
const E3: usize = 21; // (6,3)
const W1: usize = 17; // (2,3)
const LOC_53: usize = 20; // (5,3)
const LOC_11: usize = 11; // (3,2)
const LOC_25: usize = 25; // (3,4)

fn has(acts: &[Action], a: Action) -> bool {
    acts.iter().any(|x| x.encode() == a.encode())
}

fn count_kind(acts: &[Action], pred: impl Fn(&Action) -> bool) -> usize {
    acts.iter().filter(|a| pred(a)).count()
}

/// A blank state where WHITE has one hand coin of `unit`, one unit deployed at
/// `hex` (height 1), and controls a couple of locations for deploy tests.
fn one_unit(unit: u8, hex: usize) -> State {
    let mut s = State::blank(WHITE);
    s.set_unit(hex, WHITE, unit, 1);
    s.add_zone(WHITE, Z_HAND, unit, 1);
    s
}

// =========================================================== ARCHER

#[test]
fn archer_cannot_normal_attack_but_tactic_hits_at_two() {
    let mut s = one_unit(ARCHER, CENTER);
    // enemy at distance 1 (adjacent) and one at distance 2 in line.
    s.set_unit(E1, BLACK, KNIGHT, 1); // adjacent (dist 1) -- but Knight, unbolstered
    s.set_unit(E2, BLACK, SWORDSMAN, 1); // dist 2 in straight line
    let acts = s.legal_actions();
    // No normal Attack action for the Archer at all.
    assert_eq!(
        count_kind(&acts, |a| matches!(a, Action::Attack { .. })),
        0,
        "Archer must not offer a normal Attack"
    );
    // Tactic attack on the dist-2 unit is available (intervening E1 may be occupied).
    assert!(has(
        &acts,
        Action::TacArcher {
            from: CENTER as u8,
            target: E2 as u8
        }
    ));
    // Not on the adjacent unit (dist 1, not 2).
    assert!(!has(
        &acts,
        Action::TacArcher {
            from: CENTER as u8,
            target: E1 as u8
        }
    ));
}

// =========================================================== CROSSBOWMAN

#[test]
fn crossbowman_straight_line_blocked_by_intervening_unit() {
    let mut s = one_unit(CROSSBOWMAN, W1); // (2,3)=17
                                           // target at (4,3)=E1 is dist 2 from 17; intervening (3,3)=CENTER.
    s.set_unit(E1, BLACK, SWORDSMAN, 1);
    // First: intervening empty -> tactic legal.
    let acts = s.legal_actions();
    assert!(has(
        &acts,
        Action::TacCrossbow {
            from: W1 as u8,
            target: E1 as u8
        }
    ));
    // Now block the intervening hex.
    s.set_unit(CENTER, BLACK, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(
        !has(
            &acts,
            Action::TacCrossbow {
                from: W1 as u8,
                target: E1 as u8
            }
        ),
        "Crossbowman line must be blocked by an intervening unit"
    );
    // But the Crossbowman may still normal-attack an adjacent unit.
    // (CENTER is adjacent to W1.)
    assert!(has(
        &acts,
        Action::Attack {
            from: W1 as u8,
            target: CENTER as u8
        }
    ));
}

// =========================================================== KNIGHT immunity

#[test]
fn knight_immune_to_unbolstered_and_hit_by_bolstered() {
    // attacker Swordsman adjacent to a Knight.
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, KNIGHT, 1);
    // unbolstered attacker (height 1): cannot attack the Knight.
    let acts = s.legal_actions();
    assert!(!has(
        &acts,
        Action::Attack {
            from: CENTER as u8,
            target: E1 as u8
        }
    ));
    // bolster the attacker to height 2.
    s.set_unit(CENTER, WHITE, SWORDSMAN, 2);
    let acts = s.legal_actions();
    assert!(has(
        &acts,
        Action::Attack {
            from: CENTER as u8,
            target: E1 as u8
        }
    ));
}

#[test]
fn knight_immunity_applies_to_tactic_attacks() {
    // Archer (unbolstered) cannot tactic-attack a Knight at range 2.
    let mut s = one_unit(ARCHER, CENTER);
    s.set_unit(E2, BLACK, KNIGHT, 1); // dist 2
    let acts = s.legal_actions();
    assert!(!has(
        &acts,
        Action::TacArcher {
            from: CENTER as u8,
            target: E2 as u8
        }
    ));
    // bolstered Archer can.
    s.set_unit(CENTER, WHITE, ARCHER, 2);
    let acts = s.legal_actions();
    assert!(has(
        &acts,
        Action::TacArcher {
            from: CENTER as u8,
            target: E2 as u8
        }
    ));
}

// =========================================================== PIKEMAN reflex

#[test]
fn pikeman_reflex_mutual_death_and_ignores_knight() {
    // Single-coin attacker attacks single-coin Pikeman: both die.
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, PIKEMAN, 1);
    let s2 = s.apply(Action::Attack {
        from: CENTER as u8,
        target: E1 as u8,
    });
    assert_eq!(
        s2.hex_type[CENTER],
        warchest::board::NONE,
        "attacker should die to reflex"
    );
    assert_eq!(
        s2.hex_type[E1],
        warchest::board::NONE,
        "pikeman should die to the attack"
    );
    // The attacker's played coin went face-up (maneuver); its board coin was
    // eliminated by the reflex; the pikeman's coin was eliminated by the attack.
    assert_eq!(s2.zones[WHITE as usize][Z_ELIM][SWORDSMAN as usize], 1);
    assert_eq!(s2.zones[WHITE as usize][Z_FACEUP][SWORDSMAN as usize], 1);
    assert_eq!(s2.zones[BLACK as usize][Z_ELIM][PIKEMAN as usize], 1);
}

// =========================================================== ROYAL GUARD

#[test]
fn royal_guard_defender_supply_choice() {
    let mut s = one_unit(SWORDSMAN, CENTER); // white attacker
    s.set_unit(E1, BLACK, ROYAL_GUARD, 1);
    // defender has an RG coin in supply to soak with.
    s.add_zone(BLACK, Z_SUPPLY, ROYAL_GUARD, 1);
    // white attacks; the pending should become the defender's RG choice.
    let s2 = s.apply(Action::Attack {
        from: CENTER as u8,
        target: E1 as u8,
    });
    assert_eq!(s2.to_act(), BLACK, "defender chooses RG soak");
    let choices = s2.legal_actions();
    assert!(has(&choices, Action::RGSoakSupply));
    assert!(has(&choices, Action::RGSoakStack));
    // soak from supply: the on-board RG survives, a supply RG is eliminated.
    let s3 = s2.apply(Action::RGSoakSupply);
    assert_eq!(
        s3.hex_type[E1], ROYAL_GUARD,
        "RG survives when soaked from supply"
    );
    assert_eq!(s3.zones[BLACK as usize][Z_SUPPLY][ROYAL_GUARD as usize], 0);
    assert_eq!(s3.zones[BLACK as usize][Z_ELIM][ROYAL_GUARD as usize], 1);
    // soak from stack: on-board RG loses its only coin and dies.
    let s3b = s2.apply(Action::RGSoakStack);
    assert_eq!(
        s3b.hex_type[E1],
        warchest::board::NONE,
        "RG dies when soaked from stack"
    );
    assert_eq!(s3b.zones[BLACK as usize][Z_SUPPLY][ROYAL_GUARD as usize], 1);
}

#[test]
fn royal_guard_tactic_needs_royal_coin_and_ends_on_controlled_location() {
    // RG at CENTER; white controls LOC_53 (dist 2 straight) and holds a Royal Coin.
    let mut s = one_unit(ROYAL_GUARD, CENTER);
    s.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);
    s.set_marker(LOC_53, WHITE); // (5,3), two straight steps from CENTER via E1.
    let acts = s.legal_actions();
    // path CENTER(3,3)->E1(4,3)->E2/LOC_53(5,3): E1 must be empty.
    assert!(has(
        &acts,
        Action::TacRoyalGuard {
            from: CENTER as u8,
            to: LOC_53 as u8
        }
    ));
    // Without the Royal Coin, the tactic is illegal.
    let mut s2 = one_unit(ROYAL_GUARD, CENTER);
    s2.set_marker(LOC_53, WHITE);
    let acts2 = s2.legal_actions();
    assert_eq!(
        count_kind(&acts2, |a| matches!(a, Action::TacRoyalGuard { .. })),
        0
    );
    // Cannot end on a non-controlled location.
    let mut s3 = one_unit(ROYAL_GUARD, CENTER);
    s3.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1);
    // LOC_53 not controlled -> no RG tactic ending there.
    let acts3 = s3.legal_actions();
    assert_eq!(
        count_kind(&acts3, |a| matches!(a, Action::TacRoyalGuard { .. })),
        0
    );
}

// =========================================================== SCOUT

#[test]
fn scout_deploy_adjacent_to_friendly() {
    let mut s = State::blank(WHITE);
    // a friendly unit at CENTER, a Scout coin in hand, no controlled empty loc.
    s.set_unit(CENTER, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, SCOUT, 1);
    let acts = s.legal_actions();
    // Scout may deploy onto any empty hex adjacent to the Knight, e.g. E1.
    assert!(has(
        &acts,
        Action::Deploy {
            unit: SCOUT,
            hex: E1 as u8
        }
    ));
    // Not onto a non-adjacent empty hex like E3 (6,3).
    assert!(!has(
        &acts,
        Action::Deploy {
            unit: SCOUT,
            hex: E3 as u8
        }
    ));
}

// =========================================================== SWORDSMAN

#[test]
fn swordsman_optional_move_after_attack() {
    let mut s = one_unit(SWORDSMAN, CENTER);
    s.set_unit(E1, BLACK, FOOTMAN, 2); // survives the attack
    let s2 = s.apply(Action::Attack {
        from: CENTER as u8,
        target: E1 as u8,
    });
    // post-trigger: the swordsman may move (optional) or decline.
    match s2.pending() {
        Cont::SwordsmanMove { hex } => assert_eq!(*hex as usize, CENTER),
        other => panic!("expected SwordsmanMove pending, got {:?}", other),
    }
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::SwordsmanDecline));
    // can step to an empty neighbor of CENTER (e.g. W1).
    assert!(has(
        &acts,
        Action::SwordsmanMove {
            from: CENTER as u8,
            to: W1 as u8
        }
    ));
}

// =========================================================== BERSERKER

#[test]
fn berserker_v1_chain_via_any_maneuver_coin_to_faceup() {
    // Berserker height 3 at CENTER controls nothing; do a MOVE, expect a chain.
    let mut s = one_unit(BERSERKER, CENTER);
    s.set_unit(CENTER, WHITE, BERSERKER, 3);
    // give it a second coin in hand isn't needed for chain.
    let s2 = s.apply(Action::Move {
        from: CENTER as u8,
        to: W1 as u8,
    });
    // chain offered at the new hex (W1).
    match s2.pending() {
        Cont::BerserkerChain { hex, v2 } => {
            assert_eq!(*hex as usize, W1);
            assert!(!*v2);
        }
        other => panic!("expected BerserkerChain, got {:?}", other),
    }
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::BerserkStop));
    // chain with another move: discards one coin FACE-UP (recycles), height 3->2.
    // (The played maneuver coin from the initial Move also went face-up, so the
    // face-up count is 2: the played coin + the chain coin.)
    let s3 = s2.apply(Action::BerserkMove {
        from: W1 as u8,
        to: CENTER as u8,
    });
    assert_eq!(
        s3.zones[WHITE as usize][Z_FACEUP][BERSERKER as usize], 2,
        "chain coin recycles face-up, not eliminated"
    );
    assert_eq!(
        s3.zones[WHITE as usize][Z_ELIM][BERSERKER as usize], 0,
        "chain coin is NOT eliminated"
    );
    assert_eq!(s3.hex_height[CENTER], 2);
}

#[test]
fn berserker_cannot_remove_final_coin() {
    // height 1: after a maneuver, no chain is offered (needs >= 2).
    let s = one_unit(BERSERKER, CENTER);
    let s2 = s.apply(Action::Move {
        from: CENTER as u8,
        to: W1 as u8,
    });
    // no chain node; turn passes to the other player (or draws).
    assert!(!matches!(s2.pending(), Cont::BerserkerChain { .. }));
}

#[test]
fn berserker_v2_only_attack_or_move_no_control() {
    // V2 after a CONTROL: no chain (control excluded).
    let mut s = State::blank(WHITE);
    s.set_unit(LOC_11, WHITE, BERSERKER_V2, 2); // stands on a neutral location (3,2)
    s.add_zone(WHITE, Z_HAND, BERSERKER_V2, 1);
    s.set_markers_hand(WHITE, 4);
    let s2 = s.apply(Action::Control { from: LOC_11 as u8 });
    assert!(
        !matches!(s2.pending(), Cont::BerserkerChain { .. }),
        "V2 must not chain after control"
    );
    // V2 after a MOVE: chain offered, and its options exclude control.
    let mut s3 = State::blank(WHITE);
    s3.set_unit(CENTER, WHITE, BERSERKER_V2, 2);
    s3.add_zone(WHITE, Z_HAND, BERSERKER_V2, 1);
    // stand adjacent to a controllable location so a control WOULD be possible.
    s3.set_marker(LOC_25, BLACK); // (3,4) neighbor of CENTER, controllable
    let s4 = s3.apply(Action::Move {
        from: CENTER as u8,
        to: LOC_25 as u8,
    });
    match s4.pending() {
        Cont::BerserkerChain { v2, .. } => assert!(*v2),
        other => panic!("expected V2 chain, got {:?}", other),
    }
    let acts = s4.legal_actions();
    assert_eq!(
        count_kind(&acts, |a| matches!(a, Action::BerserkControl { .. })),
        0,
        "V2 chain excludes control"
    );
}

// =========================================================== LANCER

#[test]
fn lancer_must_attack_and_knight_immunity_in_line() {
    // Lancer at 17(2,3); enemy Swordsman at 19(4,3): move 1 to 18, attack 19.
    let mut s = one_unit(LANCER, W1);
    s.set_unit(E1, BLACK, SWORDSMAN, 1);
    let acts = s.legal_actions();
    assert!(has(
        &acts,
        Action::TacLancer {
            from: W1 as u8,
            to: CENTER as u8,
            target: E1 as u8
        }
    ));
    // No normal attack for a Lancer.
    assert_eq!(count_kind(&acts, |a| matches!(a, Action::Attack { .. })), 0);
    // Knight immunity: unbolstered Lancer cannot end adjacent to a Knight with
    // no other target -> tactic illegal.
    let mut s2 = one_unit(LANCER, W1);
    s2.set_unit(E1, BLACK, KNIGHT, 1);
    let acts2 = s2.legal_actions();
    assert_eq!(
        count_kind(&acts2, |a| matches!(a, Action::TacLancer { .. })),
        0,
        "unbolstered Lancer cannot attack the Knight"
    );
    // bolstered Lancer can.
    s2.set_unit(W1, WHITE, LANCER, 2);
    let acts3 = s2.legal_actions();
    assert!(has(
        &acts3,
        Action::TacLancer {
            from: W1 as u8,
            to: CENTER as u8,
            target: E1 as u8
        }
    ));
}

// =========================================================== CAVALRY

#[test]
fn cavalry_move_then_attack() {
    let mut s = one_unit(CAVALRY, W1);
    s.set_unit(E1, BLACK, FOOTMAN, 2);
    // move CENTER-ward then attack E1.
    let acts = s.legal_actions();
    assert!(has(
        &acts,
        Action::TacCavalryMove {
            from: W1 as u8,
            to: CENTER as u8
        }
    ));
    let s2 = s.apply(Action::TacCavalryMove {
        from: W1 as u8,
        to: CENTER as u8,
    });
    // mandatory follow-up attack from CENTER onto E1.
    match s2.pending() {
        Cont::CavalryAttack { hex } => assert_eq!(*hex as usize, CENTER),
        other => panic!("expected CavalryAttack, got {:?}", other),
    }
    let acts2 = s2.legal_actions();
    assert!(has(
        &acts2,
        Action::TacCavalryAttack {
            from: CENTER as u8,
            target: E1 as u8
        }
    ));
}

// =========================================================== LIGHT CAVALRY

#[test]
fn light_cavalry_moves_exactly_two_through_empty() {
    let mut s = one_unit(LIGHT_CAVALRY, W1); // (2,3)
    let acts = s.legal_actions();
    // two straight steps to E1 (4,3) via CENTER (both empty).
    assert!(has(
        &acts,
        Action::TacLightCav {
            from: W1 as u8,
            to: E1 as u8
        }
    ));
    // blocking the intermediate CENTER removes that destination.
    s.set_unit(CENTER, BLACK, FOOTMAN, 1);
    let acts2 = s.legal_actions();
    assert!(!has(
        &acts2,
        Action::TacLightCav {
            from: W1 as u8,
            to: E1 as u8
        }
    ));
}

// =========================================================== ENSIGN / MARSHAL

#[test]
fn ensign_moves_friendly_within_two() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, ENSIGN, 1);
    s.add_zone(WHITE, Z_HAND, ENSIGN, 1);
    // a friendly Footman adjacent (W1); it may step to an empty hex within 2 of ensign.
    s.set_unit(W1, WHITE, FOOTMAN, 1);
    let acts = s.legal_actions();
    // Footman at W1 can move to (1,3)=16 (dist 2 from ensign) or (2,4)=24 etc.
    assert!(
        count_kind(
            &acts,
            |a| matches!(a, Action::TacEnsign { from, .. } if *from as usize == W1)
        ) > 0
    );
}

#[test]
fn marshal_grants_normal_attack_not_archer() {
    const NB: usize = 24; // (2,4) is adjacent to W1 (2,3)
    assert_eq!(board().dist[W1][NB], 1);
    // friendly Archer adjacent to an enemy; Marshal cannot grant the Archer an attack.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, MARSHAL, 1);
    s.add_zone(WHITE, Z_HAND, MARSHAL, 1);
    s.set_unit(W1, WHITE, ARCHER, 1);
    s.set_unit(NB, BLACK, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert_eq!(
        count_kind(
            &acts,
            |a| matches!(a, Action::TacMarshal { unit_hex, .. } if *unit_hex as usize == W1)
        ),
        0,
        "Marshal must not grant an attack to an Archer"
    );
    // A friendly Swordsman adjacent to an enemy can be granted an attack.
    let mut s2 = State::blank(WHITE);
    s2.set_unit(CENTER, WHITE, MARSHAL, 1);
    s2.add_zone(WHITE, Z_HAND, MARSHAL, 1);
    s2.set_unit(W1, WHITE, SWORDSMAN, 1);
    s2.set_unit(NB, BLACK, FOOTMAN, 1);
    let acts2 = s2.legal_actions();
    assert!(
        has(
            &acts2,
            Action::TacMarshal {
                unit_hex: W1 as u8,
                target: NB as u8
            }
        ),
        "Marshal should grant the Swordsman a normal attack"
    );
}

// =========================================================== FOOTMAN

#[test]
fn footman_two_units_and_tactic_two_maneuvers() {
    let mut s = State::blank(WHITE);
    // two footman units on board + a footman coin in hand for the tactic.
    s.set_unit(CENTER, WHITE, FOOTMAN, 1);
    s.set_unit(E1, BLACK, FOOTMAN, 1); // enemy to allow an attack target adjacency test
    s.set_unit(W1, WHITE, FOOTMAN, 1);
    s.add_zone(WHITE, Z_HAND, FOOTMAN, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::TacFootman { coin: FOOTMAN }));
    // playing the tactic queues a maneuver for the first footman.
    let s2 = s.apply(Action::TacFootman { coin: FOOTMAN });
    assert!(matches!(s2.pending(), Cont::FootmanManeuver { .. }));
}

#[test]
fn footman_deploy_two_allowed_others_capped_at_one() {
    let mut s = State::blank(WHITE);
    // one Footman already deployed; controls two empty locations.
    s.set_unit(CENTER, WHITE, FOOTMAN, 1);
    s.set_marker(LOC_11, WHITE);
    s.set_marker(LOC_25, WHITE);
    s.add_zone(WHITE, Z_HAND, FOOTMAN, 1);
    let acts = s.legal_actions();
    // a second Footman may still deploy.
    assert!(count_kind(&acts, |a| matches!(a, Action::Deploy { unit: FOOTMAN, .. })) >= 1);
    // A non-Footman with one already deployed cannot deploy a second.
    let mut s2 = State::blank(WHITE);
    s2.set_unit(CENTER, WHITE, KNIGHT, 1);
    s2.set_marker(LOC_11, WHITE);
    s2.add_zone(WHITE, Z_HAND, KNIGHT, 1);
    let acts2 = s2.legal_actions();
    assert_eq!(
        count_kind(&acts2, |a| matches!(a, Action::Deploy { unit: KNIGHT, .. })),
        0
    );
}

// =========================================================== MERCENARY

#[test]
fn mercenary_free_maneuver_only_when_deployed() {
    // deployed Mercenary + recruit a Mercenary coin -> free maneuver offered.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, MERCENARY, 1);
    s.add_zone(WHITE, Z_HAND, MERCENARY, 1); // coin to spend on the recruit
    s.add_zone(WHITE, Z_SUPPLY, MERCENARY, 1); // a Mercenary in supply to recruit
    let s2 = s.apply(Action::Recruit {
        coin: MERCENARY,
        unit: MERCENARY,
    });
    assert!(matches!(s2.pending(), Cont::MercenaryManeuver { .. }));
    let acts = s2.legal_actions();
    assert!(has(&acts, Action::MercDecline));
    // Not deployed -> no free maneuver.
    let mut s3 = State::blank(WHITE);
    s3.add_zone(WHITE, Z_HAND, MERCENARY, 1);
    s3.add_zone(WHITE, Z_SUPPLY, MERCENARY, 1);
    let s4 = s3.apply(Action::Recruit {
        coin: MERCENARY,
        unit: MERCENARY,
    });
    assert!(!matches!(s4.pending(), Cont::MercenaryManeuver { .. }));
}

// =========================================================== FOOTMAN V2 recruit-deploy

#[test]
fn footman_v2_recruit_instant_deploy() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN_V2, 1); // one already deployed
    s.set_marker(LOC_11, WHITE); // an empty controlled location to deploy onto
    s.add_zone(WHITE, Z_HAND, FOOTMAN_V2, 1); // coin to spend on the recruit
    s.add_zone(WHITE, Z_SUPPLY, FOOTMAN_V2, 1); // one in supply to recruit
    let s2 = s.apply(Action::Recruit {
        coin: FOOTMAN_V2,
        unit: FOOTMAN_V2,
    });
    assert!(matches!(s2.pending(), Cont::FootmanInstantDeploy { .. }));
    let acts = s2.legal_actions();
    assert!(has(
        &acts,
        Action::FootmanInstantDeploy { hex: LOC_11 as u8 }
    ));
    assert!(has(&acts, Action::FootmanInstantDecline));
    let s3 = s2.apply(Action::FootmanInstantDeploy { hex: LOC_11 as u8 });
    assert_eq!(s3.hex_type[LOC_11], FOOTMAN_V2);
}

// =========================================================== WARRIOR PRIEST

#[test]
fn warrior_priest_draws_and_forces_use() {
    let mut s = State::blank(WHITE);
    s.set_unit(LOC_11, WHITE, WARRIOR_PRIEST, 1); // stands on a neutral location
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1); // coin to spend on the control
    s.add_zone(WHITE, Z_BAG, ARCHER, 1); // a coin to draw
    s.set_markers_hand(WHITE, 4);
    let s2 = s.apply(Action::Control { from: LOC_11 as u8 });
    // WP trigger: a draw chance node for white.
    match s2.pending() {
        Cont::WarriorPriestDraw { player, .. } => assert_eq!(*player, WHITE),
        other => panic!("expected WarriorPriestDraw, got {:?}", other),
    }
    let draws = s2.legal_actions();
    assert!(has(&draws, Action::DrawCoin { unit: ARCHER }));
    let s3 = s2.apply(Action::DrawCoin { unit: ARCHER });
    // now a forced play with the drawn Archer coin; pass always legal.
    match s3.pending() {
        Cont::WarriorPriestPlay { coin, .. } => assert_eq!(*coin, ARCHER),
        other => panic!("expected WarriorPriestPlay, got {:?}", other),
    }
    let plays = s3.legal_actions();
    assert!(has(&plays, Action::Pass { coin: ARCHER }));
}

#[test]
fn warrior_priest_v2_once_per_turn() {
    // Attack with a WP V2; it draws + forces a play. If that forced play is
    // itself a WP-V2 attack (same unit), a second trigger must NOT occur.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, WARRIOR_PRIEST_V2, 1);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST_V2, 1);
    s.set_unit(E1, BLACK, FOOTMAN, 3); // survives
    s.add_zone(WHITE, Z_BAG, WARRIOR_PRIEST_V2, 1); // draws a WP V2 coin
    let s2 = s.apply(Action::Attack {
        from: CENTER as u8,
        target: E1 as u8,
    });
    // first trigger -> draw.
    assert!(matches!(s2.pending(), Cont::WarriorPriestDraw { .. }));
    let s3 = s2.apply(Action::DrawCoin {
        unit: WARRIOR_PRIEST_V2,
    });
    // forced play: attack again with the WP V2 (drawn coin type).
    let s4 = s3.apply(Action::Attack {
        from: CENTER as u8,
        target: E1 as u8,
    });
    // V2 cap: no second WP draw this coin play.
    assert!(
        !matches!(s4.pending(), Cont::WarriorPriestDraw { .. }),
        "V2 caps at one trigger per turn"
    );
}

// =========================================================== INITIATIVE

#[test]
fn initiative_claim_rules() {
    let mut s = State::blank(WHITE);
    s.add_zone(WHITE, Z_HAND, ARCHER, 1);
    // white does NOT hold initiative and it has not moved -> claim legal.
    s.set_initiative(BLACK, false);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::ClaimInitiative { coin: ARCHER }));
    // white already holds it -> illegal.
    s.set_initiative(WHITE, false);
    let acts2 = s.legal_actions();
    assert_eq!(
        count_kind(&acts2, |a| matches!(a, Action::ClaimInitiative { .. })),
        0
    );
    // moved this round already -> illegal even if not held.
    s.set_initiative(BLACK, true);
    let acts3 = s.legal_actions();
    assert_eq!(
        count_kind(&acts3, |a| matches!(a, Action::ClaimInitiative { .. })),
        0
    );
}

// =========================================================== CONTROL / WIN

#[test]
fn control_returns_opponent_marker_and_wins_at_six() {
    let mut s = State::blank(WHITE);
    // white has 5 markers out, 1 in hand; a unit stands on an enemy-controlled loc.
    s.set_marker(1, WHITE);
    s.set_marker(4, WHITE);
    s.set_marker(8, WHITE);
    s.set_marker(11, WHITE);
    s.set_marker(16, WHITE);
    s.set_markers_hand(WHITE, 1);
    s.set_marker(LOC_25, BLACK); // enemy controls (3,4); white unit stands on it
    s.set_markers_hand(BLACK, 5);
    s.set_unit(LOC_25, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, KNIGHT, 1);
    let acts = s.legal_actions();
    assert!(has(&acts, Action::Control { from: LOC_25 as u8 }));
    let s2 = s.apply(Action::Control { from: LOC_25 as u8 });
    assert_eq!(s2.winner(), Some(WHITE), "6th marker wins");
    // black marker returned.
    assert_eq!(s2.markers_hand[BLACK as usize], 6);
}

#[test]
fn control_illegal_on_already_controlled_location() {
    let mut s = State::blank(WHITE);
    s.set_marker(LOC_11, WHITE);
    s.set_unit(LOC_11, WHITE, KNIGHT, 1);
    s.add_zone(WHITE, Z_HAND, KNIGHT, 1);
    let acts = s.legal_actions();
    assert_eq!(
        count_kind(&acts, |a| matches!(a, Action::Control { .. })),
        0
    );
}

// =========================================================== DEPLOY rules

#[test]
fn deploy_only_on_empty_controlled_locations() {
    let mut s = State::blank(WHITE);
    s.set_marker(LOC_11, WHITE); // empty controlled loc
    s.set_marker(LOC_25, WHITE);
    s.set_unit(LOC_25, WHITE, KNIGHT, 1); // occupied -> not deployable
    s.add_zone(WHITE, Z_HAND, ARCHER, 1);
    let acts = s.legal_actions();
    assert!(has(
        &acts,
        Action::Deploy {
            unit: ARCHER,
            hex: LOC_11 as u8
        }
    ));
    assert!(!has(
        &acts,
        Action::Deploy {
            unit: ARCHER,
            hex: LOC_25 as u8
        }
    ));
    // cannot deploy on a location you do not control.
    assert!(!has(
        &acts,
        Action::Deploy {
            unit: ARCHER,
            hex: LOC_53 as u8
        }
    ));
}

// =========================================================== MOVE rules

#[test]
fn move_only_into_empty_hexes() {
    let mut s = one_unit(KNIGHT, CENTER);
    s.set_unit(E1, BLACK, FOOTMAN, 1); // occupied neighbor
    let acts = s.legal_actions();
    assert!(!has(
        &acts,
        Action::Move {
            from: CENTER as u8,
            to: E1 as u8
        }
    ));
    assert!(has(
        &acts,
        Action::Move {
            from: CENTER as u8,
            to: W1 as u8
        }
    ));
}

// =========================================================== ACTION ENCODING

#[test]
fn action_encode_decode_roundtrip() {
    // Sample every variant with representative fields; assert stable roundtrip.
    let samples = [
        Action::Deploy {
            unit: ARCHER,
            hex: 5,
        },
        Action::Bolster {
            unit: KNIGHT,
            hex: 18,
        },
        Action::ClaimInitiative { coin: ROYAL_COIN },
        Action::Recruit {
            coin: ARCHER,
            unit: MARSHAL,
        },
        Action::Pass { coin: ROYAL_COIN },
        Action::Move { from: 18, to: 19 },
        Action::Control { from: 11 },
        Action::Attack {
            from: 18,
            target: 19,
        },
        Action::TacArcher {
            from: 18,
            target: 20,
        },
        Action::TacCavalryMove { from: 18, to: 19 },
        Action::TacCavalryAttack {
            from: 19,
            target: 20,
        },
        Action::TacCrossbow {
            from: 17,
            target: 19,
        },
        Action::TacEnsign { from: 18, to: 19 },
        Action::TacLancer {
            from: 17,
            to: 18,
            target: 19,
        },
        Action::TacLightCav { from: 17, to: 19 },
        Action::TacMarshal {
            unit_hex: 17,
            target: 24,
        },
        Action::TacRoyalGuard { from: 18, to: 20 },
        Action::TacFootman { coin: FOOTMAN },
        Action::FootMove { from: 18, to: 19 },
        Action::FootControl { from: 11 },
        Action::FootAttack {
            from: 18,
            target: 19,
        },
        Action::DrawCoin { unit: ARCHER },
        Action::DrawCoin {
            unit: warchest::board::NONE,
        },
        Action::RGSoakSupply,
        Action::RGSoakStack,
        Action::SwordsmanMove { from: 18, to: 19 },
        Action::SwordsmanDecline,
        Action::BerserkMove { from: 18, to: 19 },
        Action::BerserkControl { from: 11 },
        Action::BerserkAttack {
            from: 18,
            target: 19,
        },
        Action::BerserkStop,
        Action::MercMove { from: 18, to: 19 },
        Action::MercControl { from: 11 },
        Action::MercAttack {
            from: 18,
            target: 19,
        },
        Action::MercDecline,
        Action::FootmanInstantDeploy { hex: 11 },
        Action::FootmanInstantDecline,
    ];
    for a in samples {
        let code = a.encode();
        let back = Action::decode(code).expect("decode");
        assert_eq!(a.encode(), back.encode(), "roundtrip mismatch for {}", a);
        assert_eq!(a, back);
    }
}

// ================================================== census-replay fixes
// Each test below pins a rule fixed (or adjudicated) by the verify/ census
// replay against warchestonline.com data. See verify/FIXES.md.

#[test]
fn royal_guard_tactic_with_only_royal_coin_in_hand_discards_it_faceup() {
    // The RG tactic is PAID with the Royal Coin: it must be offered when only
    // the Royal Coin is in hand, and the coin is discarded FACE-UP.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, ROYAL_GUARD, 1);
    s.set_marker(LOC_11, WHITE); // adjacent controlled empty location
    s.add_zone(WHITE, Z_HAND, ROYAL_COIN, 1); // ONLY the Royal Coin in hand
    let acts = s.legal_actions();
    assert!(
        has(
            &acts,
            Action::TacRoyalGuard {
                from: CENTER as u8,
                to: LOC_11 as u8
            }
        ),
        "RG tactic must be offered with only the Royal Coin in hand"
    );
    let s2 = s.apply(Action::TacRoyalGuard {
        from: CENTER as u8,
        to: LOC_11 as u8,
    });
    assert_eq!(
        s2.zones[WHITE as usize][Z_FACEUP][ROYAL_COIN as usize], 1,
        "Royal Coin goes to the FACE-UP discard"
    );
    assert_eq!(s2.zones[WHITE as usize][Z_FACEDOWN][ROYAL_COIN as usize], 0);
    assert_eq!(s2.hex_type[LOC_11], ROYAL_GUARD);
}

#[test]
fn footman_v2_recruit_coin_stays_in_discard_until_deployed() {
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN_V2, 1);
    s.set_marker(LOC_11, WHITE);
    s.add_zone(WHITE, Z_HAND, FOOTMAN_V2, 1);
    s.add_zone(WHITE, Z_SUPPLY, FOOTMAN_V2, 1);
    let s2 = s.apply(Action::Recruit {
        coin: FOOTMAN_V2,
        unit: FOOTMAN_V2,
    });
    assert!(matches!(s2.pending(), Cont::FootmanInstantDeploy { .. }));
    // The recruited coin sits in the face-up discard, NOT in hand.
    assert_eq!(s2.zones[WHITE as usize][Z_FACEUP][FOOTMAN_V2 as usize], 1);
    assert_eq!(s2.zones[WHITE as usize][Z_HAND][FOOTMAN_V2 as usize], 0);
    // Deploying takes it out of the discard.
    let s3 = s2.apply(Action::FootmanInstantDeploy { hex: LOC_11 as u8 });
    assert_eq!(s3.hex_type[LOC_11], FOOTMAN_V2);
    assert_eq!(s3.zones[WHITE as usize][Z_FACEUP][FOOTMAN_V2 as usize], 0);
    // Declining leaves it in the discard.
    let s4 = s2.apply(Action::FootmanInstantDecline);
    assert_eq!(s4.zones[WHITE as usize][Z_FACEUP][FOOTMAN_V2 as usize], 1);
}

#[test]
fn footman_tactic_covers_both_versions_in_any_order() {
    // A Footman coin's tactic maneuvers Footman V2 units too, and the player
    // chooses which footman maneuvers first.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, FOOTMAN, 1); // hex 18
    s.set_unit(W1, WHITE, FOOTMAN_V2, 1); // hex 17
    s.add_zone(WHITE, Z_HAND, FOOTMAN, 1);
    let s2 = s.apply(Action::TacFootman { coin: FOOTMAN });
    match s2.pending() {
        Cont::FootmanManeuver { hexes } => {
            assert_eq!(
                hexes.to_vec().as_slice(),
                &[W1 as u8, CENTER as u8],
                "both footman versions owe a maneuver"
            );
        }
        other => panic!("expected FootmanManeuver, got {:?}", other),
    }
    let acts = s2.legal_actions();
    // maneuvers offered for BOTH hexes (player-chosen order).
    assert!(has(
        &acts,
        Action::FootMove {
            from: CENTER as u8,
            to: E1 as u8
        }
    ));
    assert!(has(
        &acts,
        Action::FootMove {
            from: W1 as u8,
            to: 16
        }
    ));
    // act with the V2 (second in hex order) first; the V1 is still owed.
    let s3 = s2.apply(Action::FootMove {
        from: W1 as u8,
        to: 16,
    });
    match s3.pending() {
        Cont::FootmanManeuver { hexes } => assert_eq!(hexes.to_vec().as_slice(), &[CENTER as u8]),
        other => panic!(
            "expected FootmanManeuver for the remaining footman, got {:?}",
            other
        ),
    }
}

#[test]
fn berserker_chain_attack_uses_post_payment_height_for_knight_immunity() {
    // Chain cost is discarded BEFORE the chained maneuver: a height-2
    // Berserker's chained attack is made at height 1 and cannot hit a Knight.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, BERSERKER, 2);
    s.set_unit(19, BLACK, KNIGHT, 1); // E1 (4,3); hex 12 (4,2) is adjacent to it
    s.add_zone(WHITE, Z_HAND, BERSERKER, 1);
    let s2 = s.apply(Action::Move {
        from: CENTER as u8,
        to: 12,
    });
    assert!(matches!(s2.pending(), Cont::BerserkerChain { .. }));
    let acts = s2.legal_actions();
    assert!(
        !has(
            &acts,
            Action::BerserkAttack {
                from: 12,
                target: 19
            }
        ),
        "height-2 chain attack resolves at height 1: Knight is immune"
    );
    // At height 3 the chained attack (height 2 after payment) is legal.
    let mut t = State::blank(WHITE);
    t.set_unit(CENTER, WHITE, BERSERKER, 3);
    t.set_unit(19, BLACK, KNIGHT, 1);
    t.add_zone(WHITE, Z_HAND, BERSERKER, 1);
    let t2 = t.apply(Action::Move {
        from: CENTER as u8,
        to: 12,
    });
    assert!(has(
        &t2.legal_actions(),
        Action::BerserkAttack {
            from: 12,
            target: 19
        }
    ));
}

#[test]
fn warrior_priest_draw_resolves_before_rg_soak_choice() {
    // WP attacks a Royal Guard whose defender can soak from supply: the WP
    // draw happens BEFORE the defender's choice; the forced play after it.
    let mut s = State::blank(WHITE);
    s.set_unit(CENTER, WHITE, WARRIOR_PRIEST, 1);
    s.set_unit(E1, BLACK, ROYAL_GUARD, 1);
    s.add_zone(BLACK, Z_SUPPLY, ROYAL_GUARD, 1);
    s.add_zone(WHITE, Z_HAND, WARRIOR_PRIEST, 1);
    s.add_zone(WHITE, Z_BAG, ARCHER, 1);
    let s2 = s.apply(Action::Attack {
        from: CENTER as u8,
        target: E1 as u8,
    });
    match s2.pending() {
        Cont::WarriorPriestDraw { player, rg_hex } => {
            assert_eq!(*player, WHITE);
            assert_eq!(*rg_hex, E1 as u8);
        }
        other => panic!(
            "expected WarriorPriestDraw before the soak, got {:?}",
            other
        ),
    }
    let s3 = s2.apply(Action::DrawCoin { unit: ARCHER });
    assert!(
        matches!(s3.pending(), Cont::RoyalGuardChoice { .. }),
        "defender's soak choice comes after the draw"
    );
    let s4 = s3.apply(Action::RGSoakSupply);
    assert_eq!(s4.zones[BLACK as usize][Z_SUPPLY][ROYAL_GUARD as usize], 0);
    assert_eq!(s4.zones[BLACK as usize][Z_ELIM][ROYAL_GUARD as usize], 1);
    assert_eq!(s4.hex_height[E1], 1, "stack untouched after a supply soak");
    match s4.pending() {
        Cont::WarriorPriestPlay { coin, .. } => assert_eq!(*coin, ARCHER),
        other => panic!("expected the forced play after the soak, got {:?}", other),
    }
}

#[test]
fn ensign_cannot_move_itself() {
    // 0 of 8,999 Ensign-granted moves in the census move the Ensign itself.
    let mut s = one_unit(ENSIGN, CENTER);
    s.set_unit(W1, WHITE, SWORDSMAN, 1);
    let acts = s.legal_actions();
    assert_eq!(
        count_kind(
            &acts,
            |a| matches!(a, Action::TacEnsign { from, .. } if *from == CENTER as u8)
        ),
        0,
        "Ensign must not grant itself a move"
    );
    // ...but it still grants moves to other friendly units within 2.
    assert!(
        count_kind(
            &acts,
            |a| matches!(a, Action::TacEnsign { from, .. } if *from == W1 as u8)
        ) > 0
    );
}

#[test]
fn marshal_cannot_grant_itself_an_attack() {
    // 0 of 3,490 Marshal-granted attacks in the census are by the Marshal.
    let mut s = one_unit(MARSHAL, CENTER);
    s.set_unit(E1, BLACK, SWORDSMAN, 1); // adjacent enemy
    s.set_unit(W1, WHITE, PIKEMAN, 1);
    s.set_unit(16, BLACK, SCOUT, 1); // adjacent to W1 (1,3 next to 2,3)
    let acts = s.legal_actions();
    assert_eq!(
        count_kind(
            &acts,
            |a| matches!(a, Action::TacMarshal { unit_hex, .. } if *unit_hex == CENTER as u8)
        ),
        0,
        "Marshal must not direct itself"
    );
    assert!(has(
        &acts,
        Action::TacMarshal {
            unit_hex: W1 as u8,
            target: 16
        }
    ));
    // its own plain Attack is unaffected.
    assert!(has(
        &acts,
        Action::Attack {
            from: CENTER as u8,
            target: E1 as u8
        }
    ));
}
