//! Invariant test: many seeded random playouts, checking conservation and
//! structural invariants after every action.

use warchest::board::{board, NONE, N_HEXES};
use warchest::rng::Rng;
use warchest::state::{State, BLACK, WHITE};
use warchest::units::{def, N_UNITS};

const POOL: [u16; 19] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 52, 53, 54,
];

fn random_draft(rng: &mut Rng) -> (Vec<u16>, Vec<u16>, u8) {
    let pick4 = |rng: &mut Rng| -> Vec<u16> {
        let mut chosen: Vec<u16> = Vec::new();
        while chosen.len() < 4 {
            let id = POOL[rng.below(POOL.len())];
            if !chosen.contains(&id) {
                chosen.push(id);
            }
        }
        chosen
    };
    let w = pick4(rng);
    let b = pick4(rng);
    let first = if rng.next_u64() & 1 == 0 {
        WHITE
    } else {
        BLACK
    };
    (w, b, first)
}

/// Per-type total coins each player owns at setup, derived from the draft.
fn initial_totals(units: &[u16]) -> [u16; N_UNITS] {
    let mut t = [0u16; N_UNITS];
    for &id in units {
        let u = warchest::units::index_of_id(id).unwrap();
        t[u as usize] += def(u).coins as u16;
    }
    // one Royal Coin
    t[warchest::units::ROYAL_COIN as usize] += 1;
    t
}

fn check_invariants(s: &State, init: &[[u16; N_UNITS]; 2]) {
    // 1. coin conservation per type per player.
    for p in 0..2u8 {
        for u in 0..N_UNITS {
            let got = s.total_coins(p, u) as u16;
            assert_eq!(
                got, init[p as usize][u],
                "coin conservation broke: player {} unit {} got {} want {}",
                p, u, got, init[p as usize][u]
            );
        }
    }
    // 2. board structure.
    let b = board();
    for h in 0..N_HEXES {
        if s.hex_type[h] != NONE {
            assert!(s.hex_owner[h] == WHITE || s.hex_owner[h] == BLACK);
            assert!(s.hex_height[h] >= 1, "deployed stack must be >= 1");
        } else {
            assert_eq!(s.hex_height[h], 0);
            assert_eq!(s.hex_owner[h], NONE);
        }
        // markers only on locations.
        if s.loc_marker[h] != NONE {
            assert!(b.is_location[h], "marker on a non-location hex");
        }
    }
    // 3. markers 0..6 and hand+board == 6.
    for p in 0..2u8 {
        let on = s.markers_on_board(p);
        assert!(on <= 6);
        assert_eq!(on + s.markers_hand[p as usize], 6, "marker count broke");
    }
    // 4. winner consistency.
    if let Some(w) = s.winner() {
        assert_eq!(s.markers_on_board(w), 6, "winner must have 6 markers");
    }
    // 5. the horizon payoff is zero-sum and strictly inside +/-1.
    if s.is_terminal() && s.winner().is_none() {
        let (a, b) = (s.utility(0), s.utility(1));
        assert!((a + b).abs() < 1e-6, "utility must be zero-sum");
        assert!(a.abs() < 1.0, "horizon payoff must stay inside +/-1");
    }
}

#[test]
fn invariants_random_playouts() {
    let n_games = 10_000;
    let cap = 1000u32;
    let mut rng = Rng::new(0xA11CE);
    let mut cap_hits = 0u32;
    let mut lengths_sum = 0u64;
    let mut winners = [0u32; 3]; // white, black, none(cap)

    for _ in 0..n_games {
        let (w, b, first) = random_draft(&mut rng);
        let init = [initial_totals(&w), initial_totals(&b)];
        let mut s = State::from_draft(&w, &b, first);
        check_invariants(&s, &init);

        let mut applies = 0u32;
        while !s.is_terminal() && applies < cap {
            let acts = s.legal_actions();
            assert!(
                !acts.is_empty(),
                "no legal actions at a non-terminal state: {}",
                s.pending_debug()
            );
            let a = acts[rng.below(acts.len())];
            s.apply_inplace(a);
            check_invariants(&s, &init);
            applies += 1;
        }
        lengths_sum += applies as u64;
        if s.is_terminal() {
            match s.winner() {
                Some(winner) => winners[winner as usize] += 1,
                None => winners[2] += 1,
            }
        } else {
            cap_hits += 1;
            winners[2] += 1;
        }
    }

    eprintln!(
        "invariants: {} games, avg len {:.1}, cap hits {}, winners W={} B={} none={}",
        n_games,
        lengths_sum as f64 / n_games as f64,
        cap_hits,
        winners[0],
        winners[1],
        winners[2],
    );
    // Report (not assert) cap hits per spec; but flag loudly if many.
    if cap_hits > 0 {
        eprintln!("WARNING: {} playouts hit the 1000-action cap", cap_hits);
    }
}
