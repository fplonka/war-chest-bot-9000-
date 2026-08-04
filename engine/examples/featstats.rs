//! Measured ranges of the scalar features, so their divisors are the true
//! maxima rather than estimates.
//!
//! Two of the encoding's scalars were previously normalised by guessed bounds
//! and saturated in exactly the late-game states that matter (see the
//! face-down / bag note in `rebel.rs`). This probe exists so the remaining
//! divisors are chosen from data. It reports the distribution of stack height
//! and round number under both random and one-ply-greedy play on the starter
//! draft.
//!
//! `cargo run --release --example featstats`

use warchest::rng::Rng;
use warchest::selfplay::{eval_static, make_game};
use warchest::state::{Cont, State};

/// One-ply greedy on the public evaluation, matching the reference bot's
/// shape closely enough for a range probe. Chance nodes are resolved uniformly
/// over the listed draws, which is not the true draw distribution but does not
/// bias a maximum.
fn step_greedy(s: &mut State, rng: &mut Rng, greedy: bool) {
    let acts = s.legal_actions();
    if acts.is_empty() {
        return;
    }
    if !greedy || matches!(s.pending(), Cont::Draw { .. }) {
        s.apply_inplace(acts[rng.below(acts.len())]);
        return;
    }
    let p = s.to_act();
    let mut best = f32::NEG_INFINITY;
    let mut pick = acts[0];
    for a in &acts {
        let mut t = *s;
        t.apply_inplace(*a);
        let v = eval_static(&t, p);
        if v > best {
            best = v;
            pick = *a;
        }
    }
    s.apply_inplace(pick);
}

struct Stats {
    max_round: u16,
    max_height: u8,
    height_hist: [u64; 8],
    rounds_over_40: u64,
    heights_over_3: u64,
    /// Games that ran out the step budget without reaching a terminal state.
    stalled: u64,
    /// `main_plays` of a stalled game: if this is far below the cap while
    /// `round` keeps climbing, the game is advancing through empty rounds that
    /// consume no coin play, and the horizon can never be reached.
    stalled_main_plays: Vec<u16>,
    stalled_round: Vec<u16>,
}

fn run(games: u64, greedy: bool, budget: u32) -> Stats {
    let mut st = Stats {
        max_round: 0,
        max_height: 0,
        height_hist: [0; 8],
        rounds_over_40: 0,
        heights_over_3: 0,
        stalled: 0,
        stalled_main_plays: Vec::new(),
        stalled_round: Vec::new(),
    };
    for g in 0..games {
        let mut rng = Rng::new(g * 2_654_435_761 + 7);
        let mut s = make_game(&mut Rng::new(g + 1), false);
        let mut steps = 0u32;
        while !s.is_terminal() && steps < budget {
            step_greedy(&mut s, &mut rng, greedy);
            steps += 1;
            st.max_round = st.max_round.max(s.round);
            for h in 0..warchest::board::N_HEXES {
                let ht = s.hex_height[h];
                if ht > 0 {
                    st.max_height = st.max_height.max(ht);
                    st.height_hist[(ht as usize).min(7)] += 1;
                    if ht > 3 {
                        st.heights_over_3 += 1;
                    }
                }
            }
        }
        if !s.is_terminal() {
            st.stalled += 1;
            if st.stalled_main_plays.len() < 8 {
                st.stalled_main_plays.push(s.main_plays);
                st.stalled_round.push(s.round);
            }
        }
        if s.round > 40 {
            st.rounds_over_40 += 1;
        }
    }
    st
}

fn main() {
    let budget: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000);
    for (name, greedy) in [("random", false), ("greedy", true)] {
        let games = if greedy { 200 } else { 2000 };
        let st = run(games, greedy, budget);
        let tot: u64 = st.height_hist.iter().sum();
        println!("--- {name} play, {games} games (starter draft), step budget {budget} ---");
        println!(
            "  max round        {}   (games ending past round 40: {}/{games})",
            st.max_round, st.rounds_over_40
        );
        println!(
            "  max stack height {}   (occupied-hex observations with height > 3: {})",
            st.max_height, st.heights_over_3
        );
        print!("  height histogram ");
        for (h, n) in st.height_hist.iter().enumerate().skip(1) {
            if *n > 0 {
                print!("{h}:{:.3}% ", 100.0 * *n as f64 / tot as f64);
            }
        }
        println!();
        println!(
            "  non-terminal at budget: {}/{games}   main_plays {:?} round {:?}",
            st.stalled, st.stalled_main_plays, st.stalled_round
        );
    }
}
