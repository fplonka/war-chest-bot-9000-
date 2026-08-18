//! The arena: a referee, and the protocol it speaks to bots.
//!
//! The referee owns the true position and the dice. A bot owns nothing but its
//! own head. Between them passes only public information — the same
//! information a spectator would have, plus whatever is private to the bot
//! being spoken to. A bot therefore cannot read its opponent's hand, and two
//! bots built from different engine revisions need agree on nothing except
//! this protocol and the rules.
//!
//! The two streams are independent. The referee sends work for any game it is
//! not already waiting on, and the bot answers each game as that game is ready
//! rather than a batch at a time. Nothing waits for the slowest game in a
//! batch, so a bot always has trees to build while the device is busy and
//! solves to run while the cores are — which is where a ladder's throughput
//! comes from.
//!
//! ```text
//! referee -> {"go":[{"id":7,"start":{...},"obs":[...]}],"drop":[3]}
//! bot     -> {"done":[{"id":7,"action":8452}]}
//! ```

use std::collections::BTreeMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::actions::Action;
use crate::rebel::{reserve, true_config, Config, Ctx, NSLOT};
use crate::rng::Rng;
use crate::selfplay::resolve_chance;
use crate::rebel::Belief;
use crate::state::{
    Cont, ContStack, State, BLACK, N_ZONES, WHITE, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_HAND,
    Z_SUPPLY,
};
use crate::board::{NONE, N_HEXES};
use crate::units::{N_UNITS, UNITS};
use crate::units::index_of_id;

/// Bumped whenever a message changes shape. A bot announces the version it
/// was built against and the referee refuses to play a mismatch, because a
/// frozen binary can never be taught a new one.
pub const PROTOCOL: u32 = 6;

/// A bot's first line on stdout, before it reads anything.
#[derive(Serialize, Deserialize, Debug)]
pub struct Hello {
    pub name: String,
    pub protocol: u32,
    /// `rebel::rules_table_hash`. Two builds that disagree here are playing
    /// different games, and any result between them would be meaningless.
    pub rules: u64,
}

/// The armies and who moves first. Public to both seats from the start.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Draft {
    pub white: Vec<u16>,
    pub black: Vec<u16>,
    pub first: u8,
}

impl Draft {
    pub fn state(&self) -> Result<State, String> {
        for (side, ids) in [("white", &self.white), ("black", &self.black)] {
            if ids.len() != 4 {
                return Err(format!("{} drafts 4 units, not {}", side, ids.len()));
            }
            if let Some(&id) = ids.iter().find(|&&id| index_of_id(id).is_none()) {
                return Err(format!("unitTypeId {} is out of scope", id));
            }
        }
        if self.first != WHITE && self.first != BLACK {
            return Err(format!("first must be {} or {}", WHITE, BLACK));
        }
        Ok(State::from_draft(&self.white, &self.black, self.first))
    }
}

/// Something that happened, as one seat saw it.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Obs {
    /// A player drew a coin. `code` is the draw itself and is present only in
    /// the drawing player's own stream: everyone else learns that a draw
    /// happened, which moves the public hand count, and nothing more.
    Draw {
        player: u8,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<u32>,
    },
    /// A player acted. `key` is the action's public observation when the
    /// message goes to the opponent — a face-down play hides the coin behind
    /// it — and the whole action when it goes back to the actor, which happens
    /// when a recorded game is replayed and a seat has to re-make its own
    /// moves to rebuild the beliefs it played under.
    Act { player: u8, key: u32 },
}

/// A position in the language of the game rather than of this build's memory.
///
/// Units are named by the id the rules give them and hexes by index, so a bot
/// compiled against a different internal state reads one and fills in its own.
/// The alternative — writing this build's `State` out field by field — is a
/// memcpy with extra steps, and it silently stops meaning anything the moment
/// a zone is added.
///
/// It carries exactly what the player to move can see: the board, the public
/// counts, and its own coins. The other side's hand is a *size*, not a list,
/// because which of its unseen coins sit in hand rather than bag is precisely
/// the hidden thing — and the range of hands consistent with that size is
/// something every correct bot derives for itself from the same public facts.
///
/// A *game* never carries one of these: forming a belief is the bot's own job.
/// A *benchmark question* is a position by definition, and has to mean the
/// same thing to every bot that answers it.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub draft: Draft,
    /// `[hex, unit id, owner, height]` for each occupied hex.
    pub units: Vec<[u16; 4]>,
    /// `[hex, owner]` for each control marker.
    pub marks: Vec<[u16; 2]>,
    /// `[player, zone, unit id, count]` for every coin off the board, the
    /// zone being one of the codes below.
    pub coins: Vec<[u16; 4]>,
    /// The hands each player may be holding, as the public record has it,
    /// each hand a list of `[unit id, count]`. Both sides, not just the
    /// hidden one: the player to move knows its own hand, but its *opponent*
    /// does not, and a solve that was told otherwise would be solving a game
    /// where one side plays with its coins face up.
    ///
    /// Carried rather than derived, because a question has to be quantified
    /// over exactly the set the proof was.
    pub ranges: [Vec<Vec<[u16; 2]>>; 2],
    pub markers: [u8; 2],
    pub initiative: u8,
    pub moved: bool,
    pub round: u16,
    pub first: u8,
    pub active: u8,
    pub turns: [u8; 2],
    pub plays: u16,
    pub priest: bool,
}

/// Zone codes this notation defines, and where they live in *this* build.
/// They are deliberately not the engine's own numbering: a build numbers its
/// zones however it likes, and mapping these onto them is the reader's job.
const ZONES: [(u16, usize); 6] = [
    (0, Z_BAG),
    (1, Z_HAND),
    (2, Z_FACEUP),
    (3, Z_FACEDOWN),
    (4, Z_SUPPLY),
    (5, Z_ELIM),
];

impl Position {
    /// The position as the player to move sees it. Only meaningful at a coin
    /// play: mid-turn there is a continuation stack to be in the middle of,
    /// and a benchmark question has no business being half way through a
    /// maneuver.
    pub fn of(s: &State, draft: &Draft, ranges: [&[Config]; 2]) -> Result<Position, String> {
        if !matches!(s.pending, Cont::MainPlay) || s.conts.len() > 0 {
            return Err("a position is written at a coin play, not mid-turn".into());
        }
        let mover = s.to_act();
        let ctx = Ctx::new(s);
        let mut units = Vec::new();
        let mut marks = Vec::new();
        for h in 0..N_HEXES {
            if s.hex_type[h] != NONE {
                units.push([
                    h as u16,
                    UNITS[s.hex_type[h] as usize].id,
                    s.hex_owner[h] as u16,
                    s.hex_height[h] as u16,
                ]);
            }
            if s.loc_marker[h] != NONE {
                marks.push([h as u16, s.loc_marker[h] as u16]);
            }
        }
        let mut coins = Vec::new();
        for p in 0..2usize {
            for (code, z) in ZONES {
                for u in 0..N_UNITS {
                    // The side not to move reports bag and hand as one pool.
                    let n = match (p as u8 != mover, z) {
                        (true, Z_BAG) => s.zones[p][Z_BAG][u] + s.zones[p][Z_HAND][u],
                        (true, Z_HAND) => 0,
                        _ => s.zones[p][z][u],
                    };
                    if n > 0 {
                        coins.push([p as u16, code, UNITS[u].id, n as u16]);
                    }
                }
            }
        }
        Ok(Position {
            draft: draft.clone(),
            units,
            marks,
            coins,
            ranges: [0, 1].map(|p| {
                ranges[p]
                    .iter()
                    .map(|c| {
                        (0..NSLOT)
                            .filter(|&k| c.hand[k] > 0)
                            .map(|k| [UNITS[ctx.slots[p][k] as usize].id, c.hand[k] as u16])
                            .collect()
                    })
                    .collect()
            }),
            markers: s.markers_hand,
            initiative: s.initiative,
            moved: s.initiative_moved,
            round: s.round,
            first: s.first_player,
            active: s.active,
            turns: s.turns_taken,
            plays: s.main_plays,
            priest: s.wp_v2_triggered,
        })
    }

    /// The state this describes, and the ranges it implies: a point mass on
    /// the mover's own coins, and for the other side every hand its public
    /// counts allow. Which hand the carrier state happens to hold is
    /// arbitrary and never read — the proof quantifies over all of them, and
    /// the bot is given the range, not the truth.
    pub fn state(&self) -> Result<(State, [Belief; 2]), String> {
        let mut s = self.draft.state()?;
        for p in 0..2 {
            for z in 0..N_ZONES {
                s.zones[p][z] = [0; N_UNITS];
            }
        }
        for &[p, code, id, n] in &self.coins {
            let (_, z) = ZONES
                .iter()
                .find(|&&(c, _)| c == code)
                .ok_or_else(|| format!("zone {} is not in this notation", code))?;
            let u = index_of_id(id).ok_or_else(|| format!("unitTypeId {} is out of scope", id))?;
            s.zones[p as usize][*z][u as usize] += n as u8;
        }
        s.hex_type = [NONE; N_HEXES];
        s.hex_owner = [NONE; N_HEXES];
        s.hex_height = [0; N_HEXES];
        s.loc_marker = [NONE; N_HEXES];
        for &[h, id, owner, height] in &self.units {
            let u = index_of_id(id).ok_or_else(|| format!("unitTypeId {} is out of scope", id))?;
            s.hex_type[h as usize] = u;
            s.hex_owner[h as usize] = owner as u8;
            s.hex_height[h as usize] = height as u8;
        }
        for &[h, owner] in &self.marks {
            s.loc_marker[h as usize] = owner as u8;
        }
        s.markers_hand = self.markers;
        s.initiative = self.initiative;
        s.initiative_moved = self.moved;
        s.round = self.round;
        s.first_player = self.first;
        s.active = self.active;
        s.turns_taken = self.turns;
        s.main_plays = self.plays;
        s.wp_v2_triggered = self.priest;
        s.winner = NONE;
        s.adjudicated_draw = false;
        s.interrupt = false;
        s.pending = Cont::MainPlay;
        s.conts = ContStack::default();

        // Deal the side not to move the first of the hands it might hold.
        // Which one the carrier state ends up with is arbitrary and never
        // read: the proof quantifies over all of them, and the bot is handed
        // the range rather than the truth.
        let them = 1 - s.to_act() as usize;
        for &[id, n] in self.ranges[them]
            .first()
            .ok_or("a position needs the range each side may hold")?
        {
            let u = index_of_id(id).ok_or_else(|| format!("unitTypeId {} is out of scope", id))?
                as usize;
            if s.zones[them][Z_BAG][u] < n as u8 {
                return Err("a hand asks for coins the pool does not have".into());
            }
            s.zones[them][Z_BAG][u] -= n as u8;
            s.zones[them][Z_HAND][u] += n as u8;
        }

        let ctx = Ctx::new(&s);
        let mut belief = [Belief::default(), Belief::default()];
        for p in 0..2 {
            let mut out = Vec::with_capacity(self.ranges[p].len());
            for hand in &self.ranges[p] {
                let mut c = true_config(&s, p as u8, &ctx);
                c.hand = [0; NSLOT];
                for &[id, n] in hand {
                    let u = index_of_id(id)
                        .ok_or_else(|| format!("unitTypeId {} is out of scope", id))?;
                    let k = (0..NSLOT)
                        .find(|&k| ctx.slots[p][k] == u)
                        .ok_or("a hand names a coin that side never drafted")?;
                    c.hand[k] = n as u8;
                }
                out.push((c, 1.0));
            }
            belief[p] = Belief::from_pairs(out);
        }
        Ok((s, belief))
    }
}

/// What a bot is told when a game first reaches it.
/// A position handed to a bot directly, rather than played into.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Placed {
    pub position: Position,
    pub seat: u8,
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Start {
    pub draft: Draft,
    pub seat: u8,
    pub seed: u64,
}

/// One game's move request.
#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Ask {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<Start>,
    /// A benchmark question: begin here, with these ranges, on this seat.
    /// Mutually exclusive with `start`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Placed>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obs: Vec<Obs>,
}

/// Unknown fields are refused rather than ignored. A bot binary is frozen at
/// the revision that trained it, so the only safe response to a message it was
/// not built to read is to stop: a bot that quietly skipped half a request
/// would still return moves, and they would be wrong.
#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// Games this bot must move in.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub go: Vec<Ask>,
    /// Games the *opponent* must move in. A bot models its opponent to keep a
    /// belief over their hand, and that model is a solve of the node they are
    /// sitting at — which does not need their move, only their position. Asked
    /// for here, it happens while the opponent is still thinking, so both bots
    /// work at once instead of taking turns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub watch: Vec<Ask>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<u32>,
}

/// One game brought up to date. `action` is present exactly when the ask was a
/// `go`; a `watch` is acknowledged without one, and the referee needs the
/// acknowledgement so it knows the bot is ready for what happens next.
#[derive(Serialize, Deserialize, Debug)]
pub struct Done {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<u32>,
}

/// Games the bot has finished with. A reply need not cover a whole request:
/// the bot sends what is ready, and the referee sends more work for those
/// games as soon as it arrives.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Reply {
    #[serde(default)]
    pub done: Vec<Done>,
    /// Set instead of `done` when the bot could not answer. The referee
    /// abandons the run: a bot that cannot follow the game produces no result
    /// worth having.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// In how few plies is the result of `s` settled, whatever anyone does?
///
/// This is the whole definition of an exact benchmark position. The game as a
/// whole cannot be searched — the public tree multiplies by about twenty per
/// ply and a game runs for hundreds — but a position whose result is *forced*
/// does not need the game searched, only the forced part of it. Every line
/// must end, and end the same way.
///
/// A chance node and an opponent's turn are both branches that must *all* win,
/// and the slowest of them sets the distance; only the winner's own turn needs
/// one that does, and the fastest of those sets it.
///
/// Returning the distance rather than a yes or no is what lets one search
/// replace the ladder of searches that asking "settled in two? in three? in
/// four?" would otherwise need, and it pays for itself twice over: a win found
/// at some distance tightens the cap on every branch still to be tried, so the
/// deeper half of the tree is never built.
///
/// `budget` bounds the work and makes a position that is merely large report
/// as unproven rather than hang. No value network appears anywhere here, which
/// is the point: the answer is the game's, not a model's, so it cannot flatter
/// the architecture that happened to produce it.
pub fn distance(s: &State, winner: u8, cap: usize, budget: &mut usize) -> Option<usize> {
    // One action buffer per level, reused for every node at that level. The
    // search visits millions of nodes and a vector allocated at each of them
    // costs more than the rules do.
    let mut scratch = vec![Vec::new(); cap + 1];
    walk(s, winner, cap, budget, &mut scratch)
}

fn walk(
    s: &State,
    winner: u8,
    cap: usize,
    budget: &mut usize,
    scratch: &mut [Vec<Action>],
) -> Option<usize> {
    if s.is_terminal() {
        return (s.utility(winner as usize) > 0.0).then_some(0);
    }
    if cap == 0 || *budget == 0 {
        return None;
    }
    *budget -= 1;
    let (acts, deeper) = scratch.split_first_mut().expect("a level per ply of cap");
    s.legal_actions_into(acts);
    if acts.is_empty() {
        return None;
    }
    let ours = !s.is_chance() && s.to_act() == winner;
    // Ours: the best distance found so far, and nothing slower is worth
    // building. Theirs: the worst found so far, and every branch must answer.
    let mut best: Option<usize> = if ours { None } else { Some(0) };
    for i in 0..acts.len() {
        // Nothing beats a win on the move, so stop looking for one.
        if ours && best == Some(1) {
            break;
        }
        // A branch is only worth building if it could beat the best distance
        // found so far, which is what makes this cheaper than asking the same
        // question once per depth.
        let room = if ours {
            best.map_or(cap, |b| b - 1) - 1
        } else {
            cap - 1
        };
        let mut next = *s;
        next.apply_inplace(acts[i]);
        match (ours, walk(&next, winner, room, budget, deeper)) {
            (true, Some(d)) => best = Some(d + 1),
            (true, None) => {}
            (false, Some(d)) => best = Some(best.unwrap().max(d + 1)),
            (false, None) => return None,
        }
    }
    best
}

/// Every hand the opponent could hold, given only what is public.
fn opponent_range(s: &State, them: u8, ctx: &Ctx) -> Vec<Config> {
    fn walk(
        slot: usize,
        left: u8,
        hand: &mut [u8; NSLOT],
        res: &[u8; NSLOT],
        truth: &Config,
        out: &mut Vec<Config>,
    ) {
        if slot == NSLOT {
            if left == 0 {
                out.push(Config {
                    hand: *hand,
                    fd: truth.fd,
                    inflight: truth.inflight,
                });
            }
            return;
        }
        // A hand cannot hold more of a coin than the reserve still has once
        // the face-down pile and any in-flight coin are accounted for.
        let spare = res[slot]
            .saturating_sub(truth.fd[slot] + u8::from(truth.inflight == Some(slot as u8)));
        for n in 0..=left.min(spare) {
            hand[slot] = n;
            walk(slot + 1, left - n, hand, res, truth, out);
        }
        hand[slot] = 0;
    }
    let truth = true_config(s, them, ctx);
    let res = reserve(s, them, ctx);
    let mut out = Vec::new();
    walk(0, truth.hand_size(), &mut [0; NSLOT], &res, &truth, &mut out);
    out
}

/// Can `winner` force a win from here within `depth`, against *every* hand
/// the loser could be holding?
///
/// Quantifying over the range is the whole claim. Proving it against the
/// hand they actually hold would be proving something about a game nobody
/// is playing: the winner cannot see that hand, so a plan that only works
/// against it is not forced.
///
/// The distance returned is therefore the *worst* over the range: the
/// number of plies that holds against every hand at once.
pub fn settled(s: &State, winner: u8, cap: usize, budget: usize) -> Option<usize> {
    // Against the hand they actually hold first: one cheap search, and it
    // fails for almost every position, so the range is rarely walked.
    let mut left = budget;
    distance(s, winner, cap, &mut left)?;
    let ctx = Ctx::new(s);
    opponent_range(s, 1 - winner, &ctx)
        .par_iter()
        .map(|c| {
            let mut probe = *s;
            crate::rebel::set_config(&mut probe, 1 - winner, &ctx, c);
            let mut left = budget;
            distance(&probe, winner, cap, &mut left)
        })
        .try_reduce(|| 0, |a, b| Some(a.max(b)))
}


/// How many of the legal moves here keep a win that is `plies` away, and how
/// many there are. Sharpness — how many of the moves are right — is the axis
/// that actually separates bots, far more than how deep the win is.
fn sharpness(s: &State, winner: u8, plies: usize, budget: usize) -> (usize, usize) {
    let acts = s.legal_actions();
    let wins = acts
        .par_iter()
        .filter(|a| {
            let mut next = *s;
            next.apply_inplace(**a);
            let mut left = budget;
            distance(&next, winner, plies.saturating_sub(1), &mut left).is_some()
        })
        .count();
    (wins, acts.len())
}

/// A proven position, ready to become a benchmark question.
pub struct Question {
    pub id: u32,
    pub winner: u8,
    /// Plies to the win, against every hand the loser could hold.
    pub plies: usize,
    /// How many of the legal moves keep the win, out of how many there are.
    pub wins: usize,
    pub moves: usize,
    /// How many hands the loser could hold. One means the position has no
    /// hidden information left in it.
    pub range: usize,
    pub position: Position,
}

/// The referee's view of one game.
struct Bout {
    s: State,
    rng: Rng,
    /// Which bot plays which seat.
    bots: [usize; 2],
    /// The armies, so a position can name its units.
    draft: Draft,
    /// What each seat has not been told yet.
    pending: [Vec<Obs>; 2],
    start: [Option<Start>; 2],
    /// Whether each seat has an ask outstanding. One at a time per seat keeps
    /// a game's observations in order without the bot having to sequence them.
    asked: [bool; 2],
    /// Whether each seat has already modelled the position it is looking at.
    /// A watch is a solve of a node, so asking for it twice over would buy
    /// nothing; the flags clear as soon as the position moves.
    watched: [bool; 2],
    /// The game is decided. It is kept until both seats have answered, because
    /// telling a bot to forget a game it is still thinking about would pull the
    /// position out from under it.
    over: bool,
}

/// Every game in flight. The referee resolves chance itself and hands each bot
/// only the games it must move in.
#[derive(Default)]
pub struct Table {
    bouts: BTreeMap<u32, Bout>,
    /// Every decision node the games have passed through since the last
    /// sweep. Proving one is expensive and proving many is the same work
    /// spread over every core, so positions are collected and swept in bulk
    /// rather than examined one at a time as they arrive.
    queued: Vec<(u32, State, Draft)>,
    done: Vec<(u32, [usize; 2], f32)>,
    dropped: BTreeMap<usize, Vec<u32>>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// Let `bot` see the strategy its opponents play, which makes every result
    /// it produces a measurement rather than a game.

    /// Games still being played. A decided game waiting for its last ack does
    /// not count.
    pub fn live(&self) -> usize {
        self.bouts.values().filter(|b| !b.over).count()
    }

    /// Seat a new game. `bots[seat]` is the bot that plays that seat.
    pub fn start(
        &mut self,
        id: u32,
        draft: &Draft,
        bots: [usize; 2],
        seed: u64,
    ) -> Result<(), String> {
        let s = draft.state()?;
        let start = |seat: u8| {
            Some(Start {
                draft: draft.clone(),
                seat,
                seed: seed ^ (seat as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            })
        };
        self.bouts.insert(
            id,
            Bout {
                s,
                rng: Rng::new(seed),
                bots,
                draft: draft.clone(),
                pending: [Vec::new(), Vec::new()],
                start: [start(WHITE), start(BLACK)],
                asked: [false, false],
                watched: [false, false],
                over: false,
            },
        );
        Ok(())
    }

    /// Seat a game at a benchmark position rather than at a draft. The
    /// referee needs the same position the bot was handed, so that it can say
    /// whether the win still stands after the bot has moved.
    pub fn start_at(&mut self, id: u32, at: &Position, bots: [usize; 2], seed: u64) -> Result<(), String> {
        let (state, _) = at.state()?;
        self.bouts.insert(
            id,
            Bout {
                s: state,
                rng: Rng::new(seed),
                bots,
                draft: at.draft.clone(),
                pending: [Vec::new(), Vec::new()],
                start: [None, None],
                asked: [false, false],
                watched: [false, false],
                over: false,
            },
        );
        Ok(())
    }

    /// Resolve every pending draw and retire every finished game, so that each
    /// surviving game sits at a decision belonging to one of the two bots.
    /// Draws are the referee's alone: only it knows what is in a bag.
    pub fn settle(&mut self) {
        let mut ended = Vec::new();
        let mut fresh = Vec::new();
        for (&id, b) in self.bouts.iter_mut() {
            if b.over {
                continue;
            }
            let mut drew = false;
            while !b.s.is_terminal() && b.s.is_chance() {
                drew = true;
                let player = b.s.to_act();
                let drawn = resolve_chance(&mut b.s, player, &mut b.rng);
                b.watched = [false, false];
                b.pending[player as usize].push(Obs::Draw {
                    player,
                    code: Some(drawn.encode()),
                });
                b.pending[1 - player as usize].push(Obs::Draw { player, code: None });
            }
            if b.s.is_terminal() {
                b.over = true;
                ended.push((id, b.bots, b.s.utility(WHITE as usize)));
            } else if drew {
                // Only a game that just moved is a new position to prove.
                fresh.push((id, b.s, b.draft.clone()));
            }
        }
        self.queued.append(&mut fresh);
        self.done.append(&mut ended);
        self.retire();
    }

    /// Forget the decided games both bots have finished answering about, and
    /// tell them to do the same.
    fn retire(&mut self) {
        let done: Vec<u32> = self
            .bouts
            .iter()
            .filter(|(_, b)| b.over && !b.asked[0] && !b.asked[1])
            .map(|(&id, _)| id)
            .collect();
        for id in done {
            let bout = self.bouts.remove(&id).expect("just listed");
            for bot in bout.bots {
                self.dropped.entry(bot).or_default().push(id);
            }
        }
    }

    /// Apply a bot's move to the true position and tell the other seat what it
    /// saw. An illegal move is an error, not a loss: it means the bot has lost
    /// track of the game.
    pub fn play(&mut self, id: u32, code: u32) -> Result<(), String> {
        let b = self
            .bouts
            .get_mut(&id)
            .ok_or_else(|| format!("game {} is not live", id))?;
        let action =
            Action::decode(code).ok_or_else(|| format!("action {} does not decode", code))?;
        if !b.s.legal_actions().iter().any(|x| x.encode() == code) {
            return Err(format!("game {}: illegal action {}", id, action));
        }
        let player = b.s.to_act();
        b.s.apply_inplace(action);
        b.watched = [false, false];
        // A move that leaves a decision node is a position to prove; one that
        // leaves a draw belongs to `settle`, which records it once resolved.
        let reached =
            (!b.s.is_terminal() && !b.s.is_chance()).then_some((id, b.s, b.draft.clone()));
        b.pending[1 - player as usize].push(Obs::Act {
            player,
            key: crate::rebel::obs_key(&action),
        });
        self.queued.extend(reached);
        Ok(())
    }

    /// Games that ended since the last call: the game id, its seating, and
    /// White's result in `-1..=1`.
    pub fn reap(&mut self) -> Vec<(u32, [usize; 2], f32)> {
        std::mem::take(&mut self.done)
    }

    pub fn forced(&self, id: u32, winner: u8, depth: usize, budget: usize) -> Result<bool, String> {
        let b = self
            .bouts
            .get(&id)
            .ok_or_else(|| format!("game {} is not live", id))?;
        Ok(settled(&b.s, winner, depth, budget).is_some())
    }

    /// Every live game whose result is already forced, proven and described.
    ///
    /// The whole sweep happens here rather than one call at a time from the
    /// caller, because the games are independent and this search is the only
    /// expensive thing in the program: driven from Python it ran on one core
    /// of a machine that has seventy, and held the interpreter lock while it
    /// did.
    ///
    /// Nothing happens until `batch` positions have piled up: a sweep of two
    /// or three has nothing to spread across the cores, and a position waits
    /// just as well in the queue as it would in the game. Pass zero to flush.
    pub fn harvest(
        &mut self,
        batch: usize,
        min_plies: usize,
        cap: usize,
        min_markers: u8,
        budget: usize,
    ) -> Vec<Question> {
        if self.queued.len() < batch {
            return Vec::new();
        }
        std::mem::take(&mut self.queued)
            .into_par_iter()
            .filter(|(_, s, _)| (0..2).any(|p| s.markers_on_board(p) >= min_markers))
            .filter_map(|(id, s, draft)| {
                let winner = s.to_act();
                let plies = settled(&s, winner, cap, budget)?;
                // A win already available in fewer plies is a different and
                // easier question, and `min_plies` exists to exclude it.
                if plies < min_plies {
                    return None;
                }
                let (wins, moves) = sharpness(&s, winner, plies, budget);
                let ctx = Ctx::new(&s);
                let range = opponent_range(&s, 1 - winner, &ctx);
                let ours = opponent_range(&s, winner, &ctx);
                let ranges = if winner == WHITE {
                    [&ours[..], &range[..]]
                } else {
                    [&range[..], &ours[..]]
                };
                Some(Question {
                    id,
                    winner,
                    plies,
                    wins,
                    moves,
                    range: range.len(),
                    position: Position::of(&s, &draft, ranges).ok()?,
                })
            })
            .collect()
    }

    /// Work for `bot` that is not already out with it: the games it must move
    /// in, the games it should be watching, and the games it may forget. A
    /// game whose next transition is a draw belongs to the referee, so it
    /// waits here for the next `settle`.
    pub fn request(&mut self, bot: usize) -> Request {
        let (mut go, mut watch) = (Vec::new(), Vec::new());
        for (&id, b) in self.bouts.iter_mut() {
            if b.s.is_terminal() || b.s.is_chance() {
                continue;
            }
            let Some(seat) = b.bots.iter().position(|&x| x == bot) else {
                continue;
            };
            let acting = seat == b.s.to_act() as usize;
            if b.asked[seat] || (!acting && b.watched[seat]) {
                continue;
            }
            b.asked[seat] = true;
            let ask = Ask {
                id,
                start: b.start[seat].take(),
                at: None,
                obs: std::mem::take(&mut b.pending[seat]),
            };
            if acting {
                go.push(ask);
            } else {
                b.watched[seat] = true;
                watch.push(ask);
            }
        }
        Request {
            go,
            watch,
            drop: self.dropped.remove(&bot).unwrap_or_default(),
        }
    }

    /// Apply what a bot has finished with. Each entry clears that game's ask,
    /// so the next `request` can carry it further.
    pub fn accept(&mut self, bot: usize, reply: Reply) -> Result<(), String> {
        if let Some(error) = reply.error {
            return Err(error);
        }
        for done in reply.done {
            if let Some(b) = self.bouts.get_mut(&done.id) {
                if let Some(seat) = b.bots.iter().position(|&x| x == bot) {
                    b.asked[seat] = false;
                }
            }
            if let Some(action) = done.action {
                self.play(done.id, action)?;
            }
        }
        self.retire();
        Ok(())
    }

    /// How many asks are out with the bots. The referee is finished only when
    /// this is zero and nothing is live.
    pub fn outstanding(&self) -> usize {
        self.bouts
            .values()
            .map(|b| usize::from(b.asked[0]) + usize::from(b.asked[1]))
            .sum()
    }
}

/// The referee as the ladder driver sees it. Both directions are the
/// protocol's own JSON, so the driver only moves lines between a pipe and this
/// object and never has to know what is in them.
#[cfg(feature = "python")]
#[pyo3::pyclass(name = "Table")]
pub struct PyTable(Table);

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PyTable {
    #[new]
    fn new() -> PyTable {
        PyTable(Table::new())
    }

    fn start(
        &mut self,
        id: u32,
        white: Vec<u16>,
        black: Vec<u16>,
        first: u8,
        bots: [usize; 2],
        seed: u64,
    ) -> pyo3::PyResult<()> {
        let draft = Draft {
            white,
            black,
            first,
        };
        self.0
            .start(id, &draft, bots, seed)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn settle(&mut self) {
        self.0.settle()
    }


    fn live(&self) -> usize {
        self.0.live()
    }

    /// One request line for `bot`, or `None` when it is owed nothing.
    fn request(&mut self, bot: usize) -> Option<String> {
        let request = self.0.request(bot);
        let idle = request.go.is_empty() && request.watch.is_empty() && request.drop.is_empty();
        (!idle).then(|| serde_json::to_string(&request).expect("request encodes"))
    }

    /// Seat a game at a benchmark position, given on the wire.
    fn start_at(&mut self, id: u32, position: &str, bots: [usize; 2], seed: u64) -> pyo3::PyResult<()> {
        let at: Position = serde_json::from_str(position)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.0
            .start_at(id, &at, bots, seed)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Every live game whose result is already forced: the game, who wins
    /// it, in how many plies, how many of the legal moves keep the win out of
    /// how many there are, how many hands the loser could hold, and the
    /// position itself.
    #[pyo3(signature = (batch, min_plies, cap, min_markers, budget=400_000))]
    fn harvest(
        &mut self,
        py: pyo3::Python<'_>,
        batch: usize,
        min_plies: usize,
        cap: usize,
        min_markers: u8,
        budget: usize,
    ) -> Vec<(u32, u8, usize, usize, usize, usize, String)> {
        py.allow_threads(|| {
            self.0
                .harvest(batch, min_plies, cap, min_markers, budget)
                .into_iter()
                .map(|q| {
                    let at = serde_json::to_string(&q.position)
                        .expect("a position always serialises");
                    (q.id, q.winner, q.plies, q.wins, q.moves, q.range, at)
                })
                .collect()
        })
    }

    /// Whether `winner` still has a forced win here.
    #[pyo3(signature = (id, winner, depth, budget=400_000))]
    fn forced(&self, id: u32, winner: u8, depth: usize, budget: usize) -> pyo3::PyResult<bool> {
        self.0
            .forced(id, winner, depth, budget)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Consume one reply line from `bot`.
    fn reply(&mut self, bot: usize, line: &str) -> pyo3::PyResult<()> {
        let reply: Reply = serde_json::from_str(line)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.0
            .accept(bot, reply)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Asks that are out with the bots and not yet answered.
    fn outstanding(&self) -> usize {
        self.0.outstanding()
    }

    /// Games that ended since the last call, as `(id, bot_white, bot_black,
    /// white_result)`.
    fn reap(&mut self) -> Vec<(u32, usize, usize, f32)> {
        self.0
            .reap()
            .into_iter()
            .map(|(id, bots, z)| (id, bots[0], bots[1], z))
            .collect()
    }

    /// The true position, and the actions available in it. This is the
    /// referee's own view — everything, including both hands — so a caller
    /// that shows it to one of the players must hide the other's private
    /// zones first.
    fn view(&self, py: pyo3::Python<'_>, id: u32) -> pyo3::PyResult<pyo3::PyObject> {
        use pyo3::prelude::*;
        use pyo3::types::{PyDict, PyList};
        let bout = self
            .0
            .bouts
            .get(&id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("no game {}", id)))?;
        let obj = crate::py::state_to_dict(py, &bout.s)?;
        let d = obj.downcast_bound::<PyDict>(py)?;
        let acts = PyList::empty_bound(py);
        for a in bout.s.legal_actions() {
            acts.append(crate::py::action_to_dict(py, &bout.s, &a)?)?;
        }
        d.set_item("actions", acts)?;
        Ok(d.clone().into())
    }
}
