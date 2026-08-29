use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::actions::Action;
use crate::rng::Rng;
use crate::selfplay::resolve_chance;
use crate::state::{State, BLACK, WHITE};
use crate::units::index_of_id;

pub const PROTOCOL: u32 = 6;

#[derive(Serialize, Deserialize, Debug)]
pub struct Hello {
    pub name: String,
    pub protocol: u32,
    pub rules: u64,
}

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

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Obs {
    Draw {
        player: u8,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<u32>,
    },
    Act {
        player: u8,
        key: u32,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Start {
    pub draft: Draft,
    pub seat: u8,
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Ask {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<Start>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obs: Vec<Obs>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Request {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub go: Vec<Ask>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub watch: Vec<Ask>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Done {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Reply {
    #[serde(default)]
    pub done: Vec<Done>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

struct Bout {
    s: State,
    rng: Rng,
    bots: [usize; 2],
    draft: Draft,
    pending: [Vec<Obs>; 2],
    start: [Option<Start>; 2],
    asked: [bool; 2],
    watched: [bool; 2],
    over: bool,
}

#[derive(Default)]
pub struct Table {
    bouts: BTreeMap<u32, Bout>,
    queued: Vec<(u32, State, Draft)>,
    done: Vec<(u32, [usize; 2], f32)>,
    dropped: BTreeMap<usize, Vec<u32>>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    pub fn live(&self) -> usize {
        self.bouts.values().filter(|b| !b.over).count()
    }

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
                fresh.push((id, b.s, b.draft.clone()));
            }
        }
        self.queued.append(&mut fresh);
        self.done.append(&mut ended);
        self.retire();
    }

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

    pub fn play(&mut self, id: u32, code: u32) -> Result<(), String> {
        let b = self
            .bouts
            .get_mut(&id)
            .ok_or_else(|| format!("game {} is not live", id))?;
        let action =
            Action::decode(code).ok_or_else(|| format!("action {} does not decode", code))?;
        if !b.s.legal_actions().iter().any(|x| x.encode() == code) {
            return Err(format!("game {id}: illegal action {action:?}"));
        }
        let player = b.s.to_act();
        b.s.apply_inplace(action);
        b.watched = [false, false];
        let reached =
            (!b.s.is_terminal() && !b.s.is_chance()).then_some((id, b.s, b.draft.clone()));
        b.pending[1 - player as usize].push(Obs::Act {
            player,
            key: crate::pbs::obs_key(&action),
        });
        self.queued.extend(reached);
        Ok(())
    }

    pub fn reap(&mut self) -> Vec<(u32, [usize; 2], f32)> {
        std::mem::take(&mut self.done)
    }

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
}
