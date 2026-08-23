//! An arena bot: one checkpoint, playing over stdin and stdout.
//!
//! This binary is the unit an experiment is archived as. Built once at the
//! revision that trained its weights and never rebuilt, it keeps playing after
//! the source that produced it has been rewritten — which is the only way to
//! compare an architecture with the one that replaced it.
//!
//! It reads requests from stdin and writes replies in the JSON
//! `warchest::arena` defines. All searches in one request run as one farm wave.
//!
//! ```text
//! bot --name v5-2h --weights weights.bin --s 512 --c 8 --cfr sog
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};

use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use warchest::args::Args;
use warchest::bot::{Brain, Mind, Session};
use warchest::farm::{Backend, Farm};
#[cfg(feature = "gpu")]
use warchest::farm::host_slots;
use warchest::net::Net;
use warchest::pbs::rules_table_hash;
use warchest::search::{Budget, Cfg, Cfr};

struct Options {
    name: String,
    weights: String,
    mind: String,
    s: u32,
    c: f32,
    batch: usize,
    rounds: u8,
    cfr: String,
    temp: f32,
    devices: String,
}

fn options() -> Result<Options, String> {
    let a = Args::parse(&[
        "name", "weights", "mind", "s", "c", "batch", "rounds", "cfr", "temp", "devices",
    ])?;
    Ok(Options {
        name: a.text("name", "bot"),
        weights: a.text("weights", ""),
        mind: a.text("mind", "sog"),
        cfr: a.text("cfr", "sog"),
        s: a.num("s", 512)?,
        c: a.num("c", 8.0)?,
        batch: a.num("batch", 8)?,
        rounds: a.num("rounds", 0)?,
        temp: a.num("temp", 2.0)?,
        devices: a.text("devices", ""),
    })
}

fn engine(o: &Options) -> Result<(Brain, Option<Farm>), String> {
    let mind = match o.mind.as_str() {
        "sog" => Mind::Sog,
        "random" => Mind::Random,
        "greedy" => Mind::Greedy { temp: o.temp },
        other => return Err(format!("unknown mind {}", other)),
    };
    let cfr = Cfr::named(&o.cfr).ok_or_else(|| format!("unknown cfr rule {}", o.cfr))?;
    let cfg = Cfg {
        s: o.s,
        c: o.c,
        batch: o.batch,
        rounds: o.rounds,
        cfr,
        budget: Budget::for_s(o.s),
        ..Default::default()
    };
    if !matches!(mind, Mind::Sog) {
        if !o.devices.is_empty() {
            return Err(format!("{} is CPU-only and cannot use --devices", o.mind));
        }
        return Ok((Brain { mind, nets: Default::default(), cfg }, None));
    }
    let net = Net::load_bin(&o.weights).map_err(|e| format!("{}: {}", o.weights, e))?;
    let backend = backend(o, net, cfg)?;
    let workers = std::thread::available_parallelism().map_or(8, |n| n.get());
    let farm = Farm::arena(workers, backend);
    let brain = Brain { mind, nets: farm.value(), cfg };
    Ok((brain, Some(farm)))
}

/// Every live game. The referee sends at most one ask for each game.
type Table = HashMap<u32, Session>;

/// Bring one game up to date for its next solve.
fn session(ask: &Ask, table: &mut Table, brain: &Brain) -> Result<Session, String> {
    let mut session = match (&ask.start, &ask.at) {
        (Some(start), None) => Session::new(&start.draft, start.seat, start.seed)?,
        (None, Some(at)) => {
            let (state, belief) = at.position.state()?;
            Session::at(state, belief, at.seat, at.seed)?
        }
        (Some(_), Some(_)) => return Err("a game starts one way or the other".into()),
        (None, None) => table
            .remove(&ask.id)
            .ok_or_else(|| format!("game {} was never started", ask.id))?,
    };
    for obs in &ask.obs {
        session.observe(obs, brain)?;
    }
    Ok(session)
}

/// A game the bot could not follow ends the run. Half a ladder from a bot that
/// lost the position is worse than no ladder.
fn fail(out: &mut impl Write, error: String) -> ! {
    let reply = Reply {
        done: Vec::new(),
        error: Some(error),
    };
    let _ = writeln!(out, "{}", serde_json::to_string(&reply).unwrap());
    let _ = out.flush();
    std::process::exit(1);
}

fn main() {
    let options = options().unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(2);
    });
    // Evaluation scores a game that hit the play horizon as a draw, so the
    // horizon marker is worth nothing to either side.
    warchest::state::set_cap_marker_value(0.0);
    let (brain, farm) = engine(&options).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(2);
    });

    let mut out = std::io::stdout();
    let hello = Hello {
        name: options.name.clone(),
        protocol: PROTOCOL,
        rules: rules_table_hash(),
    };
    {
        writeln!(out, "{}", serde_json::to_string(&hello).unwrap()).unwrap();
        out.flush().unwrap();
    }

    let mut table = Table::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(e) => fail(&mut out, e.to_string()),
        };
        for id in &request.drop {
            table.remove(id);
        }
        let asks: Vec<(Ask, bool)> = request
            .go
            .into_iter()
            .map(|a| (a, true))
            .chain(request.watch.into_iter().map(|a| (a, false)))
            .collect();
        let mut done = Vec::with_capacity(asks.len());
        let mut solving = Vec::with_capacity(asks.len());
        let mut acting = HashMap::with_capacity(asks.len());
        for (ask, act) in asks {
            let mut game = session(&ask, &mut table, &brain)
                .unwrap_or_else(|e| fail(&mut out, format!("game {}: {}", ask.id, e)));
            if matches!(brain.mind, Mind::Sog) {
                let solver = game.solver(&brain, act)
                    .unwrap_or_else(|e| fail(&mut out, format!("game {}: {}", ask.id, e)));
                solving.push((ask.id, solver));
                acting.insert(ask.id, act);
            } else {
                let action = if act {
                    Some(game.decide(&brain).unwrap_or_else(|e| {
                        fail(&mut out, format!("game {}: {}", ask.id, e))
                    }))
                } else {
                    game.watch(&brain);
                    None
                };
                done.push(Done { id: ask.id, action });
            }
            table.insert(ask.id, game);
        }
        if !solving.is_empty() {
            let solved = farm
                .as_ref()
                .unwrap()
                .solve(solving)
                .unwrap_or_else(|e| fail(&mut out, e));
            for solved in solved {
                let game = table.get_mut(&solved.id).unwrap();
                let action = game.finish(&solved.solver, acting[&solved.id])
                    .unwrap_or_else(|e| fail(&mut out, format!("game {}: {}", solved.id, e)));
                done.push(Done { id: solved.id, action });
            }
        }
        if !done.is_empty() {
            let reply = Reply { done, error: None };
            writeln!(out, "{}", serde_json::to_string(&reply).unwrap()).unwrap();
            out.flush().unwrap();
        }
    }
}

/// Use only the devices assigned by the referee. No assignment means CPU.
fn backend(o: &Options, net: Net, _cfg: Cfg) -> Result<Backend, String> {
    if o.devices.is_empty() {
        return Ok(Backend::Reference(net));
    }
    #[cfg(feature = "gpu")]
    {
        let devices: Result<Vec<usize>, String> = o
            .devices
            .split(',')
            .map(|name| {
                name.strip_prefix("cuda:")
                    .ok_or_else(|| format!("invalid device {name}"))?
                    .parse()
                    .map_err(|_| format!("invalid device {name}"))
            })
            .collect();
        return warchest::cuda::Device::new(&devices?, net, _cfg, host_slots(_cfg.budget))
            .map(Backend::Cuda);
    }
    #[cfg(not(feature = "gpu"))]
    Err(format!("built without GPU support; cannot open {}", o.devices))
}
