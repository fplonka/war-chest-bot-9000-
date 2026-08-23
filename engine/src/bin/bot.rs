//! An arena bot: one checkpoint, playing over stdin and stdout.
//!
//! This binary is the unit an experiment is archived as. Built once at the
//! revision that trained its weights and never rebuilt, it keeps playing after
//! the source that produced it has been rewritten — which is the only way to
//! compare an architecture with the one that replaced it.
//!
//! It reads requests from stdin and writes each game back as that game is
//! ready, in the JSON `warchest::arena` defines. Games are worked on
//! independently and in parallel. The referee sends work for any game it is
//! not already waiting on, which keeps the cores busy.
//!
//! ```text
//! bot --name v5-2h --weights weights.bin --s 512 --c 8 --cfr sog
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use warchest::args::Args;
use warchest::bot::{Brain, Mind, Session};
use warchest::farm::{Backend, Cards};
use warchest::net::Net;
use warchest::pbs::rules_table_hash;
use warchest::search::{Budget, Cfg, Cfr, Nets};

struct Options {
    name: String,
    weights: String,
    mind: String,
    s: u32,
    c: f32,
    batch: usize,
    rounds: u8,
    cfr: String,
    threads: usize,
    temp: f32,
    devices: String,
}

fn options() -> Result<Options, String> {
    let a = Args::parse(&[
        "name", "weights", "mind", "s", "c", "batch", "rounds", "cfr", "threads", "temp",
        "devices",
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
        threads: a.num("threads", 0)?,
        temp: a.num("temp", 2.0)?,
        devices: a.text("devices", ""),
    })
}

fn brain(o: &Options) -> Result<Brain, String> {
    let mind = match o.mind.as_str() {
        "sog" => Mind::Sog,
        "random" => Mind::Random,
        "greedy" => Mind::Greedy { temp: o.temp },
        other => return Err(format!("unknown mind {}", other)),
    };
    let cfr = Cfr::named(&o.cfr).ok_or_else(|| format!("unknown cfr rule {}", o.cfr))?;
    let cfg = Cfg { s: o.s, c: o.c, batch: o.batch, rounds: o.rounds, cfr, budget: Budget::for_s(o.s), ..Default::default() };
    let cards = devices(o, mind, cfg)?.map(|backend| Arc::new(Cards::new(backend)));
    let mut nets = Nets { device: cards.is_some(), ..Nets::default() };
    if matches!(mind, Mind::Sog) && cards.is_none() {
        nets.value = Net::load_bin(&o.weights).map_err(|e| format!("{}: {}", o.weights, e))?;
    }
    Ok(Brain {
        mind,
        nets: Arc::new(nets),
        cfg,
        cards,
    })
}

/// Every live game. A game is taken out of the table while it is being worked
/// on and put back when it is done; the referee never has two asks out for the
/// same game, so no two tasks ever hold the same session.
type Table = Arc<Mutex<HashMap<u32, Session>>>;

/// Bring one game up to date and, if the ask was a `go`, choose its move.
fn work(ask: &Ask, table: &Table, brain: &Brain, act: bool) -> Result<Done, String> {
    let mut session = match (&ask.start, &ask.at) {
        (Some(start), None) => Session::new(&start.draft, start.seat, start.seed)?,
        (None, Some(at)) => {
            let (state, belief) = at.position.state()?;
            Session::at(state, belief, at.seat, at.seed)?
        }
        (Some(_), Some(_)) => return Err("a game starts one way or the other".into()),
        (None, None) => table
            .lock()
            .unwrap()
            .remove(&ask.id)
            .ok_or_else(|| format!("game {} was never started", ask.id))?,
    };
    for obs in &ask.obs {
        session.observe(obs, brain)?;
    }
    let action = if act {
        Some(session.decide(brain)?)
    } else {
        session.watch(brain);
        None
    };
    table.lock().unwrap().insert(ask.id, session);
    Ok(Done { id: ask.id, action })
}

/// A game the bot could not follow ends the run. Half a ladder from a bot that
/// lost the position is worse than no ladder.
fn fail(out: &Mutex<impl Write>, error: String) -> ! {
    let reply = Reply {
        done: Vec::new(),
        error: Some(error),
    };
    let mut out = out.lock().unwrap();
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
    let brain = Arc::new(
        brain(&options).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(2);
        }),
    );
    let threads = if options.threads == 0 {
        std::thread::available_parallelism().map_or(8, |n| n.get())
    } else {
        options.threads
    };
    // A thread's own stack is its continuation: it puts a solve's calls on a
    // card's queue and waits for the round that carries them, which is shared
    // with every other thread that was ready at the same moment.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");

    let out = Arc::new(Mutex::new(std::io::stdout()));
    let hello = Hello {
        name: options.name.clone(),
        protocol: PROTOCOL,
        rules: rules_table_hash(),
    };
    {
        let mut lock = out.lock().unwrap();
        writeln!(lock, "{}", serde_json::to_string(&hello).unwrap()).unwrap();
        lock.flush().unwrap();
    }

    let table: Table = Arc::new(Mutex::new(HashMap::new()));
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("stdin");
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(e) => fail(&out, e.to_string()),
        };
        for id in &request.drop {
            table.lock().unwrap().remove(id);
        }
        // One task per game rather than per request: a game answered early
        // gets more work from the referee immediately, keeping the cores busy.
        for (ask, act) in request
            .go
            .into_iter()
            .map(|a| (a, true))
            .chain(request.watch.into_iter().map(|a| (a, false)))
        {
            let (table, brain, out) = (table.clone(), brain.clone(), out.clone());
            pool.spawn(move || match work(&ask, &table, &brain, act) {
                Ok(done) => {
                    let reply = Reply {
                        done: vec![done],
                        error: None,
                    };
                    let mut lock = out.lock().unwrap();
                    let _ = writeln!(lock, "{}", serde_json::to_string(&reply).unwrap());
                    let _ = lock.flush();
                }
                Err(e) => fail(&out, format!("game {}: {}", ask.id, e)),
            });
        }
    }
}

/// Use only the devices assigned by the referee. No assignment means CPU.
fn devices(o: &Options, mind: Mind, _cfg: Cfg) -> Result<Option<Backend>, String> {
    if !matches!(mind, Mind::Sog) || o.devices.is_empty() {
        return Ok(None);
    }
    #[cfg(feature = "gpu")]
    {
        let ordinals: Result<Vec<usize>, String> = o
            .devices
            .split(',')
            .map(|name| {
                name.strip_prefix("cuda:")
                    .ok_or_else(|| format!("invalid device {name}"))?
                    .parse()
                    .map_err(|_| format!("invalid device {name}"))
            })
            .collect();
        let net = Net::load_bin(&o.weights).map_err(|e| format!("{}: {}", o.weights, e))?;
        return warchest::cuda::Device::new(&ordinals?, net, _cfg, usize::MAX)
            .map(Backend::Cuda)
            .map(Some);
    }
    #[cfg(not(feature = "gpu"))]
    Err(format!("built without GPU support; cannot open {}", o.devices))
}
