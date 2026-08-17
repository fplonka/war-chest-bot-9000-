//! An arena bot: one checkpoint, playing over stdin and stdout.
//!
//! This binary is the unit an experiment is archived as. Built once at the
//! revision that trained its weights and never rebuilt, it keeps playing after
//! the source that produced it has been rewritten — which is the only way to
//! compare an architecture with the one that replaced it.
//!
//! It reads requests from stdin and writes each game back as that game is
//! ready, in the JSON `warchest::arena` defines. Games are worked on
//! independently and in parallel, so at any moment some are having their
//! subgame built on the cores while others are being solved on the device;
//! nothing waits for a batch to finish. The referee sends work for any game it
//! is not already waiting on, which keeps that mixture topped up.
//!
//! ```text
//! bot --name v5-2h --weights weights.bin --depth 2 --iters 64 --cfr dcfr --device 0
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use warchest::args::Args;
use warchest::bot::{Brain, Mind, Session};
use warchest::net::Net;
use warchest::rebel::rules_table_hash;
use warchest::search::{Cfg, Cfr, Nets};

struct Options {
    name: String,
    weights: String,
    mind: String,
    depth: usize,
    iters: usize,
    cfr: String,
    temp: f32,
    node_cap: usize,
    device: i32,
    inflight: usize,
}

fn options() -> Result<Options, String> {
    let a = Args::parse(&[
        "name", "weights", "mind", "depth", "iters", "cfr", "temp", "node-cap", "device",
        "inflight",
    ])?;
    Ok(Options {
        name: a.text("name", "bot"),
        weights: a.text("weights", ""),
        mind: a.text("mind", "rebel"),
        cfr: a.text("cfr", "dcfr"),
        depth: a.num("depth", 2)?,
        iters: a.num("iters", 64)?,
        temp: a.num("temp", 2.0)?,
        node_cap: a.num("node-cap", 200_000)?,
        device: a.num("device", -1)?,
        inflight: a.num("inflight", 0)?,
    })
}

fn brain(o: &Options) -> Result<Brain, String> {
    let mind = match o.mind.as_str() {
        "rebel" => Mind::Rebel,
        "greedy" => Mind::Greedy { temp: o.temp },
        "random" => Mind::Random,
        "lbr" => Mind::Lbr,
        other => return Err(format!("unknown mind {}", other)),
    };
    let cfr = Cfr::named(&o.cfr).ok_or_else(|| format!("unknown cfr rule {}", o.cfr))?;
    let mut nets = Nets::default();
    #[cfg(feature = "gpu")]
    let mut gpu = None;
    if matches!(mind, Mind::Rebel | Mind::Lbr) {
        let (dims, w, b, ln) =
            Net::load_flat_bin(&o.weights).map_err(|e| format!("{}: {}", o.weights, e))?;
        nets.value = Net::from_flat(&dims, &w, &b, &ln)?;
        #[cfg(feature = "gpu")]
        if o.device >= 0 {
            gpu = Some(warchest::gpu::service::spawn(
                o.device as usize,
                dims,
                w,
                b,
                ln,
                false,
            )?);
        }
    }
    Ok(Brain {
        mind,
        nets,
        cfg: Cfg {
            depth: o.depth,
            iters: o.iters,
            cfr,
            snapshots: false,
            node_cap: o.node_cap,
            ..Default::default()
        },
        #[cfg(feature = "gpu")]
        gpu,
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
            let (state, belief) = warchest::arena::decode_position(&at.position)?;
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
    let (action, policy) = if act {
        let (action, policy) = session.decide(brain, ask.policy)?;
        (Some(action), policy)
    } else {
        session.watch(brain);
        (None, None)
    };
    table.lock().unwrap().insert(ask.id, session);
    Ok(Done {
        id: ask.id,
        action,
        policy,
    })
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
    let brain = Arc::new(brain(&options).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(2);
    }));
    // A solve blocks its thread until the device answers, so a bot on a device
    // wants far more threads than cores: the threads are how many solves can be
    // in flight for the wave to merge. A bot solving on the CPU wants one per
    // core.
    let threads = match (options.inflight, options.device >= 0) {
        (0, false) => std::thread::available_parallelism().map_or(8, |n| n.get()),
        (0, true) => 512,
        (n, _) => n,
    };
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
        // gets more work from the referee immediately, which is what keeps
        // both the cores and the device busy.
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
