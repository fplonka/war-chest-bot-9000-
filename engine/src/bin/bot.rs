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
//! bot --name v5-2h --weights weights.bin --nodes 1024 --expand 1 --iters 64 --cfr dcfr
//! ```

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use warchest::args::Args;
use warchest::bot::{Brain, Mind, Session};
use warchest::farm::{Backend, Gate};
use warchest::net::Net;
use warchest::rebel::rules_table_hash;
use warchest::search::{Cfg, Cfr, Nets};

struct Options {
    name: String,
    weights: String,
    mind: String,
    nodes: usize,
    expand: usize,
    iters: usize,
    cfr: String,
    temp: f32,
    node_cap: usize,
    threads: usize,
}

fn options() -> Result<Options, String> {
    let a = Args::parse(&[
        "name", "weights", "mind", "nodes", "expand", "iters", "cfr", "temp",
        "node-cap", "threads",
    ])?;
    Ok(Options {
        name: a.text("name", "bot"),
        weights: a.text("weights", ""),
        mind: a.text("mind", "rebel"),
        cfr: a.text("cfr", "dcfr"),
        nodes: a.num("nodes", 1024)?,
        expand: a.num("expand", 1)?,
        iters: a.num("iters", 64)?,
        temp: a.num("temp", 2.0)?,
        node_cap: a.num("node-cap", 200_000)?,
        threads: a.num("threads", 0)?,
    })
}

fn brain(o: &Options, gate: Option<Arc<Gate>>, device: bool) -> Result<Brain, String> {
    let mind = match o.mind.as_str() {
        "rebel" => Mind::Rebel,
        "greedy" => Mind::Greedy { temp: o.temp },
        "random" => Mind::Random,
        other => return Err(format!("unknown mind {}", other)),
    };
    let cfr = Cfr::named(&o.cfr).ok_or_else(|| format!("unknown cfr rule {}", o.cfr))?;
    let mut nets = Nets { gate, device, ..Nets::default() };
    if matches!(mind, Mind::Rebel) {
        nets.value = Net::load_bin(&o.weights).map_err(|e| format!("{}: {}", o.weights, e))?;
    }
    Ok(Brain {
        mind,
        nets,
        cfg: Cfg {
            nodes: o.nodes,
            expand: o.expand,
            iters: o.iters,
            cfr,
            node_cap: o.node_cap,
            ..Default::default()
        },
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
    // The cards, if there are any. A ladder is a few thousand solves at the
    // same budget a training run uses, so it belongs on the same machinery:
    // without this a forty-game ladder is an hour of CPU, which is too dear to
    // be the thing that checks whether a run learned anything.
    let (gate, driver) = match devices(&options) {
        None => (None, None),
        Some(backend) => {
            let gate = Arc::new(Gate::default());
            let mine = gate.clone();
            // The driver holds no gate slot of its own, so it never waits on
            // itself; it stops when the gate closes, which is when stdin ends.
            let driver = std::thread::spawn(move || {
                while !mine.round_closed() {
                    mine.serve_until_idle(|calls| backend.run(calls));
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            });
            (Some(gate), Some(driver))
        }
    };
    let brain = Arc::new(
        brain(&options, gate.clone(), gate.is_some()).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(2);
        }),
    );
    let threads = if options.threads == 0 {
        std::thread::available_parallelism().map_or(8, |n| n.get())
    } else {
        options.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        // Every worker is a gate member for its whole life, so a round is
        // exactly the games in flight and a thread that is between games does
        // not hold the others up.
        .start_handler({
            let gate = gate.clone();
            move |_| {
                if let Some(g) = &gate {
                    std::mem::forget(g.enter());
                }
            }
        })
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
    if let Some(gate) = gate {
        gate.close();
    }
    if let Some(driver) = driver {
        let _ = driver.join();
    }
}

/// The backend a solve evaluates on: every card the driver can see.
///
/// Not a flag. A bot solves at the training budget, so on a machine with cards
/// it belongs on them, and there is nothing a caller could usefully decide
/// here. The CPU network answers when there are no cards, which is what makes
/// a bot runnable on a laptop -- and it is the oracle `cuda_parity` holds the
/// device to, so it is not going anywhere.
fn devices(o: &Options) -> Option<Backend> {
    if !matches!(o.mind.as_str(), "rebel") {
        return None;
    }
    #[cfg(feature = "gpu")]
    {
        let n = warchest::cuda::Device::count();
        if n > 0 {
            let net = Net::load_bin(&o.weights).ok()?;
            match warchest::cuda::Device::new(&(0..n).collect::<Vec<_>>(), net) {
                Ok(d) => return Some(Backend::Cuda(d)),
                Err(e) => eprintln!("no device backend: {e}"),
            }
        }
    }
    None
}
