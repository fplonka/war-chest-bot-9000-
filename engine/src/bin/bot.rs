use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use warchest::bot::{Brain, Session};
use warchest::cuda::Device;
use warchest::farm::Cards;
use warchest::net::Net;
use warchest::packed::Packed;
use warchest::pbs::rules_table_hash;

fn work(
    ask: Ask,
    act: bool,
    sessions: &Mutex<HashMap<u32, Session>>,
    brain: &Brain,
) -> Result<Done, String> {
    let mut session = match ask.start {
        Some(start) => Session::new(&start.draft, start.seat, start.seed)?,
        None => sessions
            .lock()
            .unwrap()
            .remove(&ask.id)
            .ok_or_else(|| format!("game {} was never started", ask.id))?,
    };
    for obs in &ask.obs {
        session.observe(obs)?;
    }
    let action = if act {
        Some(session.decide(brain)?)
    } else {
        session.watch(brain)?;
        None
    };
    sessions.lock().unwrap().insert(ask.id, session);
    Ok(Done { id: ask.id, action })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or("usage: bot BOT_DIR --devices 0,1")?;
    let packed = Packed::load(path.as_ref())?;
    let device_text = args
        .iter()
        .position(|x| x == "--devices")
        .and_then(|i| args.get(i + 1))
        .ok_or("packed bot needs --devices")?;
    let devices = device_text
        .split(',')
        .map(|x| x.parse().map_err(|_| format!("invalid device {x}")))
        .collect::<Result<Vec<_>, _>>()?;
    let cfg = packed.manifest.search.config()?;
    let weights = packed.dir.join(&packed.manifest.weights);
    let net = Net::load_bin(weights.to_str().ok_or("invalid weights path")?)
        .map_err(|e| format!("{}: {e}", weights.display()))?;
    let cards = Arc::new(Cards::new(Device::new(&devices, &net, cfg, usize::MAX)?));
    let brain = Brain {
        cards,
        net: Arc::new(net),
        cfg,
    };
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{}",
        serde_json::to_string(&Hello {
            name: packed.manifest.name,
            protocol: PROTOCOL,
            rules: rules_table_hash(),
        })
        .unwrap()
    )
    .map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    let sessions = Mutex::new(HashMap::new());
    for line in std::io::stdin().lock().lines() {
        let request: Request =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        for id in request.drop {
            sessions.lock().unwrap().remove(&id);
        }
        let asks = request
            .go
            .into_iter()
            .map(|ask| (ask, true))
            .chain(request.watch.into_iter().map(|ask| (ask, false)))
            .collect::<Vec<_>>();
        if asks.is_empty() {
            continue;
        }
        let done: Result<Vec<_>, _> = pool.install(|| {
            asks.into_par_iter()
                .map(|(ask, act)| work(ask, act, &sessions, &brain))
                .collect()
        });
        let reply = match done {
            Ok(done) => Reply { done, error: None },
            Err(error) => Reply {
                done: Vec::new(),
                error: Some(error),
            },
        };
        writeln!(out, "{}", serde_json::to_string(&reply).unwrap()).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        if reply.error.is_some() {
            return Err("session failed".into());
        }
    }
    Ok(())
}
