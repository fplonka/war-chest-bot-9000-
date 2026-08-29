use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use warchest::bot::{Brain, Mind, Session};
#[cfg(feature = "gpu")]
use warchest::cuda::Device;
#[cfg(feature = "gpu")]
use warchest::farm::Cards;
use warchest::net::Net;
use warchest::pbs::rules_table_hash;
use warchest::search::{Budget, Cfg, Cfr};

struct Options {
    name: String,
    weights: String,
    mind: String,
    temp: f32,
    cfg: Cfg,
    threads: usize,
    devices: Vec<usize>,
}

fn value<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match args.iter().position(|x| x == name) {
        Some(i) => args
            .get(i + 1)
            .ok_or_else(|| format!("{name} needs a value"))?
            .parse()
            .map_err(|_| format!("invalid value for {name}")),
        None => Ok(default),
    }
}

fn text(args: &[String], name: &str, default: &str) -> Result<String, String> {
    match args.iter().position(|x| x == name) {
        Some(i) => args
            .get(i + 1)
            .cloned()
            .ok_or_else(|| format!("{name} needs a value")),
        None => Ok(default.to_string()),
    }
}

fn options() -> Result<Options, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let s = value(&args, "--s", 512)?;
    let cfr = text(&args, "--cfr", "dcfr")?;
    let devices = text(&args, "--devices", "")?;
    Ok(Options {
        name: text(&args, "--name", "bot")?,
        weights: text(&args, "--weights", "")?,
        mind: text(&args, "--mind", "sog")?,
        temp: value(&args, "--temp", 2.0)?,
        cfg: Cfg {
            s,
            c: value(&args, "--c", 8.0)?,
            batch: value(&args, "--batch", 8)?,
            rounds: value(&args, "--rounds", 0)?,
            puct: value(&args, "--puct", 1.5)?,
            prior_temp: value(&args, "--prior-temp", 1.0)?,
            cfr: Cfr::named(&cfr).ok_or_else(|| format!("unknown cfr rule {cfr}"))?,
            budget: Budget::for_s(s),
        },
        threads: value(&args, "--threads", 0)?,
        devices: if devices.is_empty() {
            Vec::new()
        } else {
            devices
                .split(',')
                .map(|x| x.parse().map_err(|_| format!("invalid device {x}")))
                .collect::<Result<_, _>>()?
        },
    })
}

fn brain(o: &Options) -> Result<Brain, String> {
    #[cfg(not(feature = "gpu"))]
    let _ = &o.devices;
    let net = if o.mind == "sog" {
        Net::load_bin(&o.weights).map_err(|e| format!("{}: {e}", o.weights))?
    } else {
        Net::default()
    };
    let mind = match o.mind.as_str() {
        "random" => Mind::Random,
        "greedy" => Mind::Greedy { temp: o.temp },
        "sog" => {
            #[cfg(feature = "gpu")]
            {
                if o.devices.is_empty() {
                    return Err("sog needs an assigned GPU".into());
                }
                Mind::Sog(Arc::new(Cards::new(Device::new(
                    &o.devices,
                    &net,
                    o.cfg,
                    usize::MAX,
                )?)))
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err("this bot was built without GPU support".into());
            }
        }
        other => return Err(format!("unknown mind {other}")),
    };
    Ok(Brain {
        mind,
        net: Arc::new(net),
        cfg: o.cfg,
    })
}

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
        session.observe(obs, brain)?;
    }
    let action = if act {
        Some(session.decide(brain)?)
    } else {
        session.watch(brain);
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
    let options = options()?;
    let brain = brain(&options)?;
    let threads = if options.threads == 0 {
        std::thread::available_parallelism().map_or(8, |n| n.get())
    } else {
        options.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| e.to_string())?;
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{}",
        serde_json::to_string(&Hello {
            name: options.name,
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
        let asks: Vec<_> = request
            .go
            .into_iter()
            .map(|ask| (ask, true))
            .chain(request.watch.into_iter().map(|ask| (ask, false)))
            .collect();
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
