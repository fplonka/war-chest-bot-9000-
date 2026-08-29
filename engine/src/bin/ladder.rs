use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use warchest::arena::{Done, Draft, Reply, Table};
use warchest::bot::{Brain, Mind, Session};
use warchest::cuda::Device;
use warchest::farm::Cards;
use warchest::net::Net;
use warchest::rng::Rng;
use warchest::search::{Budget, Cfg, Cfr};
use warchest::selfplay::DRAFT_POOL;

struct Spec {
    name: String,
    mind: String,
    temp: f32,
    cfg: Cfg,
    weights: String,
}

fn spec_of(dir: &str) -> Result<Spec, String> {
    let raw = std::fs::read_to_string(format!("{dir}/bot.json")).map_err(|e| format!("{dir}/bot.json: {e}"))?;
    let j: serde_json::Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let text = |k: &str, d: &str| j[k].as_str().unwrap_or(d).to_string();
    let s = j["search"]["s"].as_u64().unwrap_or(512) as u32;
    let num = |k: &str, d: f64| j["search"][k].as_f64().unwrap_or(d);
    let cfr = j["search"]["cfr"].as_str().unwrap_or("dcfr");
    Ok(Spec {
        name: text("name", dir),
        mind: text("mind", "sog"),
        temp: j["temp"].as_f64().unwrap_or(2.0) as f32,
        cfg: Cfg {
            s,
            c: num("c", 8.0) as f32,
            batch: num("batch", 8.0) as usize,
            rounds: num("rounds", 0.0) as u8,
            puct: num("puct", 1.5) as f32,
            prior_temp: num("prior_temp", 1.0) as f32,
            cfr: Cfr::named(cfr).ok_or_else(|| format!("unknown cfr rule {cfr}"))?,
            budget: Budget::for_s(s),
        },
        weights: match j["weights"].as_str() {
            Some(w) => format!("{dir}/{w}"),
            None => String::new(),
        },
    })
}

fn brain_of(spec: &Spec, device: usize) -> Result<Brain, String> {
    let net = if spec.weights.is_empty() {
        Net::default()
    } else {
        Net::load_bin(&spec.weights).map_err(|e| format!("{}: {e}", spec.weights))?
    };
    let mind = match spec.mind.as_str() {
        "random" => Mind::Random,
        "greedy" => Mind::Greedy { temp: spec.temp },
        "sog" => Mind::Sog(Arc::new(Cards::new(Device::new(
            &[device],
            net.clone(),
            spec.cfg,
            0,
        )?))),
        other => return Err(format!("unknown mind {other}")),
    };
    Ok(Brain {
        mind,
        net: Arc::new(net),
        cfg: spec.cfg,
    })
}

fn drafts(seed: u64, count: usize) -> Vec<Draft> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let mut pool: Vec<u16> = DRAFT_POOL.to_vec();
            for i in (1..pool.len()).rev() {
                pool.swap(i, rng.below(i + 1));
            }
            Draft {
                white: pool[..4].to_vec(),
                black: pool[4..8].to_vec(),
                first: (rng.next_u64() & 1) as u8,
            }
        })
        .collect()
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str, d: u64| -> u64 {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let dirs: Vec<&String> = argv.iter().take_while(|a| !a.starts_with("--")).collect();
    if dirs.len() != 2 {
        eprintln!("usage: ladder <botA> <botB> [--games N] [--seed N] [--concurrent N]");
        std::process::exit(2);
    }
    let (games, seed) = (flag("--games", 200) as usize, flag("--seed", 83));
    let concurrent = flag("--concurrent", 128) as usize;
    let die = |e: String| -> ! {
        eprintln!("{e}");
        std::process::exit(2)
    };

    let specs: Vec<Spec> = dirs.iter().map(|d| spec_of(d).unwrap_or_else(|e| die(e))).collect();
    let brains: Vec<Arc<Brain>> = specs
        .iter()
        .enumerate()
        .map(|(seat, spec)| Arc::new(brain_of(spec, seat).unwrap_or_else(|e| die(e))))
        .collect();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(8, |n| n.get()))
        .build()
        .expect("thread pool");
    let half = games / 2;
    let mut table = Table::new();
    let mut sessions: [Arc<Mutex<HashMap<u32, Session>>>; 2] = Default::default();
    let mut points: Vec<f32> = Vec::new();

    for (g, draft) in drafts(seed, half).into_iter().enumerate() {
        for (round, bots) in [[0usize, 1], [1, 0]].into_iter().enumerate() {
            let id = (2 * g + round) as u32;
            table
                .start(id, &draft, bots, seed ^ id as u64)
                .unwrap_or_else(|e| die(e));
        }
        while table.live() >= concurrent {
            step(&mut table, &brains, &sessions, &pool, &mut points);
        }
    }
    while table.live() > 0 {
        step(&mut table, &brains, &sessions, &pool, &mut points);
    }

    let w = points.iter().filter(|&&z| z > 0.0).count();
    let l = points.iter().filter(|&&z| z < 0.0).count();
    let d = points.len() - w - l;
    let score = (w as f32 + 0.5 * d as f32) / points.len().max(1) as f32;
    println!(
        "{} vs {}: W{w} L{l} D{d}  score {score:.3}  over {} games",
        specs[0].name,
        specs[1].name,
        points.len()
    );
    let _ = &mut sessions;
}

fn step(
    table: &mut Table,
    brains: &[Arc<Brain>],
    sessions: &[Arc<Mutex<HashMap<u32, Session>>>; 2],
    pool: &rayon::ThreadPool,
    points: &mut Vec<f32>,
) {
    table.settle();
    for (_, _, z) in table.reap() {
        points.push(z);
    }
    for bot in 0..2 {
        let request = table.request(bot);
        for id in &request.drop {
            sessions[bot].lock().unwrap().remove(id);
        }
        let asks: Vec<_> = request
            .go
            .into_iter()
            .map(|a| (a, true))
            .chain(request.watch.into_iter().map(|a| (a, false)))
            .collect();
        let done: Vec<Done> = pool.install(|| {
            use rayon::prelude::*;
            asks.into_par_iter()
                .map(|(ask, act)| {
                    let mut seat = match &ask.start {
                        Some(start) => Session::new(&start.draft, start.seat, start.seed).expect("a session starts"),
                        None => sessions[bot].lock().unwrap().remove(&ask.id).expect("a started game"),
                    };
                    for obs in &ask.obs {
                        seat.observe(obs, &brains[bot]).expect("an observation lands");
                    }
                    let action = if act {
                        Some(seat.decide(&brains[bot]).expect("a decision"))
                    } else {
                        seat.watch(&brains[bot]);
                        None
                    };
                    sessions[bot].lock().unwrap().insert(ask.id, seat);
                    Done { id: ask.id, action }
                })
                .collect()
        });
        table
            .accept(bot, Reply { done, error: None })
            .expect("the referee accepts");
    }
}
