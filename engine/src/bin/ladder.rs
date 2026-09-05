use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde::Serialize;
use warchest::arena::{Draft, Hello, Reply, Request, Table, PROTOCOL};
use warchest::packed::{Manifest, Packed};
use warchest::pbs::rules_table_hash;
use warchest::rng::Rng;
use warchest::selfplay::DRAFT_POOL;

struct BotProcess {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl BotProcess {
    fn launch(bot: &Packed, devices: &str) -> Result<BotProcess, String> {
        let m = &bot.manifest;
        let mut command = Command::new(bot.dir.join("bot"));
        command
            .arg(&bot.dir)
            .arg("--devices")
            .arg(devices)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().map_err(|e| format!("{}: {e}", m.name))?;
        let input = child.stdin.take().ok_or("bot stdin")?;
        let mut output = BufReader::new(child.stdout.take().ok_or("bot stdout")?);
        let mut line = String::new();
        output.read_line(&mut line).map_err(|e| e.to_string())?;
        let hello: Hello = serde_json::from_str(&line)
            .map_err(|e| format!("{} sent an invalid greeting: {e}", m.name))?;
        if hello.name != m.name {
            return Err(format!("{} identified as {}", m.name, hello.name));
        }
        if hello.protocol != PROTOCOL {
            return Err(format!(
                "{} speaks protocol {}, not {}",
                m.name, hello.protocol, PROTOCOL
            ));
        }
        if hello.rules != rules_table_hash() {
            return Err(format!("{} was built against different rules", m.name));
        }
        Ok(BotProcess {
            child,
            input,
            output,
        })
    }

    fn exchange(&mut self, request: &Request) -> Result<Reply, String> {
        serde_json::to_writer(&mut self.input, request).map_err(|e| e.to_string())?;
        writeln!(self.input).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())?;
        let expected = request.go.len() + request.watch.len();
        let mut done = Vec::with_capacity(expected);
        while done.len() < expected {
            let mut line = String::new();
            self.output
                .read_line(&mut line)
                .map_err(|e| e.to_string())?;
            if line.is_empty() {
                return Err("bot exited before replying".into());
            }
            let reply: Reply =
                serde_json::from_str(&line).map_err(|e| format!("invalid bot reply: {e}"))?;
            if let Some(error) = reply.error {
                return Err(error);
            }
            done.extend(reply.done);
        }
        Ok(Reply { done, error: None })
    }
}

impl Drop for BotProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Serialize)]
struct PairResult {
    a: String,
    b: String,
    a_minutes: f64,
    b_minutes: f64,
    wins: usize,
    losses: usize,
    draws: usize,
    games: usize,
    score: f64,
    elo: f64,
    elo_low: f64,
    elo_high: f64,
    color_pairs: Vec<[f32; 2]>,
}

#[derive(Serialize)]
struct Report {
    format: u32,
    complete: bool,
    protocol: u32,
    rules: u64,
    games_per_pair: usize,
    seed: u64,
    bots: Vec<Manifest>,
    pairs: Vec<PairResult>,
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

fn play_pair(
    a: &Packed,
    b: &Packed,
    games: usize,
    seed: u64,
    concurrent: usize,
    devices: &[&str],
) -> Result<PairResult, String> {
    let mut bots = [
        BotProcess::launch(a, devices[0])?,
        BotProcess::launch(b, devices[1 % devices.len()])?,
    ];
    let mut table = Table::new();
    let mut color_pairs = vec![[0.0; 2]; games / 2];
    for (pair, draft) in drafts(seed, games / 2).into_iter().enumerate() {
        for (color, seats) in [[0, 1], [1, 0]].into_iter().enumerate() {
            let id = (2 * pair + color) as u32;
            table.start(id, &draft, seats, seed ^ id as u64)?;
        }
        while table.live() >= concurrent {
            step(&mut table, &mut bots, &mut color_pairs)?;
        }
    }
    while table.live() > 0 {
        step(&mut table, &mut bots, &mut color_pairs)?;
    }
    let wins = color_pairs.iter().flatten().filter(|&&x| x > 0.0).count();
    let losses = color_pairs.iter().flatten().filter(|&&x| x < 0.0).count();
    let draws = games - wins - losses;
    let score = (wins as f64 + 0.5 * draws as f64) / games as f64;
    let (elo, elo_low, elo_high) = paired_elo(&color_pairs);
    Ok(PairResult {
        a: a.manifest.name.clone(),
        b: b.manifest.name.clone(),
        a_minutes: a.manifest.minutes,
        b_minutes: b.manifest.minutes,
        wins,
        losses,
        draws,
        games,
        score,
        elo,
        elo_low,
        elo_high,
        color_pairs,
    })
}

fn paired_elo(outcomes: &[[f32; 2]]) -> (f64, f64, f64) {
    let point = |x: f32| f64::from(x > 0.0) + 0.5 * f64::from(x == 0.0);
    let pairs = outcomes.iter().map(|x| (point(x[0]) + point(x[1])) / 2.0);
    let n = outcomes.len() as f64;
    let p = (pairs.clone().sum::<f64>() + 0.5) / (n + 1.0);
    let variance = (pairs.map(|x| (x - p).powi(2)).sum::<f64>() + (0.5 - p).powi(2)) / n;
    let error = 1.96 * (variance / (n + 1.0)).sqrt();
    let elo =
        |x: f64| 400.0 * (x.clamp(1e-9, 1.0 - 1e-9) / (1.0 - x.clamp(1e-9, 1.0 - 1e-9))).log10();
    (elo(p), elo(p - error), elo(p + error))
}

fn step(
    table: &mut Table,
    bots: &mut [BotProcess; 2],
    outcomes: &mut [[f32; 2]],
) -> Result<(), String> {
    table.settle();
    for (id, seats, utility) in table.reap() {
        outcomes[id as usize / 2][id as usize % 2] = if seats[0] == 0 { utility } else { -utility };
    }
    for (index, bot) in bots.iter_mut().enumerate() {
        let request = table.request(index);
        let reply = bot.exchange(&request)?;
        table.accept(index, reply)?;
    }
    Ok(())
}

fn series(path: &Path) -> Result<Vec<Packed>, String> {
    let root = path.join("bots");
    let mut bots = Vec::new();
    for entry in fs::read_dir(&root).map_err(|e| format!("{}: {e}", root.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.join("bot.json").is_file() {
            bots.push(Packed::load(&path)?);
        }
    }
    bots.sort_by(|x, y| {
        x.manifest
            .minutes
            .total_cmp(&y.manifest.minutes)
            .then_with(|| x.dir.ends_with("final").cmp(&y.dir.ends_with("final")))
    });
    if bots.is_empty() {
        return Err(format!("{} contains no packed bots", path.display()));
    }
    Ok(bots)
}

fn pairings<'a>(a: &'a [Packed], b: Option<&'a [Packed]>) -> Vec<(&'a Packed, &'a Packed)> {
    match b {
        None => a.windows(2).map(|pair| (&pair[0], &pair[1])).collect(),
        Some(b) => a
            .iter()
            .map(|a| {
                let b = b
                    .iter()
                    .min_by(|x, y| {
                        (x.manifest.minutes - a.manifest.minutes)
                            .abs()
                            .total_cmp(&(y.manifest.minutes - a.manifest.minutes).abs())
                    })
                    .unwrap();
                (a, b)
            })
            .collect(),
    }
}

fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    let tmp = path.with_extension("json.tmp");
    let mut file = fs::File::create(&tmp).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, report).map_err(|e| e.to_string())?;
    writeln!(file).map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if !(1..=2).contains(&paths.len()) {
        return Err("usage: ladder <run> [other-run]".into());
    }
    let a = series(&paths[0])?;
    let b = paths.get(1).map(|path| series(path)).transpose()?;
    let pairs = pairings(&a, b.as_deref());
    if pairs.is_empty() {
        return Err("evaluation needs at least two bots".into());
    }
    let bots = pairs.iter().flat_map(|(a, b)| [&a.manifest, &b.manifest]);
    let mut manifests = bots.cloned().collect::<Vec<_>>();
    manifests.sort_by(|x, y| x.name.cmp(&y.name));
    manifests.dedup_by(|x, y| x.name == y.name);
    let games = 400;
    let seed = 83;
    let devices = ["0", "1"];
    let output = if paths.len() == 1 {
        paths[0].join("ladder.json")
    } else {
        let dir = paths[1].join("comparisons");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let baseline = paths[0].file_name().ok_or("baseline run needs a name")?;
        dir.join(format!("{}.json", baseline.to_string_lossy()))
    };
    let mut report = Report {
        format: 1,
        complete: false,
        protocol: PROTOCOL,
        rules: rules_table_hash(),
        games_per_pair: games,
        seed,
        bots: manifests,
        pairs: Vec::new(),
    };
    for (a, b) in pairs.iter() {
        let result = play_pair(a, b, games, seed, 128, &devices)?;
        println!(
            "{} vs {}: W{} L{} D{} score {:.3}, Elo {:+.0} [{:+.0}, {:+.0}]",
            result.a,
            result.b,
            result.wins,
            result.losses,
            result.draws,
            result.score,
            result.elo,
            result.elo_low,
            result.elo_high
        );
        report.pairs.push(result);
        write_report(&output, &report)?;
    }
    report.complete = true;
    write_report(&output, &report)
}
