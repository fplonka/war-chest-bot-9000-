use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use warchest::arena::{Draft, Hello, Reply, Request, Table, PROTOCOL};
use warchest::pbs::rules_table_hash;
use warchest::rng::Rng;
use warchest::selfplay::DRAFT_POOL;

#[derive(Clone, Deserialize, Serialize)]
struct Search {
    s: u32,
    c: f32,
    batch: usize,
    rounds: u8,
    puct: f32,
    prior_temp: f32,
    cfr: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct Manifest {
    format: u32,
    name: String,
    sha: String,
    binary: String,
    mind: String,
    #[serde(default)]
    weights: String,
    search: Search,
    minutes: f64,
    note: String,
}

#[derive(Clone)]
struct Packed {
    dir: PathBuf,
    manifest: Manifest,
}

impl Packed {
    fn load(path: &Path) -> Result<Packed, String> {
        let raw = fs::read_to_string(path.join("bot.json"))
            .map_err(|e| format!("{}: {e}", path.join("bot.json").display()))?;
        let manifest: Manifest = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        if manifest.format != 1 {
            return Err(format!(
                "{} has bot format {}",
                path.display(),
                manifest.format
            ));
        }
        let executable = path.join("bot");
        if !executable.is_file() {
            return Err(format!("{} has no bot executable", path.display()));
        }
        let bytes = fs::read(&executable).map_err(|e| e.to_string())?;
        let actual = format!("{:x}", Sha256::digest(bytes));
        if !actual.starts_with(&manifest.binary) {
            return Err(format!(
                "{} does not match its packed binary",
                path.display()
            ));
        }
        Ok(Packed {
            dir: path.to_path_buf(),
            manifest,
        })
    }
}

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
            .arg("--name")
            .arg(&m.name)
            .arg("--mind")
            .arg(&m.mind)
            .arg("--weights")
            .arg(bot.dir.join(&m.weights))
            .arg("--s")
            .arg(m.search.s.to_string())
            .arg("--c")
            .arg(m.search.c.to_string())
            .arg("--batch")
            .arg(m.search.batch.to_string())
            .arg("--rounds")
            .arg(m.search.rounds.to_string())
            .arg("--puct")
            .arg(m.search.puct.to_string())
            .arg("--prior-temp")
            .arg(m.search.prior_temp.to_string())
            .arg("--cfr")
            .arg(&m.search.cfr)
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
    devices: &[String],
) -> Result<PairResult, String> {
    let mut bots = [
        BotProcess::launch(a, &devices[0])?,
        BotProcess::launch(b, &devices[1 % devices.len()])?,
    ];
    let mut table = Table::new();
    let mut points = Vec::new();
    for (g, draft) in drafts(seed, games / 2).into_iter().enumerate() {
        for (round, seats) in [[0, 1], [1, 0]].into_iter().enumerate() {
            let id = (2 * g + round) as u32;
            table.start(id, &draft, seats, seed ^ id as u64)?;
        }
        while table.live() >= concurrent {
            step(&mut table, &mut bots, &mut points)?;
        }
    }
    while table.live() > 0 {
        step(&mut table, &mut bots, &mut points)?;
    }
    let wins = points.iter().filter(|&&x| x > 0.0).count();
    let losses = points.iter().filter(|&&x| x < 0.0).count();
    let draws = points.len() - wins - losses;
    Ok(PairResult {
        a: a.manifest.name.clone(),
        b: b.manifest.name.clone(),
        a_minutes: a.manifest.minutes,
        b_minutes: b.manifest.minutes,
        wins,
        losses,
        draws,
        games: points.len(),
        score: (wins as f64 + 0.5 * draws as f64) / points.len().max(1) as f64,
    })
}

fn step(
    table: &mut Table,
    bots: &mut [BotProcess; 2],
    points: &mut Vec<f32>,
) -> Result<(), String> {
    table.settle();
    for (_, seats, utility) in table.reap() {
        points.push(if seats[0] == 0 { utility } else { -utility });
    }
    for (index, bot) in bots.iter_mut().enumerate() {
        let request = table.request(index);
        let reply = bot.exchange(&request)?;
        table.accept(index, reply)?;
    }
    Ok(())
}

fn series(path: &Path) -> Result<Vec<Packed>, String> {
    if path.join("bot").is_file() {
        return Packed::load(path).map(|bot| vec![bot]);
    }
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

fn pairings(groups: &[Vec<Packed>]) -> Vec<(Packed, Packed)> {
    if groups.len() == 1 {
        return groups[0]
            .windows(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect();
    }
    let anchor = &groups[0];
    groups[1..]
        .iter()
        .flat_map(|other| {
            anchor.iter().map(move |a| {
                let b = other
                    .iter()
                    .min_by(|x, y| {
                        (x.manifest.minutes - a.manifest.minutes)
                            .abs()
                            .total_cmp(&(y.manifest.minutes - a.manifest.minutes).abs())
                    })
                    .unwrap();
                (a.clone(), b.clone())
            })
        })
        .collect()
}

fn write_report(path: &Path, report: &Report) -> Result<(), String> {
    let tmp = path.with_extension("json.tmp");
    let mut file = fs::File::create(&tmp).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, report).map_err(|e| e.to_string())?;
    writeln!(file).map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())
}

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|x| x == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(default)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<PathBuf> = args
        .iter()
        .take_while(|x| !x.starts_with("--"))
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        return Err("usage: ladder <bot-or-run>... [--games N] [--out FILE]".into());
    }
    let games = flag(&args, "--games", 200usize);
    if games < 2 || games % 2 != 0 {
        return Err("--games must be a positive even number".into());
    }
    let seed = flag(&args, "--seed", 83u64);
    let concurrent = flag(&args, "--concurrent", 128usize).max(1);
    let device_text = flag(&args, "--devices", "0,1".to_string());
    let devices: Vec<String> = device_text.split(',').map(str::to_string).collect();
    if devices.is_empty() {
        return Err("--devices cannot be empty".into());
    }
    let groups: Vec<Vec<Packed>> = paths
        .iter()
        .map(|path| series(path))
        .collect::<Result<_, _>>()?;
    let pairs = pairings(&groups);
    if pairs.is_empty() {
        return Err("evaluation needs at least two bots".into());
    }
    let mut manifests = groups
        .iter()
        .flatten()
        .map(|x| x.manifest.clone())
        .collect::<Vec<_>>();
    manifests.sort_by(|x, y| x.name.cmp(&y.name));
    manifests.dedup_by(|x, y| x.name == y.name);
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
    let output = args
        .iter()
        .position(|x| x == "--out")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .or_else(|| {
            (paths.len() == 1 && paths[0].join("bots").is_dir())
                .then(|| paths[0].join("ladder.json"))
        });
    for (index, (a, b)) in pairs.iter().enumerate() {
        let result = play_pair(
            a,
            b,
            games,
            seed.wrapping_add(index as u64),
            concurrent,
            &devices,
        )?;
        println!(
            "{} vs {}: W{} L{} D{} score {:.3}",
            result.a, result.b, result.wins, result.losses, result.draws, result.score
        );
        report.pairs.push(result);
        if let Some(path) = &output {
            write_report(path, &report)?;
        }
    }
    report.complete = true;
    if let Some(path) = &output {
        write_report(path, &report)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    }
    Ok(())
}
