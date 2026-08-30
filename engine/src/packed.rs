use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::search::{Budget, Cfg, Cfr};

#[derive(Clone, Deserialize, Serialize)]
pub struct Search {
    pub s: u32,
    pub c: f32,
    pub batch: usize,
    pub rounds: u8,
    pub puct: f32,
    pub prior_temp: f32,
    pub cfr: String,
}

impl Search {
    pub fn config(&self) -> Result<Cfg, String> {
        Ok(Cfg {
            s: self.s,
            c: self.c,
            batch: self.batch,
            rounds: self.rounds,
            puct: self.puct,
            prior_temp: self.prior_temp,
            cfr: Cfr::named(&self.cfr).ok_or_else(|| format!("unknown cfr rule {}", self.cfr))?,
            budget: Budget::for_s(self.s),
        })
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Manifest {
    pub format: u32,
    pub name: String,
    pub sha: String,
    pub binary: String,
    pub weights: String,
    pub weights_sha: String,
    pub search: Search,
    pub minutes: f64,
    pub note: String,
}

pub struct Packed {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Packed {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = path.join("bot.json");
        let manifest: Manifest = serde_json::from_str(
            &fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?,
        )
        .map_err(|e| format!("{}: {e}", file.display()))?;
        if manifest.format != 2 {
            return Err(format!(
                "{} has bot format {}",
                path.display(),
                manifest.format
            ));
        }
        for (name, expected) in [
            ("bot", &manifest.binary),
            (&manifest.weights, &manifest.weights_sha),
        ] {
            let file = path.join(name);
            let bytes = fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            let actual = format!("{:x}", Sha256::digest(bytes));
            if &actual != expected {
                return Err(format!("{} does not match bot.json", file.display()));
            }
        }
        Ok(Self {
            dir: path.into(),
            manifest,
        })
    }
}
