//! Command-line flags for the binaries, in the one shape they all use.
//!
//! Every tool here takes `--name value` and nothing else: no positionals, no
//! short forms, no flags that stand alone. That is enough for a program whose
//! arguments are written down in a bot manifest or a build script rather than
//! typed, and it means an unknown or malformed option is an error rather than
//! a silently different run.

use std::collections::BTreeMap;

pub struct Args(BTreeMap<String, String>);

impl Args {
    /// Parse this process's arguments, refusing anything not in `known`.
    ///
    /// A misspelled flag would otherwise be ignored, and a benchmark or a
    /// suite build that quietly ran with the default instead of what was asked
    /// for is worse than one that did not run.
    pub fn parse(known: &[&str]) -> Result<Args, String> {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut out = BTreeMap::new();
        let mut i = 0;
        while i < argv.len() {
            let name = argv[i]
                .strip_prefix("--")
                .ok_or_else(|| format!("expected an option, found {}", argv[i]))?;
            if !known.contains(&name) {
                return Err(format!(
                    "unknown option --{}; known: --{}",
                    name,
                    known.join(" --")
                ));
            }
            let value = argv
                .get(i + 1)
                .ok_or_else(|| format!("--{} needs a value", name))?;
            out.insert(name.to_string(), value.clone());
            i += 2;
        }
        Ok(Args(out))
    }

    pub fn text(&self, name: &str, fallback: &str) -> String {
        self.0
            .get(name)
            .cloned()
            .unwrap_or_else(|| fallback.to_string())
    }

    pub fn num<T: std::str::FromStr>(&self, name: &str, fallback: T) -> Result<T, String> {
        match self.0.get(name) {
            None => Ok(fallback),
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{} is not a number: {}", name, v)),
        }
    }
}
