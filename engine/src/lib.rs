//! War Chest (2-player ranked) rules engine.
//!
//! Architecture: the game is a sequence of DECISION NODES. `State::legal_actions`
//! returns the actions for whoever is to act (including chance draws), and
//! `State::apply` returns the successor state. All randomness enters through
//! chance-node `DrawCoin` actions; the core is RNG-free and deterministic, so a
//! replay can force observed draws.

#[cfg(test)]
extern crate self as warchest;

pub mod actions;
pub mod arena;
pub mod args;
pub mod board;
pub mod contract;
#[cfg(feature = "gpu")]
pub mod cuda;
pub mod farm;
pub mod bot;
pub mod net;
pub mod policy;
pub mod prof;
pub mod pbs;
pub mod rng;
pub mod roots;
pub mod rules;
pub mod search;
pub mod selfplay;
pub mod state;
pub mod units;

pub use actions::Action;
pub use state::{State, BLACK, WHITE};

#[cfg(feature = "python")]
mod py;

#[cfg(test)]
#[path = "../tests/pbs.rs"]
mod pbs_integration;

#[cfg(test)]
#[path = "../tests/sog_solver.rs"]
mod sog_solver;

#[cfg(all(test, feature = "gpu"))]
#[path = "../tests/cuda_parity.rs"]
mod cuda_parity;
