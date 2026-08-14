//! War Chest (2-player ranked) rules engine.
//!
//! Architecture: the game is a sequence of DECISION NODES. `State::legal_actions`
//! returns the actions for whoever is to act (including chance draws), and
//! `State::apply` returns the successor state. All randomness enters through
//! chance-node `DrawCoin` actions; the core is RNG-free and deterministic, so a
//! replay can force observed draws.

pub mod actions;
pub mod board;
pub mod gpu;
pub mod net;
pub mod prof;
pub mod rebel;
pub mod rng;
pub mod roots;
pub mod rules;
pub mod search;
pub mod selfplay;
pub mod serialize;
pub mod state;
pub mod units;

pub use actions::Action;
pub use state::{State, BLACK, WHITE};

#[cfg(feature = "python")]
mod live;
#[cfg(feature = "python")]
mod py;
