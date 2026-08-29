#[cfg(test)]
extern crate self as warchest;

pub mod actions;
pub mod arena;
pub mod board;
pub mod bot;
pub mod contract;
#[cfg(feature = "gpu")]
pub mod cuda;
#[cfg(feature = "gpu")]
pub mod farm;
pub mod net;
pub mod pbs;
pub mod policy;
pub mod rng;
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
