//! The GPU wave executor. Packed sparse solves are cost-bucketed into
//! contiguous waves; a completion contains everything the actor needs and no
//! solve remains resident while the public game walks its tree.

pub mod client;
pub(crate) mod wave;

#[cfg(feature = "gpu")]
mod device;
#[cfg(feature = "gpu")]
pub mod service;
#[cfg(all(test, feature = "gpu"))]
mod tests;

pub use client::{CarriedBeliefs, CarryStore, GpuClient, SolveHandle, SolveResult};
#[cfg(feature = "gpu")]
pub use service::Service;
