//! The GPU solve service (work package B): solves run on the CUDA device.
//!
//! One process, three roles (docs/arch plan, section 8): the workers build
//! trees on the CPU and submit them as serialized jobs; the GPU service (one
//! thread, owns GPU-0) keeps a live set of solves resident and advances it
//! with ticks; the trainer (Python) publishes weights. The worker-side
//! client (`client`) compiles always; the service itself is behind the `gpu`
//! cargo feature, so without CUDA the engine builds and runs exactly as
//! before and the client's calls fail at runtime.

pub mod client;

#[cfg(feature = "gpu")]
pub mod layout;
#[cfg(feature = "gpu")]
pub mod service;
#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests;

pub use client::{GpuClient, SolveHandle, Trip1, Trip2};
#[cfg(feature = "gpu")]
pub use service::Service;
