//! The GPU solve service (work package B): solves run on the CUDA device.
//!
//! One process, three roles (docs/arch plan, section 8): the workers build
//! trees on the CPU and submit them as serialized jobs; the GPU service (one
//! thread, owns GPU-0) keeps a live set of solves resident and advances it
//! with ticks; the trainer (Python) publishes weights. This module is the
//! service + the worker-side client. It is behind the `gpu` cargo feature;
//! without CUDA the engine builds and runs exactly as before.
//!
//! A tick advances every live solve by one step (a CFR iteration, a
//! fixed-policy value pass, or a snapshot propagation — the per-solve stage
//! decides). Each phase is one kernel over the solves that need it, with the
//! wide math through cuBLAS GEMMs. The kernels live in `kernels.cu`
//! (compiled by NVRTC at startup) and each one ports one Rust function of
//! `search.rs`/`net.rs`; the CPU solver is the oracle.

pub mod client;
pub mod service;
#[cfg(test)]
mod tests;

pub use client::{GpuClient, Trip1, Trip2};
pub use service::Service;
