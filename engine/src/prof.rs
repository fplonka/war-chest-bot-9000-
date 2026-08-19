//! Phase timers for the generation loop, compiled out unless the `prof`
//! feature is on.
//!
//! Sampling profilers under-report the BLAS calls here (the AMX coprocessor
//! time does not land on the calling thread's stack the way scalar work does),
//! which is exactly backwards for deciding what to optimise — so the phases are
//! timed directly instead.

#[cfg(feature = "prof")]
mod imp {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    macro_rules! counters {
        ($($name:ident),* $(,)?) => {
            $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
            pub fn dump() {
                $(println!("  {:<10} {:>9.0} cpu-ms", stringify!($name),
                           $name.load(Relaxed) as f64 / 1e6);)*
            }
        };
    }

    counters!(
        BUILD, BACTS, BDRAW, BCELLS, BOBS, BSUP, DGEN, DSORT, DCOMP, BAPPLY,
        BPUSH, REACH, PUBFEAT, PUBNET, BELFEAT, NET, LEAFPOST, BACK, AVG,
    );

    pub static PUBLIC_ROWS: AtomicU64 = AtomicU64::new(0);
    pub static CONFIG_ROWS: AtomicU64 = AtomicU64::new(0);
    pub static JOIN_ROWS: AtomicU64 = AtomicU64::new(0);
    pub static READOUT_CONFIGS: AtomicU64 = AtomicU64::new(0);

    pub fn work(public_rows: usize, config_rows: usize, join_rows: usize, readout_configs: usize) {
        PUBLIC_ROWS.fetch_add(public_rows as u64, Relaxed);
        CONFIG_ROWS.fetch_add(config_rows as u64, Relaxed);
        JOIN_ROWS.fetch_add(join_rows as u64, Relaxed);
        READOUT_CONFIGS.fetch_add(readout_configs as u64, Relaxed);
    }

    pub fn dump_work() {
        println!(
            "  network work: public-rows={} config-rows={} join-rows={} readout-configs={}",
            PUBLIC_ROWS.load(Relaxed),
            CONFIG_ROWS.load(Relaxed),
            JOIN_ROWS.load(Relaxed),
            READOUT_CONFIGS.load(Relaxed),
        );
    }

    pub struct Timer(std::time::Instant, &'static AtomicU64);

    impl Timer {
        pub fn new(c: &'static AtomicU64) -> Timer {
            Timer(std::time::Instant::now(), c)
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            self.1
                .fetch_add(self.0.elapsed().as_nanos() as u64, Relaxed);
        }
    }
}

#[cfg(feature = "prof")]
pub use imp::*;

#[cfg(not(feature = "prof"))]
pub fn dump() {}

#[cfg(not(feature = "prof"))]
#[inline]
pub fn work(_: usize, _: usize, _: usize, _: usize) {}

#[cfg(not(feature = "prof"))]
pub fn dump_work() {}

/// `let _t = timed!(REACH);` — a no-op without the `prof` feature.
#[macro_export]
macro_rules! timed {
    ($c:ident) => {{
        #[cfg(feature = "prof")]
        let t = $crate::prof::Timer::new(&$crate::prof::$c);
        #[cfg(not(feature = "prof"))]
        let t = $crate::prof::NoTimer;
        t
    }};
}

/// Stand-in for a `Timer` when the `prof` feature is off. A plain `()` would
/// make every `drop(t)` that ends a timed region a no-op on a `Copy` value,
/// which the compiler rightly warns about six times over.
#[cfg(not(feature = "prof"))]
pub struct NoTimer;
