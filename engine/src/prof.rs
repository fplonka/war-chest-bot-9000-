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
        BUILD, BACTS, BDRAW, BAPPLY, BPUSH, ALLOC, REACH, PUBFEAT, PUBNET, BELFEAT, NET, LEAFPOST, BACK, RM, AVG, WALK,
    );

    pub struct Timer(std::time::Instant, &'static AtomicU64);

    impl Timer {
        pub fn new(c: &'static AtomicU64) -> Timer {
            Timer(std::time::Instant::now(), c)
        }
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            self.1.fetch_add(self.0.elapsed().as_nanos() as u64, Relaxed);
        }
    }
}

#[cfg(feature = "prof")]
pub use imp::*;

#[cfg(not(feature = "prof"))]
pub fn dump() {}

/// `let _t = timed!(REACH);` — a no-op without the `prof` feature.
#[macro_export]
macro_rules! timed {
    ($c:ident) => {{
        #[cfg(feature = "prof")]
        let t = $crate::prof::Timer::new(&$crate::prof::$c);
        #[cfg(not(feature = "prof"))]
        let t = ();
        t
    }};
}

// -------------------------------------------------------------- tree shape

#[cfg(feature = "prof")]
mod shape {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static SOLVES: AtomicU64 = AtomicU64::new(0);
    pub static NODES: AtomicU64 = AtomicU64::new(0);
    pub static LEAVES: AtomicU64 = AtomicU64::new(0);
    pub static CHANCE: AtomicU64 = AtomicU64::new(0);
    pub static INNER_CA: AtomicU64 = AtomicU64::new(0);
    pub static CFGSUM: AtomicU64 = AtomicU64::new(0);

    pub fn add(c: &AtomicU64, v: u64) {
        c.fetch_add(v, Relaxed);
    }

    pub fn dump_shape() {
        let s = SOLVES.load(Relaxed).max(1) as f64;
        println!(
            "  per solve: nodes {:.1} leaves {:.1} chance {:.1} inner c*a {:.0} cfg-slots {:.0}",
            NODES.load(Relaxed) as f64 / s,
            LEAVES.load(Relaxed) as f64 / s,
            CHANCE.load(Relaxed) as f64 / s,
            INNER_CA.load(Relaxed) as f64 / s,
            CFGSUM.load(Relaxed) as f64 / s,
        );
    }
}

#[cfg(feature = "prof")]
pub use shape::*;

#[cfg(not(feature = "prof"))]
pub fn dump_shape() {}

/// `shape!(NODES, n)` — a no-op without the `prof` feature.
#[macro_export]
macro_rules! shape {
    ($c:ident, $v:expr) => {{
        #[cfg(feature = "prof")]
        $crate::prof::add(&$crate::prof::$c, $v as u64);
    }};
}
