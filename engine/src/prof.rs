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
        BUILD, BACTS, BDRAW, BCELLS, BOBS, BSUP, DGEN, DSORT, DCOMP, BAPPLY, BPUSH, ALLOC, SERIAL,
        REACH, PUBFEAT, PUBNET, BELFEAT, NET, LEAFDOT, LEAFPOST, BACK, RM, AVG, WALK, SNAP, P2,
        ADVANCE, TRIP1, TRIP2, DLIDLE, DLSYNC, DLBUSY, SVCTICK, SVCHDR, SVCADMIT, SVCFENCE,
    );

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
    /// Distinct configs interned per solve — the height of the readout table,
    /// against `CFGSUM`'s total slots, which is what decides whether the
    /// per-config readout is better done leaf by leaf or as one matmul.
    pub static NCFG: AtomicU64 = AtomicU64::new(0);
    pub static S_ROWS: AtomicU64 = AtomicU64::new(0);
    pub static S_LEVELS: AtomicU64 = AtomicU64::new(0);
    pub static S_REACH: AtomicU64 = AtomicU64::new(0);
    pub static S_CELLS: AtomicU64 = AtomicU64::new(0);
    pub static S_DRAW_NNZ: AtomicU64 = AtomicU64::new(0);
    pub static S_DEC_REV_NNZ: AtomicU64 = AtomicU64::new(0);
    pub static S_DRAW_REV_NNZ: AtomicU64 = AtomicU64::new(0);
    pub static S_SNAP_CFG: AtomicU64 = AtomicU64::new(0);
    pub static CH_PARENT_ROOT: AtomicU64 = AtomicU64::new(0);
    pub static CH_PARENT_DECISION: AtomicU64 = AtomicU64::new(0);
    pub static CH_PARENT_CHANCE: AtomicU64 = AtomicU64::new(0);
    pub static CH_CHILD_LEAF: AtomicU64 = AtomicU64::new(0);
    pub static CH_CHILD_DECISION: AtomicU64 = AtomicU64::new(0);
    pub static CH_CHILD_CHANCE: AtomicU64 = AtomicU64::new(0);

    pub fn add(c: &AtomicU64, v: u64) {
        c.fetch_add(v, Relaxed);
    }

    pub fn dump_shape() {
        let s = SOLVES.load(Relaxed).max(1) as f64;
        println!(
            "  per solve: nodes {:.1} leaves {:.1} chance {:.1} inner c*a {:.0} cfg-slots {:.0} distinct-cfg {:.1}",
            NODES.load(Relaxed) as f64 / s,
            LEAVES.load(Relaxed) as f64 / s,
            CHANCE.load(Relaxed) as f64 / s,
            INNER_CA.load(Relaxed) as f64 / s,
            CFGSUM.load(Relaxed) as f64 / s,
            NCFG.load(Relaxed) as f64 / s,
        );
        println!(
            "  serialized: rows {:.1} levels {:.1} reach-slots {:.0} cells {:.0} draw-nnz {:.0} decision-reverse-nnz {:.0} draw-reverse-nnz {:.0} snapshot-cfg {:.0}",
            S_ROWS.load(Relaxed) as f64 / s,
            S_LEVELS.load(Relaxed) as f64 / s,
            S_REACH.load(Relaxed) as f64 / s,
            S_CELLS.load(Relaxed) as f64 / s,
            S_DRAW_NNZ.load(Relaxed) as f64 / s,
            S_DEC_REV_NNZ.load(Relaxed) as f64 / s,
            S_DRAW_REV_NNZ.load(Relaxed) as f64 / s,
            S_SNAP_CFG.load(Relaxed) as f64 / s,
        );
        let chance = CH_PARENT_ROOT.load(Relaxed)
            + CH_PARENT_DECISION.load(Relaxed)
            + CH_PARENT_CHANCE.load(Relaxed);
        let ch = chance.max(1) as f64;
        println!(
            "  chance topology: parent root/decision/chance {:.1}/{:.1}/{:.1}%, child leaf/decision/chance {:.1}/{:.1}/{:.1}%",
            100.0 * CH_PARENT_ROOT.load(Relaxed) as f64 / ch,
            100.0 * CH_PARENT_DECISION.load(Relaxed) as f64 / ch,
            100.0 * CH_PARENT_CHANCE.load(Relaxed) as f64 / ch,
            100.0 * CH_CHILD_LEAF.load(Relaxed) as f64 / ch,
            100.0 * CH_CHILD_DECISION.load(Relaxed) as f64 / ch,
            100.0 * CH_CHILD_CHANCE.load(Relaxed) as f64 / ch,
        );
    }
}

#[cfg(feature = "prof")]
pub use shape::*;

#[cfg(not(feature = "prof"))]
pub fn dump_shape() {}

// ---------------------------------------------------------- GPU live-set shape

#[cfg(feature = "prof")]
mod gpu_shape {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    static TICKS: AtomicU64 = AtomicU64::new(0);
    static RESIDENT_SUM: AtomicU64 = AtomicU64::new(0);
    static RESIDENT_MAX: AtomicU64 = AtomicU64::new(0);
    static ACTIVE_SUM: AtomicU64 = AtomicU64::new(0);
    static ACTIVE_MAX: AtomicU64 = AtomicU64::new(0);
    static VALUE_SUM: AtomicU64 = AtomicU64::new(0);
    static CARRY_SUM: AtomicU64 = AtomicU64::new(0);
    static DRAIN_SUM: AtomicU64 = AtomicU64::new(0);
    static WAITING_SUM: AtomicU64 = AtomicU64::new(0);
    static WAITING_MAX: AtomicU64 = AtomicU64::new(0);
    static ACTIVE_ROWS_SUM: AtomicU64 = AtomicU64::new(0);
    static ROW_SPAN_SUM: AtomicU64 = AtomicU64::new(0);
    static ROW_SPAN_MAX: AtomicU64 = AtomicU64::new(0);
    static ADMIT_BATCHES: AtomicU64 = AtomicU64::new(0);
    static ADMIT_JOBS: AtomicU64 = AtomicU64::new(0);
    static ADMIT_ROWS: AtomicU64 = AtomicU64::new(0);
    static ADMIT_CFGS: AtomicU64 = AtomicU64::new(0);
    static ADMIT_MAX: AtomicU64 = AtomicU64::new(0);

    #[allow(clippy::too_many_arguments)]
    pub fn gpu_tick(
        resident: usize,
        iterate: usize,
        value: usize,
        carry: usize,
        drain: usize,
        waiting: usize,
        active_rows: usize,
        row_span: usize,
    ) {
        let active = iterate + value;
        TICKS.fetch_add(1, Relaxed);
        RESIDENT_SUM.fetch_add(resident as u64, Relaxed);
        RESIDENT_MAX.fetch_max(resident as u64, Relaxed);
        ACTIVE_SUM.fetch_add(active as u64, Relaxed);
        ACTIVE_MAX.fetch_max(active as u64, Relaxed);
        VALUE_SUM.fetch_add(value as u64, Relaxed);
        CARRY_SUM.fetch_add(carry as u64, Relaxed);
        DRAIN_SUM.fetch_add(drain as u64, Relaxed);
        WAITING_SUM.fetch_add(waiting as u64, Relaxed);
        WAITING_MAX.fetch_max(waiting as u64, Relaxed);
        ACTIVE_ROWS_SUM.fetch_add(active_rows as u64, Relaxed);
        ROW_SPAN_SUM.fetch_add(row_span as u64, Relaxed);
        ROW_SPAN_MAX.fetch_max(row_span as u64, Relaxed);
    }

    pub fn gpu_admit(jobs: usize, rows: usize, cfgs: usize) {
        ADMIT_BATCHES.fetch_add(1, Relaxed);
        ADMIT_JOBS.fetch_add(jobs as u64, Relaxed);
        ADMIT_ROWS.fetch_add(rows as u64, Relaxed);
        ADMIT_CFGS.fetch_add(cfgs as u64, Relaxed);
        ADMIT_MAX.fetch_max(jobs as u64, Relaxed);
    }

    pub fn dump_gpu() {
        let ticks = TICKS.load(Relaxed).max(1) as f64;
        let batches = ADMIT_BATCHES.load(Relaxed).max(1) as f64;
        let active_rows = ACTIVE_ROWS_SUM.load(Relaxed) as f64 / ticks;
        let row_span = ROW_SPAN_SUM.load(Relaxed) as f64 / ticks;
        println!(
            "  gpu live set: ticks {} resident avg/max {:.1}/{} active avg/max {:.1}/{} value {:.1} carry {:.1} drain {:.1}",
            TICKS.load(Relaxed),
            RESIDENT_SUM.load(Relaxed) as f64 / ticks,
            RESIDENT_MAX.load(Relaxed),
            ACTIVE_SUM.load(Relaxed) as f64 / ticks,
            ACTIVE_MAX.load(Relaxed),
            VALUE_SUM.load(Relaxed) as f64 / ticks,
            CARRY_SUM.load(Relaxed) as f64 / ticks,
            DRAIN_SUM.load(Relaxed) as f64 / ticks,
        );
        println!(
            "  gpu queues: waiting avg/max {:.1}/{} active rows {:.0}, row span {:.0} ({:.2}x), max span {}",
            WAITING_SUM.load(Relaxed) as f64 / ticks,
            WAITING_MAX.load(Relaxed),
            active_rows,
            row_span,
            row_span / active_rows.max(1.0),
            ROW_SPAN_MAX.load(Relaxed),
        );
        println!(
            "  gpu admission: batches {} jobs/batch {:.2} max {} rows/batch {:.0} cfgs/batch {:.0}",
            ADMIT_BATCHES.load(Relaxed),
            ADMIT_JOBS.load(Relaxed) as f64 / batches,
            ADMIT_MAX.load(Relaxed),
            ADMIT_ROWS.load(Relaxed) as f64 / batches,
            ADMIT_CFGS.load(Relaxed) as f64 / batches,
        );
    }
}

#[cfg(feature = "prof")]
pub use gpu_shape::{dump_gpu, gpu_admit, gpu_tick};

#[cfg(not(feature = "prof"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_tick(
    _resident: usize,
    _iterate: usize,
    _value: usize,
    _carry: usize,
    _drain: usize,
    _waiting: usize,
    _active_rows: usize,
    _row_span: usize,
) {
}

#[cfg(not(feature = "prof"))]
pub fn gpu_admit(_jobs: usize, _rows: usize, _cfgs: usize) {}

#[cfg(not(feature = "prof"))]
pub fn dump_gpu() {}

/// `shape!(NODES, n)` — a no-op without the `prof` feature.
#[macro_export]
macro_rules! shape {
    ($c:ident, $v:expr) => {{
        #[cfg(feature = "prof")]
        $crate::prof::add(&$crate::prof::$c, $v as u64);
    }};
}
