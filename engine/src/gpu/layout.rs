//! The device layout — the one place a solve's device memory is described.
//!
//! A solve owns two blobs: a byte blob of tables uploaded from the job, and a
//! float blob of arenas the kernels write. Both are described exactly once,
//! by the `tables!` and `arenas!` invocations below. From those this module
//! derives three things that used to be written by hand and had to agree:
//!
//!   * the host packing (which bytes go where),
//!   * the offset slots the solve descriptor carries,
//!   * the CUDA accessors `kernels.cu` reads them through.
//!
//! Nothing else names an array. `kernels.cu` never declares a pointer field
//! and the host never writes an offset, so the two cannot drift: adding an
//! array means adding one line here. The descriptor's scalars are declared
//! the same way, by `scalars!`, and emitted into both languages.
//!
//! The generated CUDA is prepended to `kernels.cu` at NVRTC compile time, so
//! a mismatch is a compile error on the device, not a wrong answer.

use crate::serialize::TreeTables;

/// Element types an uploaded table may have. The CUDA spelling is what the
/// generated accessor casts to.
#[derive(Clone, Copy, PartialEq)]
pub enum Ty {
    U8,
    U32,
    I32,
    F32,
}

impl Ty {
    fn size(self) -> usize {
        match self {
            Ty::U8 => 1,
            _ => 4,
        }
    }

    fn cuda(self) -> &'static str {
        match self {
            Ty::U8 => "unsigned char",
            Ty::U32 => "unsigned int",
            Ty::I32 => "int",
            Ty::F32 => "float",
        }
    }
}

/// Tables the host derives rather than reads from the job: the solver's own
/// per-node offset arrays, and the belief vectors flattened for upload.
pub struct Derived {
    pub voff: Vec<u32>,
    pub act_off: Vec<u32>,
    /// Both players' root beliefs, concatenated (`nc_root[0]` then `[1]`).
    pub root: Vec<f32>,
    /// The carried roots Phase 2 values, each laid out like `root`.
    pub carried: Vec<f32>,
}

impl Derived {
    /// `voff`: the value arena's per-node offset, `max(nc0, nc1)` per node.
    /// `act_off`: the action-feature arena's per-node offset. Both mirror
    /// the layout `Solver` uses on the CPU.
    pub fn new(job: &crate::serialize::Job) -> Derived {
        let t = &job.tables;
        let mut root = job.root[0].clone();
        root.extend_from_slice(&job.root[1]);
        let mut carried = Vec::with_capacity(job.carried.len() * root.len());
        for c in &job.carried {
            carried.extend_from_slice(&c[0]);
            carried.extend_from_slice(&c[1]);
        }
        // An empty table would still need a valid offset; one zero keeps
        // every pointer inside the blob.
        if carried.is_empty() {
            carried.push(0.0);
        }
        let mut voff = Vec::with_capacity(t.nodes + 1);
        let mut acc = 0u32;
        for i in 0..t.nodes {
            voff.push(acc);
            let n0 = t.cfg_off[2 * i + 1] - t.cfg_off[2 * i];
            let n1 = t.cfg_off[2 * i + 2] - t.cfg_off[2 * i + 1];
            acc += n0.max(n1);
        }
        voff.push(acc);
        let mut act_off = Vec::with_capacity(t.nodes + 1);
        let mut acc = 0u32;
        for i in 0..t.nodes {
            act_off.push(acc);
            let (a0, a1) = (t.obs_off[i] as usize, t.obs_off[i + 1] as usize);
            if a1 > a0 {
                acc += t.obs_start[a1 - 1];
            }
        }
        act_off.push(acc);
        Derived { voff, act_off, root, carried }
    }

    /// The value arena's total length (the last `voff`).
    pub fn vals_len(&self) -> usize {
        *self.voff.last().unwrap() as usize
    }
}

/// Append `v` to `blob` at its natural alignment; return the byte offset.
fn put<T: Copy>(blob: &mut Vec<u8>, v: &[T]) -> u32 {
    let align = std::mem::size_of::<T>();
    while blob.len() % align != 0 {
        blob.push(0);
    }
    let at = blob.len();
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    blob.extend_from_slice(bytes);
    at as u32
}

/// Declare the uploaded tables. Each line is `name: type = slice`, where the
/// slice is read from the job `t` or the host-derived `d`.
macro_rules! tables {
    ($t:ident, $d:ident, $($name:ident : $ty:ident = $src:expr),* $(,)?) => {
        /// How many tables a descriptor carries offsets for.
        pub const N_TABLES: usize = [$(stringify!($name)),*].len();

        /// Pack every table into one blob, in declaration order.
        pub fn pack_tables($t: &TreeTables, $d: &Derived) -> (Vec<u8>, [u32; N_TABLES]) {
            let mut blob = Vec::new();
            let off = [$(put(&mut blob, $src)),*];
            (blob, off)
        }

        /// `#define T_name(d)` for each table: a typed pointer into the blob.
        fn table_defs() -> String {
            let mut s = String::new();
            for (i, (name, ty)) in [$((stringify!($name), Ty::$ty)),*].iter().enumerate() {
                s += &format!(
                    "#define T_{n}(d) ((const {c}*)((d)->tbl + (d)->toff[{i}]))\n",
                    n = name, c = ty.cuda(), i = i,
                );
            }
            s
        }

        /// The element type of each table, for the alignment self-check.
        #[cfg(test)]
        pub fn table_types() -> Vec<(&'static str, Ty)> {
            vec![$((stringify!($name), Ty::$ty)),*]
        }
    };
}

tables! { t, d,
    node_kind: U8 = &t.node_kind,
    node_player: U8 = &t.node_player,
    node_leaf: U8 = &t.node_leaf,
    node_child_start: U32 = &t.node_child_start,
    node_child: U32 = &t.node_child,
    obs_off: U32 = &t.obs_off,
    obs_start: U32 = &t.obs_start,
    obs_act: U32 = &t.obs_act,
    obs_child: U32 = &t.obs_child,
    legal_bits: U8 = &t.legal_bits,
    trans: I32 = &t.trans,
    draw_off: U32 = &t.draw_off,
    draw_to: U32 = &t.draw_to,
    draw_p: F32 = &t.draw_p,
    draw_steps: U8 = &t.draw_steps,
    draw_row_off: U32 = &t.draw_row_off,
    draw_row_start: U32 = &t.draw_row_start,
    cfg_off: U32 = &t.cfg_off,
    reach_off: U32 = &t.reach_off,
    soff: U32 = &t.soff,
    voff: U32 = &d.voff,
    act_off: U32 = &d.act_off,
    leaf_rows: U32 = &t.leaf_rows,
    term_leaves: U32 = &t.term_leaves,
    terminal_utility: F32 = &t.terminal_utility,
    leaf_coff: U32 = &t.leaf_coff,
    leaf_cidx: U32 = &t.leaf_cidx,
    bfs_order: U32 = &t.bfs_order,
    level_start: U32 = &t.level_start,
    leaf_xpub: F32 = &t.leaf_xpub,
    cphi: F32 = &t.cphi,
    psi_off: U32 = &t.psi_off,
    psi: F32 = &t.psi,
    ids: U8 = &t.ids,
    root: F32 = &d.root,
    carried: F32 = &d.carried,
}

/// The sizes a solve's arenas are cut from. Everything here is known at
/// admission, before any device memory is touched.
pub struct Sizes {
    pub reach_len: usize,
    pub vals_len: usize,
    pub ncells: usize,
    pub nsnaps: usize,
    pub ncfg: usize,
    pub nact: usize,
    pub nroot_cfg: usize,
    pub nroots: usize,
    pub rows: usize,
    pub dg: usize,
    pub rk: usize,
    pub de: usize,
}

/// Declare the per-solve float arenas. Each line is `name = length`.
macro_rules! arenas {
    ($n:ident, $($name:ident = $len:expr),* $(,)?) => {
        /// How many arenas a descriptor carries offsets for.
        pub const N_ARENAS: usize = [$(stringify!($name)),*].len();

        /// Cumulative arena offsets, in declaration order, and the total.
        pub fn arena_offsets($n: &Sizes) -> ([u32; N_ARENAS], usize) {
            let mut at = 0usize;
            let off = [$({ let a = at; at += $len; a as u32 }),*];
            (off, at)
        }

        /// `#define A_name(d)` for each arena: a float pointer into the blob.
        fn arena_defs() -> String {
            let mut s = String::new();
            for (i, name) in [$(stringify!($name)),*].iter().enumerate() {
                s += &format!("#define A_{n}(d) ((d)->arena + (d)->aoff[{i}])\n", n = name, i = i);
            }
            s
        }
    };
}

arenas! { n,
    reach = n.reach_len,
    vals = n.vals_len,
    regret = n.ncells,
    inst = n.ncells,
    cur = n.ncells,
    sum_strat = n.ncells,
    avg = n.ncells,
    snaps = n.nsnaps * n.ncells,
    e = crate::rebel::NTYPE * n.de,
    z = n.ncfg * n.dg,
    g = n.ncfg * (n.rk + 1),
    q = n.nact * n.rk,
    ph = n.rows * crate::rebel::NTYPE * n.de,
    beliefs = n.nsnaps * 2 * n.nroot_cfg,
    root_vals = n.nroots * 2 * n.nroot_cfg,
}

/// Declare the descriptor's scalars. Each becomes a field of the Rust struct
/// and a field of the CUDA struct, in the same order.
/// One scalar field, as CUDA. Accepts `i32`, `f32`, and fixed arrays of
/// them, which `stringify!` renders as `[i32 ; 2]`.
fn cuda_field(name: &str, ty: &str) -> String {
    let prim = |t: &str| if t.trim() == "f32" { "float" } else { "int" };
    match ty.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        Some(inner) => {
            let (elem, n) = inner.split_once(';').expect("array scalar");
            format!("  {} {}[{}];\n", prim(elem), name, n.trim())
        }
        None => format!("  {} {};\n", prim(ty), name),
    }
}

macro_rules! scalars {
    ($($name:ident : $ty:tt),* $(,)?) => {
        /// Per-solve state the kernels read. The pointer and offset fields
        /// come first and are fixed; everything after is declared above.
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Desc {
            pub tbl: *const u8,
            pub arena: *mut f32,
            pub toff: [u32; N_TABLES],
            pub aoff: [u32; N_ARENAS],
            $(pub $name: $ty,)*
        }

        fn desc_def() -> String {
            let mut s = format!(
                "typedef struct {{\n  const unsigned char* tbl;\n  float* arena;\n  \
                 unsigned int toff[{}];\n  unsigned int aoff[{}];\n",
                N_TABLES, N_ARENAS,
            );
            $(s += &cuda_field(stringify!($name), stringify!($ty));)*
            s + "} Desc;\n"
        }
    };
}

scalars! {
    nodes: i32,
    rows: i32,
    nleaf: i32,
    nterm: i32,
    ncells: i32,
    ncfg: i32,
    nlevels: i32,
    nsnaps: i32,
    nroots: i32,
    nact: i32,
    // The solve's stable row range in the service's row pool.
    row0: i32,
    // Config counts at the root and at the trip-2 exit leaf.
    nc_root: [i32; 2],
    nc_leaf: [i32; 2],
    leaf: i32,
    // Advancing state.
    t: i32,
    stage: i32,
    step: i32,
    traverser: i32,
    snap_t: i32,
    steps: [i32; 2],
    first_query: i32,
    snapshots: i32,
    // Per-phase switches the tick sets.
    mode: i32,
    p_player: i32,
    nplayers: i32,
    strat_src: i32,
    // CFR variant.
    alpha: f32,
    beta: f32,
    gamma: f32,
    predict: f32,
    warm: f32,
}

// SAFETY: `Desc` is a plain-old-data mirror of the CUDA struct. The raw
// pointers are device addresses, never dereferenced by the host.
unsafe impl cudarc::driver::DeviceRepr for Desc {}
unsafe impl cudarc::driver::ValidAsZeroBits for Desc {}
unsafe impl Send for Desc {}

impl Default for Desc {
    /// All-zero: the descriptor of an empty slot. Kernels never read one,
    /// because a slot only enters a launch group when it holds a solve.
    fn default() -> Desc {
        // SAFETY: every field is an integer, float, or device pointer.
        unsafe { std::mem::zeroed() }
    }
}

/// Solve stages. A tick advances every live solve by one step of its stage.
pub const STAGE_ITERATE: i32 = 0;
pub const STAGE_VALUE: i32 = 1;
pub const STAGE_CARRY: i32 = 2;

/// The generated CUDA preamble: the descriptor struct and every accessor.
/// Prepended to `kernels.cu` before NVRTC sees it.
pub fn cuda_preamble() -> String {
    format!(
        "// Generated by gpu/layout.rs — do not edit.\n{}{}{}{}",
        geometry_defs(),
        desc_def(),
        table_defs(),
        arena_defs(),
    )
}

/// Board and feature geometry. These are frozen by the job format; emitting
/// them from the Rust constants keeps `kernels.cu` from restating them.
fn geometry_defs() -> String {
    format!(
        "#define N_HEXES {}\n#define NSLOT {}\n#define NTYPE {}\n#define HEX_FACTS {}\n\
         #define HEX_CH {}\n#define OFF_LOOSE {}\n#define LOOSE {}\n#define OFF_PILES {}\n\
         #define PILE_COUNTS {}\n#define OFF_CARDS {}\n#define CARD_FEATS {}\n\
         #define AFEAT {}\n#define CFEAT {}\n#define N_UNITS {}\n",
        crate::board::N_HEXES,
        crate::rebel::NSLOT,
        crate::rebel::NTYPE,
        crate::rebel::HEX_FACTS,
        crate::rebel::HEX_CH,
        crate::rebel::OFF_LOOSE,
        crate::rebel::LOOSE,
        crate::rebel::OFF_PILES,
        crate::rebel::PILE_COUNTS,
        crate::rebel::OFF_CARDS,
        crate::units::CARD_FEATS,
        crate::rebel::AFEAT,
        crate::rebel::CFEAT,
        crate::units::N_UNITS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table's byte offset must be aligned for its element type, or the
    /// generated cast produces a misaligned device pointer. The packer
    /// aligns as it goes; this pins that it actually does.
    #[test]
    fn table_offsets_are_aligned() {
        let job = crate::serialize::Job::stub();
        let d = Derived::new(&job);
        let (_, off) = pack_tables(&job.tables, &d);
        for (i, (name, ty)) in table_types().into_iter().enumerate() {
            assert_eq!(off[i] as usize % ty.size(), 0, "{name} misaligned");
        }
    }

    #[test]
    fn preamble_dump() {
        println!("{}", cuda_preamble());
    }
}
