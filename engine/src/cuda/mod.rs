//! The device backend: one round of the farm, on the GPU.
//!
//! The farm hands this a whole round — every solve in flight, one call each.
//! Those calls are not run one at a time. Calls of a kind are concatenated into
//! a single batch and the network runs **once per kind per round**, so a round
//! costs three chains of large GEMMs instead of a hundred small ones. That is
//! the entire reason the farm exists; a solve on its own is a couple of hundred
//! rows, which no accelerator is interested in.
//!
//! Two conventions carry the concatenation into the kernels:
//!
//! * a leaf's physical `xpub` row is `2 * r`. The paired canonical queries stay
//!   adjacent when calls are joined, so the copy `net::board` makes to pick the
//!   physical rows becomes a stride;
//! * anything that was constant within a call and varies across a batch — the
//!   card table a leaf reads, the seat a join asks about — becomes an index
//!   array.
//!
//! The arithmetic is `net.rs`, in the same order, and `tests/cuda_parity.rs`
//! holds it to `Backend::Reference` on the same weights.
//!
//! Every scratch buffer is allocated per pass. With event tracking off and one
//! stream per context, `CudaSlice` is a stream-ordered pool allocation, which
//! costs about as much as a kernel launch and keeps the code the same shape as
//! the CPU network.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchArgs, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use rayon::prelude::*;

use crate::board::{board, N_HEXES, NONE};
use crate::farm::{Call, Reply, CARD_ROWS};
use crate::net::{
    ln_block, Net, NetLayout, NormSpan, Span, C, CFGH, D, JBLOCKS, JOIN_IN, JW, LN_CFG, LN_H,
    LN_JOIN, LN_JOUT, LN_TRUNK, POOL, TYPE,
};
use crate::rebel::{
    CFEAT, HEX_CH, HEX_FACTS, LOOSE, NSLOT, NTYPE, OFF_LOOSE, OFF_PILES, PILE_COUNTS, PUBFEAT,
};

type Res<T> = Result<T, String>;

const KERNELS: &str = include_str!("kernels.cu");

/// Everything in `kernels.cu`, resolved once at startup so a name that does not
/// exist is an error there rather than a wrong answer later.
struct Kernels {
    gelu: CudaFunction,
    add: CudaFunction,
    layernorm: CudaFunction,
    bias: CudaFunction,
    group_bias: CudaFunction,
    window: CudaFunction,
    tokens: CudaFunction,
    hex_facts: CudaFunction,
    type_pool: CudaFunction,
    stem: CudaFunction,
    neighbour_mix: CudaFunction,
    pool: CudaFunction,
    board_input: CudaFunction,
    cfg_slots: CudaFunction,
    sum_slots: CudaFunction,
    bag: CudaFunction,
    join_input: CudaFunction,
}

impl Kernels {
    fn load(m: &Arc<CudaModule>) -> Res<Kernels> {
        let get = |name: &str| {
            m.load_function(name)
                .map_err(|e| format!("kernel {name}: {e:?}"))
        };
        Ok(Kernels {
            gelu: get("k_gelu")?,
            add: get("k_add")?,
            layernorm: get("k_layernorm")?,
            bias: get("k_bias")?,
            group_bias: get("k_group_bias")?,
            window: get("k_window")?,
            tokens: get("k_tokens")?,
            hex_facts: get("k_hex_facts")?,
            type_pool: get("k_type_pool")?,
            stem: get("k_stem")?,
            neighbour_mix: get("k_neighbour_mix")?,
            pool: get("k_pool")?,
            board_input: get("k_board_input")?,
            cfg_slots: get("k_cfg_slots")?,
            sum_slots: get("k_sum_slots")?,
            bag: get("k_bag")?,
            join_input: get("k_join_input")?,
        })
    }
}

/// `LaunchArgs::launch` hands back the events it may have recorded. Event
/// tracking is off here, so it never records any and no call site wants them.
trait LaunchUnit {
    /// # Safety
    /// The same contract as `LaunchArgs::launch`: the arguments must match the
    /// kernel's signature and stay in bounds.
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError>;
}

impl LaunchUnit for LaunchArgs<'_> {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.launch(cfg).map(|_| ())
    }
}

/// `launch!(self, kernel, elements, args...)` — one kernel over `elements`
/// work items. The builder is the same nine lines every time, and spelling it
/// out buries the arithmetic it is there to express.
macro_rules! launch {
    ($card:expr, $kernel:ident, $n:expr, $($arg:expr),+ $(,)?) => {{
        let n = $n;
        unsafe {
            $card.stream
                .launch_builder(&$card.k.$kernel)
                $(.arg($arg))+
                .launch_unit(spread(n))
        }
        .map_err(err)
    }};
}

/// Threads per block for the elementwise kernels.
const THREADS: u32 = 256;

fn spread(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One block per row, and a power-of-two block width so the reduction inside
/// `k_layernorm` halves cleanly.
fn per_row(rows: usize, width: usize) -> LaunchConfig {
    let threads = width.next_power_of_two().clamp(32, 256) as u32;
    LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 4 * threads,
    }
}

/// The GPUs a run evaluates on.
///
/// A round is split across the cards by call, so each card builds and runs a
/// self-contained batch and nothing crosses the bus between them.
pub struct Device {
    cards: Vec<Card>,
    net: Net,
}

struct Card {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    /// The weights exactly as `NetLayout` describes them.
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    /// Hex adjacency, `NONE` folded to `-1`.
    nb: CudaSlice<i32>,
    layout: NetLayout,
}

impl Device {
    /// Bring up one card per ordinal and upload the weights to each.
    pub fn new(ordinals: &[usize], net: Net) -> Res<Device> {
        if ordinals.is_empty() {
            return Err("no cuda device ordinals given".into());
        }
        if net.is_empty() {
            return Err("cannot start the device backend without weights".into());
        }
        let cards = ordinals
            .iter()
            .map(|&o| Card::new(o, &net))
            .collect::<Res<Vec<_>>>()?;
        Ok(Device { cards, net })
    }

    /// How many cards the driver can see.
    pub fn count() -> usize {
        CudaContext::device_count().unwrap_or(0).max(0) as usize
    }

    pub fn net(&self) -> &Net {
        &self.net
    }

    /// Evaluate a round. A device error is not recoverable and not worth
    /// limping past, so it stops the run.
    pub fn run(&self, calls: &[Call]) -> Vec<Reply> {
        match self.try_run(calls) {
            Ok(replies) => replies,
            Err(e) => panic!("cuda: {e}"),
        }
    }

    fn try_run(&self, calls: &[Call]) -> Res<Vec<Reply>> {
        // Deal each kind round-robin, so the cards get the same mix of work
        // and not, say, every trunk on one of them.
        let n = self.cards.len();
        let mut shards: Vec<Vec<usize>> = vec![Vec::new(); n];
        for kind in 0..3 {
            let mut at = 0;
            for (i, c) in calls.iter().enumerate() {
                if c.kind() == kind {
                    shards[at % n].push(i);
                    at += 1;
                }
            }
        }
        let mut out: Vec<Reply> = (0..calls.len()).map(|_| Reply::default()).collect();
        let done = self
            .cards
            .par_iter()
            .zip(shards)
            .map(|(card, mine)| card.round(calls, &mine))
            .collect::<Res<Vec<_>>>()?;
        for part in done {
            for (i, reply) in part {
                out[i] = reply;
            }
        }
        Ok(out)
    }
}

impl Card {
    fn new(ordinal: usize, net: &Net) -> Res<Card> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("device {ordinal}: {e:?}"))?;
        // One stream per context and no sharing between them, so the read/write
        // events cudarc would otherwise create on every allocation buy nothing
        // and cost two event creations per buffer.
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let ptx = compile_ptx_with_opts(
            KERNELS,
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e:?}"))?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(ptx).map_err(err)?;
        let k = Kernels::load(&module)?;
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let flat = net.flat();
        let nb: Vec<i32> = board()
            .neighbors
            .iter()
            .flatten()
            .map(|&n| if n == NONE { -1 } else { n as i32 })
            .collect();
        Ok(Card {
            w: stream.memcpy_stod(&flat.w).map_err(err)?,
            b: stream.memcpy_stod(&flat.b).map_err(err)?,
            ln: stream.memcpy_stod(&flat.ln).map_err(err)?,
            nb: stream.memcpy_stod(&nb).map_err(err)?,
            stream,
            blas,
            k,
            layout: NetLayout::new(),
        })
    }

    fn round(&self, calls: &[Call], mine: &[usize]) -> Res<Vec<(usize, Reply)>> {
        self.stream.context().bind_to_thread().map_err(err)?;
        let pick = |kind: usize| -> Vec<usize> {
            mine.iter()
                .copied()
                .filter(|&i| calls[i].kind() == kind)
                .collect()
        };
        let mut out = Vec::with_capacity(mine.len());
        self.trunk(calls, &pick(0), &mut out)?;
        self.configs(calls, &pick(1), &mut out)?;
        self.join(calls, &pick(2), &mut out)?;
        Ok(out)
    }

    // ------------------------------------------------------------ primitives

    /// `out[rows, o] = inp[rows, i] @ w[i, o] + beta * out[rows, o]`, the
    /// row-major shape of `Lin::run`.
    ///
    /// cuBLAS is column-major, so the very same buffers read as their own
    /// transposes give this with no transposes and no repacking: computing
    /// `outᵀ[o, rows] = wᵀ[o, i] @ inpᵀ[i, rows]`.
    fn lin<A: DevicePtr<f32>, O: DevicePtrMut<f32>>(
        &self,
        s: Span,
        inp: &A,
        rows: usize,
        beta: f32,
        out: &mut O,
    ) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let cfg = GemmConfig {
            transa: CUBLAS_OP_N,
            transb: CUBLAS_OP_N,
            m: s.o as i32,
            n: rows as i32,
            k: s.i as i32,
            alpha: 1.0,
            lda: s.o as i32,
            ldb: s.i as i32,
            beta,
            ldc: s.o as i32,
        };
        let w = self.w.slice(s.w..s.w + s.i * s.o);
        unsafe { self.blas.gemm(cfg, &w, inp, out) }.map_err(err)
    }

    /// The per-column bias the GEMM does not carry. A span with no bias is a
    /// no-op, which is how `Lin::bias` behaves on an empty bias.
    fn bias(&self, s: Span, rows: usize, out: &mut CudaSlice<f32>) -> Res<()> {
        if s.b == usize::MAX || rows == 0 {
            return Ok(());
        }
        let bias = self.b.slice(s.b..s.b + s.o);
        let (rows, width) = (rows as i32, s.o as i32);
        launch!(self, bias, rows as usize * s.o, out, &bias, &rows, &width)
    }

    /// `Lin::run`: the GEMM and then the bias.
    fn run<A: DevicePtr<f32>>(
        &self,
        s: Span,
        inp: &A,
        rows: usize,
        out: &mut CudaSlice<f32>,
    ) -> Res<()> {
        self.lin(s, inp, rows, 0.0, out)?;
        self.bias(s, rows, out)
    }

    /// `Norm::apply` when `act`, `Norm::plain` when not.
    fn norm(&self, s: NormSpan, rows: usize, act: bool, x: &mut CudaSlice<f32>) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let g = self.ln.slice(s.g..s.g + s.width);
        let b = self.ln.slice(s.b..s.b + s.width);
        let (rows_i, width, act) = (rows as i32, s.width as i32, act as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.layernorm)
                .arg(x)
                .arg(&g)
                .arg(&b)
                .arg(&rows_i)
                .arg(&width)
                .arg(&act)
                .launch_unit(per_row(rows, s.width))
        }
        .map_err(err)
    }

    fn add(&self, x: &mut CudaSlice<f32>, y: &CudaSlice<f32>, n: usize) -> Res<()> {
        let n_i = n as i32;
        launch!(self, add, n, x, y, &n_i)
    }

    fn alloc(&self, n: usize) -> Res<CudaSlice<f32>> {
        self.stream.alloc_zeros::<f32>(n.max(1)).map_err(err)
    }

    fn up<T: cudarc::driver::DeviceRepr>(&self, host: &[T]) -> Res<CudaSlice<T>> {
        self.stream.memcpy_stod(host).map_err(err)
    }

    fn down(&self, d: &CudaSlice<f32>, n: usize) -> Res<Vec<f32>> {
        self.stream.memcpy_dtov(&d.slice(0..n)).map_err(err)
    }

    // ----------------------------------------------------------------- trunk

    /// Every new leaf in the round: the board vector and the join cache.
    fn trunk(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        // Concatenate. `card_of_row` is what replaces `board`'s modulo: a leaf
        // reads the physical view of the card table its own solve drafted.
        let (mut xpub, mut cards, mut card_of_row) = (Vec::new(), Vec::new(), Vec::new());
        let mut rows = 0usize;
        for &i in mine {
            let Call::Trunk { xpub: xp, cards: cd, rows: n } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            // Concatenation only works if a call carries exactly its own rows.
            // A trailing tail from a caller's scratch buffer would shift every
            // later call in the batch and is invisible when a call runs alone.
            assert_eq!(xp.len(), 2 * n * PUBFEAT, "trunk xpub is not 2 rows a leaf");
            assert_eq!(cd.len(), CARD_ROWS * NTYPE * TYPE, "trunk card table");
            let base = (cards.len() / (NTYPE * TYPE)) as i32;
            xpub.extend_from_slice(xp);
            cards.extend_from_slice(cd);
            card_of_row.extend((0..*n).map(|r| base + ((2 * r) % CARD_ROWS) as i32));
            rows += n;
        }
        let cells = rows * N_HEXES;
        let stride = (2 * PUBFEAT) as i32;
        let (rows_i, cells_i) = (rows as i32, cells as i32);
        let (nhex, ntype, chan, nslot) = (N_HEXES as i32, NTYPE as i32, C as i32, NSLOT as i32);
        let l = &self.layout;

        let xpub = self.up(&xpub)?;
        let cards = self.up(&cards)?;
        let card_of_row = self.up(&card_of_row)?;

        // Tokens: projected pile counts, then the card token and seat on top.
        let mut piles = self.alloc(rows * NTYPE * PILE_COUNTS)?;
        let (off, width) = (OFF_PILES as i32, (NTYPE * PILE_COUNTS) as i32);
        launch!(self, window, rows * NTYPE * PILE_COUNTS, &xpub, &mut piles, &rows_i, &stride, &off, &width)?;
        let mut tokens = self.alloc(rows * NTYPE * TYPE)?;
        self.lin(l.pile, &piles, rows * NTYPE, 0.0, &mut tokens)?;
        let seat = self.w.slice(l.seat..l.seat + 2 * TYPE);
        let type_i = TYPE as i32;
        launch!(self, tokens, rows * NTYPE * TYPE, &cards, &card_of_row, &seat, &mut tokens, &rows_i, &ntype, &type_i, &nslot)?;

        // Stem.
        let mut projected = self.alloc(rows * NTYPE * C)?;
        self.run(l.tok_stem, &tokens, rows * NTYPE, &mut projected)?;
        let mut type_pool = self.alloc(rows * C)?;
        launch!(self, type_pool, rows * C, &projected, &mut type_pool, &rows_i, &ntype, &chan)?;
        let mut loose = self.alloc(rows * LOOSE)?;
        let (off, width) = (OFF_LOOSE as i32, LOOSE as i32);
        launch!(self, window, rows * LOOSE, &xpub, &mut loose, &rows_i, &stride, &off, &width)?;
        let mut glob = self.alloc(rows * C)?;
        self.run(l.glob_stem, &loose, rows, &mut glob)?;
        let mut facts = self.alloc(cells * HEX_FACTS)?;
        let mut occupant = self.stream.alloc_zeros::<i32>(cells.max(1)).map_err(err)?;
        let (hex_ch, hex_facts) = (HEX_CH as i32, HEX_FACTS as i32);
        launch!(self, hex_facts, cells, &xpub, &mut facts, &mut occupant, &rows_i, &stride, &nhex, &hex_ch, &hex_facts, &ntype)?;
        let mut x = self.alloc(cells * C)?;
        self.run(l.hex_stem, &facts, cells, &mut x)?;
        let pos = self.w.slice(l.pos..l.pos + N_HEXES * C);
        launch!(self, stem, cells * C, &mut x, &projected, &occupant, &pos, &glob, &type_pool, &cells_i, &nhex, &ntype, &chan)?;

        // Residual blocks over the board's adjacency.
        let mut a = self.alloc(cells * C)?;
        let mut mixed = self.alloc(cells * 2 * C)?;
        let mut y = self.alloc(cells * C)?;
        let mut pooled = self.alloc(rows * 2 * C)?;
        let mut gb = self.alloc(rows * C)?;
        let mut z = self.alloc(cells * C)?;
        for (i, blk) in l.blocks.iter().enumerate() {
            self.stream
                .memcpy_dtod(&x.slice(0..cells * C), &mut a)
                .map_err(err)?;
            self.norm(l.norms[ln_block(i, 0)], cells, true, &mut a)?;
            launch!(self, neighbour_mix, cells * C, &a, &self.nb, &mut mixed, &cells_i, &nhex, &chan)?;
            self.run(blk.mix, &mixed, cells, &mut y)?;
            launch!(self, pool, rows * C, &a, &mut pooled, &rows_i, &nhex, &chan)?;
            self.run(blk.pool, &pooled, rows, &mut gb)?;
            launch!(self, group_bias, cells * C, &mut y, &gb, &cells_i, &chan, &nhex)?;
            self.norm(l.norms[ln_block(i, 1)], cells, true, &mut y)?;
            self.run(blk.out, &y, cells, &mut z)?;
            self.add(&mut x, &z, cells * C)?;
        }
        self.norm(l.norms[LN_TRUNK], cells, true, &mut x)?;

        // The board head, and the half of the join that does not move between
        // CFR iterations.
        let width = 2 * C + LOOSE;
        let mut input = self.alloc(rows * width)?;
        let (off, loose_i) = (OFF_LOOSE as i32, LOOSE as i32);
        launch!(self, board_input, rows * width, &x, &xpub, &mut input, &rows_i, &nhex, &chan, &stride, &off, &loose_i)?;
        let mut p = self.alloc(rows * D)?;
        self.run(l.board_out, &input, rows, &mut p)?;
        let mut jp = self.alloc(rows * JW)?;
        self.run(l.join_p, &p, rows, &mut jp)?;

        let (p, jp) = (self.down(&p, rows * D)?, self.down(&jp, rows * JW)?);
        let mut at = 0;
        for &i in mine {
            let n = calls[i].rows();
            out.push((
                i,
                Reply {
                    a: p[at * D..(at + n) * D].to_vec(),
                    b: jp[at * JW..(at + n) * JW].to_vec(),
                },
            ));
            at += n;
        }
        Ok(())
    }

    // --------------------------------------------------------------- configs

    /// `f(c)` for the readout and `g(c)` for the pooling, for every config the
    /// round asked about.
    fn configs(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let (mut phi, mut owner, mut cards) = (Vec::new(), Vec::new(), Vec::new());
        let mut n = 0usize;
        for &i in mine {
            let Call::Configs { phi: ph, owner: ow, cards: cd, n: k } = &calls[i] else {
                unreachable!("config shard holds only config calls")
            };
            assert_eq!(ph.len(), k * CFEAT, "config phi is not one row a config");
            assert_eq!(ow.len(), *k, "config owner is not one entry a config");
            let base = (cards.len() / (NTYPE * TYPE)) as u32;
            phi.extend_from_slice(ph);
            owner.extend(ow.iter().map(|&q| q + base));
            cards.extend_from_slice(cd);
            n += k;
        }
        let views = cards.len() / (NTYPE * TYPE);
        let l = &self.layout;
        let (n_i, nslot, cfeat) = (n as i32, NSLOT as i32, CFEAT as i32);
        let (ntype, type_i, pool_i) = (NTYPE as i32, TYPE as i32, POOL as i32);

        let phi = self.up(&phi)?;
        let owner = self.up(&owner)?;
        let cards = self.up(&cards)?;

        let width = 3 + TYPE;
        let mut slots = self.alloc(n * NSLOT * width)?;
        launch!(self, cfg_slots, n * NSLOT * width, &phi, &owner, &cards, &mut slots, &n_i, &nslot, &cfeat, &ntype, &type_i)?;
        let mut hidden = self.alloc(n * NSLOT * CFGH)?;
        self.run(l.cfg1, &slots, n * NSLOT, &mut hidden)?;
        let hid = (n * NSLOT * CFGH) as i32;
        launch!(self, gelu, n * NSLOT * CFGH, &mut hidden, &hid)?;
        let mut u = self.alloc(n * CFGH)?;
        let cfgh = CFGH as i32;
        launch!(self, sum_slots, n * CFGH, &hidden, &mut u, &n_i, &nslot, &cfgh)?;
        self.norm(l.norms[LN_CFG], n, true, &mut u)?;
        let mut f = self.alloc(n * D)?;
        let mut g = self.alloc(n * POOL)?;
        self.run(l.cfg_f, &u, n, &mut f)?;
        self.run(l.cfg_g, &u, n, &mut g)?;

        // The linear half of `g`, which pooling carries exactly.
        let mut bag = self.alloc(views * NTYPE * 3 * POOL)?;
        self.run(l.cfg_m, &cards, views * NTYPE, &mut bag)?;
        launch!(self, bag, n * POOL, &bag, &phi, &owner, &mut g, &n_i, &nslot, &ntype, &cfeat, &pool_i)?;

        let (f, g) = (self.down(&f, n * D)?, self.down(&g, n * POOL)?);
        let mut at = 0;
        for &i in mine {
            let k = calls[i].rows();
            out.push((
                i,
                Reply {
                    a: f[at * D..(at + k) * D].to_vec(),
                    b: g[at * POOL..(at + k) * POOL].to_vec(),
                },
            ));
            at += k;
        }
        Ok(())
    }

    // ------------------------------------------------------------------ join

    /// The per-iteration head. Every leaf of every solve in the round shares
    /// one pass; the queried seat varies by row because the solves do.
    fn join(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let (mut p, mut jp, mut pooled, mut player) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut rows = 0usize;
        for &i in mine {
            let Call::Join { p: pv, jp: jv, pooled: bv, rows: n, player: q } = &calls[i] else {
                unreachable!("join shard holds only join calls")
            };
            assert_eq!(pv.len(), n * D, "join board vectors");
            assert_eq!(jv.len(), n * JW, "join cache");
            assert_eq!(bv.len(), 2 * n * POOL, "join pooled beliefs");
            p.extend_from_slice(pv);
            jp.extend_from_slice(jv);
            pooled.extend_from_slice(bv);
            player.extend(std::iter::repeat(*q as i32).take(*n));
            rows += n;
        }
        let l = &self.layout;
        let (rows_i, pool_i) = (rows as i32, POOL as i32);

        let p = self.up(&p)?;
        let jp = self.up(&jp)?;
        let pooled = self.up(&pooled)?;
        let player = self.up(&player)?;

        let mut input = self.alloc(rows * JOIN_IN)?;
        launch!(self, join_input, rows * JOIN_IN, &pooled, &player, &mut input, &rows_i, &pool_i)?;

        let mut z = self.alloc(rows * JW)?;
        self.stream
            .memcpy_dtod(&jp.slice(0..rows * JW), &mut z)
            .map_err(err)?;
        self.lin(l.join_b, &input, rows, 1.0, &mut z)?;
        self.bias(l.join_b, rows, &mut z)?;

        let mut t = self.alloc(rows * JW)?;
        let mut d = self.alloc(rows * JW)?;
        for i in 0..JBLOCKS {
            self.stream
                .memcpy_dtod(&z.slice(0..rows * JW), &mut t)
                .map_err(err)?;
            self.norm(l.norms[LN_JOIN + i], rows, true, &mut t)?;
            self.run(l.join_w[i], &t, rows, &mut d)?;
            self.add(&mut z, &d, rows * JW)?;
        }
        self.norm(l.norms[LN_JOUT], rows, true, &mut z)?;

        let mut h = self.alloc(rows * D)?;
        self.stream
            .memcpy_dtod(&p.slice(0..rows * D), &mut h)
            .map_err(err)?;
        self.lin(l.join_out, &z, rows, 1.0, &mut h)?;
        self.bias(l.join_out, rows, &mut h)?;
        self.norm(l.norms[LN_H], rows, false, &mut h)?;

        let h = self.down(&h, rows * D)?;
        let mut at = 0;
        for &i in mine {
            let n = calls[i].rows();
            out.push((
                i,
                Reply {
                    a: h[at * D..(at + n) * D].to_vec(),
                    b: Vec::new(),
                },
            ));
            at += n;
        }
        Ok(())
    }
}

/// The driver and cuBLAS error types are `Debug` only.
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
