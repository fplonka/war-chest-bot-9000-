//! Host packing for one immutable v5 wave.
//!
//! Every solve-local offset is patched once into wave-global SoA space. Hot
//! kernels consume direct `(node, config)` task records; they never search a
//! live-slot prefix or recover ownership from a flattened index.

use std::borrow::Borrow;

use crate::rebel::{CFEAT, GPU_ROW_BYTES};
use crate::serialize::{device_value_layout, IndexWidth, PackedJob, PackedMeta, WorkVector};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Task {
    pub node: u32,
    pub config: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReadTask {
    pub node: u32,
    /// Network row, or `u32::MAX` for a terminal leaf.
    pub row: u32,
}

#[derive(Clone, Debug)]
pub struct JobSlice {
    pub nodes: std::ops::Range<usize>,
    pub rows: std::ops::Range<usize>,
    pub network_leaves: usize,
    pub configs: std::ops::Range<usize>,
    pub cells: std::ops::Range<usize>,
    pub reach: std::ops::Range<usize>,
    pub vals: std::ops::Range<usize>,
    pub root: std::ops::Range<usize>,
    pub carried: std::ops::Range<usize>,
    pub root_values: std::ops::Range<usize>,
    pub root_nc: [usize; 2],
    pub nroots: usize,
    pub exits: std::ops::Range<usize>,
    pub exit_configs: std::ops::Range<usize>,
}

#[derive(Clone, Debug)]
pub struct Wave {
    pub meta: PackedMeta,
    pub width: IndexWidth,
    pub jobs: Vec<JobSlice>,
    pub work: WorkVector,

    // Tree and sparse-cell SoA.
    pub node_kind: Vec<u8>,
    pub node_player: Vec<u8>,
    pub node_nc: Vec<u32>,
    pub node_child_start: Vec<u32>,
    pub node_child: Vec<u32>,
    pub legal_row_of: Vec<u32>,
    pub legal_off: Vec<u32>,
    /// Direct `A_VALS` index for each legal cell, or `u32::MAX` when the
    /// action has no successor in this depth-limited tree.
    pub legal_value: Vec<u32>,
    pub draw_off: Vec<u32>,
    pub draw_to: Vec<u32>,
    pub draw_p: Vec<f32>,
    pub draw_row_off: Vec<u32>,
    pub draw_row_start: Vec<u32>,
    /// Arena base for each `(node, player)` reach vector. A player's vector
    /// aliases its parent's base across public edges where that player did not
    /// act, so only changed beliefs occupy device storage.
    pub reach_base: Vec<u32>,
    pub reach_len: usize,
    /// Arena base for each `(node, player)` value vector. The two player
    /// layouts overlap because traversal is sequential, and an unchanged
    /// vector across an opponent chance edge aliases its sole child.
    pub vals_base: Vec<u32>,
    pub vals_len: usize,
    pub soff: Vec<u32>,
    pub node_parent: Vec<u32>,
    pub rev_row_of: Vec<u32>,
    pub rev_start: Vec<u32>,
    pub rev_src: Vec<u32>,
    pub rev_cell: Vec<u32>,
    pub rvd_row_of: Vec<u32>,
    pub rvd_start: Vec<u32>,
    pub rvd_src: Vec<u32>,
    pub rvd_p: Vec<f32>,

    // Network and roots.
    pub row_node: Vec<u32>,
    pub row_job: Vec<u32>,
    pub row_cfg_off: Vec<u32>,
    pub row_cfg: Vec<u32>,
    pub raw_rows: Vec<u8>,
    pub card_feat: Vec<f32>,
    pub ids: Vec<u8>,
    pub config_job: Vec<u32>,
    pub cphi: Vec<f32>,
    pub roots: Vec<f32>,
    pub carried: Vec<f32>,
    /// Terminal payoff in the terminal node's player perspective; zero for
    /// every non-terminal node. Keeping this node-addressable removes the last
    /// leaf-list search from readout.
    pub node_utility: Vec<f32>,
    pub terminal: Vec<ReadTask>,
    pub terminal_utility: Vec<f32>,

    // Result store shape.
    pub exit_nodes: Vec<u32>,
    pub exit_coff: Vec<u32>,
    pub snapshot_configs: usize,

    // Direct work maps. Boundaries index the corresponding task vector.
    pub decision: [Vec<Task>; 2],
    pub reach_task: [Vec<Task>; 2],
    pub reach_level: [Vec<u32>; 2],
    pub back_task: [Vec<Task>; 2],
    pub back_level: [Vec<u32>; 2],
    pub readout: Vec<ReadTask>,
}

impl Wave {
    pub fn compatible(a: &PackedJob, b: &PackedJob) -> bool {
        a.index_width() == b.index_width() && same_meta(&a.meta, &b.meta)
    }

    pub fn pack<J: Borrow<PackedJob>>(submitted: &[J]) -> Result<Wave, String> {
        let jobs: Vec<&PackedJob> = submitted.iter().map(Borrow::borrow).collect();
        let first = jobs.first().ok_or("cannot pack an empty wave")?;
        let width = first.index_width();
        for &j in &jobs[1..] {
            if !same_meta(&first.meta, &j.meta) {
                return Err("wave mixes different CFR schedules".into());
            }
            if j.index_width() != width {
                return Err("wave mixes narrow and wide local indices".into());
            }
        }
        let max_levels = jobs.iter().map(|j| j.tables.nlevels).max().unwrap_or(0);
        let mut w = Wave {
            meta: first.meta.clone(),
            width,
            jobs: Vec::with_capacity(jobs.len()),
            work: WorkVector::default(),
            node_kind: Vec::new(),
            node_player: Vec::new(),
            node_nc: Vec::new(),
            node_child_start: vec![0],
            node_child: Vec::new(),
            legal_row_of: Vec::new(),
            legal_off: vec![0],
            legal_value: Vec::new(),
            draw_off: Vec::new(),
            draw_to: Vec::new(),
            draw_p: Vec::new(),
            draw_row_off: Vec::new(),
            draw_row_start: Vec::new(),
            reach_base: Vec::new(),
            reach_len: 0,
            vals_base: Vec::new(),
            vals_len: 0,
            soff: vec![0],
            node_parent: Vec::new(),
            rev_row_of: Vec::new(),
            rev_start: vec![0],
            rev_src: Vec::new(),
            rev_cell: Vec::new(),
            rvd_row_of: Vec::new(),
            rvd_start: vec![0],
            rvd_src: Vec::new(),
            rvd_p: Vec::new(),
            row_node: Vec::new(),
            row_job: Vec::new(),
            row_cfg_off: vec![0],
            row_cfg: Vec::new(),
            raw_rows: Vec::new(),
            card_feat: Vec::new(),
            ids: Vec::new(),
            config_job: Vec::new(),
            cphi: Vec::new(),
            roots: Vec::new(),
            carried: Vec::new(),
            node_utility: Vec::new(),
            terminal: Vec::new(),
            terminal_utility: Vec::new(),
            exit_nodes: Vec::new(),
            exit_coff: vec![0],
            snapshot_configs: 0,
            decision: [Vec::new(), Vec::new()],
            reach_task: [Vec::new(), Vec::new()],
            reach_level: [vec![0; max_levels + 1], vec![0; max_levels + 1]],
            back_task: [Vec::new(), Vec::new()],
            back_level: [vec![0; max_levels + 1], vec![0; max_levels + 1]],
            readout: Vec::new(),
        };
        w.reserve_tables(&jobs);

        // Task maps are level-major across the whole wave. Pack the immutable
        // arrays first, then derive maps from the already-patched node ids.
        for (job_id, &job) in jobs.iter().enumerate() {
            w.push_job(job_id, job)?;
            add_work(&mut w.work, job.work());
        }
        w.build_tasks(&jobs, max_levels);
        w.validate()?;
        Ok(w)
    }

    /// Reserve the final concatenated sizes before patching. Large tail waves
    /// otherwise repeatedly copy tens or hundreds of MiB while each of these
    /// independent SoA vectors grows geometrically.
    fn reserve_tables(&mut self, jobs: &[&PackedJob]) {
        macro_rules! table_len {
            ($field:ident) => {
                jobs.iter().map(|j| j.tables.$field.len()).sum::<usize>()
            };
        }
        macro_rules! reserve {
            ($dst:ident, $n:expr) => {
                self.$dst.reserve($n)
            };
        }

        let nodes = table_len!(node_kind);
        let rows = jobs.iter().map(|j| j.tables.rows).sum::<usize>();
        let cfgs = jobs.iter().map(|j| j.tables.ncfg).sum::<usize>();
        reserve!(jobs, jobs.len());
        reserve!(node_kind, nodes);
        reserve!(node_player, nodes);
        reserve!(node_nc, 2 * nodes);
        reserve!(node_child_start, nodes);
        reserve!(node_child, table_len!(node_child));
        reserve!(legal_row_of, table_len!(legal_row_of));
        reserve!(legal_off, table_len!(legal_off).saturating_sub(jobs.len()));
        reserve!(legal_value, table_len!(legal_child));
        reserve!(draw_off, nodes);
        reserve!(draw_to, table_len!(draw_to));
        reserve!(draw_p, table_len!(draw_p));
        reserve!(draw_row_off, nodes);
        reserve!(draw_row_start, table_len!(draw_row_start));
        reserve!(reach_base, 2 * nodes);
        reserve!(vals_base, 2 * nodes);
        reserve!(soff, nodes);
        reserve!(node_parent, nodes);
        reserve!(rev_row_of, nodes);
        reserve!(rev_start, table_len!(rev_start).saturating_sub(jobs.len()));
        reserve!(rev_src, table_len!(rev_src));
        reserve!(rev_cell, table_len!(rev_cell));
        reserve!(rvd_row_of, nodes);
        reserve!(rvd_start, table_len!(rvd_start).saturating_sub(jobs.len()));
        reserve!(rvd_src, table_len!(rvd_src));
        reserve!(rvd_p, table_len!(rvd_p));
        reserve!(row_node, rows);
        reserve!(row_job, rows);
        reserve!(
            row_cfg_off,
            table_len!(leaf_coff).saturating_sub(jobs.len())
        );
        reserve!(row_cfg, table_len!(leaf_cidx));
        reserve!(raw_rows, table_len!(leaf_raw));
        reserve!(card_feat, table_len!(card_feat));
        reserve!(ids, table_len!(ids));
        reserve!(config_job, cfgs);
        reserve!(cphi, table_len!(cphi));
        reserve!(
            roots,
            jobs.iter()
                .map(|j| j.root[0].len() + j.root[1].len())
                .sum::<usize>()
        );
        reserve!(
            carried,
            jobs.iter()
                .flat_map(|j| &j.carried)
                .map(|r| r[0].len() + r[1].len())
                .sum::<usize>()
        );
        reserve!(node_utility, nodes);
        let terminals = table_len!(term_leaves);
        reserve!(terminal, terminals);
        reserve!(terminal_utility, terminals);
        reserve!(exit_nodes, table_len!(leaf_rows) + terminals);
        reserve!(exit_coff, table_len!(snap_coff).saturating_sub(jobs.len()));
        reserve!(readout, table_len!(leaf_rows) + terminals);
    }

    fn push_job(&mut self, job_id: usize, job: &PackedJob) -> Result<(), String> {
        let t = &job.tables;
        let node0 = self.node_kind.len();
        let child0 = self.node_child.len();
        let legal_row0 = self.legal_off.len() - 1;
        let cell0 = self.legal_value.len();
        let draw0 = self.draw_to.len();
        let draw_row0 = self.draw_row_start.len();
        let reach0 = self.reach_len;
        let vals0 = self.vals_len;
        let rev_row0 = self.rev_start.len() - 1;
        let rev0 = self.rev_src.len();
        let rvd_row0 = self.rvd_start.len() - 1;
        let rvd0 = self.rvd_src.len();
        let row0 = self.row_node.len();
        let member0 = self.row_cfg.len();
        let config0 = self.config_job.len();
        let root0 = self.roots.len();
        let carried0 = self.carried.len();
        let exit0 = self.exit_nodes.len();
        let exit_cfg0 = self.snapshot_configs;

        self.node_kind.extend_from_slice(&t.node_kind);
        self.node_player.extend_from_slice(&t.node_player);
        self.node_utility.resize(node0 + t.nodes, 0.0);
        for i in 0..t.nodes {
            self.node_nc.push(t.cfg_off[2 * i + 1] - t.cfg_off[2 * i]);
            self.node_nc
                .push(t.cfg_off[2 * i + 2] - t.cfg_off[2 * i + 1]);
        }
        self.node_child_start.extend(
            t.node_child_start
                .iter()
                .skip(1)
                .map(|&x| child0 as u32 + x),
        );
        self.node_child
            .extend(t.node_child.iter().map(|&x| node0 as u32 + x));
        self.legal_row_of.extend(t.legal_row_of.iter().map(|&x| {
            if x == u32::MAX {
                x
            } else {
                legal_row0 as u32 + x
            }
        }));
        self.legal_off
            .extend(t.legal_off.iter().skip(1).map(|&x| cell0 as u32 + x));
        self.draw_off
            .extend(t.draw_off.iter().take(t.nodes).map(|&x| draw0 as u32 + x));
        self.draw_to.extend_from_slice(&t.draw_to);
        self.draw_p.extend_from_slice(&t.draw_p);
        self.draw_row_off.extend(
            t.draw_row_off
                .iter()
                .take(t.nodes)
                .map(|&x| draw_row0 as u32 + x),
        );
        self.draw_row_start.extend_from_slice(&t.draw_row_start);
        self.reach_base.resize(2 * (node0 + t.nodes), 0);
        let mut reach_at = reach0;
        for &local_u in &t.bfs_order {
            let local = local_u as usize;
            let node = node0 + local;
            for p in 0..2 {
                let n = (t.cfg_off[2 * local + p + 1] - t.cfg_off[2 * local + p]) as usize;
                if local == 0 || t.node_player[t.node_parent[local] as usize] as usize == p {
                    self.reach_base[2 * node + p] = reach_at as u32;
                    reach_at += n;
                } else {
                    let parent = node0 + t.node_parent[local] as usize;
                    self.reach_base[2 * node + p] = self.reach_base[2 * parent + p];
                }
            }
        }
        self.reach_len = reach_at;
        self.soff
            .extend(t.soff.iter().skip(1).map(|&x| cell0 as u32 + x));
        let (local_vals, vals_len) = device_value_layout(t).ok_or("invalid value alias layout")?;
        self.vals_base.resize(2 * (node0 + t.nodes), 0);
        for local in 0..t.nodes {
            for p in 0..2 {
                self.vals_base[2 * (node0 + local) + p] = u32::try_from(
                    vals0
                        .checked_add(local_vals[p][local] as usize)
                        .ok_or("wave value offset overflow")?,
                )
                .map_err(|_| "wave value offset exceeds u32")?;
            }
        }
        self.vals_len = vals0
            .checked_add(vals_len)
            .ok_or("wave value length overflow")?;
        self.legal_value.resize(cell0 + t.ncells, u32::MAX);
        for local in 0..t.nodes {
            if t.node_kind[local] != 0 {
                continue;
            }
            let p = t.node_player[local] as usize;
            let nc = (t.cfg_off[2 * local + p + 1] - t.cfg_off[2 * local + p]) as usize;
            let row0 = t.legal_row_of[local] as usize;
            for row in row0..row0 + nc {
                let lo = t.legal_off[row] as usize;
                let hi = t.legal_off[row + 1] as usize;
                for cell in lo..hi {
                    let trans = t.legal_trans[cell];
                    if trans != u32::MAX {
                        let child = node0 + t.legal_child[cell] as usize;
                        self.legal_value[cell0 + cell] = self.vals_base[2 * child + p]
                            .checked_add(trans)
                            .ok_or("legal value offset overflow")?;
                    }
                }
            }
        }
        self.node_parent.extend(t.node_parent.iter().map(|&x| {
            if x == u32::MAX {
                x
            } else {
                node0 as u32 + x
            }
        }));
        self.rev_row_of.extend(t.rev_row_of.iter().map(|&x| {
            if x == u32::MAX {
                x
            } else {
                rev_row0 as u32 + x
            }
        }));
        self.rev_start
            .extend(t.rev_start.iter().skip(1).map(|&x| rev0 as u32 + x));
        self.rev_src.extend_from_slice(&t.rev_src);
        self.rev_cell
            .extend(t.rev_cell.iter().map(|&x| cell0 as u32 + x));
        self.rvd_row_of.extend(t.rvd_row_of.iter().map(|&x| {
            if x == u32::MAX {
                x
            } else {
                rvd_row0 as u32 + x
            }
        }));
        self.rvd_start
            .extend(t.rvd_start.iter().skip(1).map(|&x| rvd0 as u32 + x));
        self.rvd_src.extend_from_slice(&t.rvd_src);
        self.rvd_p.extend_from_slice(&t.rvd_p);

        self.row_node.extend(
            t.leaf_rows
                .iter()
                .chain(&t.inner_rows)
                .map(|&x| node0 as u32 + x),
        );
        self.row_job
            .resize(self.row_job.len() + t.rows, job_id as u32);
        self.row_cfg_off
            .extend(t.leaf_coff.iter().skip(1).map(|&x| member0 as u32 + x));
        self.row_cfg
            .extend(t.leaf_cidx.iter().map(|&x| config0 as u32 + x));
        self.raw_rows.extend_from_slice(&t.leaf_raw);
        self.card_feat.extend_from_slice(&t.card_feat);
        self.ids.extend_from_slice(&t.ids);
        self.config_job
            .resize(self.config_job.len() + t.ncfg, job_id as u32);
        self.cphi.extend_from_slice(&t.cphi);
        self.roots.extend_from_slice(&job.root[0]);
        self.roots.extend_from_slice(&job.root[1]);
        for root in &job.carried {
            self.carried.extend_from_slice(&root[0]);
            self.carried.extend_from_slice(&root[1]);
        }
        for (k, &n) in t.term_leaves.iter().enumerate() {
            self.node_utility[node0 + n as usize] = t.terminal_utility[k];
            self.terminal.push(ReadTask {
                node: node0 as u32 + n,
                row: u32::MAX,
            });
            self.terminal_utility.push(t.terminal_utility[k]);
        }
        self.exit_nodes.extend(
            t.leaf_rows
                .iter()
                .chain(&t.term_leaves)
                .map(|&x| node0 as u32 + x),
        );
        self.exit_coff
            .extend(t.snap_coff.iter().skip(1).map(|&x| exit_cfg0 as u32 + x));
        self.snapshot_configs += t.snapshot_configs;

        let root_n = job.root[0].len() + job.root[1].len();
        let carried_n = job.carried.len() * root_n;
        let root_value0 = self.jobs.last().map_or(0, |j| j.root_values.end);
        self.jobs.push(JobSlice {
            nodes: node0..node0 + t.nodes,
            rows: row0..row0 + t.rows,
            network_leaves: t.nleaf,
            configs: config0..config0 + t.ncfg,
            cells: cell0..cell0 + t.ncells,
            reach: reach0..reach_at,
            vals: vals0..self.vals_len,
            root: root0..root0 + root_n,
            carried: carried0..carried0 + carried_n,
            root_values: root_value0..root_value0 + carried_n,
            root_nc: [job.root[0].len(), job.root[1].len()],
            nroots: job.carried.len(),
            exits: exit0..exit0 + t.nleaf + t.nterm,
            exit_configs: exit_cfg0..exit_cfg0 + t.snapshot_configs,
        });
        debug_assert_eq!(self.raw_rows.len(), self.row_node.len() * GPU_ROW_BYTES);
        debug_assert_eq!(self.cphi.len(), self.config_job.len() * CFEAT);
        Ok(())
    }

    fn build_tasks(&mut self, jobs: &[&PackedJob], levels: usize) {
        let mut reach_caps = vec![[[0usize; 3]; 2]; levels];
        let mut decision_caps = [0usize; 2];
        let mut back_caps = [0usize; 2];
        for job in jobs {
            let t = &job.tables;
            for level in 0..t.nlevels {
                let lo = t.level_start[level] as usize;
                let hi = t.level_start[level + 1] as usize;
                for &local in &t.bfs_order[lo..hi] {
                    let node = local as usize;
                    let nc = [
                        (t.cfg_off[2 * node + 1] - t.cfg_off[2 * node]) as usize,
                        (t.cfg_off[2 * node + 2] - t.cfg_off[2 * node + 1]) as usize,
                    ];
                    if node != 0 {
                        let parent = t.node_parent[node] as usize;
                        for p in 0..2 {
                            if t.node_player[parent] as usize != p {
                                continue;
                            }
                            let mode = if t.rev_row_of[node] != u32::MAX { 1 } else { 2 };
                            reach_caps[level][p][mode] += nc[p];
                        }
                    }
                    if t.node_kind[node] != 2 {
                        for p in 0..2 {
                            if t.node_kind[node] == 1 && t.node_player[node] as usize != p {
                                continue;
                            }
                            back_caps[p] += nc[p];
                        }
                    }
                    if t.node_kind[node] == 0 {
                        let p = t.node_player[node] as usize;
                        decision_caps[p] += nc[p];
                    }
                }
            }
        }
        for p in 0..2 {
            self.decision[p].reserve(decision_caps[p]);
            self.reach_task[p].reserve(
                reach_caps
                    .iter()
                    .map(|level| level[p].iter().sum::<usize>())
                    .sum(),
            );
            self.back_task[p].reserve(back_caps[p]);
        }
        for level in 0..levels {
            // Keep the compact `(node, config)` record, but make neighbouring
            // CUDA threads take the same reach branch: strategy reverse
            // gathers, then chance reverse gathers. Unchanged opponent beliefs
            // alias their parent and need no task.
            let mut reach: [[Vec<Task>; 3]; 2] = std::array::from_fn(|p| {
                std::array::from_fn(|mode| Vec::with_capacity(reach_caps[level][p][mode]))
            });
            for (job_id, job) in jobs.iter().enumerate() {
                let t = &job.tables;
                if level >= t.nlevels {
                    continue;
                }
                let node0 = self.jobs[job_id].nodes.start as u32;
                let lo = t.level_start[level] as usize;
                let hi = t.level_start[level + 1] as usize;
                for &local in &t.bfs_order[lo..hi] {
                    let i = node0 + local;
                    if i != node0 {
                        for p in 0..2 {
                            let nc = self.nc(i as usize, p);
                            let node = i as usize;
                            let parent = self.node_parent[node] as usize;
                            if self.node_player[parent] as usize != p {
                                continue;
                            }
                            let mode = if self.rev_row_of[node] != u32::MAX {
                                1
                            } else {
                                2
                            };
                            for c in 0..nc {
                                reach[p][mode].push(Task {
                                    node: i,
                                    config: c as u32,
                                });
                            }
                        }
                    }
                    if t.node_kind[local as usize] != 2 {
                        for p in 0..2 {
                            if t.node_kind[local as usize] == 1
                                && t.node_player[local as usize] as usize != p
                            {
                                continue;
                            }
                            let nc = self.nc(i as usize, p);
                            for c in 0..nc {
                                self.back_task[p].push(Task {
                                    node: i,
                                    config: c as u32,
                                });
                            }
                        }
                    }
                }
            }
            for p in 0..2 {
                for bucket in &mut reach[p] {
                    self.reach_task[p].append(bucket);
                }
                self.reach_level[p][level + 1] = self.reach_task[p].len() as u32;
                self.back_level[p][level + 1] = self.back_task[p].len() as u32;
            }
        }
        for i in 0..self.node_kind.len() {
            if self.node_kind[i] == 0 {
                let p = self.node_player[i] as usize;
                for c in 0..self.nc(i, p) {
                    self.decision[p].push(Task {
                        node: i as u32,
                        config: c as u32,
                    });
                }
            }
        }
        for (job_id, job) in jobs.iter().enumerate() {
            let node0 = self.jobs[job_id].nodes.start as u32;
            let row0 = self.jobs[job_id].rows.start as u32;
            for (r, &node) in job.tables.leaf_rows.iter().enumerate() {
                self.readout.push(ReadTask {
                    node: node0 + node,
                    row: row0 + r as u32,
                });
            }
        }
        self.readout.extend_from_slice(&self.terminal);
    }

    #[inline]
    pub fn nc(&self, node: usize, player: usize) -> usize {
        self.node_nc[2 * node + player] as usize
    }

    fn validate(&self) -> Result<(), String> {
        let nodes = self.node_kind.len();
        if self.node_player.len() != nodes
            || self.node_utility.len() != nodes
            || self.node_nc.len() != 2 * nodes
            || self.node_child_start.len() != nodes + 1
            || self.reach_base.len() != 2 * nodes
            || self.vals_base.len() != 2 * nodes
            || self.soff.len() != nodes + 1
        {
            return Err("wave node arrays disagree".into());
        }
        if *self.node_child_start.last().unwrap() as usize != self.node_child.len()
            || *self.legal_off.last().unwrap() as usize != self.legal_value.len()
            || *self.soff.last().unwrap() as usize != self.legal_value.len()
        {
            return Err("wave sparse-cell arrays disagree".into());
        }
        if self.row_cfg_off.len() != 2 * self.row_node.len() + 1
            || *self.row_cfg_off.last().unwrap() as usize != self.row_cfg.len()
            || self.raw_rows.len() != self.row_node.len() * GPU_ROW_BYTES
            || self.cphi.len() != self.config_job.len() * CFEAT
        {
            return Err("wave network arrays disagree".into());
        }
        if self.exit_coff.len() != 2 * self.exit_nodes.len() + 1
            || *self.exit_coff.last().unwrap() as usize != self.snapshot_configs
        {
            return Err("wave carry arrays disagree".into());
        }
        for (p, tasks) in self.reach_task.iter().enumerate() {
            for task in tasks {
                if task.node as usize >= nodes
                    || task.config as usize >= self.nc(task.node as usize, p)
                {
                    return Err("wave reach task out of bounds".into());
                }
            }
        }
        for node in 0..nodes {
            for p in 0..2 {
                let end = self.reach_base[2 * node + p] as usize + self.nc(node, p);
                if end > self.reach_len {
                    return Err("wave reach alias out of bounds".into());
                }
                let end = self.vals_base[2 * node + p] as usize + self.nc(node, p);
                if end > self.vals_len {
                    return Err("wave value alias out of bounds".into());
                }
            }
        }
        if self
            .legal_value
            .iter()
            .any(|&x| x != u32::MAX && x as usize >= self.vals_len)
        {
            return Err("wave legal value out of bounds".into());
        }
        Ok(())
    }
}

fn same_meta(a: &PackedMeta, b: &PackedMeta) -> bool {
    a.depth == b.depth
        && a.iters == b.iters
        && a.snapshots == b.snapshots
        && a.cfr.alpha.to_bits() == b.cfr.alpha.to_bits()
        && a.cfr.beta.to_bits() == b.cfr.beta.to_bits()
        && a.cfr.gamma.to_bits() == b.cfr.gamma.to_bits()
        && a.cfr.predict.to_bits() == b.cfr.predict.to_bits()
        && a.warm.to_bits() == b.warm.to_bits()
        && a.snap_iters == b.snap_iters
        && a.net_dims == b.net_dims
}

fn add_work(dst: &mut WorkVector, x: WorkVector) {
    dst.network_rows += x.network_rows;
    dst.legal_cells += x.legal_cells;
    dst.reach_slots += x.reach_slots;
    dst.reverse_nonzeros += x.reverse_nonzeros;
    dst.table_bytes += x.table_bytes;
    dst.mutable_bytes += x.mutable_bytes;
    dst.carried_output_bytes += x.carried_output_bytes;
    dst.levels = dst.levels.max(x.levels);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_wave_patches_every_space() {
        let a = PackedJob::stub();
        let b = PackedJob::stub();
        let w = Wave::pack(&[a, b]).expect("pack");
        assert_eq!(w.jobs.len(), 2);
        assert_eq!(w.node_kind.len(), 2);
        assert_eq!(w.row_node, vec![0, 1]);
        assert_eq!(w.reach_base, vec![0, 1, 2, 3]);
        assert_eq!(w.reach_len, 4);
        assert_eq!(w.exit_nodes, vec![0, 1]);
        assert_eq!(w.exit_coff, vec![0, 1, 2, 3, 4]);
        assert_eq!(w.snapshot_configs, 4);
        assert_eq!(w.work.network_rows, 2);
    }
}
