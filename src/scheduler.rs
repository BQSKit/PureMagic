use crate::accum_start;
use crate::astar::{AStar, PathResult};
use crate::circuit::Circuit;
use crate::cultivation::CultivationManager;
use crate::debug_sched;
use crate::fn_timer;
use crate::frame::{CliffordFrame, FrameGate, SPauli};
use crate::info_sched;
use crate::node::NodeType;
use crate::pauliproduct::{GateType, Operator, PauliProduct};
use crate::steinertree::SteinerTree;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use crate::utils::AccumTimers;
use colored::{Color, Colorize};

use indexmap::{IndexMap, IndexSet};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

/// Per-lcycle and cumulative qubit utilisation counters.
pub(crate) struct ScheduleStats {
    data_qubits: usize,
    bus_qubits: usize,
    magic_qubits: usize,
    /// Cumulative totals across all lcycles (reset only at construction).
    tot_data_used: usize,
    tot_bus_used: usize,
    tot_magic_used: usize,
    /// Per-lcycle counters; reset to 0 at the end of each `update()` call.
    bus_used: usize,
    data_used: usize,
    magic_used: usize,
    /// T products that consumed a magic node this lcycle (first attempt only).
    t_scheduled: usize,
    /// Ready magic nodes used in any path this lcycle (routing or T-terminal).
    magic_ready_used: usize,
    tot_magic_unused: usize,
    plot_info: String,
}

impl ScheduleStats {
    pub(crate) fn new(data_qubits: usize, bus_qubits: usize, magic_qubits: usize) -> Self {
        ScheduleStats {
            data_qubits,
            bus_qubits,
            magic_qubits,
            tot_data_used: 0,
            tot_bus_used: 0,
            tot_magic_used: 0,
            bus_used: 0,
            data_used: 0,
            magic_used: 0,
            t_scheduled: 0,
            magic_ready_used: 0,
            tot_magic_unused: 0,
            plot_info: String::new(),
        }
    }

    pub(crate) fn summarize(&self, n_lcycles: usize) {
        let data_frac = self.tot_data_used as f64 / (self.data_qubits * n_lcycles) as f64;
        let bus_frac = self.tot_bus_used as f64 / (self.bus_qubits * n_lcycles) as f64;
        let magic_frac = self.tot_magic_used as f64 / (self.magic_qubits * n_lcycles) as f64;
        let magic_unused_frac =
            self.tot_magic_unused as f64 / (self.magic_qubits * n_lcycles) as f64;
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("Magic unused {:.3}", magic_unused_frac);
    }

    pub(crate) fn update(
        &mut self, lcycle_i: usize, pp_paths_len: usize, tot_available: usize, magic_ready: usize,
        magic_unused: usize, plotting: bool,
    ) {
        self.tot_data_used += self.data_used;
        self.tot_bus_used += self.bus_used;
        self.tot_magic_used += self.magic_used;
        self.tot_magic_unused += magic_unused;

        let tot_qubits = self.data_qubits + self.bus_qubits + self.magic_qubits;
        let tot_qubits_used = self.data_used + self.bus_used + self.magic_used;

        info_sched!("Scheduling results:");
        let frac_paths =
            if tot_available == 0 { 1.0 } else { pp_paths_len as f64 / tot_available as f64 };
        let frac_qubits =
            if tot_qubits == 0 { 0.0 } else { tot_qubits_used as f64 / tot_qubits as f64 };
        // magic_denom excludes nodes consumed purely as routing intermediaries (magic-routing mode).
        let magic_ready_routing = self.magic_ready_used.saturating_sub(self.t_scheduled);
        let magic_denom = magic_ready.saturating_sub(magic_ready_routing);
        let frac_magic =
            if magic_denom == 0 { 0.0 } else { self.t_scheduled as f64 / magic_denom as f64 };
        info_sched!("  products:    {}/{} ({:.2})", pp_paths_len, tot_available, frac_paths);
        info_sched!("  qubits:      {}/{} ({:.2})", tot_qubits_used, tot_qubits, frac_qubits);
        info_sched!("  magic:       {}/{} ({:.2})", self.t_scheduled, magic_denom, frac_magic);
        if plotting {
            self.plot_info = format!(
                "lcycle {}: products scheduled {}/{} ({:.2}), qubits {}/{} ({:.2}), magic {}/{} ({:.2})",
                lcycle_i,
                pp_paths_len,
                tot_available,
                frac_paths,
                tot_qubits_used,
                tot_qubits,
                frac_qubits,
                self.t_scheduled,
                magic_denom,
                frac_magic,
            );
        }

        self.data_used = 0;
        self.bus_used = 0;
        self.magic_used = 0;
        self.t_scheduled = 0;
        self.magic_ready_used = 0;
    }

    pub(crate) fn inc(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::Bus => self.bus_used += 1,
            NodeType::Magic => self.magic_used += 1,
            NodeType::Data => self.data_used += 1,
        }
    }

    pub(crate) fn inc_with_cultivation(&mut self, node_type: NodeType, cultivation_time: i32) {
        self.inc(node_type);
        if node_type == NodeType::Magic && cultivation_time == 0 {
            self.magic_ready_used += 1;
        }
    }

    pub(crate) fn inc_t(&mut self) {
        self.t_scheduled += 1;
    }

    pub(crate) fn plot_info_str(&self) -> &str {
        &self.plot_info
    }
}

/// Read-only inputs grouped so methods can borrow `input` and mutable state
/// fields simultaneously without conflicting borrows.
pub(crate) struct SchedulerInput {
    pub circuit: Circuit,
    pub topo: TopoGraph,
}

impl SchedulerInput {
    pub(crate) fn circuit_stem(&self) -> &str {
        use std::path::Path;
        Path::new(&self.circuit.circuit_fname)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("circuit")
    }
}

/// Lcycles a Litinski injection waits for a free neighbour to borrow as its |0> helper before
/// its magic patch is measured in place instead.
const HELPER_MAX_WAIT: usize = 2;

/// How a failed T injection is corrected (see docs/t-gate-corrections.md).
///
/// A T is a pi/8 rotation about a Pauli P, done by measuring P (x) Z_m with a magic patch m. Half
/// the time it comes out as -pi/8 and owes a pi/4 rotation about P. That correction is a rotation
/// about the whole of P (entangling when P has several qubits), so it cannot be recorded as
/// independent one-qubit S corrections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum TCorrection {
    /// Auto-corrected pi/8 rotation (Litinski, Game of Surface Codes). In the
    /// injection lcycle a free neighbour of the magic patch is borrowed as a |0> helper;
    /// Z_m (x) Y_helper is measured alongside P (x) Z_m, then the magic patch in X, which frees
    /// it. Once the injection outcome is decoded the helper is measured in X or Z, which applies
    /// the correction or not. No Clifford ever reaches the data: later products are never
    /// changed and no S gate is emitted. The helper is held until that decision (reaction
    /// latency, chained through earlier T gates whose Paulis anticommute). The helper is a bus
    /// patch, else a cultivating magic patch, else a ready magic state, whose state is wasted.
    /// An injection that finds no helper within `HELPER_MAX_WAIT` lcycles has its magic patch
    /// measured in place instead (in X, or in Y for a failed T; Gidney, "Inplace Access to the
    /// Surface Code Y Basis", Quantum 2024); with ready states allowed this is rare.
    LitinskiReady,
    /// As `litinski-ready`, but a ready magic state is never borrowed as the helper. Injections
    /// whose free neighbours are all ready states wait for a helper, and after `HELPER_MAX_WAIT`
    /// lcycles fall back to measuring the magic patch in place in X or Y (Gidney). The default:
    /// it never wastes a magic state, and is cheaper overall on the benchmarks
    /// (docs/t-gate-corrections.md).
    LitinskiY,
    /// Exact Clifford frame (src/frame.rs): a failed T's correction R_P(pi/4) is never applied.
    /// It is folded into a frame W, and every later T and measurement about Q runs about
    /// W.Q.W^dag, possibly on more qubits than Q. Physical S/SX/CX run unchanged and
    /// conjugate the frame. No helper and no hold, but a T or measurement whose rewritten
    /// Pauli depends on an undecoded injection outcome (it anticommutes with a T injected in
    /// the last L lcycles) waits until that outcome is known.
    Frame,
}

/// Counters for `--t-correction frame`, reported after scheduling.
#[derive(Default, Debug, Clone)]
pub(crate) struct FrameStats {
    /// Failed T gates folded into the frame, and physical Cliffords that conjugated it.
    pub(crate) failures: usize,
    pub(crate) cliffords: usize,
    /// Product-lcycles T gates and measurements waited for an undecoded outcome.
    pub(crate) waits: usize,
    /// Executed T gates and measurements, how many ran rewritten, their total weight as written
    /// and as executed, and the largest executed weight.
    pub(crate) products: usize,
    pub(crate) rewritten: usize,
    pub(crate) ideal_weight: usize,
    pub(crate) exec_weight: usize,
    pub(crate) max_exec_weight: usize,
    /// One entry per executed T or measurement: (lcycle, product, executed operators), and one
    /// per failed T: (lcycle, product). Kept in test and debug builds only.
    pub(crate) log: Vec<(usize, i32, Vec<Operator>)>,
    pub(crate) failed_log: Vec<(usize, i32)>,
}

/// Counters for `--t-correction litinski-ready` and `litinski-y`, reported after scheduling.
#[derive(Default, Debug, Clone)]
pub(crate) struct TCorrectionStats {
    /// Injections handled by the auto-correction model.
    pub(crate) injections: usize,
    /// |0> helper patches borrowed, and how many had to wait for a free neighbour.
    pub(crate) helpers: usize,
    pub(crate) helpers_deferred: usize,
    /// Helpers that were ready magic states (the state is wasted).
    pub(crate) helpers_ready: usize,
    /// Injections that found no helper within `HELPER_MAX_WAIT` lcycles (or whose magic patch
    /// has no neighbour that could ever be one). Their magic patch is measured in place (the
    /// in-place Y measurement) instead; see docs/t-gate-corrections.md.
    pub(crate) helpers_missing: usize,
    /// Magic patches measured in place, and how many of those needed the (slower) Y basis.
    pub(crate) inplace: usize,
    pub(crate) inplace_y: usize,
    /// Patch-lcycles spent holding magic and helper patches after their injection lcycle.
    pub(crate) hold_patch_lcycles: usize,
    /// Largest gap between an injection and its correction decision, in lcycles.
    pub(crate) max_decision_delay: usize,
    /// One entry per injection: (lcycle, T product, magic patch, decision lcycle). Helpers are
    /// logged separately as (lcycle, magic patch, helper), since a helper can come later.
    pub(crate) injection_log: Vec<(usize, i32, u16, usize)>,
    pub(crate) helper_log: Vec<(usize, u16, u16)>,
}

/// Assigns Pauli products to lcycles and routes them through the topology.
pub(crate) struct Scheduler {
    pub(crate) input: SchedulerInput,
    rng_uniform: StdRng,
    magic_state_lambda: f64,
    plot_option: String,
    pub(crate) cultivation: CultivationManager,
    pub(crate) stats: ScheduleStats,
    /// (lcycle_index, product_ids) for every lcycle that scheduled at least one product.
    pub(crate) lcycle_scheduled: Vec<(usize, Vec<i32>)>,
    pub(crate) scheduled_products: IndexSet<i32>,
    /// Per-node occupancy bitmap for the current lcycle; reset to all-false each lcycle.
    used: Vec<bool>,
    /// In-progress multi-lcycle Clifford gates: maps product_id →
    /// (remaining_lcycles, product, node_ids, opt_tree).
    /// CX occupies 2 lcycles; S/SX occupy 3.
    clifford_paths: IndexMap<i32, (usize, PauliProduct, Vec<u16>, Option<Rc<TreeGraph>>)>,
    pub(crate) t_gate_failures: usize,
    pub(crate) stree_computation: SteinerTree,
    pub(crate) astar: AStar,
    no_t_failures: bool,
    record_cultivation_dist: bool,
    /// Reusable buffers to avoid per-lcycle allocations.
    terminals_buf: Vec<u16>,
    scheduled_ids_buf: Vec<i32>,
    children_buf: Vec<i32>,
    precomputed_clifford_trees: HashMap<i32, Rc<TreeGraph>>,
    remaining_ids_buf: Vec<i32>,
    /// Indexed by product_id; built once before scheduling begins.
    precomputed_terminals: Vec<Vec<u16>>,
    /// (is_y_paired, preferred_roots, side_roots) per terminal per product.
    precomputed_root_info: Vec<Vec<(bool, Vec<u16>, Vec<u16>)>>,
    timers: AccumTimers,
    loop_timer: usize,
    other_timer: usize,
    pps_pending: Vec<PauliProduct>,
    /// Remaining unresolved parent count per product; decremented as parents complete.
    remaining_parents: Vec<usize>,
    pub(crate) pp_paths: Vec<(i32, Option<Rc<TreeGraph>>)>,
    current_lcycle: usize,
    /// Lcycles from a T injection until its outcome is decoded and can be acted on (L).
    reaction_latency: usize,
    /// How failed T injections are corrected.
    t_correction: TCorrection,
    /// Patches held after a T injection until its correction is decided: (last lcycle, node).
    held: Vec<(usize, u16)>,
    /// Litinski injections whose magic patch had no free neighbour for the |0> helper yet:
    /// (T product, magic node, decision lcycle, give-up lcycle). The magic patch is held until
    /// a helper is found.
    helper_wanted: Vec<(i32, u16, usize, usize)>,
    /// Magic patches measured in place, by T product: (magic node, decision lcycle). If the T
    /// failed, the patch is measured in Y, which takes `y_meas_lcycles` more.
    inplace_pending: HashMap<i32, (u16, usize)>,
    /// Failed T products still waiting for a helper (their measurement basis is not known yet).
    failed_waiting: HashSet<i32>,
    /// Extra lcycles an in-place Y measurement of a magic patch takes.
    y_meas_lcycles: usize,
    /// True if `hold_t_patches` occupied any patch, or a product waited for an undecoded
    /// outcome, in the current lcycle (an empty lcycle is then not a dead end).
    holding_now: bool,
    /// Per qubit, recent T products and the lcycle their correction is decided (decision chain).
    t_decisions: Vec<Vec<(i32, usize)>>,
    pub(crate) t_stats: TCorrectionStats,
    /// Frame mode: the Clifford frame (None in Litinski mode and with -F).
    frame: Option<CliffordFrame>,
    /// Frame mode: T injections whose outcome is not decoded yet, as (last undecided lcycle,
    /// ideal axis conjugated by the physical Cliffords run since).
    frame_undecided: Vec<(usize, SPauli)>,
    /// Frame mode: rewritten Paulis of products currently routed on them.
    frame_exec: HashMap<i32, Vec<Operator>>,
    /// Frame mode: physical Cliffords already folded into the frame (they span several lcycles).
    frame_applied: HashSet<i32>,
    pub(crate) frame_stats: FrameStats,
}

impl Scheduler {
    pub(crate) fn new(
        circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_level: &str,
        plot_option: String, rseed: u32, no_t_failures: bool, record_cultivation_dist: bool,
    ) -> Self {
        if log_level != "none" {
            // Installed regardless of build profile, but debug_sched!/info_sched! (see
            // utils.rs) are compiled out entirely in release, so a release build still
            // creates trace_fname, it just never gets anything written to it.
            let trace_fname = format!("{}.sched_trace", circuit_stem(&circuit.circuit_fname));
            let level_filter = match log_level.to_lowercase().as_str() {
                "debug" => log::LevelFilter::Debug,
                "info" => log::LevelFilter::Info,
                _ => log::LevelFilter::Off,
            };
            simple_logging::log_to_file(&trace_fname, level_filter)
                .expect("Failed to initialize logging");
        }
        let n_data_qubits = topo.n_data_qubits;
        let n_bus_qubits = topo.n_bus_qubits;
        let n_magic_qubits = topo.n_magic_qubits;
        let n_nodes = topo.n_nodes;
        let mut timers = AccumTimers::new();
        let loop_timer = timers.add_or_get("schedule loop");
        let other_timer = timers.add_or_get("other ");
        Scheduler {
            input: SchedulerInput { circuit, topo },
            rng_uniform: StdRng::seed_from_u64(rseed as u64),
            magic_state_lambda,
            plot_option,
            cultivation: CultivationManager::new(rseed),
            stats: ScheduleStats::new(n_data_qubits, n_bus_qubits, n_magic_qubits),
            lcycle_scheduled: Vec::new(),
            scheduled_products: IndexSet::new(),
            used: vec![false; n_nodes],
            clifford_paths: IndexMap::new(),
            t_gate_failures: 0,
            stree_computation: SteinerTree::new(n_nodes),
            astar: AStar::new(n_nodes),
            no_t_failures,
            record_cultivation_dist,
            terminals_buf: Vec::new(),
            scheduled_ids_buf: Vec::new(),
            children_buf: Vec::new(),
            precomputed_clifford_trees: HashMap::new(),
            remaining_ids_buf: Vec::new(),
            precomputed_terminals: Vec::new(),
            precomputed_root_info: Vec::new(),
            timers: timers,
            loop_timer: loop_timer,
            other_timer: other_timer,
            pps_pending: Vec::new(),
            remaining_parents: Vec::new(),
            pp_paths: Vec::new(),
            current_lcycle: 0,
            reaction_latency: 0,
            t_correction: TCorrection::LitinskiY,
            held: Vec::new(),
            helper_wanted: Vec::new(),
            inplace_pending: HashMap::new(),
            failed_waiting: HashSet::new(),
            y_meas_lcycles: 1,
            holding_now: false,
            t_decisions: Vec::new(),
            t_stats: TCorrectionStats::default(),
            frame: None,
            frame_undecided: Vec::new(),
            frame_exec: HashMap::new(),
            frame_applied: HashSet::new(),
            frame_stats: FrameStats::default(),
        }
    }

    /// Sets the reaction latency L in lcycles: the time from a T injection until its outcome is
    /// decoded and the correction can be decided (default 0, ideal).
    pub(crate) fn set_reaction_latency(&mut self, lcycles: usize) {
        self.reaction_latency = lcycles;
    }

    /// Sets how failed T injections are corrected (default: Litinski auto-correction).
    pub(crate) fn set_t_correction(&mut self, mode: TCorrection) {
        self.t_correction = mode;
    }

    /// Sets how many extra lcycles an in-place Y measurement of a magic patch takes (default 1:
    /// floor(d/2)+2 rounds, rounded up to whole lcycles).
    pub(crate) fn set_y_meas_lcycles(&mut self, lcycles: usize) {
        self.y_meas_lcycles = lcycles;
    }

    /// True if patches are held or wanted for T corrections, or a product is waiting for an
    /// undecoded outcome: an lcycle that schedules nothing is then not a dead end.
    fn holds_outstanding(&self) -> bool {
        self.holding_now || !self.held.is_empty() || !self.helper_wanted.is_empty()
    }

    // ── Frame mode ─────────────────────────────────────────────────────────────────────────

    /// Frame mode, start of an lcycle: forget injections whose outcome is now decoded.
    fn frame_begin_lcycle(&mut self) {
        let now = self.current_lcycle;
        self.frame_undecided.retain(|&(last, _)| last >= now);
    }

    /// Frame mode: prepares T or measurement `pp_id` to run in the current frame. Returns false
    /// if it must wait (its rewritten Pauli depends on an undecoded outcome). Otherwise its
    /// terminals and root candidates are those of W.Q.W^dag.
    fn frame_prepare(&mut self, pp_id: i32) -> bool {
        let Some(frame) = self.frame.as_ref() else {
            return true;
        };
        let ideal = SPauli::from_ops(frame.words, &self.input.circuit.product(pp_id).operators);
        if self.frame_undecided.iter().any(|(_, axis)| axis.anticommutes(&ideal)) {
            info_sched!("  {} waits for an undecoded T outcome", pp_id);
            self.frame_stats.waits += 1;
            self.holding_now = true; // not a dead end: the outcome arrives
            return false;
        }
        let exec = frame.image(&ideal);
        if exec == ideal {
            if self.frame_exec.remove(&pp_id).is_some() {
                // Back to the Pauli as written (the frame changed since the last attempt).
                let ops = self.input.circuit.product(pp_id).operators.clone();
                self.set_terminals(pp_id, &ops);
            }
            return true;
        }
        let ops = exec.to_ops();
        if self.frame_exec.get(&pp_id) != Some(&ops) {
            self.set_terminals(pp_id, &ops);
            self.frame_exec.insert(pp_id, ops);
        }
        true
    }

    /// Routes product `pp_id` on the data patches of `ops` instead of its own operators.
    fn set_terminals(&mut self, pp_id: i32, ops: &[Operator]) {
        let terminals = operators_to_node_ids(&self.input.topo, ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals[pp_id as usize] = terminals;
        self.precomputed_root_info[pp_id as usize] = root_info;
    }

    /// Frame mode: the operators product `pp_id` runs on (its own unless rewritten).
    fn exec_ops(&self, pp_id: i32) -> &[Operator] {
        match self.frame_exec.get(&pp_id) {
            Some(ops) => ops,
            None => &self.input.circuit.product(pp_id).operators,
        }
    }

    /// Frame mode: bookkeeping for T or measurement `pp_id`, scheduled this lcycle.
    fn frame_executed(&mut self, pp_id: i32) {
        if self.frame.is_none() {
            return;
        }
        let ideal_w = self.input.circuit.product(pp_id).operators.len();
        let ops = self.exec_ops(pp_id).to_vec();
        let st = &mut self.frame_stats;
        st.products += 1;
        st.ideal_weight += ideal_w;
        st.exec_weight += ops.len();
        st.max_exec_weight = st.max_exec_weight.max(ops.len());
        if self.frame_exec.contains_key(&pp_id) {
            st.rewritten += 1;
        }
        #[cfg(debug_assertions)]
        {
            let frame = self.frame.as_ref().unwrap();
            let label = SPauli::from_ops(frame.words, &ops).to_label(frame.n);
            info_sched!("  product {} executed as {}", pp_id, label);
        }
        if cfg!(any(test, debug_assertions)) {
            self.frame_stats.log.push((self.current_lcycle, pp_id, ops));
        }
        if self.input.circuit.product(pp_id).gate_type.is_t() {
            // Until decoded, its outcome decides the Pauli of anything that anticommutes with it.
            let frame = self.frame.as_ref().unwrap();
            let axis = SPauli::from_ops(frame.words, &self.input.circuit.product(pp_id).operators);
            self.frame_undecided.push((self.current_lcycle + self.reaction_latency, axis));
        }
    }

    /// Frame mode, end of an lcycle: fold failed T gates and newly started physical Cliffords
    /// into the frame.
    fn frame_end_lcycle(&mut self, failed: &[i32]) {
        let Some(frame) = self.frame.as_mut() else {
            return;
        };
        for &pp_id in failed {
            let axis = SPauli::from_ops(frame.words, &self.input.circuit.product(pp_id).operators);
            frame.absorb_failed_t(&axis);
            self.frame_stats.failures += 1;
            if cfg!(any(test, debug_assertions)) {
                self.frame_stats.failed_log.push((self.current_lcycle, pp_id));
            }
        }
        for i in 0..self.pp_paths.len() {
            let pp_id = self.pp_paths[i].0;
            if self.frame_applied.contains(&pp_id) {
                continue;
            }
            let pp = self.input.circuit.product(pp_id);
            let gate = match pp.gate_type {
                GateType::S => FrameGate::S(pp.operators[0].qubit as usize),
                GateType::SX => FrameGate::SX(pp.operators[0].qubit as usize),
                GateType::CX => {
                    let c = pp.operators.iter().find(|op| op.basis == 'Z').unwrap().qubit;
                    let t = pp.operators.iter().find(|op| op.basis == 'X').unwrap().qubit;
                    FrameGate::CX(c as usize, t as usize)
                }
                _ => continue,
            };
            frame.absorb_physical_clifford(gate);
            info_sched!("  physical Clifford {} folded into the frame", pp_id);
            for (_, axis) in &mut self.frame_undecided {
                axis.conjugate_by(gate);
            }
            self.frame_applied.insert(pp_id);
            self.frame_stats.cliffords += 1;
        }
    }

    // ── Litinski auto-correction ───────────────────────────────────────────────────────────

    /// True if the Paulis of products `a` and `b` anticommute (they differ on an odd number of
    /// shared qubits).
    fn anticommute(&self, a: i32, b: i32) -> bool {
        let pa = &self.input.circuit.product(a).operators;
        let pb = &self.input.circuit.product(b).operators;
        let mut odd = false;
        for oa in pa {
            if let Some(ob) = pb.iter().find(|ob| ob.qubit == oa.qubit) {
                odd ^= oa.basis != ob.basis;
            }
        }
        odd
    }

    /// The lcycle in which the correction of T product `pp_id`, injected this lcycle, is decided.
    ///
    /// Its own injection outcome is decoded after the reaction latency L. Whether a correction is
    /// owed also depends on the Pauli-frame bits of earlier T gates whose Paulis anticommute with
    /// this one (their byproduct flips this rotation's sign), and those bits are known only L
    /// lcycles after their own decision. So decisions chain: with L = 0 the decision is immediate.
    fn decision_lcycle(&self, pp_id: i32) -> usize {
        let now = self.current_lcycle;
        let l = self.reaction_latency;
        let mut decision = now + l;
        if l == 0 {
            return decision;
        }
        for op in &self.input.circuit.product(pp_id).operators {
            for &(j, d) in self.t_decisions.get(op.qubit as usize).map_or(&[][..], |v| v) {
                // An earlier decision binds until its frame bits are known (d + L).
                if d + l > now && d + l > decision && self.anticommute(j, pp_id) {
                    decision = d + l;
                }
            }
        }
        decision
    }

    /// Records T product `pp_id`'s decision lcycle on its qubits for later T gates.
    fn record_decision(&mut self, pp_id: i32, decision: usize) {
        let (now, l) = (self.current_lcycle, self.reaction_latency);
        if l == 0 {
            return;
        }
        for i in 0..self.input.circuit.product(pp_id).operators.len() {
            let q = self.input.circuit.product(pp_id).operators[i].qubit as usize;
            if self.t_decisions.len() <= q {
                self.t_decisions.resize(q + 1, Vec::new());
            }
            // Entries whose frame bits are already known can no longer bind.
            self.t_decisions[q].retain(|&(_, d)| d + l > now);
            self.t_decisions[q].push((pp_id, decision));
        }
    }

    /// Rank of node `nb` as a |0> helper for a T injection, lower is better, or None if it can
    /// never be one. A free routing (bus) patch comes first, then the cultivating magic patch
    /// with the least progress (its cultivation restarts), and last, with `litinski-ready` only,
    /// a ready magic state, which is wasted. A data patch never is. With magic routing (-u)
    /// there are no bus patches, so the helper is always a magic patch. Without the last resort
    /// (`litinski-y`), the free neighbours of a magic patch are often all ready states, and
    /// injections wait for a helper or fall back to measuring the magic patch in place.
    fn helper_rank(&self, nb: u16) -> Option<(u8, i32)> {
        let topo = &self.input.topo;
        match topo.node(nb).node_type {
            NodeType::Bus => Some((0, 0)),
            NodeType::Magic if topo.cultivation_times[nb as usize] > 0 => {
                Some((1, topo.busy_counts[nb as usize]))
            }
            NodeType::Magic if self.t_correction == TCorrection::LitinskiReady => Some((2, 0)),
            NodeType::Magic => None,
            NodeType::Data => None,
        }
    }

    /// A free neighbour of `magic` to borrow as a |0> helper (best `helper_rank`).
    fn pick_helper(&self, magic: u16) -> Option<u16> {
        let mut best: Option<((u8, i32), u16)> = None;
        for &nb in self.input.topo.node(magic).nbs_slice() {
            if self.used[nb as usize] {
                continue;
            }
            if let Some(rank) = self.helper_rank(nb) {
                if best.map_or(true, |(r, _)| rank < r) {
                    best = Some((rank, nb));
                }
            }
        }
        best.map(|(_, nb)| nb)
    }

    /// Occupies helper `h` (a neighbour of `magic`) until `decision` and logs it.
    fn take_helper(&mut self, magic: u16, h: u16, decision: usize) {
        info_sched!("    helper |0> patch {}", self.input.topo.label(h));
        if self.helper_rank(h).map_or(false, |r| r.0 == 2) {
            self.t_stats.helpers_ready += 1;
        }
        self.t_stats.helper_log.push((self.current_lcycle, magic, h));
        self.t_stats.helpers += 1;
        self.mark_nodes_used(&[h]);
        self.hold(h, decision);
    }

    /// True if some neighbour of `magic` could ever serve as its helper.
    fn helper_possible(&self, magic: u16) -> bool {
        self.input.topo.node(magic).nbs_slice().iter().any(|&nb| self.helper_rank(nb).is_some())
    }

    /// Keeps patch `node` occupied through lcycle `last` (nothing to do if `last` is now).
    fn hold(&mut self, node: u16, last: usize) {
        let now = self.current_lcycle;
        if last > now {
            self.held.push((last, node));
            self.t_stats.hold_patch_lcycles += last - now;
        }
    }

    /// Pushes T product `pp_id`'s recorded decision back by `by` lcycles: its frame bit comes
    /// from a measurement that ends that much later.
    fn delay_decision(&mut self, pp_id: i32, by: usize) {
        for i in 0..self.input.circuit.product(pp_id).operators.len() {
            let q = self.input.circuit.product(pp_id).operators[i].qubit as usize;
            if let Some(v) = self.t_decisions.get_mut(q) {
                for e in v.iter_mut().filter(|e| e.0 == pp_id) {
                    e.1 += by;
                }
            }
        }
    }

    /// Fallback when no helper can be found: the magic patch of T product `pp_id` is measured in
    /// place once the correction is decided (lcycle `decision`), in X, or in Y if the T failed.
    /// `failed` is None while the injection outcome has not been drawn yet (the injection
    /// lcycle); `process_t_gate_outcomes` then adds the Y time if needed.
    fn measure_in_place(&mut self, pp_id: i32, magic: u16, decision: usize, failed: Option<bool>) {
        self.t_stats.inplace += 1;
        let y = self.y_meas_lcycles;
        match failed {
            Some(true) => {
                self.t_stats.inplace_y += 1;
                self.delay_decision(pp_id, y);
                self.hold(magic, decision + y);
            }
            Some(false) => self.hold(magic, decision),
            None => {
                self.hold(magic, decision);
                self.inplace_pending.insert(pp_id, (magic, decision));
            }
        }
    }

    /// Bookkeeping for a T product injected this lcycle from magic patch `magic`.
    ///
    /// P (x) Z_m and Z_m (x) Y_h are measured together, then the magic patch in X, so the magic
    /// patch is free again after this lcycle. Only the helper h waits for the decision (it is
    /// measured in X to apply the correction, in Z not to).
    fn inject_t(&mut self, pp_id: i32, magic: u16) {
        let now = self.current_lcycle;
        let decision = self.decision_lcycle(pp_id);
        self.record_decision(pp_id, decision);
        self.t_stats.injections += 1;
        self.t_stats.injection_log.push((now, pp_id, magic, decision));
        self.t_stats.max_decision_delay = self.t_stats.max_decision_delay.max(decision - now);
        info_sched!(
            "  T {} injected from magic {}; correction decided in lcycle {}",
            pp_id,
            self.input.topo.label(magic),
            decision
        );
        match self.pick_helper(magic) {
            Some(h) => self.take_helper(magic, h, decision),
            None if self.helper_possible(magic) => {
                // The magic patch keeps its state until a helper is free (hold_t_patches).
                info_sched!("    no free neighbour for the helper yet");
                self.t_stats.helpers_deferred += 1;
                self.helper_wanted.push((pp_id, magic, decision, now + HELPER_MAX_WAIT));
            }
            None => {
                info_sched!("    no neighbour can be a helper; magic patch measured in place");
                self.t_stats.helpers_missing += 1;
                self.measure_in_place(pp_id, magic, decision, None);
            }
        }
    }

    /// Start of an lcycle: release expired holds, mark the rest used, and give injections that
    /// are still waiting for a |0> helper the first pick of free neighbours.
    fn hold_t_patches(&mut self) {
        let now = self.current_lcycle;
        self.held.retain(|&(last, _)| last >= now);
        let nodes: Vec<u16> = self.held.iter().map(|&(_, n)| n).collect();
        self.holding_now = !nodes.is_empty() || !self.helper_wanted.is_empty();
        if self.holding_now {
            info_sched!(
                "  held patches {:?}; waiting for a helper: {:?}",
                nodes.iter().map(|&n| self.input.topo.label(n)).collect::<Vec<_>>(),
                self.helper_wanted
                    .iter()
                    .map(|&(_, m, _, _)| self.input.topo.label(m))
                    .collect::<Vec<_>>()
            );
        }
        self.mark_nodes_used(&nodes);
        let wanted = std::mem::take(&mut self.helper_wanted);
        // Every waiting magic patch still holds its state: occupy them all before any helper is
        // picked, so no waiting patch is borrowed as another one's helper.
        for &(_, magic, _, _) in &wanted {
            self.mark_nodes_used(&[magic]);
            self.t_stats.hold_patch_lcycles += 1;
        }
        for (pp_id, magic, decision, give_up) in wanted {
            match self.pick_helper(magic) {
                Some(h) => {
                    // Z_m (x) Y_h and X_m now; the magic patch is free from the next lcycle.
                    self.take_helper(magic, h, decision);
                    self.failed_waiting.remove(&pp_id);
                }
                None if now < give_up => self.helper_wanted.push((pp_id, magic, decision, give_up)),
                None => {
                    // Waiting longer could deadlock (two magic patches each waiting on the
                    // other); measure the magic patch in place instead.
                    info_sched!(
                        "  no helper for {}; measured in place",
                        self.input.topo.label(magic)
                    );
                    self.t_stats.helpers_missing += 1;
                    let failed = self.failed_waiting.remove(&pp_id);
                    self.measure_in_place(pp_id, magic, decision, Some(failed));
                }
            }
        }
    }

    pub(crate) fn count_t_products(&self) -> usize {
        (0..self.input.circuit.n_products())
            .filter(|&id| self.input.circuit.product(id as i32).gate_type.is_t())
            .count()
    }

    /// Greedily assigns products to lcycles. Returns (total lcycles, total scheduled products).
    pub(crate) fn sched_circuit(&mut self) -> io::Result<(usize, usize)> {
        let _timer = fn_timer!();
        self.cultivation
            .set_lambda(self.magic_state_lambda)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        if self.t_correction == TCorrection::Frame && !self.no_t_failures {
            self.frame = Some(CliffordFrame::new(self.input.circuit.n_qubits));
        }
        self.init_magic_nodes();
        let n_t_products = self.count_t_products();
        self.cultivation.t_products_remaining = n_t_products;
        // Pre-generate a large pool: ~120 samples per T gate plus one per topology node.
        self.cultivation.fill_pool(120 * n_t_products.max(1) + self.input.topo.n_nodes);
        self.precompute_terminals_and_roots();
        self.precompute_multi_term_clifford_trees();
        self.pps_pending = self.input.circuit.initial_products().cloned().collect();
        self.remaining_parents = (0..self.input.circuit.n_products())
            .map(|id| self.input.circuit.product(id as i32).parents.len())
            .collect();
        debug_sched!("Initial pps_to_sched len {}", self.pps_pending.len());
        let mut plot_lcycles = 0usize;
        let mut path_dir: Option<String> = None;
        if self.plot_option.contains("paths") {
            let dir_name = format!("{}.paths", circuit_stem(&self.input.circuit.circuit_fname));
            std::fs::create_dir_all(&dir_name)?;
            path_dir = Some(dir_name);
            plot_lcycles = 30;
        }
        let plotting = path_dir.is_some();
        let tot_pps_to_sched = self.input.circuit.n_products();
        let mut prev_pct_complete = 0usize;
        self.current_lcycle = 0;
        self.pp_paths = Vec::new();
        while !self.pps_pending.is_empty() || !self.clifford_paths.is_empty() {
            self.timers.start(self.loop_timer);
            self.current_lcycle += 1;
            info_sched!(
                "{}",
                format!(
                    "lcycle {}: {:?}",
                    self.current_lcycle,
                    self.pps_pending
                        .iter()
                        .map(|pp| format!("{}:{}", pp.id, pp.to_operator_str()))
                        .collect::<Vec<_>>(),
                )
                .cyan()
            );
            if self.schedule_lcycle(plotting) {
                self.complete_lcycle()?;
                let n_scheduled = self.scheduled_products.len();
                // Show a progress bar during the non-plotting middle section of the run.
                // The first `plot_lcycles` and last `plot_lcycles` are plotted instead.
                if self.current_lcycle >= plot_lcycles
                    && n_scheduled + plot_lcycles <= tot_pps_to_sched
                {
                    if self.current_lcycle == plot_lcycles {
                        print!("Scheduling {} products:    ", tot_pps_to_sched);
                    }
                    let pct_complete = ((n_scheduled * 100) / tot_pps_to_sched).min(100);
                    if pct_complete > prev_pct_complete {
                        // \x08 is backspace; overwrite the previous "XX%" in-place.
                        print!("\x08\x08\x08{:02}%", pct_complete);
                        std::io::stdout().flush()?;
                        prev_pct_complete = pct_complete;
                    }
                    if n_scheduled + plot_lcycles == tot_pps_to_sched {
                        print!("\n");
                    }
                } else if plotting {
                    let plot_info_str = self.stats.plot_info_str();
                    assert!(!plot_info_str.is_empty());
                    let fname_added = format!(".{}", self.current_lcycle);
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir.as_ref().unwrap())?;
                    let pp_paths_plot: Vec<(PauliProduct, Rc<TreeGraph>, u32)> = self
                        .pp_paths
                        .iter()
                        .filter_map(|(pp_id, opt_tree)| {
                            opt_tree.as_ref().map(|t| {
                                let pp = self.input.circuit.product(*pp_id);
                                // Determine which cycle of a multi-lcycle product this is;
                                // clifford_paths already reflects post-advance_clifford_state.
                                let cycle: u32 = if pp.gate_type.is_clifford() {
                                    match self.clifford_paths.get(pp_id) {
                                        Some((c, _, _, _)) => {
                                            let initial =
                                                if pp.gate_type.is_cx() { 1u32 } else { 2u32 };
                                            initial - *c as u32 + 1
                                        }
                                        None => {
                                            if pp.gate_type.is_cx() {
                                                2
                                            } else {
                                                3
                                            }
                                        }
                                    }
                                } else if pp.gate_type.is_t() {
                                    1
                                } else {
                                    1
                                };
                                (pp.clone(), Rc::clone(t), cycle)
                            })
                        })
                        .collect();
                    self.input
                        .topo
                        .plot(&fname_added, &pp_paths_plot, &plot_info_str)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    std::env::set_current_dir(curr_dir)?;
                }
            } else {
                debug_sched!("Could not schedule anything on lcycle {}", self.current_lcycle);
                // Only error if T gates are pending and nothing can change: no patch is held for
                // a T correction (holds expire), no product waits for an undecoded outcome, and
                // no magic state is cultivating.
                let has_pending_t = self.pps_pending.iter().any(|pp| pp.gate_type.is_t());
                if has_pending_t
                    && !self.holds_outstanding()
                    && !(0..self.input.topo.n_nodes)
                        .any(|node_i| self.input.topo.is_cultivating(node_i as u16))
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Cannot schedule on current layout".red().to_string(),
                    ));
                }
            }
            self.timers.stop(self.loop_timer);
        }
        self.print_sched_stats(self.current_lcycle);
        #[cfg(debug_assertions)]
        self.check_clifford_repetitions()?;
        #[cfg(debug_assertions)]
        self.check_schedule()?;
        Ok((self.current_lcycle, self.scheduled_products.len()))
    }

    fn init_magic_nodes(&mut self) {
        self.cultivation.init_magic_nodes(&mut self.input.topo);
    }

    /// Multi-terminal non-T products have topology-independent routing trees that
    /// can be computed once on an empty grid and reused every lcycle.
    fn should_precompute(pp: &PauliProduct) -> bool {
        !pp.gate_type.is_t() && pp.operators.len() > 1
    }

    fn precompute_steiner_tree(&mut self, pp: &PauliProduct) -> Option<TreeGraph> {
        if !self.terminal_nodes(pp.id) {
            return None;
        }
        let root_ids = self.root_nodes(pp.id as usize, &self.terminals_buf[..]);
        if root_ids.is_empty() {
            return None;
        }
        self.stree_computation.compute(
            &self.input.topo,
            &self.used,
            &root_ids,
            &self.terminals_buf,
            pp.gate_type,
        )
    }

    /// Fills `terminals_buf` from precomputed IDs. Returns false early if any terminal
    /// is occupied or all its root candidates are occupied (cheap pre-check before `root_nodes`).
    #[inline]
    fn terminal_nodes(&mut self, pp_id: i32) -> bool {
        let pp_id = pp_id as usize;
        self.terminals_buf.clear();
        let terminals = &self.precomputed_terminals[pp_id];
        let root_info = &self.precomputed_root_info[pp_id];
        for (i, &node_id) in terminals.iter().enumerate() {
            if self.used[node_id as usize] {
                info_sched!("  Node {} is already used", node_id);
                return false;
            }
            let (_, preferred, side) = &root_info[i];
            // Only reject when candidates exist and all are occupied; empty preferred+side
            // means the gate uses sched_s_sx (no root needed), so don't reject vacuously.
            if (!preferred.is_empty() || !side.is_empty())
                && preferred.iter().all(|&rid| self.used[rid as usize])
                && side.iter().all(|&rid| self.used[rid as usize])
            {
                info_sched!("  No unused root candidates for node {}", node_id);
                return false;
            }
            self.terminals_buf.push(node_id);
        }
        true
    }

    /// Returns routing nodes adjacent to each terminal.
    /// For Y-basis operators the paired X/Z data nodes can share a single routing node
    /// (the "preferred" root), decrementing `unmatched_count` by 2; if none is free,
    /// each terminal falls back to a side nb.
    fn root_nodes(&self, pp_id: usize, terminals: &[u16]) -> Vec<u16> {
        let root_info = &self.precomputed_root_info[pp_id];
        let mut root_ids: Vec<u16> = Vec::new();
        let mut unmatched_count: usize = terminals.len();
        for (i, _node_id) in terminals.iter().enumerate() {
            let (is_paired, preferred, side) = &root_info[i];
            let mut pair_found = false;
            if *is_paired {
                for &nb_id in preferred {
                    if self.used[nb_id as usize] {
                        continue;
                    }
                    if !root_ids.contains(&nb_id) {
                        root_ids.push(nb_id);
                    }
                    // One root serves both X and Z terminals of the Y pair.
                    unmatched_count = unmatched_count.saturating_sub(2);
                    pair_found = true;
                    break;
                }
            }
            if !pair_found {
                let fallback = if *is_paired { side.as_slice() } else { preferred.as_slice() };
                for &nb_id in fallback {
                    if self.used[nb_id as usize] {
                        continue;
                    }
                    if !root_ids.contains(&nb_id) {
                        root_ids.push(nb_id);
                    }
                    unmatched_count = unmatched_count.saturating_sub(1);
                    break;
                }
            }
        }
        if unmatched_count > 0 {
            debug_sched!(
                "    could not find root nodes for {} unmatched terminals",
                unmatched_count
            );
            return Vec::new();
        }
        root_ids
    }

    fn precompute_multi_term_clifford_trees(&mut self) {
        let _timer = fn_timer!("precompute_clifford_trees");
        self.used.fill(false);
        let n_products = self.input.circuit.n_products();
        let mut n_precomputed = 0;
        for pp_id in 0..n_products {
            let pp = self.input.circuit.product(pp_id as i32).clone();
            if Self::should_precompute(&pp) {
                if let Some(tree) = self.precompute_steiner_tree(&pp) {
                    self.precomputed_clifford_trees.insert(pp.id, Rc::new(tree));
                    n_precomputed += 1;
                } else {
                    eprintln!(
                        "{}",
                        format!("Warning: failed to precompute tree for {}", pp).yellow()
                    );
                }
            }
        }
        println!("Precomputed {} multi-term Clifford trees", n_precomputed);
    }

    /// Computes root candidate info for a set of already-computed terminal node IDs.
    /// Extracted from `precompute_terminals_and_roots` so it can be called for dynamically
    /// constructed operator lists (a product rewritten by the Clifford frame).
    fn compute_root_info_for_terminals(
        topo: &TopoGraph, terminals: &[u16],
    ) -> Vec<(bool, Vec<u16>, Vec<u16>)> {
        let mut root_info = Vec::with_capacity(terminals.len());
        for &term_id in terminals {
            let node = topo.node(term_id);
            let is_paired =
                node.paired_data_id.map(|pid| terminals.contains(&pid)).unwrap_or(false);
            let mut preferred = Vec::new();
            let mut side = Vec::new();
            if is_paired {
                let is_x = topo.label(term_id).contains('X');
                for &nb_id in node.nbs_slice() {
                    let nb = topo.node(nb_id);
                    if !nb.is_routing() {
                        continue;
                    }
                    if (is_x && nb.pos.1 < node.pos.1) || (!is_x && nb.pos.1 > node.pos.1) {
                        preferred.push(nb_id);
                    } else if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        side.push(nb_id);
                    }
                }
            } else {
                for &nb_id in node.nbs_slice() {
                    let nb = topo.node(nb_id);
                    if nb.is_routing() && nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        preferred.push(nb_id);
                    }
                }
            }
            root_info.push((is_paired, preferred, side));
        }
        root_info
    }

    /// Precomputes terminal node IDs and root candidates for every product.
    ///
    /// For each terminal data node the root candidates are split into:
    /// - `preferred`: the routing node between the X and Z patches of a Y-pair
    ///   (X looks downward toward Z; Z looks upward toward X), or the same-row
    ///   side nb for unpaired terminals.
    /// - `side`: same-row side nbs used as fallback for Y-paired terminals.
    fn precompute_terminals_and_roots(&mut self) {
        let _timer = fn_timer!("precompute_terminals_and_roots");
        let n_products = self.input.circuit.n_products();
        self.precomputed_terminals = vec![Vec::new(); n_products];
        self.precomputed_root_info = vec![Vec::new(); n_products];
        for pp_id in 0..n_products {
            let pp = self.input.circuit.product(pp_id as i32).clone();
            let terminals = operators_to_node_ids(&self.input.topo, &pp.operators);
            let mut root_info: Vec<(bool, Vec<u16>, Vec<u16>)> =
                Vec::with_capacity(terminals.len());
            for &term_id in &terminals {
                let node = self.input.topo.node(term_id);
                // is_paired: the operator is Y-basis, so both its X and Z data nodes are terminals.
                let is_paired =
                    node.paired_data_id.map(|pid| terminals.contains(&pid)).unwrap_or(false);
                let mut preferred: Vec<u16> = Vec::new();
                let mut side: Vec<u16> = Vec::new();
                if is_paired {
                    // X nodes look downward (toward paired Z), Z nodes look upward.
                    let is_x = self.input.topo.label(term_id).contains('X');
                    for &nb_id in node.nbs_slice() {
                        let nb = self.input.topo.node(nb_id);
                        if !nb.is_routing() {
                            continue;
                        }
                        if (is_x && nb.pos.1 < node.pos.1) || (!is_x && nb.pos.1 > node.pos.1) {
                            preferred.push(nb_id);
                        } else if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                            side.push(nb_id);
                        }
                    }
                } else {
                    // Unpaired terminal: only same-row side nbs are valid roots.
                    for &nb_id in node.nbs_slice() {
                        let nb = self.input.topo.node(nb_id);
                        if nb.is_routing() && nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                            preferred.push(nb_id);
                        }
                    }
                }
                root_info.push((is_paired, preferred, side));
            }
            self.precomputed_terminals[pp_id] = terminals;
            self.precomputed_root_info[pp_id] = root_info;
        }
        println!("Precomputed terminals and root candidates for {} products", n_products);
    }

    fn mark_nodes_used(&mut self, node_ids: &[u16]) {
        for &node_id in node_ids {
            self.used[node_id as usize] = true;
            let node = self.input.topo.node(node_id);
            self.stats.inc_with_cultivation(
                node.node_type,
                self.input.topo.cultivation_times[node_id as usize],
            );
        }
    }

    /// Reserves the nodes of an in-progress multi-lcycle product for this lcycle.
    fn carry_forward_path(
        &mut self, pp_id: i32, node_ids: &[u16], opt_tree: Option<Rc<TreeGraph>>,
    ) {
        self.mark_nodes_used(node_ids);
        self.pp_paths.push((pp_id, opt_tree));
    }

    /// Schedules as many products as possible in one lcycle; returns false if nothing scheduled.
    fn schedule_lcycle(&mut self, plotting: bool) -> bool {
        let _timer = accum_start!(self.timers);
        self.timers.start(self.other_timer);
        let mut n_avail_magic = self.update_cultivators();
        let initial_magic = n_avail_magic;
        self.pp_paths.clear();
        self.used.fill(false);
        // Collect carry-forward data before the loop to release borrows on the map.
        let clifford_carry: Vec<(i32, Vec<u16>, Option<Rc<TreeGraph>>)> = self
            .clifford_paths
            .values()
            .map(|(_, pp, node_ids, opt_tree)| {
                (pp.id, node_ids.clone(), opt_tree.as_ref().map(Rc::clone))
            })
            .collect();
        for (pp_id, node_ids, opt_tree) in clifford_carry {
            self.carry_forward_path(pp_id, &node_ids, opt_tree);
        }
        match self.t_correction {
            TCorrection::LitinskiReady | TCorrection::LitinskiY => self.hold_t_patches(),
            TCorrection::Frame => {
                self.holding_now = false;
                self.frame_begin_lcycle();
            }
        }
        let carry_forward_count = self.clifford_paths.len();
        let tot_available = carry_forward_count + self.pps_pending.len();
        info_sched!("  Remaining to schedule: {}", self.pps_pending.len());
        self.sched_precomputed(plotting);
        self.timers.stop(self.other_timer);
        self.sched_remaining(&mut n_avail_magic, plotting);
        self.stats.update(
            self.current_lcycle,
            self.pp_paths.len(),
            tot_available,
            initial_magic,
            n_avail_magic,
            plotting,
        );
        if self.pp_paths.is_empty() {
            if n_avail_magic > 0 {
                // Only panic when T gates couldn't consume available magic. Patches held after
                // earlier injections can block every route for a few lcycles; that resolves
                // itself when the holds expire.
                let has_pending_t = self.pps_pending.iter().any(|pp| pp.gate_type.is_t());
                if has_pending_t && !self.holds_outstanding() {
                    panic!(
                        "{}",
                        format!(
                            "lcycle {}: Cannot schedule products [{}] on current layout ({} magic)",
                            self.current_lcycle,
                            self.pps_pending
                                .iter()
                                .map(|pp| pp.to_operator_str())
                                .collect::<Vec<_>>()
                                .join(", "),
                            n_avail_magic,
                        )
                        .red()
                    );
                }
            }
            false
        } else {
            true
        }
    }

    pub(crate) fn update_cultivators(&mut self) -> usize {
        let _timer = accum_start!(self.timers);
        let n_avail_magic = self.cultivation.update_cultivators(&mut self.input.topo, &self.used);
        info_sched!("  Available magic {}", n_avail_magic);
        n_avail_magic
    }

    /// First pass: schedule products with precomputed Clifford trees.
    /// Products whose tree nodes are all free are scheduled immediately.
    /// Products whose data nodes are blocked mark those nodes as used to prevent
    /// other products from routing through them this lcycle.
    fn sched_precomputed(&mut self, plotting: bool) {
        let _timer = accum_start!(self.timers);
        self.remaining_ids_buf.clear();
        self.remaining_ids_buf.extend(self.pps_pending.iter().map(|pp| pp.id));
        let mut to_remove: Vec<i32> = Vec::new();
        // Clone into a local Vec to release the borrow on `remaining_ids_buf`
        // before the loop body calls `&mut self` methods.
        let ids_to_process: Vec<i32> = self.remaining_ids_buf.clone();
        for &pp_id in &ids_to_process {
            // Frame mode: a multi-qubit measurement the frame rewrites cannot use its tree
            // (precomputed for the Pauli as written): route it now, on its rewritten terminals.
            if self.frame.is_some()
                && self.input.circuit.product(pp_id).gate_type.is_m()
                && self.precomputed_clifford_trees.contains_key(&pp_id)
            {
                if !self.frame_prepare(pp_id) {
                    continue;
                }
                if self.frame_exec.contains_key(&pp_id) {
                    let result = if !self.terminal_nodes(pp_id) {
                        PathResult::NoPath
                    } else if self.terminals_buf.len() == 1 {
                        self.sched_measurement(pp_id, plotting)
                    } else {
                        self.sched_t_or_multi(pp_id, plotting)
                    };
                    match result {
                        PathResult::PathFound(opt_graph) => {
                            to_remove.push(pp_id);
                            if let Some(ref g) = opt_graph {
                                let node_ids: Vec<u16> = g.iter_nodes().collect();
                                self.mark_nodes_used(&node_ids);
                            }
                            let keep = if plotting { opt_graph.map(Rc::new) } else { None };
                            self.pp_paths.push((pp_id, keep));
                            self.frame_executed(pp_id);
                        }
                        PathResult::NoPath => self.mark_terminals_used(pp_id),
                    }
                    continue;
                }
                // The frame leaves it as written: the precomputed tree applies.
            }
            // Clone the Rc to end the immutable borrow on `precomputed_clifford_trees`.
            let Some(tree) = self.precomputed_clifford_trees.get(&pp_id).map(Rc::clone) else {
                continue;
            };
            let all_free = tree.iter_nodes().all(|nid| !self.used[nid as usize]);
            if all_free {
                to_remove.push(pp_id);
                let node_ids: Vec<u16> = tree.iter_nodes().collect();
                let (_tree_n_nodes, _tree_n_edges) = (tree.n_nodes, tree.n_edges);
                let opt_tree: Option<Rc<TreeGraph>> =
                    if plotting { Some(Rc::clone(&tree)) } else { None };
                // Drop `tree` before the `&mut self` call to `mark_nodes_used`.
                drop(tree);
                self.mark_nodes_used(&node_ids);
                info_sched!(
                    "  Scheduled product {} (precomputed) with {} nodes and {} edges",
                    self.input.circuit.product(pp_id),
                    _tree_n_nodes,
                    _tree_n_edges
                );
                self.pp_paths.push((pp_id, opt_tree));
                if self.input.circuit.product(pp_id).gate_type.is_m() {
                    self.frame_executed(pp_id);
                }
            } else {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
            }
        }
        self.pps_pending.retain(|pp| !to_remove.contains(&pp.id));
    }

    /// Marks the data nodes product `pp_id` would run on (its current terminals, which the
    /// frame may have rewritten) as used, so other products cannot route through them.
    fn mark_terminals_used(&mut self, pp_id: i32) {
        for &nid in &self.precomputed_terminals[pp_id as usize] {
            self.used[nid as usize] = true;
        }
    }

    /// Marks the data nodes of a blocked product as used so that other products
    /// cannot route through them this lcycle (takes separate params to avoid
    /// borrow conflicts in caller loops).
    fn mark_blocked_product_as_used(used: &mut Vec<bool>, topo: &TopoGraph, pp: &PauliProduct) {
        for node_id in operators_to_node_ids(topo, &pp.operators) {
            used[node_id as usize] = true;
        }
    }

    /// Second pass: greedily schedule T gates, measurements, and S/SX gates.
    /// T gates are skipped when no magic state is available.
    fn sched_remaining(&mut self, n_avail_magic: &mut usize, plotting: bool) {
        let _timer = accum_start!(self.timers);
        for i in 0..self.pps_pending.len() {
            let pp_id = self.pps_pending[i].id;
            let pp = self.input.circuit.product(pp_id);
            if Self::should_precompute(pp) {
                continue;
            }
            let (pp_id, gate_type) = (pp.id, pp.gate_type);
            // Frame mode: a T or measurement runs about W.Q.W^dag, or waits if that depends on
            // an undecoded outcome. Physical S/SX run as written and conjugate the frame.
            if (gate_type.is_t() || gate_type.is_m()) && !self.frame_prepare(pp_id) {
                continue;
            }
            if *n_avail_magic > 0 || !gate_type.is_t() {
                info_sched!("  Trying to schedule product {}", self.input.circuit.product(pp_id));
                let result = if !self.terminal_nodes(pp_id) {
                    info_sched!(
                        "    Cannot schedule {}: no data nodes found in working graph",
                        pp_id
                    );
                    PathResult::NoPath
                } else if self.terminals_buf.len() == 1 && gate_type.is_m() {
                    self.sched_measurement(pp_id, plotting)
                } else if gate_type.is_s() || gate_type.is_sx() {
                    self.sched_s_sx(pp_id, plotting)
                } else {
                    self.sched_t_or_multi(pp_id, plotting)
                };
                if let PathResult::PathFound(opt_graph) = result {
                    info_sched!("  Scheduled product {}", self.input.circuit.product(pp_id));
                    if let Some(ref pp_graph) = opt_graph {
                        let node_ids: Vec<u16> = pp_graph.iter_nodes().collect();
                        self.mark_nodes_used(&node_ids);
                    }
                    let magic = opt_graph
                        .as_ref()
                        .and_then(|g| g.root_node_id)
                        .unwrap_or(self.astar.last_goal);
                    self.pp_paths.push((pp_id, opt_graph.map(Rc::new)));
                    if gate_type.is_t() || gate_type.is_m() {
                        self.frame_executed(pp_id);
                    }
                    if gate_type.is_t() {
                        *n_avail_magic -= 1;
                        self.stats.inc_t();
                        // With T failures switched off (-F) no correction is ever owed, so the
                        // auto-correction needs no helper and no hold.
                        if self.t_correction != TCorrection::Frame && !self.no_t_failures {
                            self.inject_t(pp_id, magic);
                        }
                    }
                    continue;
                }
            }
            info_sched!("  Could not schedule {} on graph", pp_id);
            if self.frame_exec.contains_key(&pp_id) {
                self.mark_terminals_used(pp_id);
            } else {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
            }
        }
    }

    fn sched_measurement(&mut self, _pp_id: i32, plotting: bool) -> PathResult {
        let node_id = self.terminals_buf[0];
        let node = self.input.topo.node(node_id);
        if self.used[node.id as usize] {
            info_sched!(
                "    Cannot schedule {}: node for M {} is used",
                _pp_id,
                self.input.topo.label(node_id)
            );
            return PathResult::NoPath;
        }
        if !plotting {
            self.used[node_id as usize] = true;
            self.stats.inc_with_cultivation(
                node.node_type,
                self.input.topo.cultivation_times[node_id as usize],
            );
            return PathResult::PathFound(None);
        }
        let mut g = TreeGraph::new(self.input.topo.n_nodes);
        g.add_node(node, self.input.topo.label(node_id));
        PathResult::PathFound(Some(g))
    }

    /// Schedules an S or SX gate: data node plus one same-row ancilla nb.
    /// S/SX require two adjacent nodes in the same row (the data qubit and its
    /// horizontal routing nb).
    fn sched_s_sx(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        let node_id = self.terminals_buf[0];
        let node = self.input.topo.node(node_id);
        if self.used[node.id as usize] {
            let _gate_type = self.input.circuit.product(pp_id).gate_type;
            info_sched!(
                "    Cannot schedule {}: node for {:?} {} is used",
                pp_id,
                _gate_type,
                self.input.topo.label(node_id)
            );
            return PathResult::NoPath;
        }
        for &nb_id in node.nbs_slice() {
            let nb = self.input.topo.node(nb_id);
            if nb.pos.1 == node.pos.1 {
                info_sched!(
                    "    product {} on node {} has available ancilla {}",
                    self.input.circuit.product(pp_id),
                    self.input.topo.label(node_id),
                    self.input.topo.label(nb_id)
                );
                if !self.used[nb_id as usize] {
                    if !plotting {
                        self.used[node_id as usize] = true;
                        self.used[nb_id as usize] = true;
                        self.stats.inc_with_cultivation(
                            node.node_type,
                            self.input.topo.cultivation_times[node_id as usize],
                        );
                        self.stats.inc_with_cultivation(
                            nb.node_type,
                            self.input.topo.cultivation_times[nb_id as usize],
                        );
                        return PathResult::PathFound(None);
                    }
                    let mut g = TreeGraph::new(self.input.topo.n_nodes);
                    g.add_node(node, self.input.topo.label(node_id));
                    g.add_node(nb, self.input.topo.label(nb_id));
                    g.add_edge(node_id, nb_id);
                    return PathResult::PathFound(Some(g));
                }
            }
        }
        info_sched!("    Cannot schedule S/SX {}: no available ancilla", pp_id);
        PathResult::NoPath
    }

    fn sched_t_or_multi(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        debug_assert!(!self.terminals_buf.iter().any(|node_id| self.used[*node_id as usize]));
        let root_ids = self.root_nodes(pp_id as usize, &self.terminals_buf[..]);
        if root_ids.is_empty() {
            info_sched!("    Cannot schedule {}: no roots available", pp_id);
            return PathResult::NoPath;
        }
        let gate_type = self.input.circuit.product(pp_id).gate_type;
        let n_operators = self.exec_ops(pp_id).len();
        let result = if gate_type.is_t() && n_operators == 1 {
            self.astar.compute(
                &self.terminals_buf[..],
                &root_ids[..],
                &self.input.topo,
                &mut self.used,
                &self.cultivation.ready_magic_positions,
                plotting,
            )
        } else {
            debug_assert!(
                !Self::should_precompute(self.input.circuit.product(pp_id))
                    || self.frame_exec.contains_key(&pp_id),
                "should_precompute product {:?} reached Steiner path",
                pp_id
            );
            match self.stree_computation.compute(
                &self.input.topo,
                &self.used,
                &root_ids,
                &self.terminals_buf,
                gate_type,
            ) {
                Some(tree) => PathResult::PathFound(Some(tree)),
                None => PathResult::NoPath,
            }
        };
        if let PathResult::PathFound(opt_g) = result {
            return PathResult::PathFound(opt_g);
        }
        info_sched!("    Cannot schedule {}: no path found", pp_id);
        PathResult::NoPath
    }

    /// Post-lcycle bookkeeping: remove scheduled products, unlock children,
    /// advance Clifford state, and process T-gate coin-flip outcomes.
    fn complete_lcycle(&mut self) -> io::Result<()> {
        let _timer = accum_start!(self.timers);
        self.scheduled_ids_buf.clear();
        self.scheduled_ids_buf.extend(self.pp_paths.iter().map(|(id, _)| *id));
        self.pps_pending.retain(|pp| !self.scheduled_ids_buf.contains(&pp.id));
        debug_sched!("After purge, pps_to_sched len {}", self.pps_pending.len());
        let t_newly_scheduled = self
            .pp_paths
            .iter()
            .filter(|(id, _)| self.input.circuit.product(*id).gate_type.is_t())
            .count();
        self.cultivation.t_products_remaining =
            self.cultivation.t_products_remaining.saturating_sub(t_newly_scheduled);
        self.process_t_gate_outcomes();
        self.unlock_children();
        self.advance_clifford_state();
        debug_sched!(
            "After inserting previous lcycle cliffords, pps_to_sched len {}",
            self.pps_pending.len()
        );
        self.pps_pending
            .extend(self.children_buf.iter().map(|&id| self.input.circuit.product(id).clone()));
        debug_sched!(
            "After adding {} children, pps_to_sched len {}",
            self.children_buf.len(),
            self.pps_pending.len()
        );
        let lcycle_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        self.lcycle_scheduled.push((self.current_lcycle, lcycle_ids));
        #[cfg(debug_assertions)]
        self.check_lcycle()?;
        self.scheduled_products.extend(self.pp_paths.iter().map(|(id, _)| *id));
        Ok(())
    }

    /// Coin-flip T gate outcomes.
    ///
    /// T gates complete in one lcycle regardless of outcome; a failed one owes R_P(pi/4) about
    /// its whole Pauli P. Litinski: the correction is a choice of helper measurement basis, so
    /// nothing is owed on the data (a magic patch measured in place instead needs the slower Y
    /// basis). Frame: the correction is folded into the Clifford frame, with the physical
    /// Cliffords that started this lcycle.
    fn process_t_gate_outcomes(&mut self) {
        let pp_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        let mut failed = Vec::new();
        for pp_id in pp_ids {
            if !self.input.circuit.product(pp_id).gate_type.is_t() {
                continue;
            }
            if self.no_t_failures || self.rng_uniform.gen_bool(0.5) {
                info_sched!("  T gate {} succeeded", pp_id);
                self.inplace_pending.remove(&pp_id);
                continue;
            }
            self.t_gate_failures += 1;
            if self.frame.is_some() {
                info_sched!("  T gate {} failed (50% probability), folded into the frame", pp_id);
                failed.push(pp_id);
                continue;
            }
            info_sched!("  T gate {} failed (50% probability), auto-corrected", pp_id);
            if let Some((magic, decision)) = self.inplace_pending.remove(&pp_id) {
                let y = self.y_meas_lcycles;
                self.t_stats.inplace_y += 1;
                self.delay_decision(pp_id, y);
                // Already held through `decision`; extend by the Y measurement.
                let now = self.current_lcycle;
                let start = decision.max(now);
                self.held.push((decision + y, magic));
                self.t_stats.hold_patch_lcycles += decision + y - start;
            } else if self.helper_wanted.iter().any(|w| w.0 == pp_id) {
                self.failed_waiting.insert(pp_id);
            }
        }
        self.frame_end_lcycle(&failed);
    }

    /// Decrements `remaining_parents` for each completed product and collects
    /// newly-ready children into `children_buf`.
    fn unlock_children(&mut self) {
        self.children_buf.clear();
        for i in 0..self.pp_paths.len() {
            let pp_id = self.pp_paths[i].0;
            let pp = self.input.circuit.product(pp_id);
            let gate_type = pp.gate_type;
            if gate_type.is_clifford() {
                match self.clifford_paths.get(&pp_id) {
                    // S/SX: count==2 means this is the second of three lcycles;
                    // children are not unlocked until the third (final) lcycle.
                    Some((count, _, _, _)) if *count == 2 => {
                        debug_assert!(gate_type.is_s() || gate_type.is_sx());
                        continue;
                    }
                    // count==1 (CX) or count==1 (S/SX final): fall through to unlock.
                    // None means this is the first lcycle: children not yet unlocked.
                    None => continue,
                    _ => {}
                }
            }
            let children: Vec<i32> = self.input.circuit.product(pp_id).children.clone();
            for child_id in children {
                self.remaining_parents[child_id as usize] -= 1;
                if self.remaining_parents[child_id as usize] == 0
                    && !self.children_buf.contains(&child_id)
                {
                    self.children_buf.push(child_id);
                }
            }
        }
    }

    /// Advances multi-lcycle Clifford state: decrements remaining-lcycle counters
    /// and inserts new entries for first-seen Clifford products.
    ///
    /// CX requires 2 consecutive lcycles (initial count = 1, removed when it hits 0).
    /// S/SX require 3 consecutive lcycles (initial count = 2).
    fn advance_clifford_state(&mut self) {
        let pp_paths_snapshot: Vec<(i32, Option<Rc<TreeGraph>>)> =
            self.pp_paths.iter().map(|(id, opt)| (*id, opt.as_ref().map(Rc::clone))).collect();
        for (pp_id, opt_pp_path) in &pp_paths_snapshot {
            let pp_id = *pp_id;
            let pp = self.input.circuit.product(pp_id);
            if !pp.gate_type.is_clifford() {
                continue;
            }
            let gate_type = pp.gate_type;
            if let Some(clifford_path) = self.clifford_paths.get_mut(&pp_id) {
                clifford_path.0 -= 1;
                if clifford_path.0 == 0 {
                    self.clifford_paths.swap_remove(&pp_id);
                }
            } else {
                // First lcycle for this Clifford: insert with remaining count.
                let count = if gate_type.is_cx() { 1 } else { 2 };
                let node_ids: Vec<u16> = if let Some(tree) = opt_pp_path {
                    tree.iter_nodes().collect()
                } else {
                    self.precomputed_clifford_trees
                        .get(&pp_id)
                        .map(|t| t.iter_nodes().collect())
                        .unwrap_or_default()
                };
                let pp_owned = self.input.circuit.product(pp_id).clone();
                self.clifford_paths.insert(
                    pp_id,
                    (count, pp_owned, node_ids, opt_pp_path.as_ref().map(Rc::clone)),
                );
            }
        }
    }

    fn print_sched_stats(&mut self, n_lcycles: usize) {
        self.stats.summarize(n_lcycles);
        let tot_t = self.count_t_products();
        let fail_pct =
            if tot_t > 0 { 100.0 * self.t_gate_failures as f64 / tot_t as f64 } else { 0.0 };
        println!("Magic state cultivation time:");
        let mean = self.cultivation.cultivation_times_log.iter().sum::<i32>() as f64
            / self.cultivation.cultivation_times_log.len() as f64;
        let min = self.cultivation.cultivation_times_log.iter().min().copied().unwrap_or(0);
        let max = self.cultivation.cultivation_times_log.iter().max().copied().unwrap_or(0);
        println!("  number:  {}", self.cultivation.cultivation_times_log.len());
        println!("  average: {:.2}", mean);
        println!("  min:     {}", min);
        println!("  max:     {}", max);
        println!("T gate failures: {}/{} ({:.1}%)", self.t_gate_failures, tot_t, fail_pct);
        match self.t_correction {
            TCorrection::Frame => {
                let f = &self.frame_stats;
                let avg =
                    |w: usize| if f.products > 0 { w as f64 / f.products as f64 } else { 0.0 };
                println!(
                    "T correction: Frame; {} failures and {} physical Cliffords folded into the \
                     frame, {} T/M rewritten of {}, mean weight {:.2} as written -> {:.2} executed, \
                     max {}, {} product-lcycles waiting for decoding",
                    f.failures,
                    f.cliffords,
                    f.rewritten,
                    f.products,
                    avg(f.ideal_weight),
                    avg(f.exec_weight),
                    f.max_exec_weight,
                    f.waits
                );
            }
            mode => println!(
                "T correction: {:?}; {} injections, {} helper patches ({} ready states \
                 wasted, {} waited for one, {} found none), {} patch-lcycles held, max decision \
                 delay {} lcycles, {} measured in place ({} in Y)",
                mode,
                self.t_stats.injections,
                self.t_stats.helpers,
                self.t_stats.helpers_ready,
                self.t_stats.helpers_deferred,
                self.t_stats.helpers_missing,
                self.t_stats.hold_patch_lcycles,
                self.t_stats.max_decision_delay,
                self.t_stats.inplace,
                self.t_stats.inplace_y
            ),
        }
        println!("Steiner tree computation called {} times", self.stree_computation.n_calls);
        println!("A* computation called {} times", self.astar.n_calls);

        if self.record_cultivation_dist {
            let dist_fname = format!("{}.cultivation_dist", self.input.circuit_stem());
            match self.write_cultivation_dist(&dist_fname, min, max) {
                Ok(()) => println!("Cultivation time distribution written to {}", dist_fname),
                Err(e) => eprintln!("Warning: could not write cultivation dist: {}", e),
            }
        }
    }

    /// Writes the normalized distribution of cultivation times to `fname`.
    /// Each line contains: `<cultivation_time> <normalized_count>` where
    /// `normalized_count = count(t) / total_events` for t in [min, max].
    fn write_cultivation_dist(&self, fname: &str, min: i32, max: i32) -> io::Result<()> {
        let times = &self.cultivation.cultivation_times_log;
        let total = times.len();
        if total == 0 || min > max {
            return Ok(());
        }
        let f = File::create(fname)?;
        let mut buf_f = BufWriter::new(f);
        for t in min..=max {
            let count = times.iter().filter(|&&v| v == t).count();
            let normalized = count as f64 / total as f64;
            writeln!(buf_f, "{} {:.6}", t, normalized)?;
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_lcycle(&self) -> io::Result<()> {
        let mut lcycle_used = vec![false; self.input.topo.n_nodes];
        for &(pp_id, ref opt_tree) in &self.pp_paths {
            let Some(tree) = opt_tree else { continue };
            let tree = tree.as_ref();
            let pp = self.input.circuit.product(pp_id);
            if self.scheduled_products.contains(&pp_id) && !pp.gate_type.is_clifford() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("product {} scheduled twice", pp_id),
                ));
            }
            for &parent_id in &pp.parents {
                if !self.scheduled_products.contains(&parent_id) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("product {} scheduled before parent {}", pp_id, parent_id),
                    ));
                }
            }
            for &nid in &self.precomputed_terminals[pp_id as usize] {
                if !tree.contains_node(nid) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "product {} terminal node {} missing from tree",
                            pp_id,
                            self.input.topo.label(nid)
                        ),
                    ));
                }
            }
            if pp.gate_type.is_t() {
                match tree.root_node_id {
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("product {}: T gate has no magic root node", pp_id),
                        ));
                    }
                    Some(magic_id) => {
                        if self.input.topo.node(magic_id).node_type != NodeType::Magic {
                            return Err(io::Error::new(
                                io::ErrorKind::Other,
                                format!(
                                    "product {}: root node {} is not a Magic node",
                                    pp_id, magic_id
                                ),
                            ));
                        }
                    }
                }
            }
            for node_id in tree.iter_nodes() {
                if lcycle_used[node_id as usize] {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "product {} shares node '{}' with another \
                                                       product in the same lcycle",
                            pp_id,
                            self.input.topo.label(node_id)
                        ),
                    ));
                }
                lcycle_used[node_id as usize] = true;
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_clifford_repetitions(&self) -> io::Result<()> {
        let mut cx_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        let mut s_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        for (lcycle_i, lcycle_ids) in &self.lcycle_scheduled {
            for &pp_id in lcycle_ids {
                let pp = self.input.circuit.product(pp_id);
                if pp.gate_type.is_cx() {
                    let lcycles = cx_counts.entry(pp_id).or_insert(Vec::new());
                    lcycles.push(*lcycle_i);
                } else if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                    let lcycles = s_counts.entry(pp_id).or_insert(Vec::new());
                    lcycles.push(*lcycle_i);
                }
            }
        }
        let mut errors = Vec::new();
        for (pp_id, lcycles) in &cx_counts {
            let pp = self.input.circuit.product(*pp_id);
            if pp.gate_type.is_cx() {
                if lcycles.len() != 2 || lcycles[0] != lcycles[1] - 1 {
                    errors.push(format!("  product {} not scheduled 2x {:?}", pp, lcycles));
                }
            }
        }
        for (pp_id, lcycles) in &s_counts {
            let pp = self.input.circuit.product(*pp_id);
            if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                if lcycles.len() != 3
                    || lcycles[0] != lcycles[1] - 1
                    || lcycles[1] != lcycles[2] - 1
                {
                    errors.push(format!("  product {} not scheduled 3x {:?}", pp, lcycles));
                }
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Clifford repetition errors:\n{}", errors.join("\n")),
            ));
        }
        println!(
            "Clifford repetition check passed ({} CX, {} S/SX products)",
            cx_counts.len(),
            s_counts.len()
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_schedule(&self) -> io::Result<()> {
        let n_products = self.input.circuit.n_products();
        let mut errors: Vec<String> = Vec::new();
        for pp_id in 0..n_products as i32 {
            if !self.scheduled_products.contains(&pp_id) {
                errors.push(format!("  product {} was never scheduled", pp_id));
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Completeness errors:\n{}", errors.join("\n")),
            ));
        }
        println!("Schedule check passed: all {} products scheduled", n_products);
        Ok(())
    }

    pub(crate) fn print_schedule(&self, hdr: &str) -> io::Result<()> {
        let _timer = fn_timer!();
        debug_sched!("Printing schedule");
        let output_fname = format!("{}.schedule", self.input.circuit_stem());

        let f = File::create(&output_fname)?;
        let mut buf_f = BufWriter::new(f);

        let max_lcycle: usize =
            self.lcycle_scheduled.last().map(|(lcycle_i, _)| *lcycle_i).unwrap_or(0);
        let max_width = max_lcycle.to_string().len();
        let tot_products = self.lcycle_scheduled.iter().map(|(_, v)| v.len()).sum::<usize>();
        writeln!(buf_f, "{}", hdr)?;
        writeln!(buf_f, "# Total active logical cycles: {}", self.lcycle_scheduled.len())?;
        writeln!(buf_f, "# Total logical cycles: {}", max_lcycle)?;
        writeln!(buf_f, "# Total products: {}", tot_products)?;
        writeln!(buf_f, "# Parallelism: {:.2}", tot_products as f64 / max_lcycle as f64)?;

        let colors = [
            Color::Green,
            Color::Red,
            Color::Yellow,
            Color::Blue,
            Color::Magenta,
            Color::Cyan,
            Color::White,
            Color::BrightGreen,
            Color::BrightRed,
            Color::BrightYellow,
            Color::BrightBlue,
            Color::BrightMagenta,
            Color::BrightCyan,
            Color::BrightWhite,
        ];

        // prev_cx tracks CX products seen in the previous lcycle so we can
        // suppress the second operator on the correct lcycle in the output.
        let mut prev_cx: IndexSet<i32> = IndexSet::new();
        for (lcycle_i, lcycle_ids) in &self.lcycle_scheduled {
            let mut sorted_ids = lcycle_ids.clone();
            sorted_ids.sort_by_key(|&id| {
                self.input
                    .circuit
                    .product(id)
                    .operators
                    .iter()
                    .map(|op| op.qubit)
                    .min()
                    .unwrap_or(u16::MAX)
            });
            let mut combined_chars = vec!['_'; self.input.circuit.n_qubits];
            let mut combined_colors: Vec<Option<Color>> = vec![None; self.input.circuit.n_qubits];
            for (idx, &pp_id) in sorted_ids.iter().enumerate() {
                let pp = self.input.circuit.product(pp_id);
                let color = colors[idx % colors.len()];
                for op in &pp.operators {
                    if op.qubit < self.input.circuit.n_qubits as u16 {
                        combined_chars[op.qubit as usize] = op.basis;
                        combined_colors[op.qubit as usize] = Some(color);
                    }
                }
                if pp.gate_type.is_cx() {
                    // CX spans 2 lcycles. On the first lcycle show only operator[0]
                    // (the control); on the second show only operator[1] (the target).
                    if !prev_cx.swap_remove(&pp_id) {
                        debug_sched!("  first lcycle of CX {} {}", pp_id, pp);
                        prev_cx.insert(pp_id);
                        let qubit = pp.operators[1].qubit;
                        combined_colors[qubit as usize] = None;
                        combined_chars[qubit as usize] = '_';
                    } else {
                        debug_sched!("  second lcycle of CX {} {}", pp_id, pp);
                        let qubit = pp.operators[0].qubit;
                        combined_colors[qubit as usize] = None;
                        combined_chars[qubit as usize] = '_';
                    }
                }
            }
            write!(buf_f, "{:width$}: ", lcycle_i, width = max_width)?;
            for i in 0..self.input.circuit.n_qubits {
                let ch = combined_chars[i].to_string();
                let colored_ch = match combined_colors[i] {
                    Some(c) => ch.color(c).to_string(),
                    None => ch,
                };
                write!(buf_f, "{}", colored_ch)?;
            }
            let mut id_string = String::new();
            for (idx, &pp_id) in sorted_ids.iter().enumerate() {
                let pp = self.input.circuit.product(pp_id);
                let color = colors[idx % colors.len()];
                id_string.push_str(&format!(
                    " {}",
                    format!("{}<{:?}>", pp_id, pp.gate_type).color(color)
                ));
            }
            writeln!(buf_f, "{}", id_string)?;
        }
        println!("Scheduled products written to {}", output_fname);
        Ok(())
    }
}

fn circuit_stem(fname: &str) -> &str {
    Path::new(fname).file_stem().and_then(|s| s.to_str()).unwrap_or("circuit")
}

/// Expands operators into data node IDs; Y-basis operators expand to two nodes
/// (X patch + Z patch) since a Y measurement needs both stabiliser data qubits.
fn operators_to_node_ids(topo: &TopoGraph, operators: &[Operator]) -> Vec<u16> {
    let mut node_ids = Vec::with_capacity(operators.len());
    for op in operators {
        if op.basis == 'Y' {
            node_ids.push(topo.get_data_node_id(op.qubit, 'X'));
            node_ids.push(topo.get_data_node_id(op.qubit, 'Z'));
        } else {
            node_ids.push(topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase()));
        }
    }
    node_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::node::Node;
    use crate::topograph::TopoGraph;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn run_scheduler(lines: &[&str], rseed: u32) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let mut sched =
            Scheduler::new(circuit, topo, 0.0387396, "none", String::new(), rseed, false, false);
        sched.sched_circuit().expect("sched_circuit failed");
        sched
    }

    #[test]
    fn t_gate_failures_bounded_by_total_t_gates() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 0);
        let tot_t = 4usize;
        assert!(
            sched.t_gate_failures <= tot_t,
            "t_gate_failures {} exceeds total T gates {}",
            sched.t_gate_failures,
            tot_t
        );
    }

    #[test]
    fn t_gate_failures_deterministic_with_fixed_seed() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched1 = run_scheduler(lines, 42);
        let sched2 = run_scheduler(lines, 42);
        assert_eq!(
            sched1.t_gate_failures, sched2.t_gate_failures,
            "t_gate_failures differs between runs with the same seed"
        );
    }

    #[test]
    fn t_gate_failures_varies_with_seed() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let counts: Vec<usize> =
            (0u32..20).map(|s| run_scheduler(lines, s).t_gate_failures).collect();
        let distinct = counts.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 1, "t_gate_failures never varied across 20 seeds: {:?}", counts);
    }

    #[test]
    fn all_products_appear_exactly_once_in_lcycle_scheduled() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let mut id_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for (_, ids) in &sched.lcycle_scheduled {
            for &id in ids {
                *id_counts.entry(id).or_insert(0) += 1;
            }
        }
        let n_products = 4;
        for pp_id in 0..n_products as i32 {
            let count = id_counts.get(&pp_id).copied().unwrap_or(0);
            assert_eq!(
                count, 1,
                "product {} appears {} times in lcycle_scheduled (expected 1)",
                pp_id, count
            );
        }
    }

    #[test]
    fn lcycle_scheduled_tot_entries_equals_n_products() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        let n_products = 4usize;
        assert_eq!(
            tot_entries, n_products,
            "total lcycle_scheduled entries {} != n_products {}",
            tot_entries, n_products
        );
    }

    #[test]
    fn lcycle_count_not_inflated_by_t_gate_failures() {
        // T gate failures are auto-corrected on helper patches (no product is added), so
        // they do not add extra scheduling lcycles.
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let n_t = 4usize;
        let active_lcycles = sched.lcycle_scheduled.len();
        assert!(
            active_lcycles <= n_t,
            "active lcycles {} > n_t {} (failures should not add extra lcycles)",
            active_lcycles,
            n_t,
        );
    }

    fn build_scheduler(lines: &[&str], rseed: u32) -> Scheduler {
        build_scheduler_f(lines, rseed, false)
    }

    fn build_scheduler_f(lines: &[&str], rseed: u32, no_t_failures: bool) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        Scheduler::new(circuit, topo, 0.0387396, "none", String::new(), rseed, no_t_failures, false)
    }

    // ── T-gate corrections ──────────────────────────────────────────────────────────────

    const T_HEAVY_LINES: &[&str] = &[
        "+ZZ__<T>", "+X___<T>", "+_XX_<T>", "-Z__Z<T>", "+_Y__<T>", "+XXXX<T>", "+Z___<T>",
        "+__ZZ<T>", "-_X__<T>", "+X__X<T>", "+ZZZZ<T>", "+_Z_X<T>", "+X___<M>", "+_Z__<M>",
        "+__X_<M>", "+___Z<M>",
    ];

    /// T gates, then physical Cliffords and measurements that the frame must carry.
    const CLIFFORD_LINES: &[&str] = &[
        "+Z___<T>",
        "+_X__<T>",
        "+Z___<S>",
        "+_X__<SX>",
        "+ZX__<CX>",
        "+X___<T>",
        "+_Z__<T>",
        "+X___<M>",
        "+_Z__<M>",
        "+ZZ__<M>",
    ];

    #[test]
    fn litinski_leaves_every_later_product_unchanged() {
        for rseed in 0u32..8 {
            let mut sched = build_scheduler(T_HEAVY_LINES, rseed);
            sched.set_reaction_latency(2);
            let n_products = sched.input.circuit.n_products();
            let (_, n) = sched.sched_circuit().expect("sched_circuit failed");
            assert!(sched.t_gate_failures > 0, "seed {}: want some failures", rseed);
            assert_eq!(n, n_products, "seed {}: no correction products may be added", rseed);
            assert!(sched.frame_exec.is_empty(), "seed {}: a product was relabelled", rseed);
            assert_eq!(sched.t_stats.injections, 12);
        }
    }

    #[test]
    fn litinski_helpers_are_free_neighbours_and_holds_are_respected() {
        for latency in [0usize, 1, 3] {
            for rseed in 0u32..8 {
                let mut sched = build_scheduler(T_HEAVY_LINES, rseed);
                sched.set_reaction_latency(latency);
                sched.sched_circuit().expect("sched_circuit failed");
                let st = &sched.t_stats;
                assert_eq!(st.helpers + st.helpers_missing, st.injections);
                let topo = &sched.input.topo;
                for &(t, magic, h) in &st.helper_log {
                    assert!(topo.node(magic).nbs_slice().contains(&h), "helper not adjacent");
                    assert_ne!(topo.node(h).node_type, NodeType::Data, "helper on a data patch");
                    // The magic patch is held until its helper arrives (lcycle t); the helper
                    // until the decision d. Neither may serve another T as its magic state then.
                    let (t0, _, _, d) =
                        *st.injection_log.iter().rev().find(|e| e.2 == magic && e.0 <= t).unwrap();
                    for &(t1, pp, m1, _) in &st.injection_log {
                        if t1 > t0 && t1 <= t {
                            assert_ne!(m1, magic, "magic patch reused while held (T {})", pp);
                        }
                        if t1 >= t && t1 <= d {
                            assert_ne!(m1, h, "helper used as a magic state while held");
                        }
                    }
                }
                // Holds: one helper per injection for its decision delay, plus the lcycles a
                // magic patch waited for a helper (or held in place when none was found).
                let min_hold: usize = st.injection_log.iter().map(|&(t, _, _, d)| d - t).sum();
                assert!(
                    st.hold_patch_lcycles >= min_hold,
                    "L={} seed={} held {} < {} (helpers {} waited {} missing {})",
                    latency,
                    rseed,
                    st.hold_patch_lcycles,
                    min_hold,
                    st.helpers,
                    st.helpers_deferred,
                    st.helpers_missing
                );
                for &(t, pp, magic, d) in &st.injection_log {
                    assert!(d >= t + latency, "T {} decided before its outcome is decoded", pp);
                    assert_eq!(topo.node(magic).node_type, NodeType::Magic);
                }
            }
        }
    }

    #[test]
    fn litinski_helper_rank_prefers_bus_then_cultivating_then_ready() {
        let mut sched = build_scheduler(T_HEAVY_LINES, 0);
        sched.set_t_correction(TCorrection::LitinskiReady);
        sched.sched_circuit().expect("sched_circuit failed");
        let topo = &mut sched.input.topo;
        let magic: Vec<u16> = (0..topo.n_nodes as u16)
            .filter(|&n| topo.node(n).node_type == NodeType::Magic)
            .collect();
        let data =
            (0..topo.n_nodes as u16).find(|&n| topo.node(n).node_type == NodeType::Data).unwrap();
        let (a, b) = (magic[0], magic[1]);
        topo.cultivation_times[a as usize] = 0; // ready
        topo.cultivation_times[b as usize] = 7; // cultivating
        topo.busy_counts[b as usize] = 3;
        assert_eq!(sched.helper_rank(data), None, "a data patch is never a helper");
        let (ra, rb) = (sched.helper_rank(a).unwrap(), sched.helper_rank(b).unwrap());
        assert!(rb < ra, "a cultivating patch is preferred to wasting a ready state");
        sched.set_t_correction(TCorrection::LitinskiY);
        assert_eq!(sched.helper_rank(a), None, "litinski-y never borrows a ready state");
        assert!(sched.helper_rank(b).is_some());
    }

    #[test]
    fn litinski_y_never_wastes_a_ready_state_and_falls_back_in_place() {
        // Over a few seeds and latencies, litinski-y borrows no ready state, and every injection
        // gets a helper or an in-place measurement; litinski-ready wastes some ready states.
        let (mut wasted_ready, mut any_inplace_or_wait) = (0, false);
        for latency in [0usize, 2] {
            for rseed in 0u32..8 {
                let mut y = build_scheduler(T_HEAVY_LINES, rseed);
                y.set_t_correction(TCorrection::LitinskiY);
                y.set_reaction_latency(latency);
                y.sched_circuit().expect("sched_circuit failed");
                let st = &y.t_stats;
                assert_eq!(st.helpers_ready, 0);
                assert_eq!(st.helpers + st.helpers_missing, st.injections);
                assert_eq!(st.inplace, st.helpers_missing);
                any_inplace_or_wait |= st.helpers_missing + st.helpers_deferred > 0;
                let mut r = build_scheduler(T_HEAVY_LINES, rseed);
                r.set_t_correction(TCorrection::LitinskiReady);
                r.set_reaction_latency(latency);
                r.sched_circuit().expect("sched_circuit failed");
                wasted_ready += r.t_stats.helpers_ready;
            }
        }
        assert!(any_inplace_or_wait, "litinski-y should have to wait or fall back somewhere");
        assert!(wasted_ready > 0, "litinski-ready should borrow a ready state somewhere");
    }

    #[test]
    fn litinski_decisions_chain_only_through_anticommuting_t_gates() {
        // Z then X on qubit 0 anticommute: X's correction depends on Z's frame bit.
        let mut sched = build_scheduler(&["+Z___<T>", "+X___<T>"], 3);
        sched.set_reaction_latency(2);
        sched.sched_circuit().expect("sched_circuit failed");
        let log = &sched.t_stats.injection_log;
        assert_eq!(log.len(), 2);
        let (t0, _, _, d0) = log[0];
        let (t1, _, _, d1) = log[1];
        assert_eq!(d0, t0 + 2);
        assert_eq!(d1, (t1 + 2).max(d0 + 2));
        // Z then Z commute: each decision is just its own decoding delay.
        let mut sched = build_scheduler(&["+Z___<T>", "+Z___<T>"], 3);
        sched.set_reaction_latency(2);
        sched.sched_circuit().expect("sched_circuit failed");
        for &(t, _, _, d) in &sched.t_stats.injection_log {
            assert_eq!(d, t + 2);
        }
    }

    #[test]
    fn litinski_without_failures_borrows_no_helpers() {
        let mut sched = build_scheduler_f(T_HEAVY_LINES, 0, true);
        sched.set_reaction_latency(3);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.t_stats.injections, 0);
        assert_eq!(sched.t_stats.hold_patch_lcycles, 0);
    }

    #[test]
    fn frame_mode_runs_every_product_once_and_rewrites_after_failures() {
        let mut rewrote = false;
        for lines in [T_HEAVY_LINES, CLIFFORD_LINES] {
            let n_tm = lines.iter().filter(|l| l.ends_with("<T>") || l.ends_with("<M>")).count();
            for latency in [0usize, 2] {
                for rseed in 0u32..8 {
                    let mut sched = build_scheduler(lines, rseed);
                    sched.set_t_correction(TCorrection::Frame);
                    sched.set_reaction_latency(latency);
                    let n_products = sched.input.circuit.n_products();
                    let (_, n) = sched.sched_circuit().expect("sched_circuit failed");
                    let f = &sched.frame_stats;
                    assert_eq!(n, n_products, "no correction products in frame mode");
                    assert_eq!(f.products, n_tm, "every T and measurement logged once");
                    assert_eq!(f.failures, sched.t_gate_failures);
                    assert_eq!(sched.t_stats.injections, 0, "no helper machinery");
                    if latency == 0 {
                        assert_eq!(f.waits, 0, "nothing waits when outcomes are known at once");
                    }
                    // Weights are counted from the log.
                    let w: usize = f.log.iter().map(|(_, _, ops)| ops.len()).sum();
                    assert_eq!(w, f.exec_weight);
                    rewrote |= f.rewritten > 0;
                }
            }
        }
        assert!(rewrote, "some seed must rewrite a product after a failure");
    }

    #[test]
    fn frame_mode_executes_the_conjugated_pauli() {
        // Replays each run with a fresh frame: every T and measurement must have run about
        // W.Q.W^dag, for W built from the failures and physical Cliffords of earlier lcycles.
        let mut rewrote = false;
        for lines in [T_HEAVY_LINES, CLIFFORD_LINES] {
            for latency in [0usize, 1, 3] {
                for rseed in 0u32..16 {
                    let mut sched = build_scheduler(lines, rseed);
                    sched.set_t_correction(TCorrection::Frame);
                    sched.set_reaction_latency(latency);
                    sched.sched_circuit().expect("sched_circuit failed");
                    let circuit = &sched.input.circuit;
                    let mut w = CliffordFrame::new(circuit.n_qubits);
                    let mut started = HashSet::new();
                    for (t, ids) in &sched.lcycle_scheduled {
                        for &(_, pp_id, ref ops) in
                            sched.frame_stats.log.iter().filter(|e| e.0 == *t)
                        {
                            let q = SPauli::from_ops(w.words, &circuit.product(pp_id).operators);
                            let got = SPauli::from_ops(w.words, ops);
                            assert_eq!(
                                got,
                                w.image(&q),
                                "L={} seed={} product {}",
                                latency,
                                rseed,
                                pp_id
                            );
                            rewrote |= got != q;
                        }
                        for &(_, pp_id) in sched.frame_stats.failed_log.iter().filter(|e| e.0 == *t)
                        {
                            let p = SPauli::from_ops(w.words, &circuit.product(pp_id).operators);
                            w.absorb_failed_t(&p);
                        }
                        for &pp_id in ids {
                            let pp = circuit.product(pp_id);
                            if !pp.gate_type.is_clifford() || !started.insert(pp_id) {
                                continue;
                            }
                            let q0 = pp.operators[0].qubit as usize;
                            w.absorb_physical_clifford(match pp.gate_type {
                                GateType::S => FrameGate::S(q0),
                                GateType::SX => FrameGate::SX(q0),
                                _ => {
                                    let c = pp.operators.iter().find(|o| o.basis == 'Z').unwrap();
                                    let x = pp.operators.iter().find(|o| o.basis == 'X').unwrap();
                                    FrameGate::CX(c.qubit as usize, x.qubit as usize)
                                }
                            });
                        }
                    }
                    assert_eq!(sched.frame_stats.failed_log.len(), sched.t_gate_failures);
                }
            }
        }
        assert!(rewrote);
    }

    #[test]
    fn frame_mode_waits_for_an_anticommuting_outcome() {
        // T about Z then a measurement of X on the same qubit: with latency 3 the measurement
        // starts at least 4 lcycles after the T, whatever the T's outcome.
        for rseed in 0u32..8 {
            let mut sched = build_scheduler(&["+Z___<T>", "+X___<M>"], rseed);
            sched.set_t_correction(TCorrection::Frame);
            sched.set_reaction_latency(3);
            sched.sched_circuit().expect("sched_circuit failed");
            let log = &sched.frame_stats.log;
            let t_at = log.iter().find(|e| e.1 == 0).unwrap().0;
            let m_at = log.iter().find(|e| e.1 == 1).unwrap().0;
            assert!(m_at >= t_at + 4, "seed {}: T at {}, M at {}", rseed, t_at, m_at);
            // A commuting measurement need not wait.
            let mut sched = build_scheduler(&["+Z___<T>", "+Z___<M>"], rseed);
            sched.set_t_correction(TCorrection::Frame);
            sched.set_reaction_latency(3);
            sched.sched_circuit().expect("sched_circuit failed");
            assert_eq!(sched.frame_stats.waits, 0);
        }
    }

    #[test]
    fn frame_mode_rewrites_a_measurement_after_a_failed_t() {
        // A failed T about Z_0 leaves S_Z^dag in the frame, so a later X_0 measurement runs
        // as Y_0 (both of qubit 0's data patches); a successful one leaves it as X_0.
        let mut seen = [false; 2];
        for rseed in 0u32..32 {
            let mut sched = build_scheduler(&["+Z___<T>", "+X___<M>"], rseed);
            sched.set_t_correction(TCorrection::Frame);
            sched.sched_circuit().expect("sched_circuit failed");
            let m = sched.frame_stats.log.iter().find(|e| e.1 == 1).unwrap();
            let letter = m.2[0].basis;
            if sched.t_gate_failures == 1 {
                assert_eq!(letter, 'Y', "seed {}", rseed);
                seen[1] = true;
            } else {
                assert_eq!(letter, 'X', "seed {}", rseed);
                seen[0] = true;
            }
        }
        assert!(seen[0] && seen[1], "want both outcomes over 32 seeds");
    }

    #[test]
    fn frame_mode_without_failures_matches_litinski() {
        // With -F nothing is owed, so both models give the same schedule.
        for lines in [T_HEAVY_LINES, CLIFFORD_LINES] {
            for rseed in 0u32..4 {
                let lc = |mode: TCorrection| {
                    let mut s = build_scheduler_f(lines, rseed, true);
                    s.set_t_correction(mode);
                    s.set_reaction_latency(2);
                    s.sched_circuit().unwrap().0
                };
                assert_eq!(lc(TCorrection::Frame), lc(TCorrection::LitinskiReady));
                assert_eq!(lc(TCorrection::LitinskiY), lc(TCorrection::LitinskiReady));
            }
        }
    }
}
