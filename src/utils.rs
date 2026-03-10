use std::time::{Duration, Instant};

use indexmap::IndexMap;

/// ANSI escape codes for terminal colour output.
pub const _RED: &str = "\x1b[31m";
pub const _GREEN: &str = "\x1b[32m";
pub const _YELLOW: &str = "\x1b[33m";
pub const _BLUE: &str = "\x1b[34m";
pub const _MAGENTA: &str = "\x1b[35m";
pub const _CYAN: &str = "\x1b[36m";
pub const _WHITE: &str = "\x1b[37m";
pub const _LRED: &str = "\x1b[91m";
pub const _LGREEN: &str = "\x1b[92m";
pub const _LYELLOW: &str = "\x1b[93m";
pub const _LBLUE: &str = "\x1b[94m";
pub const _LMAGENTA: &str = "\x1b[95m";
pub const _LCYAN: &str = "\x1b[96m";
pub const _LWHITE: &str = "\x1b[97m";
/// Resets all ANSI terminal formatting.
pub const _RESET: &str = "\x1b[0m";

/// RAII timer that prints the elapsed wall-clock time when it goes out of scope.
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    pub fn new(name: &str) -> Self {
        Timer { name: name.to_string(), start: Instant::now() }
    }
}

impl Drop for Timer {
    /// Prints elapsed time in seconds on drop.
    fn drop(&mut self) {
        println!("{}Timing: {} took {:.2} s{}",
                 _CYAN,
                 self.name,
                 self.start.elapsed().as_secs_f64(),
                 _RESET);
    }
}

/// Creates a [`Timer`] scoped to the enclosing function.
/// With no arguments, the function name is inferred automatically (crate prefix stripped).
/// With a string argument, that string is used as the timer label instead.
#[macro_export]
macro_rules! fn_timer {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let full_name = type_name_of(f);
        // Remove the trailing "::f"
        let name = &full_name[..full_name.len() - 3];
        // Remove the crate prefix to get "Circuit::load_circuit" instead of "puremagic::circuit::Circuit::load_circuit"
        let short_name = name.split("::").skip(2).collect::<Vec<_>>().join("::");
        $crate::utils::Timer::new(&short_name)
    }};
    ($custom_name:expr) => {{
        $crate::utils::Timer::new($custom_name)
    }};
}

/// Emits a `log::debug!` message only in debug builds (no-op in release).
#[macro_export]
macro_rules! debug_sched {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        log::debug!($($arg)*);
    };
}

/// Emits a `log::info!` message only in debug builds (no-op in release).
#[macro_export]
macro_rules! info_sched {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        log::info!($($arg)*);
    };
}

/// A single accumulated timer entry.
struct AccumTimer {
    start_time: Option<Instant>,
    total_elapsed: Duration,
    max_interval: Duration,
    num_intervals: usize,
}

impl AccumTimer {
    fn new() -> Self {
        AccumTimer { start_time: None,
                     total_elapsed: Duration::ZERO,
                     max_interval: Duration::ZERO,
                     num_intervals: 0 }
    }

    fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            let elapsed = start.elapsed();
            self.total_elapsed += elapsed;
            if elapsed > self.max_interval {
                self.max_interval = elapsed;
            }
            self.num_intervals += 1;
        }
    }
}

/// A collection of named accumulated timers. Timers are created automatically
/// on first use via [`accum_start!`] and the summary is printed when this
/// collection drops.
pub struct AccumTimers {
    timers: IndexMap<String, AccumTimer>,
}

impl AccumTimers {
    pub fn new() -> Self {
        AccumTimers { timers: IndexMap::new() }
    }

    /// Register a timer by name and return its index for fast subsequent access.
    /// If already registered, just returns the existing index.
    pub fn add_or_get(&mut self, name: &'static str) -> usize {
        if let Some(idx) = self.timers.get_index_of(name) {
            idx
        } else {
            self.timers.insert(name.to_string(), AccumTimer::new());
            self.timers.len() - 1
        }
    }

    /// Start a timer by pre-looked-up index. O(1), no string lookup.
    pub fn start(&mut self, idx: usize) {
        if let Some((_, t)) = self.timers.get_index_mut(idx) {
            t.start();
        }
    }

    /// Stop a timer by pre-looked-up index. O(1), no string lookup.
    pub fn stop(&mut self, idx: usize) {
        if let Some((_, t)) = self.timers.get_index_mut(idx) {
            t.stop();
        }
    }
}

impl Default for AccumTimers {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AccumTimers {
    fn drop(&mut self) {
        if self.timers.is_empty() {
            return;
        }
        let format_dur = |d: Duration| -> String {
            let secs = d.as_secs_f64();
            if secs >= 1.0 {
                format!("{:.2} s", secs)
                //format!("{:.1}", secs * 1_000.0)
            } else if secs >= 0.001 {
                format!("{:.1} ms", secs * 1_000.0)
            } else {
                format!("{:.2} μs", secs * 1_000_000.0)
                //format!("{:.6}", secs * 1_000.0)
            }
        };
        println!("{}Accumulated timings (ms):{}", _CYAN, _RESET);
        for (name, t) in &self.timers {
            if t.num_intervals == 0 {
                continue;
            }
            let avg = t.total_elapsed / t.num_intervals as u32;
            println!("{}  {:<25} total: {:>10}  avg: {:>10}  max: {:>10}  calls: {}{}",
                     _CYAN,
                     name,
                     format_dur(t.total_elapsed),
                     format_dur(avg),
                     format_dur(t.max_interval),
                     t.num_intervals,
                     _RESET);
        }
    }
}

/// RAII guard that stops a timer by index when dropped.
pub struct AccumTimerGuard {
    pub timers: *mut AccumTimers,
    pub idx: usize,
}

impl Drop for AccumTimerGuard {
    fn drop(&mut self) {
        // SAFETY: AccumTimers always outlives this guard.
        unsafe {
            (*self.timers).stop(self.idx);
        }
    }
}

/// Start an accumulated timer using the enclosing function name as the key.
/// Returns an [`AccumTimerGuard`] that stops the timer automatically when it
/// drops at the end of the enclosing scope.
///
/// The returned guard **must be bound to a named variable** (e.g. `_timer`);
/// binding to `_` alone will drop it immediately.
///
/// # Example
/// ```rust
/// fn my_function(&mut self) {
///     let _timer = accum_start!(self.timers);
///     // ... do work ...
/// }  // timer stops here automatically, even on early return or panic
/// ```
#[macro_export]
macro_rules! accum_start {
    ($timers:expr) => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        // Compute fn_name at compile-ish time (zero cost after first call).
        let full_name = type_name_of(f);
        let fn_name: &'static str = {
            let trimmed = match full_name.strip_suffix("::f") {
                Some(s) => s,
                None => full_name,
            };
            match trimmed.rfind("::") {
                Some(pos) => &full_name[pos + 2..full_name.len() - 3],
                None => trimmed,
            }
        };
        // Cache the index in a per-call-site static so the IndexMap lookup
        // only happens once, no matter how many times this line is executed.
        static IDX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let idx = *IDX.get_or_init(|| $timers.add_or_get(fn_name));
        $timers.start(idx);
        $crate::utils::AccumTimerGuard { timers: &mut $timers as *mut _, idx }
    }};
}
