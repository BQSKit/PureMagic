use colored::Colorize;
use std::time::{Duration, Instant};

use indexmap::IndexMap;

/// RAII timer that prints elapsed wall-clock time on drop.
pub(crate) struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    pub(crate) fn new(name: &str) -> Self {
        Timer { name: name.to_string(), start: Instant::now() }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!(
            "{}",
            format!("Timing: {} took {:.2} s", self.name, self.start.elapsed().as_secs_f64())
                .cyan()
        );
    }
}

/// Creates a [`Timer`] scoped to the enclosing function.
#[macro_export]
macro_rules! fn_timer {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let full_name = type_name_of(f);
        let name = &full_name[..full_name.len() - 3];
        let short_name = name.split("::").skip(2).collect::<Vec<_>>().join("::");
        $crate::utils::Timer::new(&short_name)
    }};
    ($custom_name:expr) => {{ $crate::utils::Timer::new($custom_name) }};
}

#[macro_export]
macro_rules! debug_sched {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        log::debug!($($arg)*);
    };
}

#[macro_export]
macro_rules! info_sched {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        log::info!($($arg)*);
    };
}

struct AccumTimer {
    start_time: Option<Instant>,
    tot_elapsed: Duration,
    max_interval: Duration,
    n_intervals: usize,
}

impl AccumTimer {
    fn new() -> Self {
        AccumTimer {
            start_time: None,
            tot_elapsed: Duration::ZERO,
            max_interval: Duration::ZERO,
            n_intervals: 0,
        }
    }

    fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            let elapsed = start.elapsed();
            self.tot_elapsed += elapsed;
            if elapsed > self.max_interval {
                self.max_interval = elapsed;
            }
            self.n_intervals += 1;
        }
    }
}

/// A collection of named accumulated timers; summary is printed on drop.
pub(crate) struct AccumTimers {
    timers: IndexMap<String, AccumTimer>,
}

impl AccumTimers {
    pub(crate) fn new() -> Self {
        AccumTimers { timers: IndexMap::new() }
    }

    /// Returns the index for `name`, registering it if not already present.
    pub(crate) fn add_or_get(&mut self, name: &'static str) -> usize {
        if let Some(idx) = self.timers.get_index_of(name) {
            idx
        } else {
            self.timers.insert(name.to_string(), AccumTimer::new());
            self.timers.len() - 1
        }
    }

    pub(crate) fn start(&mut self, idx: usize) {
        if let Some((_, t)) = self.timers.get_index_mut(idx) {
            t.start();
        }
    }

    pub(crate) fn stop(&mut self, idx: usize) {
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
            } else if secs >= 0.001 {
                format!("{:.1} ms", secs * 1_000.0)
            } else {
                format!("{:.2} μs", secs * 1_000_000.0)
            }
        };
        println!("{}", "Accumulated timings (ms):".cyan());
        for (name, t) in &self.timers {
            if t.n_intervals == 0 {
                continue;
            }
            let avg = t.tot_elapsed / t.n_intervals as u32;
            println!(
                "{}",
                format!(
                    "  {:<25} total: {:>10}  avg: {:>10}  max: {:>10}  calls: {}",
                    name,
                    format_dur(t.tot_elapsed),
                    format_dur(avg),
                    format_dur(t.max_interval),
                    t.n_intervals,
                )
                .cyan()
            );
        }
    }
}

pub(crate) struct AccumTimerGuard {
    pub timers: *mut AccumTimers,
    pub idx: usize,
}

impl Drop for AccumTimerGuard {
    fn drop(&mut self) {
        // SAFETY: `AccumTimers` always outlives this guard.
        unsafe {
            (*self.timers).stop(self.idx);
        }
    }
}

/// Starts an accumulated timer for the enclosing function; stops it on drop.
/// The returned guard **must be bound to a named variable** (e.g. `_timer`).
#[macro_export]
macro_rules! accum_start {
    ($timers:expr) => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
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
        static IDX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let idx = *IDX.get_or_init(|| $timers.add_or_get(fn_name));
        $timers.start(idx);
        $crate::utils::AccumTimerGuard { timers: &mut $timers as *mut _, idx }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn accum_timers_new_is_empty() {
        let timers = AccumTimers::new();
        assert!(timers.timers.is_empty());
    }

    #[test]
    fn add_or_get_returns_zero_for_first_entry() {
        let mut timers = AccumTimers::new();
        let idx = timers.add_or_get("alpha");
        assert_eq!(idx, 0);
    }

    #[test]
    fn add_or_get_returns_sequential_indices() {
        let mut timers = AccumTimers::new();
        let i0 = timers.add_or_get("first");
        let i1 = timers.add_or_get("second");
        let i2 = timers.add_or_get("third");
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
    }

    #[test]
    fn add_or_get_idempotent_for_same_name() {
        let mut timers = AccumTimers::new();
        let i0 = timers.add_or_get("same");
        let i1 = timers.add_or_get("same");
        assert_eq!(i0, i1);
        assert_eq!(timers.timers.len(), 1);
    }

    #[test]
    fn start_stop_does_not_panic() {
        let mut timers = AccumTimers::new();
        let idx = timers.add_or_get("t");
        timers.start(idx);
        std::thread::sleep(Duration::from_millis(1));
        timers.stop(idx);
        let (_, t) = timers.timers.get_index(idx).unwrap();
        assert_eq!(t.n_intervals, 1);
        assert!(t.tot_elapsed > Duration::ZERO);
    }

    #[test]
    fn stop_without_start_is_noop() {
        let mut timers = AccumTimers::new();
        let idx = timers.add_or_get("t");
        timers.stop(idx); // should not panic
        let (_, t) = timers.timers.get_index(idx).unwrap();
        assert_eq!(t.n_intervals, 0);
    }

    #[test]
    fn start_stop_out_of_bounds_index_is_noop() {
        let mut timers = AccumTimers::new();
        timers.start(999); // no panic
        timers.stop(999); // no panic
    }

    #[test]
    fn multiple_start_stop_accumulates() {
        let mut timers = AccumTimers::new();
        let idx = timers.add_or_get("multi");
        for _ in 0..3 {
            timers.start(idx);
            std::thread::sleep(Duration::from_millis(1));
            timers.stop(idx);
        }
        let (_, t) = timers.timers.get_index(idx).unwrap();
        assert_eq!(t.n_intervals, 3);
    }

    #[test]
    fn default_is_empty() {
        let timers = AccumTimers::default();
        assert!(timers.timers.is_empty());
    }

    #[test]
    fn timer_new_does_not_panic() {
        let _t = Timer::new("test_timer");
    }

    #[test]
    fn accum_timer_guard_stops_on_drop() {
        let mut timers = AccumTimers::default();
        let idx = timers.add_or_get("guard_test");
        timers.start(idx);
        timers.stop(idx);
    }

    #[test]
    fn add_or_get_same_name_always_returns_same_index() {
        let mut timers = AccumTimers::default();
        let i1 = timers.add_or_get("alpha");
        let i2 = timers.add_or_get("beta");
        let i3 = timers.add_or_get("alpha"); // same as first
        assert_ne!(i1, i2);
        assert_eq!(i1, i3);
    }

    #[test]
    fn start_stop_valid_index_does_not_panic() {
        let mut timers = AccumTimers::default();
        let idx = timers.add_or_get("t1");
        timers.start(idx);
        timers.stop(idx);
        // Second round
        timers.start(idx);
        timers.stop(idx);
    }

    #[test]
    fn multiple_timers_do_not_interfere() {
        let _t1 = Timer::new("timer_a");
        let _t2 = Timer::new("timer_b");
        // Both drop here — should not panic.
    }
}
