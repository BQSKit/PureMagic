use std::time::{Duration, Instant};

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

/// Timer for repeated, non-contiguous intervals (e.g. timing a function called in a loop).
/// Accumulates total, average, and maximum elapsed time across all start/stop pairs.
/// Call [`done`](Self::done) at the end to print a summary.
pub struct IntermittentTimer {
    start_time: Option<Instant>,
    total_elapsed: Duration,
    last_interval: Duration,
    max_interval: Duration,
    num_intervals: usize,
    name: String,
    interval_label: String,
}

impl IntermittentTimer {
    /// Creates a new timer. `interval_label`, if non-empty, is printed on each `start`/`stop` pair.
    pub fn new(name: &str, interval_label: &str) -> Self {
        IntermittentTimer { start_time: None,
                            total_elapsed: Duration::new(0, 0),
                            last_interval: Duration::new(0, 0),
                            max_interval: Duration::new(0, 0),
                            num_intervals: 0,
                            name: name.to_string(),
                            interval_label: interval_label.to_string() }
    }

    /// Prints a summary line with total, average, and maximum interval times and call count.
    pub fn done(&self) {
        let total_secs = self.total_elapsed.as_secs_f64();
        let avg_secs = total_secs / self.num_intervals as f64;
        let max_secs = self.max_interval.as_secs_f64();
        // Helper function to format time with appropriate unit
        let format_time = |secs: f64| -> String {
            if secs >= 1.0 {
                format!("{:.2} s", secs)
            } else if secs >= 0.001 {
                format!("{:.2} ms", secs * 1000.0)
            } else {
                format!("{:.2} μs", secs * 1_000_000.0)
            }
        };
        println!("{}Timing: {} took {} (avg {} max {} over {} calls){}",
                 _CYAN,
                 self.name,
                 format_time(total_secs),
                 format_time(avg_secs),
                 format_time(max_secs),
                 self.num_intervals,
                 _RESET);
    }

    /// Returns a compact summary string with the timer name and total elapsed seconds.
    #[allow(dead_code)]
    pub fn get_final(&self) -> String {
        format!("{}: {:.2}", self.name, self.total_elapsed.as_secs_f64())
    }

    /// Starts a new timing interval. Optionally prints the interval label if set.
    pub fn start(&mut self) {
        if !self.interval_label.is_empty() {
            println!("{:<40}:", self.interval_label);
        }
        self.start_time = Some(Instant::now());
    }

    /// Stops the current interval and accumulates it into the running totals.
    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.last_interval = start.elapsed();
            self.total_elapsed += self.last_interval;
            if self.last_interval > self.max_interval {
                self.max_interval = self.last_interval;
            }
            self.num_intervals += 1;

            if !self.interval_label.is_empty() {
                println!("{}{:.2} s{}", _CYAN, self.last_interval.as_secs_f64(), _RESET);
            }
        }
    }

    /// Returns the duration of the most recently completed interval in seconds.
    #[allow(dead_code)]
    pub fn get_interval(&self) -> f64 {
        self.last_interval.as_secs_f64()
    }
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
