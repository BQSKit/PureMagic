use std::time::{Duration, Instant};

// Add these near the top of scheduler.rs, before the struct definitions
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const BLUE: &str = "\x1b[34m";
pub const MAGENTA: &str = "\x1b[35m";
pub const CYAN: &str = "\x1b[36m";
pub const WHITE: &str = "\x1b[37m";
pub const LRED: &str = "\x1b[91m";
pub const LGREEN: &str = "\x1b[92m";
pub const LYELLOW: &str = "\x1b[93m";
pub const LBLUE: &str = "\x1b[94m";
pub const LMAGENTA: &str = "\x1b[95m";
pub const LCYAN: &str = "\x1b[96m";
pub const LWHITE: &str = "\x1b[97m";
pub const RESET: &str = "\x1b[0m";

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
    fn drop(&mut self) {
        println!("{}Timing: {} took {:.2} s{}",
                 CYAN,
                 self.name,
                 self.start.elapsed().as_secs_f64(),
                 RESET);
    }
}

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
    pub fn new(name: &str, interval_label: &str) -> Self {
        IntermittentTimer { start_time: None,
                            total_elapsed: Duration::new(0, 0),
                            last_interval: Duration::new(0, 0),
                            max_interval: Duration::new(0, 0),
                            num_intervals: 0,
                            name: name.to_string(),
                            interval_label: interval_label.to_string() }
    }

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
                 CYAN,
                 self.name,
                 format_time(total_secs),
                 format_time(avg_secs),
                 format_time(max_secs),
                 self.num_intervals,
                 RESET);
    }

    #[allow(dead_code)]
    pub fn get_final(&self) -> String {
        format!("{}: {:.2}", self.name, self.total_elapsed.as_secs_f64())
    }

    pub fn start(&mut self) {
        if !self.interval_label.is_empty() {
            println!("{:<40}:", self.interval_label);
        }
        self.start_time = Some(Instant::now());
    }

    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.last_interval = start.elapsed();
            self.total_elapsed += self.last_interval;
            if self.last_interval > self.max_interval {
                self.max_interval = self.last_interval;
            }
            self.num_intervals += 1;

            if !self.interval_label.is_empty() {
                println!("{}{:.2} s{}", CYAN, self.last_interval.as_secs_f64(), RESET);
            }
        }
    }

    #[allow(dead_code)]
    pub fn get_interval(&self) -> f64 {
        self.last_interval.as_secs_f64()
    }
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
