use std::time::{Duration, Instant};

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
        println!("\x1b[36mTiming: {} took {:.2} s\x1b[0m",
                 self.name,
                 self.start.elapsed().as_secs_f64());
    }
}

pub struct IntermittentTimer {
    start_time: Option<Instant>,
    total_elapsed: Duration,
    last_interval: Duration,
    name: String,
    interval_label: String,
}

impl IntermittentTimer {
    pub fn new(name: &str, interval_label: &str) -> Self {
        IntermittentTimer { start_time: None,
                            total_elapsed: Duration::new(0, 0),
                            last_interval: Duration::new(0, 0),
                            name: name.to_string(),
                            interval_label: interval_label.to_string() }
    }

    pub fn done(&self) {
        println!("\x1b[36mTiming: {} took {:.2} s\x1b[0m",
                 self.name,
                 self.total_elapsed.as_secs_f64());
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

            if !self.interval_label.is_empty() {
                println!("\x1b[34m{:.2} s\x1b[0m", self.last_interval.as_secs_f64());
            }
        }
    }

    #[allow(dead_code)]
    pub fn get_interval(&self) -> f64 {
        self.last_interval.as_secs_f64()
    }
}
