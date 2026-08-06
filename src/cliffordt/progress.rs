//! Live progress reporting for the long-running per-block stages (Stage 2's
//! windowed resynthesis, Stage 3's TRbO, Stage 4's final synthesis), modeled
//! on `data_processing/compile_cliffordt.py`'s `_progress_line`/
//! `_with_progress`/`_with_block_progress`:
//!
//! - Written straight to `/dev/tty`, not through stdout, so a `\r`-overwritten
//!   line is visible on the terminal watching the process but invisible to
//!   anything capturing stdout (a redirect, `tee`, a log file) -- checking
//!   `stdout.is_terminal()` isn't enough, since piping stdout through `tee`
//!   makes it not a terminal even though a real one is watching on the other
//!   end. Falls back to one plain line per update if there's no controlling
//!   terminal at all (headless: CI, a detached job).
//! - Time-throttled to at most one update per `PROGRESS_INTERVAL`, so a stage
//!   with many cheap blocks doesn't spam a line per block.
//! - The tracked count is only ever an estimate of the stage's real total
//!   (see each call site's own reasoning for what it counts), so reaching
//!   `total` exactly isn't guaranteed; callers print an unconditional final
//!   100% via `finish()` rather than relying on the count to get there itself.

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const PROGRESS_INTERVAL: Duration = Duration::from_millis(1000);

fn progress_tty() -> Option<&'static Mutex<File>> {
    static TTY: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    TTY.get_or_init(|| File::options().write(true).open("/dev/tty").ok().map(Mutex::new)).as_ref()
}

fn progress_line(label: &str, pct: u64, final_line: bool) {
    match progress_tty() {
        Some(tty) => {
            let mut f = tty.lock().unwrap();
            let _ = if final_line {
                writeln!(f, "\r  {label}: {pct}%   ")
            } else {
                write!(f, "\r  {label}: {pct}%   ")
            };
            let _ = f.flush();
        }
        None => println!("  {label}: {pct}%"),
    }
}

/// Tracks progress through an estimated `total` units of work, reporting a
/// throttled percentage as units complete via `add`. Safe to share across
/// rayon's parallel per-block closures (`Sync` via atomics only, no lock
/// held across a call) -- concurrent `add` calls just race harmlessly on
/// which one wins the throttle check, the same as a dropped update would.
pub struct ProgressTracker {
    label: String,
    total: usize,
    count: AtomicUsize,
    start: Instant,
    last_report_millis: AtomicU64,
}

impl ProgressTracker {
    /// `total == 0` makes every method a no-op, mirroring `_with_progress`'s
    /// own early return when there's nothing to report against -- a stage
    /// that touches no blocks at all shouldn't print a 0%...100% line.
    pub fn new(label: &str, total: usize) -> Self {
        ProgressTracker { label: label.to_string(), total, count: AtomicUsize::new(0), start: Instant::now(), last_report_millis: AtomicU64::new(0) }
    }

    /// Record `n` newly-completed units of work (one block, or however many
    /// new entries a call actually added to a shared cache -- see each call
    /// site), printing a throttled percentage update.
    pub fn add(&self, n: usize) {
        if self.total == 0 || n == 0 {
            return;
        }
        let count = self.count.fetch_add(n, Ordering::Relaxed) + n;
        let count = count.min(self.total);
        let now_millis = self.start.elapsed().as_millis() as u64;
        let last = self.last_report_millis.load(Ordering::Relaxed);
        if now_millis.saturating_sub(last) >= PROGRESS_INTERVAL.as_millis() as u64
            && self.last_report_millis.compare_exchange(last, now_millis, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            progress_line(&self.label, (count * 100 / self.total) as u64, false);
        }
    }

    /// Print the unconditional final 100% line -- not inferred from `count`
    /// reaching `total` on its own, since `total` is only an estimate (see
    /// module docs).
    pub fn finish(&self) {
        if self.total > 0 {
            progress_line(&self.label, 100, true);
        }
    }
}
