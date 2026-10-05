// SPDX-License-Identifier: Apache-2.0
//! RocketKV P1f / A8 (`tok_dynamo.md` §3.3): measure how the engines' step loop fares when it shares the frontend's
//! host cores, as on a GPU node (arXiv 2603.22774). Nothing is added: the loop does exactly the stock work and sleeps
//! the stock way; with `DYN_MOCKER_ENGINE_CPU=1` it is only timed (off = stock, untimed).
//!
//! Per scheduler pass (one engine step):
//!   sched_us  wall time of the pass's own CPU work (scheduling, KV bookkeeping, output publication) before its wait
//!   late_us   how long after the simulated GPU end the loop actually woke: time it waited for a core (T9)
//!   wait_us   the wait itself
//! `DYN_MOCKER_GPU_WAIT=spin` is a labelled diagnostic only (a CUDA-sync-style spin instead of the timer sleep).
//! Each worker logs `enginecpu w= passes= sched_us= late_us= wait_us= lh=` every 256 passes (sums; lh = the passes'
//! lateness histogram, bucket i: late < 2^i us, i = 0..23, for its percentiles: RocketKV W2-V).
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub struct Config {
    pub spin_wait: bool,
}

pub fn config() -> Option<&'static Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    C.get_or_init(|| {
        if std::env::var("DYN_MOCKER_ENGINE_CPU").ok().as_deref() != Some("1") {
            return None;
        }
        let spin_wait = std::env::var("DYN_MOCKER_GPU_WAIT").ok().as_deref() == Some("spin");
        tracing::info!(spin_wait, "mocker engine CPU on (DYN_MOCKER_ENGINE_CPU=1): step loop timed, nothing added");
        Some(Config { spin_wait })
    })
    .as_ref()
}

/// Busy-poll until `deadline` (diagnostic `DYN_MOCKER_GPU_WAIT=spin` only).
pub fn spin_until(deadline: Instant) {
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

/// Running sums for one scheduler (one simulated GPU).
#[derive(Default)]
pub struct Stats {
    worker: u32,
    passes: u64,
    sched_us: u64,
    late_us: u64,
    wait_us: u64,
    lh: [u64; 24],
}

impl Stats {
    pub fn new(worker: u32) -> Self {
        Self { worker, ..Default::default() }
    }
    pub fn add(&mut self, sched: Duration, late: Duration, wait: Duration) {
        self.passes += 1;
        self.sched_us += sched.as_micros() as u64;
        self.late_us += late.as_micros() as u64;
        self.wait_us += wait.as_micros() as u64;
        let l = late.as_micros() as u64;
        self.lh[((64 - l.leading_zeros()) as usize).min(23)] += 1;
        if self.passes % 256 == 0 {
            let lh: Vec<String> = self.lh.iter().map(|x| x.to_string()).collect();
            tracing::info!(
                "enginecpu w={} passes={} sched_us={} late_us={} wait_us={} lh={}",
                self.worker,
                self.passes,
                self.sched_us,
                self.late_us,
                self.wait_us,
                lh.join(",")
            );
        }
    }
}
