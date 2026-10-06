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
//! Each worker logs `enginecpu w= passes= sched_us= late_us= wait_us= lh=` every 256 passes or 5 s, whichever comes
//! first (sums; lh = the passes' lateness histogram, bucket i: late < 2^i us, i = 0..23, for its percentiles: RocketKV
//! W2-V). Aggregated and disaggregated (P/D) workers alike (W2-V.1 found P/D passes untimed).
//! `starve` (W2-V.2): whether the simulated GPU sits idle while requests wait in the worker's image work.
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
    last: Option<Instant>,
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
        let due = self.last.is_none_or(|t| t.elapsed() >= Duration::from_secs(5));
        if self.passes % 256 == 0 || due {
            self.last = Some(Instant::now());
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

/// RocketKV W2-V.2 (`tok_dynamo.md` §1.8): does the simulated GPU wait on the host's image work? One worker process =
/// one scheduler (dp 1), so process-wide sums. `gpu(true/false)` brackets each pass's simulated GPU time (the timed
/// wait in scheduler/vllm/live.rs); `img(true/false)` brackets one request's image work (lib/llm/src/mocker/images.rs).
/// From the first event: idle = the GPU not busy; starved = idle while >= 1 request is in image work (the GPU waits on
/// the CPU, or on the CMM); img = >= 1 request in image work. Telemetry only: nothing waits on it.
pub mod starve {
    use std::sync::Mutex;
    use std::time::Instant;

    struct S {
        gpu: bool,
        img: u32,
        since: Option<Instant>,
        idle_us: u64,
        starved_us: u64,
        img_us: u64,
        total_us: u64,
    }

    static ST: Mutex<S> =
        Mutex::new(S { gpu: false, img: 0, since: None, idle_us: 0, starved_us: 0, img_us: 0, total_us: 0 });

    fn tick(s: &mut S) {
        let now = Instant::now();
        if let Some(t) = s.since {
            let d = now.duration_since(t).as_micros() as u64;
            s.total_us += d;
            if !s.gpu {
                s.idle_us += d;
                if s.img > 0 {
                    s.starved_us += d;
                }
            }
            if s.img > 0 {
                s.img_us += d;
            }
        }
        s.since = Some(now);
    }

    pub fn gpu(busy: bool) {
        let mut s = ST.lock().unwrap();
        tick(&mut s);
        s.gpu = busy;
    }

    pub fn img(enter: bool) {
        let mut s = ST.lock().unwrap();
        tick(&mut s);
        s.img = if enter { s.img + 1 } else { s.img.saturating_sub(1) };
    }

    /// (idle_us, starved_us, img_us, total_us) since the first event.
    pub fn snapshot() -> (u64, u64, u64, u64) {
        let mut s = ST.lock().unwrap();
        tick(&mut s);
        (s.idle_us, s.starved_us, s.img_us, s.total_us)
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn starved_counts_idle_time_with_images_in_flight() {
            use std::thread::sleep;
            use std::time::Duration;
            super::gpu(true); // GPU busy, no images: neither idle nor starved
            sleep(Duration::from_millis(20));
            super::img(true); // busy with an image in flight: img only
            sleep(Duration::from_millis(20));
            super::gpu(false); // idle with an image in flight: starved
            sleep(Duration::from_millis(30));
            super::img(false); // idle, no image: idle only
            sleep(Duration::from_millis(20));
            let (idle, starved, img, total) = super::snapshot();
            assert!((88_000..130_000).contains(&total), "total {total}");
            assert!((48_000..80_000).contains(&idle), "idle {idle}");
            assert!((28_000..45_000).contains(&starved), "starved {starved}");
            assert!((48_000..70_000).contains(&img), "img {img}");
        }
    }
}
