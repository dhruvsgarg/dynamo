// SPDX-License-Identifier: Apache-2.0
//! RocketKV W2-V (`tok_dynamo.md` §1.8): a VLM worker's image work, in the mocker. Off unless `DYN_MOCKER_MM` is set
//! (off = stock: nothing scanned, nothing waited).
//!
//! The workload (`tok/dyn/w2.py --images`) writes each screenshot into the prompt as an image block of N tokens, as a
//! VLM's prompt carries N image placeholders: `start`, 8 hex-digit tokens (the image's uid), `pad` x (N - 10), `end`.
//! The frontend tokenizes them like any text; the mocker counts them as prompt tokens (GPU prefill from AIC, KV prefix
//! reuse by block hash, like the placeholders of a real VLM). A prefill (or aggregated) worker additionally, per
//! request, before scheduling it:
//!   1 finds the image blocks (one linear scan of the token IDs, every mode alike);
//!   2 looks each uid up in its processor cache (an LRU of `DYN_MOCKER_MM_CACHE_N` images per worker: vLLM's 4 GiB per
//!     process holds ~190 processed Qwen3-VL 720p images); a hit costs nothing, as in vLLM;
//!   3 preprocesses the misses, per `DYN_MOCKER_MM`:
//!       rust     for real, on this process's cores (the worker shares the frontend's k cores under A8): base64 decode
//!                of the image's data URI + decode (Dynamo's decoder) + the model's processor (llm-multimodal, the
//!                MM.9 library), on `DYN_MOCKER_MM_THREADS` threads per worker (1 = one processor per engine process,
//!                as vLLM runs it); corpus image = uid % the corpus's images of `DYN_MOCKER_MM_CLASS`
//!       emulate  the CMM (EMULATED, labelled): no host work, the request waits `DYN_MOCKER_MM_EMU_MS` per miss (MM.9's
//!                CMM latency per image + the pool read) on one of `DYN_MOCKER_MM_EMU_LANES` lanes per worker (the
//!                CMM's 16 cores shared by the prefill workers: 16 / workers; 0 = unlimited, W2-V.1), queueing for a
//!                lane like rust queues for its threads
//!       cmm      the CMM, REAL (tok_dynamo.md P11 CMM-Img): the miss's data-URI payload goes into one of this worker's lanes
//!                in the pool (ABI5, RocketKV tok/mm/src/lanes.rs), `mmsvc serve` (on the CMM's Arm cores, or on x86 as the
//!                loopback) decodes and processes it and leaves the tensor in the pool; the worker polls the lane's
//!                completion every `DYN_MOCKER_MM_CMM_POLL_US` (default 100) with a sleep, so the host core is free in
//!                between. Env: `DYN_MOCKER_MM_CMM_POOL` (/dev/dax0.0 or a shared file), `_BASE` (byte offset of the image
//!                region, 2 MiB aligned), `_LANES` (lanes this worker owns, default 4), `_CMO` (auto|on|off); lanes are
//!                claimed with flock files in `DYN_MOCKER_MM_CMM_LOCKDIR` (/dev/shm) so the prefill workers share the 16.
//!                Corpus as `rust` (DYN_MOCKER_MM_CORPUS / _CLASS); the tensor is never copied back (the GPU reads it from
//!                the pool: modelled by the encoder wait)
//!       ideal    nothing (the ceiling)
//!   4 waits the vision encoder, `DYN_MOCKER_MM_ENC_MS` per miss (A10, modelled; an encode stage ahead of the prefill
//!     GPU, as Dynamo's E/P/D runs it), in every mode.
//! Placement (W2-V.3, M2): `DYN_MOCKER_MM_CPUS` (a cpu list, e.g. `40,41` or `42-57`) pins the image threads (`rust`'s
//! processors, `cmm`'s lane clients) to the host budget's image cores, apart from the engine's own core; unset = they
//! inherit the process's cores (W2-V.1 / V.2).
//! Telemetry (cumulative, every `DYN_MOCKER_MM_EVERY` requests; nothing per request is logged):
//!   `mmwork w=<pid> mode= threads= reqs= imgs= miss= queue_us= prep_us= cpu_us= emu_us= enc_us= wait_us= idle_us= starved_us=
//!    img_us= total_us= cold= cold_us= warm= warm_us= h=<log2 ms buckets>`
//! where threads = the worker's concurrency for its misses (rust: processor threads; cmm: lanes it owns; emulate: lanes,
//! 0 = unlimited; ideal: 0) (W2-V.3, M3: equal concurrency, AP4);
//! where wait = what the request waited for steps 3 + 4, h its histogram (bucket i: wait < 2^i ms, i = 0..15); idle /
//! starved / img (W2-V.2, dynamo_mocker::engine_cpu::starve, needs DYN_MOCKER_ENGINE_CPU=1): the worker's GPU idle,
//! idle while >= 1 request was in image work (the GPU waiting on the image work), >= 1 request in image work, and the
//! time since the first such event; cold / warm (W2-V.3, C33, `rust` only): the misses processed by a thread that had been
//! idle >= `DYN_MOCKER_MM_COLD_MS` (default 1000) before the job / the others, and their processing wall (an idle x86 core
//! clocks down, TD18: the check that the image cores are warm).
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Rust,
    Emulate,
    Cmm,
    Ideal,
}

pub struct Config {
    pub mode: Mode,
    start: u32,
    pad: u32,
    end: u32,
    hex0: u32,
    emu: Duration,
    lanes: usize,
    enc: Duration,
    cache_n: usize,
    every: u64,
    threads: u64,
    cold: Duration,
}

/// `DYN_MOCKER_MM_CPUS` as a list of cpus (`a,b,c-d`); empty when unset.
fn mm_cpus() -> Vec<usize> {
    parse_cpus(&std::env::var("DYN_MOCKER_MM_CPUS").unwrap_or_default())
}

fn parse_cpus(spec: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => v.extend(
                a.parse::<usize>().expect("DYN_MOCKER_MM_CPUS: a cpu list")..=b.parse::<usize>().expect("DYN_MOCKER_MM_CPUS: a cpu list"),
            ),
            None => v.push(part.parse().expect("DYN_MOCKER_MM_CPUS: a cpu list")),
        }
    }
    v
}

/// Pin the calling thread to `cpus` (none = leave it where it is); false if the kernel refused.
#[cfg(target_os = "linux")]
fn pin_thread(cpus: &[usize]) -> bool {
    if cpus.is_empty() {
        return true;
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            libc::CPU_SET(c, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}

/// Off Linux (a laptop build, tests only) a thread cannot be pinned: only an empty list succeeds.
#[cfg(not(target_os = "linux"))]
fn pin_thread(cpus: &[usize]) -> bool {
    cpus.is_empty()
}

pub fn config() -> Option<&'static Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    C.get_or_init(|| {
        let mode = match std::env::var("DYN_MOCKER_MM").ok().as_deref() {
            None | Some("") | Some("off") => return None,
            Some("rust") => Mode::Rust,
            Some("emulate") => Mode::Emulate,
            Some("cmm") => Mode::Cmm,
            Some("ideal") => Mode::Ideal,
            Some(o) => panic!("DYN_MOCKER_MM={o}: rust, emulate, cmm, ideal or off"),
        };
        let toks: Vec<u32> = std::env::var("DYN_MOCKER_MM_TOKENS")
            .expect("DYN_MOCKER_MM needs DYN_MOCKER_MM_TOKENS=start,pad,end,hex0 (w2.py's meta)")
            .split(',')
            .map(|t| t.trim().parse().expect("DYN_MOCKER_MM_TOKENS: four token ids"))
            .collect();
        assert_eq!(toks.len(), 4, "DYN_MOCKER_MM_TOKENS: start,pad,end,hex0");
        let ms = |k: &str, d: f64| -> Duration {
            Duration::from_secs_f64(std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d) / 1e3)
        };
        let num = |k: &str, d: u64| -> u64 { std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d) };
        let c = Config {
            mode,
            start: toks[0],
            pad: toks[1],
            end: toks[2],
            hex0: toks[3],
            emu: ms("DYN_MOCKER_MM_EMU_MS", 8.1),
            lanes: num("DYN_MOCKER_MM_EMU_LANES", 0) as usize,
            enc: ms("DYN_MOCKER_MM_ENC_MS", 0.0),
            cache_n: num("DYN_MOCKER_MM_CACHE_N", 190) as usize,
            every: num("DYN_MOCKER_MM_EVERY", 16).max(1),
            cold: ms("DYN_MOCKER_MM_COLD_MS", 1000.0),
            threads: match mode {
                Mode::Rust => num("DYN_MOCKER_MM_THREADS", 1).max(1),
                Mode::Cmm => num("DYN_MOCKER_MM_CMM_LANES", 4),
                Mode::Emulate => num("DYN_MOCKER_MM_EMU_LANES", 0),
                Mode::Ideal => 0,
            },
        };
        if mode == Mode::Rust {
            pool::start().unwrap_or_else(|e| panic!("DYN_MOCKER_MM=rust: {e}"));
        }
        if mode == Mode::Cmm {
            cmm::start().unwrap_or_else(|e| panic!("DYN_MOCKER_MM=cmm: {e}"));
        }
        tracing::info!(
            mode = ?mode, threads = c.threads, cpus = ?mm_cpus(), emu_ms = c.emu.as_secs_f64() * 1e3, emu_lanes = c.lanes, enc_ms = c.enc.as_secs_f64() * 1e3, cache_n = c.cache_n,
            "mocker images on (DYN_MOCKER_MM): image blocks preprocessed in the worker"
        );
        Some(c)
    })
    .as_ref()
}

/// The uids of the image blocks in a prompt, in order.
pub fn scan(c: &Config, tokens: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] == c.start && i + 9 <= tokens.len() {
            let mut uid = 0u32;
            let mut ok = true;
            for &t in &tokens[i + 1..i + 9] {
                match t.checked_sub(c.hex0) {
                    Some(d) if d < 16 => uid = uid << 4 | d,
                    _ => ok = false,
                }
            }
            if ok {
                out.push(uid);
                i += 9;
                while i < tokens.len() && tokens[i] == c.pad {
                    i += 1;
                }
                if i < tokens.len() && tokens[i] == c.end {
                    i += 1;
                }
                continue;
            }
        }
        i += 1;
    }
    out
}

#[derive(Default)]
struct Stats {
    reqs: u64,
    imgs: u64,
    miss: u64,
    queue_us: u64,
    prep_us: u64,
    cpu_us: u64,
    emu_us: u64,
    enc_us: u64,
    wait_us: u64,
    cold: u64,
    cold_us: u64,
    warm: u64,
    warm_us: u64,
    h: [u64; 16],
}

impl Stats {
    /// C33: a `rust` job's misses go to `cold` when its thread had idled >= `cold` before it, else to `warm`.
    fn idle_split(&mut self, cold: Duration, idle_us: u64, misses: u64, prep_us: u64) {
        if idle_us >= cold.as_micros() as u64 {
            self.cold += misses;
            self.cold_us += prep_us;
        } else {
            self.warm += misses;
            self.warm_us += prep_us;
        }
    }
}

struct Worker {
    cache: VecDeque<u32>,
    st: Stats,
}

fn worker() -> &'static Mutex<Worker> {
    static W: OnceLock<Mutex<Worker>> = OnceLock::new();
    W.get_or_init(|| Mutex::new(Worker { cache: VecDeque::new(), st: Stats::default() }))
}

fn lanes(n: usize) -> &'static tokio::sync::Semaphore {
    static L: OnceLock<tokio::sync::Semaphore> = OnceLock::new();
    L.get_or_init(|| tokio::sync::Semaphore::new(n))
}

/// The CMM, EMULATED: one lane (if limited) for the request's misses, `emu` each; (queue, emulated) in us.
async fn emulate(c: &Config, misses: usize) -> (u64, u64) {
    let t = Instant::now();
    let _lane = if c.lanes > 0 { Some(lanes(c.lanes).acquire().await.expect("lanes open")) } else { None };
    let queue = t.elapsed().as_micros() as u64;
    let d = c.emu * misses as u32;
    tokio::time::sleep(d).await;
    (queue, d.as_micros() as u64)
}

/// Steps 2-4 for one request's prompt; returns when the request may be scheduled.
pub async fn before_prefill(c: &Config, tokens: &[u32]) {
    let uids = scan(c, tokens);
    let t0 = Instant::now();
    dynamo_mocker::engine_cpu::starve::img(true);
    let misses: Vec<u32> = {
        let mut w = worker().lock().unwrap();
        let mut m = Vec::new();
        for &u in &uids {
            if let Some(p) = w.cache.iter().position(|&x| x == u) {
                w.cache.remove(p);
            } else {
                m.push(u);
            }
            w.cache.push_back(u);
            if w.cache.len() > c.cache_n {
                w.cache.pop_front();
            }
        }
        m
    };
    let (mut queue, mut prep, mut cpu, mut emu, mut idle_before) = (0u64, 0u64, 0u64, 0u64, None);
    if !misses.is_empty() {
        match c.mode {
            Mode::Rust => {
                let r = pool::run(misses.clone()).await;
                queue = r.0;
                prep = r.1;
                cpu = r.2;
                idle_before = Some(r.3);
            }
            Mode::Emulate => (queue, emu) = emulate(c, misses.len()).await,
            Mode::Cmm => {
                let r = cmm::run(misses.clone()).await;
                (queue, prep, cpu, emu) = (r.0, r.1, r.2, r.3); // emu_us = the service's own time per request (its t_end - t_start)
            }
            Mode::Ideal => {}
        }
        if !c.enc.is_zero() {
            tokio::time::sleep(c.enc * misses.len() as u32).await;
        }
    }
    let enc = (c.enc * misses.len() as u32).as_micros() as u64;
    let wait = t0.elapsed().as_micros() as u64;
    dynamo_mocker::engine_cpu::starve::img(false);
    let mut w = worker().lock().unwrap();
    let s = &mut w.st;
    s.reqs += 1;
    s.imgs += uids.len() as u64;
    s.miss += misses.len() as u64;
    s.queue_us += queue;
    s.prep_us += prep;
    s.cpu_us += cpu;
    s.emu_us += emu;
    s.enc_us += enc;
    s.wait_us += wait;
    if let Some(idle) = idle_before {
        s.idle_split(c.cold, idle, misses.len() as u64, prep);
    }
    let b = (64 - (wait / 1000).leading_zeros()) as usize; // wait < 2^b ms
    s.h[b.min(15)] += 1;
    if s.reqs % c.every == 0 {
        let h: Vec<String> = s.h.iter().map(|x| x.to_string()).collect();
        let (idle, starved, img, total) = dynamo_mocker::engine_cpu::starve::snapshot();
        tracing::info!(
            "mmwork w={} mode={:?} threads={} reqs={} imgs={} miss={} queue_us={} prep_us={} cpu_us={} emu_us={} enc_us={} wait_us={} idle_us={} starved_us={} img_us={} total_us={} cold={} cold_us={} warm={} warm_us={} h={}",
            std::process::id(),
            c.mode,
            c.threads,
            s.reqs,
            s.imgs,
            s.miss,
            s.queue_us,
            s.prep_us,
            s.cpu_us,
            s.emu_us,
            s.enc_us,
            s.wait_us,
            idle,
            starved,
            img,
            total,
            s.cold,
            s.cold_us,
            s.warm,
            s.warm_us,
            h.join(",")
        );
    }
}

#[cfg(feature = "mm-routing")]
mod pool {
    //! The worker's image threads: each owns a processor; jobs = one request's misses, processed in order.
    use std::io::Cursor;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use llm_multimodal::{PreProcessorConfig, VisionProcessorRegistry};

    struct Job {
        uids: Vec<u32>,
        enq: Instant,
        tx: tokio::sync::oneshot::Sender<(u64, u64, u64, u64)>,
    }

    struct Corpus {
        uris: Vec<String>,
        cfg: PreProcessorConfig,
        model: String,
    }

    static TX: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();

    fn thread_cpu_us() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        ts.tv_sec as u64 * 1_000_000 + ts.tv_nsec as u64 / 1000
    }

    fn load() -> Result<Corpus, String> {
        let dir = std::env::var("DYN_MOCKER_MM_CORPUS").map_err(|_| "DYN_MOCKER_MM_CORPUS (a directory of images)")?;
        let class = std::env::var("DYN_MOCKER_MM_CLASS").unwrap_or_else(|_| "agent720".into());
        let cfgp = std::env::var("DYN_MOCKER_MM_CFG").map_err(|_| "DYN_MOCKER_MM_CFG (a processor config json)")?;
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{dir}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&format!("{class}-"))))
            .collect();
        paths.sort();
        let mut uris = Vec::new();
        for p in &paths {
            let mime = match p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                _ => continue,
            };
            let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
            uris.push(format!("data:{mime};base64,{}", STANDARD.encode(&b)));
        }
        if uris.is_empty() {
            return Err(format!("no {class}-* images in {dir}"));
        }
        let cfg = PreProcessorConfig::from_json(&std::fs::read_to_string(&cfgp).map_err(|e| format!("{cfgp}: {e}"))?)
            .map_err(|e| format!("{cfgp}: {e}"))?;
        let model = std::path::Path::new(&cfgp).file_stem().unwrap().to_string_lossy().to_string();
        VisionProcessorRegistry::with_defaults().find(&model, None).ok_or(format!("no llm-multimodal processor for {model}"))?;
        tracing::info!(images = uris.len(), class, model, "mocker images: corpus loaded");
        Ok(Corpus { uris, cfg, model })
    }

    /// The corpus as base64 payloads (the data URI's text after the comma), for the CMM lanes: no processor config needed.
    pub fn payloads() -> Result<Arc<Vec<Vec<u8>>>, String> {
        let dir = std::env::var("DYN_MOCKER_MM_CORPUS").map_err(|_| "DYN_MOCKER_MM_CORPUS (a directory of images)")?;
        let class = std::env::var("DYN_MOCKER_MM_CLASS").unwrap_or_else(|_| "agent720".into());
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{dir}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&format!("{class}-"))))
            .filter(|p| matches!(p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
            .collect();
        paths.sort();
        let mut v = Vec::new();
        for p in &paths {
            v.push(STANDARD.encode(std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).into_bytes());
        }
        if v.is_empty() {
            return Err(format!("no {class}-* images in {dir}"));
        }
        Ok(Arc::new(v))
    }

    pub fn start() -> Result<(), String> {
        let corpus = Arc::new(load()?);
        let n: usize = std::env::var("DYN_MOCKER_MM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
        let cpus = Arc::new(super::mm_cpus());
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..n {
            let (rx, corpus, cpus) = (rx.clone(), corpus.clone(), cpus.clone());
            std::thread::Builder::new()
                .name(format!("mm-prep-{i}"))
                .spawn(move || {
                    assert!(super::pin_thread(&cpus), "mm-prep-{i}: cannot pin to DYN_MOCKER_MM_CPUS {cpus:?}");
                    let reg = VisionProcessorRegistry::with_defaults();
                    let p = reg.find(&corpus.model, None).expect("checked at load");
                    let mut last = Instant::now();
                    loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        };
                        let (t, c) = (Instant::now(), thread_cpu_us());
                        let queue = t.duration_since(job.enq).as_micros() as u64;
                        let idle = t.duration_since(last).as_micros() as u64; // C33: how long this thread sat idle
                        for u in &job.uids {
                            let uri = &corpus.uris[*u as usize % corpus.uris.len()];
                            let bytes = STANDARD.decode(&uri[uri.find(',').unwrap() + 1..]).expect("corpus base64");
                            let img = image::ImageReader::new(Cursor::new(&bytes))
                                .with_guessed_format()
                                .expect("corpus image")
                                .decode()
                                .expect("corpus image decodes");
                            std::hint::black_box(p.preprocess(std::slice::from_ref(&img), &corpus.cfg).expect("preprocess"));
                        }
                        last = Instant::now();
                        let _ = job.tx.send((queue, last.duration_since(t).as_micros() as u64, thread_cpu_us() - c, idle));
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        TX.set(Mutex::new(tx)).map_err(|_| "image pool started twice".to_string())
    }

    /// (queue, processing wall, processing CPU, the thread's idle time before the job) in us for one request's misses.
    pub async fn run(uids: Vec<u32>) -> (u64, u64, u64, u64) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        TX.get().expect("pool started").lock().unwrap().send(Job { uids, enq: Instant::now(), tx }).expect("pool alive");
        rx.await.expect("pool answers")
    }
}

#[cfg(feature = "mm-routing")]
mod cmm {
    //! DYN_MOCKER_MM=cmm: the client half of ABI5 (RocketKV tok/mm/src/lanes.rs holds the protocol and `mmsvc serve` the
    //! service; the constants below mirror it, and `lane_abi_constants` pins them). One thread per lane this worker owns;
    //! a job = one request's misses, processed in order on one lane, like `emulate` and `rust`.
    use std::fs::File;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;
    use std::ptr;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    const MAGIC: u64 = 0x3558_4e4c_4d49_314b;
    const ABI: u32 = 5;
    const HDR: usize = 4096;
    const HUGE: usize = 2 << 20;
    const LANE_CTL: usize = 4096;
    const OFF_REQ: usize = 0;
    const OFF_PARAM: usize = 64;
    const OFF_DONE: usize = 128;
    const OFF_STATE: usize = 64;
    const RESULT_OFF: usize = 256;
    const RESULT_BYTES: usize = 192;

    #[derive(Clone, Copy)]
    struct Geometry {
        lanes: usize,
        in_cap: usize,
        out_cap: usize,
    }
    impl Geometry {
        fn stride(&self) -> usize {
            (LANE_CTL + self.in_cap + self.out_cap).div_ceil(HUGE) * HUGE
        }
        fn total(&self) -> usize {
            HDR.next_multiple_of(HUGE) + self.lanes * self.stride()
        }
        fn lane_off(&self, i: usize) -> usize {
            HDR.next_multiple_of(HUGE) + i * self.stride()
        }
    }

    struct Pool {
        base: *mut u8,
        len: usize,
        cmo: bool,
    }
    unsafe impl Send for Pool {}
    unsafe impl Sync for Pool {}
    impl Pool {
        fn open(path: &str, off: u64, len: usize, cmo: bool) -> Result<Pool, String> {
            let f = std::fs::OpenOptions::new().read(true).write(true).open(path).map_err(|e| format!("{path}: {e}"))?;
            let p = unsafe {
                libc::mmap(ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, f.as_raw_fd(), off as libc::off_t)
            };
            if p == libc::MAP_FAILED {
                return Err(format!("mmap {path}+{off:#x} ({len} B): {}", std::io::Error::last_os_error()));
            }
            Ok(Pool { base: p as *mut u8, len, cmo })
        }
        fn at(&self, off: usize) -> *mut u8 {
            assert!(off < self.len);
            unsafe { self.base.add(off) }
        }
        fn rd64(&self, off: usize) -> u64 {
            unsafe { ptr::read_volatile(self.at(off) as *const u64) }
        }
        fn wr64(&self, off: usize, v: u64) {
            unsafe { ptr::write_volatile(self.at(off) as *mut u64, v) }
        }
        /// Write back and drop the lines of [off, off+n): what a writer owes before the CMM reads, and a reader before
        /// it reads what the CMM wrote (clflushopt: CLFLUSH serialises every line against the device, 66 MB/s, S2f).
        fn sync(&self, off: usize, n: usize) {
            if !self.cmo {
                return;
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let mut a = (self.at(off) as usize) & !63;
                let e = self.at(off) as usize + n.max(1);
                while a < e {
                    std::arch::asm!("clflushopt [{0}]", in(reg) a, options(nostack, preserves_flags));
                    a += 64;
                }
                std::arch::asm!("mfence", options(nostack, preserves_flags));
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = (off, n);
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
        }
        /// A bulk write that leaves nothing in this core's cache (non-temporal stores), so the CMM can read it.
        fn put(&self, off: usize, src: &[u8]) {
            let dst = self.at(off);
            assert!(off + src.len() <= self.len);
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                let mut i = 0;
                if (dst as usize) & 15 == 0 {
                    while i + 64 <= src.len() {
                        let s = src.as_ptr().add(i) as *const __m128i;
                        let d = dst.add(i) as *mut __m128i;
                        _mm_stream_si128(d, _mm_loadu_si128(s));
                        _mm_stream_si128(d.add(1), _mm_loadu_si128(s.add(1)));
                        _mm_stream_si128(d.add(2), _mm_loadu_si128(s.add(2)));
                        _mm_stream_si128(d.add(3), _mm_loadu_si128(s.add(3)));
                        i += 64;
                    }
                }
                if i < src.len() {
                    ptr::copy_nonoverlapping(src.as_ptr().add(i), dst.add(i), src.len() - i);
                }
                _mm_sfence();
                if i < src.len() {
                    self.sync(off + i, src.len() - i);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
                self.sync(off, src.len());
            }
        }
    }

    fn thread_cpu_us() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        ts.tv_sec as u64 * 1_000_000 + ts.tv_nsec as u64 / 1000
    }

    struct Job {
        uids: Vec<u32>,
        enq: Instant,
        tx: tokio::sync::oneshot::Sender<(u64, u64, u64, u64)>,
    }

    static TX: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    static LOCKS: OnceLock<Vec<File>> = OnceLock::new();

    /// The first `want` lanes of `total` that no other worker holds (flock, released when this process exits).
    fn claim(dir: &str, total: usize, want: usize) -> Result<Vec<usize>, String> {
        let (mut got, mut files) = (Vec::new(), Vec::new());
        for i in 0..total {
            if got.len() == want {
                break;
            }
            let f = std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(format!("{dir}/k1img-lane-{i}.lock"))
                .map_err(|e| format!("lane lock {dir}/k1img-lane-{i}.lock: {e}"))?;
            if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                got.push(i);
                files.push(f);
            }
        }
        if got.len() < want {
            return Err(format!("only {} of {want} lanes free (of {total}): the other workers hold the rest", got.len()));
        }
        LOCKS.set(files).map_err(|_| "lanes claimed twice".to_string())?;
        Ok(got)
    }

    pub fn start() -> Result<(), String> {
        let path = std::env::var("DYN_MOCKER_MM_CMM_POOL").map_err(|_| "DYN_MOCKER_MM_CMM_POOL (/dev/dax0.0 or a shared file)")?;
        let base: u64 = std::env::var("DYN_MOCKER_MM_CMM_BASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let want: usize = std::env::var("DYN_MOCKER_MM_CMM_LANES").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let total: usize = std::env::var("DYN_MOCKER_MM_CMM_LANES_TOTAL").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        let dir = std::env::var("DYN_MOCKER_MM_CMM_LOCKDIR").unwrap_or_else(|_| "/dev/shm".into());
        let poll = Duration::from_micros(std::env::var("DYN_MOCKER_MM_CMM_POLL_US").ok().and_then(|v| v.parse().ok()).unwrap_or(100));
        let cmo = match std::env::var("DYN_MOCKER_MM_CMM_CMO").unwrap_or_else(|_| "auto".into()).as_str() {
            "on" => true,
            "off" => false,
            _ => path.starts_with("/dev/"),
        };
        // the service must be up: header, ABI, state 1 (it may still be creating the pool file); then map the lanes
        let t0 = Instant::now();
        let g = loop {
            match Pool::open(&path, base, HUGE, cmo) {
                Ok(probe) => {
                    probe.sync(0, 64);
                    if probe.rd64(0) == MAGIC {
                        probe.sync(0, 128);
                        let ab = probe.rd64(8);
                        probe.sync(OFF_STATE, 64);
                        if (ab & 0xffff_ffff) as u32 == ABI && probe.rd64(OFF_STATE) == 1 {
                            break Geometry { lanes: (ab >> 32) as usize, in_cap: probe.rd64(24) as usize, out_cap: probe.rd64(32) as usize };
                        }
                    }
                }
                Err(e) if t0.elapsed().as_secs() > 60 => return Err(e),
                Err(_) => {}
            }
            if t0.elapsed().as_secs() > 60 {
                return Err(format!("no ready ABI5 service at {path}+{base:#x} (mmsvc serve --pool {path} --base {base})"));
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        if g.lanes != total {
            return Err(format!("the service has {} lanes, DYN_MOCKER_MM_CMM_LANES_TOTAL says {total}", g.lanes));
        }
        let mine = claim(&dir, total, want)?;
        let pool = Arc::new(Pool::open(&path, base, g.total(), cmo)?);
        let uris = super::pool::payloads()?;
        let cpus = Arc::new(super::mm_cpus());
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for &lane in &mine {
            let (rx, pool, uris, cpus) = (rx.clone(), pool.clone(), uris.clone(), cpus.clone());
            std::thread::Builder::new()
                .name(format!("mm-cmm-{lane}"))
                .spawn(move || {
                    assert!(super::pin_thread(&cpus), "mm-cmm-{lane}: cannot pin to DYN_MOCKER_MM_CPUS {cpus:?}");
                    let lo = g.lane_off(lane);
                    loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        };
                        let (t, c) = (Instant::now(), thread_cpu_us());
                        let queue = t.duration_since(job.enq).as_micros() as u64;
                        let mut svc_ns = 0u64;
                        for u in &job.uids {
                            let payload = &uris[*u as usize % uris.len()];
                            assert!(payload.len() <= g.in_cap, "payload {} B > the lane's in_cap {}", payload.len(), g.in_cap);
                            pool.sync(lo + OFF_REQ, 64);
                            let seq = pool.rd64(lo + OFF_REQ) + 1;
                            pool.put(lo + LANE_CTL, payload);
                            pool.wr64(lo + OFF_PARAM, payload.len() as u64);
                            pool.wr64(lo + OFF_PARAM + 8, 0);
                            pool.wr64(lo + OFF_PARAM + 16, 0);
                            pool.sync(lo + OFF_PARAM, 64);
                            pool.wr64(lo + OFF_REQ, seq);
                            pool.sync(lo + OFF_REQ, 64);
                            loop {
                                pool.sync(lo + OFF_DONE, 64);
                                if pool.rd64(lo + OFF_DONE) >= seq {
                                    break;
                                }
                                std::thread::sleep(poll);
                            }
                            pool.sync(lo + RESULT_OFF, RESULT_BYTES);
                            let status = pool.rd64(lo + RESULT_OFF) as u32 as i32;
                            assert_eq!(status, 0, "mmsvc lane {lane}: image {u} status {status}");
                            svc_ns += pool.rd64(lo + RESULT_OFF + 88) - pool.rd64(lo + RESULT_OFF + 80); // t_end - t_start
                        }
                        let _ = job.tx.send((queue, t.elapsed().as_micros() as u64, thread_cpu_us() - c, svc_ns / 1000));
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        let _ = std::io::stderr().flush();
        tracing::info!(pool = %path, base, lanes = ?mine, poll_us = poll.as_micros() as u64, cmo, "mocker images: CMM lanes claimed (ABI5)");
        TX.set(Mutex::new(tx)).map_err(|_| "cmm lanes started twice".to_string())
    }

    /// (queue, submit -> done wall, this thread's CPU, the service's own time) in us for one request's misses.
    pub async fn run(uids: Vec<u32>) -> (u64, u64, u64, u64) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        TX.get().expect("cmm lanes started").lock().unwrap().send(Job { uids, enq: Instant::now(), tx }).expect("lane threads alive");
        rx.await.expect("a lane answers")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The real client against the real service over a file pool (the loopback): RK_MMSVC=<RocketKV>/tok/mm/target/release/mmsvc
        /// RK_MM_CORPUS=<RocketKV>/tok/mm/corpus RK_MM_CFG=<RocketKV>/tok/mm/cfg/qwen3-vl-cap1280-bicubic.json; skipped without them.
        /// One test per process: `start()` is a OnceLock.
        #[tokio::test]
        async fn lanes_roundtrip_against_mmsvc() {
            let (Ok(bin), Ok(dir), Ok(cfg)) = (std::env::var("RK_MMSVC"), std::env::var("RK_MM_CORPUS"), std::env::var("RK_MM_CFG")) else {
                eprintln!("skipped: set RK_MMSVC, RK_MM_CORPUS and RK_MM_CFG");
                return;
            };
            let pool = format!("/tmp/rk-mmsvc-test-{}.pool", std::process::id());
            let lockdir = format!("/tmp/rk-mmsvc-locks-{}", std::process::id());
            std::fs::create_dir_all(&lockdir).unwrap();
            let mut svc = std::process::Command::new(bin)
                .args(["serve", "--pool", &pool, "--base", "0", "--create", "--cmo", "off", "--cfg", &cfg, "--lanes", "4", "--report-s", "0"])
                .stdout(std::process::Stdio::null())
                .spawn()
                .expect("mmsvc starts");
            unsafe {
                std::env::set_var("DYN_MOCKER_MM_CMM_POOL", &pool);
                std::env::set_var("DYN_MOCKER_MM_CMM_BASE", "0");
                std::env::set_var("DYN_MOCKER_MM_CMM_LANES", "2");
                std::env::set_var("DYN_MOCKER_MM_CMM_LANES_TOTAL", "4");
                std::env::set_var("DYN_MOCKER_MM_CMM_CMO", "off");
                std::env::set_var("DYN_MOCKER_MM_CMM_LOCKDIR", &lockdir);
                std::env::set_var("DYN_MOCKER_MM_CORPUS", dir);
                std::env::set_var("DYN_MOCKER_MM_CLASS", "shot1080");
            }
            let _ = cfg;
            start().expect("the lanes are claimed against the running service");
            // two requests of two images each: the lanes of this worker serve them in parallel
            let (a, b) = tokio::join!(run(vec![0, 1]), run(vec![2, 3]));
            for (queue, wall, host_cpu, svc_us) in [a, b] {
                assert!(wall > 0 && svc_us > 0, "the service worked: wall {wall} us, service {svc_us} us");
                assert!(wall >= svc_us, "wall {wall} < the service's own time {svc_us}");
                assert!(host_cpu < wall, "the client polls with sleeps, it does not spin: cpu {host_cpu} us of {wall} us (queue {queue})");
            }
            unsafe { libc::kill(svc.id() as i32, libc::SIGTERM) };
            let _ = svc.wait();
            let _ = std::fs::remove_file(&pool);
        }

        /// The constants this client mirrors from tok/mm/src/lanes.rs: a lane's byte layout must not drift.
        #[test]
        fn lane_abi_constants() {
            let g = Geometry { lanes: 16, in_cap: 8 << 20, out_cap: 64 << 20 };
            assert_eq!(MAGIC, 0x3558_4e4c_4d49_314b);
            assert_eq!((HDR, LANE_CTL, OFF_REQ, OFF_PARAM, OFF_DONE, OFF_STATE, RESULT_OFF, RESULT_BYTES), (4096, 4096, 0, 64, 128, 64, 256, 192));
            assert_eq!(g.stride(), 74 << 20);
            assert_eq!(g.lane_off(0), 2 << 20);
            assert_eq!(g.total(), (2 << 20) + 16 * (74 << 20));
        }
    }
}

#[cfg(not(feature = "mm-routing"))]
mod cmm {
    pub fn start() -> Result<(), String> {
        Err("this build has no mm-routing (tok/dyn/build.sh dynamo)".into())
    }
    pub async fn run(_uids: Vec<u32>) -> (u64, u64, u64, u64) {
        unreachable!("start() refused")
    }
}

#[cfg(not(feature = "mm-routing"))]
mod pool {
    pub fn start() -> Result<(), String> {
        Err("this build has no llm-multimodal: build the bindings with --features mm-routing (tok/dyn/build.sh dynamo)".into())
    }
    pub async fn run(_uids: Vec<u32>) -> (u64, u64, u64, u64) {
        unreachable!("start() refused")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            mode: Mode::Ideal,
            start: 100,
            pad: 101,
            end: 102,
            hex0: 110,
            emu: Duration::ZERO,
            lanes: 0,
            enc: Duration::ZERO,
            cache_n: 2,
            every: 1,
            threads: 0,
            cold: Duration::from_millis(1000),
        }
    }

    fn block(uid: u32, n: usize) -> Vec<u32> {
        let mut v = vec![100];
        v.extend((0..8).rev().map(|i| 110 + (uid >> (4 * i) & 15)));
        v.extend(std::iter::repeat_n(101, n - 10));
        v.push(102);
        v
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mm_cpus_parse_and_pin() {
        assert_eq!(parse_cpus("40,41"), vec![40, 41]);
        assert_eq!(parse_cpus(" 42-45 ,50"), vec![42, 43, 44, 45, 50]);
        assert!(parse_cpus("").is_empty());
        assert!(pin_thread(&[]), "no list: the thread stays where it is");
        // pin a fresh thread to the first cpu this process may use, and read it back
        let mine = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
            (0..libc::CPU_SETSIZE as usize).find(|&c| libc::CPU_ISSET(c, &set)).unwrap()
        };
        let got = std::thread::spawn(move || {
            assert!(pin_thread(&[mine]));
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
                (0..libc::CPU_SETSIZE as usize).filter(|&c| libc::CPU_ISSET(c, &set)).collect::<Vec<_>>()
            }
        })
        .join()
        .unwrap();
        assert_eq!(got, vec![mine]);
    }

    #[test]
    fn c33_splits_misses_by_idle_before() {
        let mut s = Stats::default();
        let cold = Duration::from_millis(1000);
        s.idle_split(cold, 1_500_000, 1, 40_000); // idle 1.5 s: cold
        s.idle_split(cold, 999_999, 2, 60_000); // just under: warm
        s.idle_split(cold, 0, 1, 30_000);
        assert_eq!((s.cold, s.cold_us, s.warm, s.warm_us), (1, 40_000, 3, 90_000));
    }

    #[test]
    fn scan_finds_blocks_and_uids() {
        let mut t = vec![1, 2, 3];
        t.extend(block(0xdeadbeef, 40));
        t.extend([4, 100, 5]); // a lone start token followed by text is not a block
        t.extend(block(7, 12));
        assert_eq!(scan(&cfg(), &t), vec![0xdeadbeef, 7]);
        assert!(scan(&cfg(), &[1, 2, 3]).is_empty());
    }

    /// The real path (rust mode): RK_MM_CORPUS=<RocketKV>/tok/mm/corpus RK_MM_CFG=<RocketKV>/tok/mm/cfg/qwen3-vl-cap1280-bicubic.json
    /// (class shot1080: the committed originals); skipped without them.
    #[cfg(feature = "mm-routing")]
    #[tokio::test]
    async fn rust_pool_preprocesses_corpus_images() {
        let (Ok(dir), Ok(cfg)) = (std::env::var("RK_MM_CORPUS"), std::env::var("RK_MM_CFG")) else {
            eprintln!("skipped: set RK_MM_CORPUS and RK_MM_CFG");
            return;
        };
        unsafe {
            std::env::set_var("DYN_MOCKER_MM_CORPUS", dir);
            std::env::set_var("DYN_MOCKER_MM_CFG", cfg);
            std::env::set_var("DYN_MOCKER_MM_CLASS", "shot1080");
        }
        pool::start().expect("pool starts");
        let (queue, prep, cpu, _idle) = pool::run(vec![0, 5]).await; // two images, the second wraps around the corpus
        assert!(prep > 0 && cpu > 0 && queue < prep, "queue {queue} prep {prep} cpu {cpu}");
    }

    #[tokio::test]
    async fn emulated_cmm_queues_for_its_lanes() {
        let c = Config { mode: Mode::Emulate, emu: Duration::from_millis(40), lanes: 2, ..cfg() };
        let t = Instant::now();
        let r = futures::future::join_all((0..4).map(|_| emulate(&c, 1))).await;
        let ms = t.elapsed().as_millis();
        assert!((80..140).contains(&ms), "4 misses on 2 lanes of 40 ms took {ms} ms");
        assert_eq!(r.iter().filter(|(q, _)| *q > 30_000).count(), 2, "two waited for a lane: {r:?}");
    }

    #[tokio::test]
    async fn processor_cache_is_lru_per_worker() {
        let c = cfg();
        let mut t = block(1, 12);
        t.extend(block(2, 12));
        before_prefill(&c, &t).await; // 2 misses
        before_prefill(&c, &block(1, 12)).await; // hit
        before_prefill(&c, &block(3, 12)).await; // miss, evicts 2
        before_prefill(&c, &block(2, 12)).await; // miss again
        let w = worker().lock().unwrap();
        assert_eq!((w.st.reqs, w.st.imgs, w.st.miss), (4, 5, 4));
    }
}
