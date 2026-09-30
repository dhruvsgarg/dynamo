// SPDX-License-Identifier: Apache-2.0
//! RocketKV tokenizer backends for the CMM experiment (RocketKV `tok_dynamo.md` §4.5).
//!
//! Selected in [`crate::model_card::ModelDeploymentCard::tokenizer`] by environment:
//!
//! | env | effect | mode |
//! |---|---|---|
//! | `DYN_TOKENIZER_CACHE=1` + `DYN_TOKENIZER_CACHE_POLICY=v2` | [`CachedTokenizerV2`]: host-DRAM cache, policy v2 | B, Bf |
//! | `DYN_TOKENIZER_ENCODER=remote` + `DYN_TOKENIZER_REMOTE=host:port` | [`RemoteTokenizer`]: TCP to `tok/dyn/toksvc` | C |
//! | `DYN_TOKENIZER_ENCODER=cmm` + `DYN_TOKENIZER_CMM_LIBRARY=<.so>` | [`CmmTokenizer`]: ABI4 lanes on the CMM | D0, D (`DYN_TOKENIZER_CMM_CACHE=1`) |
//! | `DYN_TOKENIZER_PARITY=1` | [`ParityTokenizer`]: shadow uncached HF encode per request, off the timed path (C1) | diagnostic |
//! | `DYN_TOKENIZER_OFFLOAD=1` | [`encode_off_runtime`]: prompt encode on `DYN_TOKENIZER_THREADS` named threads, not a tokio worker (A9) | every mode |
//!
//! Decode always stays on the host HF tokenizer. None of this changes stock behaviour when unset.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::tokenizer_prefix_v2::{Lookup, PrefixCacheV2};
use crate::tokenizers::{
    Encoding, HuggingFaceTokenizer, TokenIdType,
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
};

pub type CacheEventFn = Arc<dyn Fn() + Send + Sync>;

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_on(name: &str) -> bool {
    matches!(std::env::var(name).ok().as_deref(), Some("1"))
}

/// Encoder backend picked by `DYN_TOKENIZER_ENCODER` (unset = host).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    Host,
    Remote,
    Cmm,
}

pub fn encoder_kind() -> Result<EncoderKind> {
    match std::env::var("DYN_TOKENIZER_ENCODER").ok().as_deref() {
        None | Some("") | Some("host") => Ok(EncoderKind::Host),
        Some("remote") => Ok(EncoderKind::Remote),
        Some("cmm") => Ok(EncoderKind::Cmm),
        Some(other) => bail!("DYN_TOKENIZER_ENCODER={other}: expected host, remote or cmm"),
    }
}

/// `DYN_TOKENIZER_CACHE_POLICY=v2` selects [`CachedTokenizerV2`] for host caches (v1 = stock).
pub fn cache_policy_v2() -> Result<bool> {
    match std::env::var("DYN_TOKENIZER_CACHE_POLICY").ok().as_deref() {
        None | Some("") | Some("v1") => Ok(false),
        Some("v2") => Ok(true),
        Some(other) => bail!("DYN_TOKENIZER_CACHE_POLICY={other}: expected v1 or v2"),
    }
}

fn ids_of(e: Encoding) -> Vec<TokenIdType> {
    match e {
        Encoding::Sp(ids) => ids,
        Encoding::Hf(inner) => inner.get_ids().to_vec(),
    }
}

/// Running per-backend telemetry, logged as one `tokstats` line every `DYN_TOKENIZER_STATS_EVERY`
/// requests (default 16). `tok/dyn/summarize.py` reads the last line of the run.
struct TokStats {
    name: &'static str,
    every: u64,
    n: AtomicU64,
    /// Summed over requests.
    fields: Vec<(&'static str, AtomicU64)>,
    /// Latest snapshot (cache size, evictions).
    last: Vec<(&'static str, AtomicU64)>,
}

impl TokStats {
    fn new(name: &'static str, fields: &[&'static str]) -> Self {
        Self::with_last(name, fields, &[])
    }
    fn with_last(name: &'static str, fields: &[&'static str], last: &[&'static str]) -> Self {
        Self {
            name,
            every: env_u64("DYN_TOKENIZER_STATS_EVERY", 16).max(1),
            n: AtomicU64::new(0),
            fields: fields.iter().map(|f| (*f, AtomicU64::new(0))).collect(),
            last: last.iter().map(|f| (*f, AtomicU64::new(0))).collect(),
        }
    }
    fn add(&self, values: &[u64]) {
        self.add_last(values, &[]);
    }
    fn add_last(&self, values: &[u64], last: &[u64]) {
        for ((_, a), v) in self.fields.iter().zip(values) {
            a.fetch_add(*v, Ordering::Relaxed);
        }
        for ((_, a), v) in self.last.iter().zip(last) {
            a.store(*v, Ordering::Relaxed);
        }
        let n = self.n.fetch_add(1, Ordering::Relaxed) + 1;
        if n % self.every == 0 {
            let body: Vec<String> = self
                .fields
                .iter()
                .chain(&self.last)
                .map(|(k, a)| format!("{k}={}", a.load(Ordering::Relaxed)))
                .collect();
            tracing::info!("tokstats {} n={n} {}", self.name, body.join(" "));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// B, Bf: host cache, policy v2
// ---------------------------------------------------------------------------------------------

/// Host-DRAM token cache with policy v2 around any inner encoder (HF or fastokens).
pub struct CachedTokenizerV2 {
    inner: Arc<dyn Tokenizer>,
    cache: PrefixCacheV2,
    on_hit: CacheEventFn,
    on_miss: CacheEventFn,
    stats: TokStats,
}

impl CachedTokenizerV2 {
    /// `policy_source` is the merged HF tokenizer the segments are encoded with (or, for
    /// fastokens, the HF instance it must agree with).
    pub fn new(
        inner: Arc<dyn Tokenizer>,
        policy_source: &tokenizers::Tokenizer,
        cache_bytes: u64,
        on_hit: CacheEventFn,
        on_miss: CacheEventFn,
    ) -> Self {
        let cache = PrefixCacheV2::new(policy_source, cache_bytes);
        match cache.disabled_reason() {
            None => tracing::info!(cache_bytes, "token cache policy v2 (host): prefix reuse on"),
            Some(reason) => tracing::warn!(
                cache_bytes,
                reason,
                "token cache policy v2 (host): prefix reuse disabled, every request is a full encode"
            ),
        }
        Self {
            inner,
            cache,
            on_hit,
            on_miss,
            stats: TokStats::new(
                "v2",
                &["hits", "misses", "reused_tokens", "encoded_tokens", "encoded_bytes"],
            ),
        }
    }
}

impl Encoder for CachedTokenizerV2 {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let (ids, out) = self
            .cache
            .encode(input, |s| self.inner.encode(s).map(ids_of))?;
        match out.lookup {
            Lookup::Hit => (self.on_hit)(),
            Lookup::Miss => (self.on_miss)(),
            Lookup::Bypass => {}
        }
        self.stats.add(&[
            (out.lookup == Lookup::Hit) as u64,
            (out.lookup == Lookup::Miss) as u64,
            out.reused_tokens as u64,
            (ids.len() - out.reused_tokens) as u64,
            out.encoded_bytes as u64,
        ]);
        Ok(Encoding::Sp(ids))
    }
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|i| self.encode(i)).collect()
    }
}

impl Decoder for CachedTokenizerV2 {
    fn decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.inner.decode(ids, skip_special_tokens)
    }
}
impl Tokenizer for CachedTokenizerV2 {}

// ---------------------------------------------------------------------------------------------
// C: x86 tokenizer service over TCP (tok/dyn/toksvc). Wire format, little-endian:
//   request  u32 magic "TKS1", u32 len, len bytes UTF-8
//   response u32 magic, i32 status, u32 n_ids, u32 reused, u32 lookup (0 bypass, 1 hit, 2 miss),
//            u32 pad, u64 service_ns, n_ids u32
// ---------------------------------------------------------------------------------------------

pub const REMOTE_MAGIC: u32 = u32::from_le_bytes(*b"TKS1");

pub struct RemoteTokenizer {
    addr: String,
    pool: Mutex<Vec<TcpStream>>,
    decoder: Arc<dyn Tokenizer>,
    on_hit: CacheEventFn,
    on_miss: CacheEventFn,
    stats: TokStats,
}

impl RemoteTokenizer {
    pub fn new(
        addr: String,
        decoder: Arc<dyn Tokenizer>,
        on_hit: CacheEventFn,
        on_miss: CacheEventFn,
    ) -> Result<Self> {
        // Fail at startup, not on the first request, when the service is not there.
        let probe = TcpStream::connect(&addr)
            .with_context(|| format!("DYN_TOKENIZER_REMOTE={addr}: tokenizer service not reachable"))?;
        probe.set_nodelay(true)?;
        tracing::info!(%addr, "remote tokenizer (mode C): connected");
        Ok(Self {
            addr,
            pool: Mutex::new(vec![probe]),
            decoder,
            on_hit,
            on_miss,
            stats: TokStats::new(
                "remote",
                &[
                    "rtt_us", "service_us", "bytes_out", "bytes_in", "hits", "misses", "reused_tokens",
                    "acct_bad",
                ],
            ),
        })
    }

    fn roundtrip(stream: &mut TcpStream, text: &str) -> std::io::Result<([u32; 8], Vec<u32>)> {
        let mut req = Vec::with_capacity(8 + text.len());
        req.extend_from_slice(&REMOTE_MAGIC.to_le_bytes());
        req.extend_from_slice(&(text.len() as u32).to_le_bytes());
        req.extend_from_slice(text.as_bytes());
        stream.write_all(&req)?;
        let mut head = [0u8; 32];
        stream.read_exact(&mut head)?;
        let mut h = [0u32; 8];
        for (i, w) in h.iter_mut().enumerate() {
            *w = u32::from_le_bytes(head[4 * i..4 * i + 4].try_into().unwrap());
        }
        if h[0] != REMOTE_MAGIC {
            return Err(std::io::Error::other("bad magic from tokenizer service"));
        }
        let mut body = vec![0u8; h[2] as usize * 4];
        stream.read_exact(&mut body)?;
        let ids = body
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok((h, ids))
    }
}

impl Encoder for RemoteTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        if input.len() > u32::MAX as usize {
            bail!("prompt too large for the tokenizer service");
        }
        let t0 = Instant::now();
        let pooled = self.pool.lock().unwrap().pop();
        let fresh = pooled.is_none();
        let mut stream = match pooled {
            Some(s) => s,
            None => {
                let s = TcpStream::connect(&self.addr)?;
                s.set_nodelay(true)?;
                s
            }
        };
        let (h, ids) = match Self::roundtrip(&mut stream, input) {
            Ok(r) => r,
            // A pooled connection may have been closed by the peer; retry once on a fresh one.
            Err(_) if !fresh => {
                stream = TcpStream::connect(&self.addr)?;
                stream.set_nodelay(true)?;
                Self::roundtrip(&mut stream, input)?
            }
            Err(e) => return Err(e.into()),
        };
        let rtt = t0.elapsed();
        let status = h[1] as i32;
        if status != 0 {
            bail!("tokenizer service returned status {status}");
        }
        self.pool.lock().unwrap().push(stream);
        match h[4] {
            1 => (self.on_hit)(),
            2 => (self.on_miss)(),
            _ => {}
        }
        let service_ns = h[6] as u64 | ((h[7] as u64) << 32);
        // C23 over the wire: a hit reused some but not all IDs; a miss or bypass reused none.
        let reused = h[3] as usize;
        let acct_ok = match h[4] {
            1 => reused > 0 && reused < ids.len(),
            0 | 2 => reused == 0,
            _ => false,
        };
        if !acct_ok {
            tracing::warn!(lookup = h[4], reused, ids = ids.len(), "remote tokenizer accounting violates C23");
        }
        self.stats.add(&[
            rtt.as_micros() as u64,
            service_ns / 1000,
            8 + input.len() as u64,
            32 + 4 * ids.len() as u64,
            (h[4] == 1) as u64,
            (h[4] == 2) as u64,
            h[3] as u64,
            !acct_ok as u64,
        ]);
        Ok(Encoding::Sp(ids))
    }
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|i| self.encode(i)).collect()
    }
}

impl Decoder for RemoteTokenizer {
    fn decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.decoder.decode(ids, skip_special_tokens)
    }
}
impl Tokenizer for RemoteTokenizer {}

// ---------------------------------------------------------------------------------------------
// D0, D: CMM ABI4 lanes through RocketKV's host lane library (k1probe/cpu/host), loaded with
// dlopen so Dynamo builds without it. One client per process (the library owns the pool).
// ---------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct TokPrefixStats {
    pub abi: u64,
    pub hits: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub encoded_tokens: u64,
    pub encoded_bytes: u64,
    pub input_bytes: u64,
    pub capacity_bytes: u64,
    pub entries: u64,
    pub memory_bytes: u64,
    pub evictions: u64,
    pub admission_drops: u64,
}

/// `k1_tok_stream_res` = `k1_tok_parallel_res` (wire/k1_tok_parallel.h, 208 B).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct StreamRes {
    pub cache: TokPrefixStats,
    pub encode_wall_ns: u64,
    pub encode_cpu_ns: u64,
    pub staging_wall_ns: u64,
    pub writeback_wall_ns: u64,
    pub encode_start_ns: u64,
    pub encode_end_ns: u64,
    pub batch_start_ns: u64,
    pub batch_end_ns: u64,
    pub total_tokens: u64,
    pub crc: u32,
    pub status: i32,
    pub abi: u32,
    pub physical_cores: u32,
    pub logical_cpus: u32,
    pub development_platform: u32,
    pub workers: u32,
    pub worker_index: u32,
    pub batch_size: u32,
    pub batch_id: u32,
}
const _: () = assert!(std::mem::size_of::<StreamRes>() == 208);
const _: () = assert!(std::mem::size_of::<TokPrefixStats>() == 96);

/// `K1TokHostTiming` (k1_tok_session_client.h, host-timing ABI 1, 120 B; the builders' layout):
/// the host side of one lane call. setup..finish sum exactly to `total_ns`; poll reads and sleeps
/// are nested inside `completion_wait_ns`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct HostTiming {
    pub abi: u64,
    pub total_ns: u64,
    pub setup_ns: u64,
    pub input_put_ns: u64,
    pub request_publish_ns: u64,
    pub completion_wait_ns: u64,
    pub result_get_ns: u64,
    pub result_validate_ns: u64,
    pub output_get_ns: u64,
    pub output_crc_ns: u64,
    pub finish_ns: u64,
    pub poll_count: u64,
    pub sleep_count: u64,
    pub poll_read_ns: u64,
    pub sleep_ns: u64,
}
const _: () = assert!(std::mem::size_of::<HostTiming>() == 120);

impl HostTiming {
    fn phases(&self) -> [u64; 9] {
        [
            self.setup_ns,
            self.input_put_ns,
            self.request_publish_ns,
            self.completion_wait_ns,
            self.result_get_ns,
            self.result_validate_ns,
            self.output_get_ns,
            self.output_crc_ns,
            self.finish_ns,
        ]
    }
    /// The builders' closure rules: exact sum, at least one poll, nested waits inside the wait phase.
    fn valid(&self) -> bool {
        let sum = self.phases().iter().try_fold(0u64, |a, &n| a.checked_add(n));
        self.abi == 1
            && sum == Some(self.total_ns)
            && self.poll_count > 0
            && self.sleep_count < self.poll_count
            && self
                .poll_read_ns
                .checked_add(self.sleep_ns)
                .is_some_and(|n| n <= self.completion_wait_ns)
    }
}

/// C23: one request's cache accounting must add up (the builders check the same per request).
/// `input` = prompt bytes, `total` = returned IDs.
fn cache_accounting_ok(cached: bool, c: &TokPrefixStats, input: u64, total: u64) -> bool {
    if !cached {
        return c.hits == 0 && c.misses == 0 && c.reused_tokens == 0;
    }
    let hit = c.hits == 1 && c.misses == 0;
    let miss = c.hits == 0 && c.misses == 1;
    let bypass = c.hits == 0 && c.misses == 0;
    c.reused_tokens + c.encoded_tokens == total
        && c.input_bytes == input
        && c.encoded_bytes <= input
        && ((hit && c.reused_tokens > 0 && c.encoded_bytes < input)
            || ((miss || bypass) && c.reused_tokens == 0 && c.encoded_bytes == input))
}

type OpenStream = unsafe extern "C" fn(usize, usize, usize) -> *mut libc::c_void;
type StreamStart = unsafe extern "C" fn(*mut libc::c_void, u32) -> i32;
type StreamEncode = unsafe extern "C" fn(
    *mut libc::c_void,
    u32,
    *const u8,
    usize,
    u32,
    *mut u32,
    usize,
    *mut StreamRes,
    *mut HostTiming,
) -> i32;

const LANES: usize = 16;

struct CmmClient {
    encode: StreamEncode,
    client: *mut libc::c_void,
    free: Mutex<Vec<u32>>,
    ready: Condvar,
    bufs: Vec<Mutex<Vec<u32>>>,
    max_text: usize,
    max_ids: usize,
}
// The library allows one caller per lane concurrently; lanes are handed out under `free`.
unsafe impl Send for CmmClient {}
unsafe impl Sync for CmmClient {}

fn cmm_client() -> Result<&'static CmmClient> {
    static CLIENT: OnceLock<std::result::Result<CmmClient, String>> = OnceLock::new();
    CLIENT
        .get_or_init(|| open_cmm().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("CMM tokenizer backend: {e}"))
}

fn open_cmm() -> Result<CmmClient> {
    let path = std::env::var("DYN_TOKENIZER_CMM_LIBRARY")
        .context("DYN_TOKENIZER_ENCODER=cmm needs DYN_TOKENIZER_CMM_LIBRARY")?;
    let max_text = env_u64("DYN_TOKENIZER_CMM_MAX_TEXT", 1 << 20) as usize;
    let max_ids = env_u64("DYN_TOKENIZER_CMM_MAX_IDS", 1 << 18) as usize;
    // The stream ABI needs an arena even for D0 (uncached lanes never touch it).
    let cache_bytes = env_u64("DYN_TOKENIZER_CMM_CACHE_BYTES", 64 << 20) as usize;
    let dev = env_u64("DYN_TOKENIZER_CMM_DEV", 0) as u32; // 1 = loopback / development platform
    let cpath = std::ffi::CString::new(path.clone())?;
    unsafe {
        let h = libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if h.is_null() {
            let err = std::ffi::CStr::from_ptr(libc::dlerror()).to_string_lossy().into_owned();
            bail!("dlopen {path}: {err}");
        }
        let sym = |name: &str| -> Result<*mut libc::c_void> {
            let c = std::ffi::CString::new(name).unwrap();
            let p = libc::dlsym(h, c.as_ptr());
            if p.is_null() {
                bail!("{path} does not export {name}");
            }
            Ok(p)
        };
        let open: OpenStream = std::mem::transmute(sym("k1ts_open_stream")?);
        let start: StreamStart = std::mem::transmute(sym("k1ts_stream_start")?);
        let encode: StreamEncode = std::mem::transmute(sym("k1ts_stream_encode_timed")?);
        let client = open(max_text, max_ids, cache_bytes);
        if client.is_null() {
            bail!(
                "k1ts_open_stream(max_text {max_text}, max_ids {max_ids}, cache {cache_bytes}) failed: \
                 pool/transport not available (is the daemon running? K1_LOOPBACK for the loopback)"
            );
        }
        let st = start(client, dev);
        if st != 0 {
            bail!("k1ts_stream_start(flags {dev}) = {st} (loopback needs DYN_TOKENIZER_CMM_DEV=1)");
        }
        tracing::info!(
            %path, max_text, max_ids, cache_bytes, dev,
            "CMM tokenizer: 16 ABI4 lanes started"
        );
        Ok(CmmClient {
            encode,
            client,
            free: Mutex::new((0..LANES as u32).rev().collect()),
            ready: Condvar::new(),
            bufs: (0..LANES).map(|_| Mutex::new(vec![0u32; max_ids])).collect(),
            max_text,
            max_ids,
        })
    }
}

pub struct CmmTokenizer {
    client: &'static CmmClient,
    cached: u32,
    decoder: Arc<dyn Tokenizer>,
    on_hit: CacheEventFn,
    on_miss: CacheEventFn,
    stats: TokStats,
}

impl CmmTokenizer {
    pub fn new(
        cached: bool,
        decoder: Arc<dyn Tokenizer>,
        on_hit: CacheEventFn,
        on_miss: CacheEventFn,
    ) -> Result<Self> {
        Ok(Self {
            client: cmm_client()?,
            cached: cached as u32,
            decoder,
            on_hit,
            on_miss,
            // T3/T4: lane wait (queueing for a free lane), host round trip, ARM service parts, bytes;
            // the host round trip by phase (HostTiming, ns); C23/timing violations (must stay 0).
            stats: TokStats::with_last(
                if cached { "cmm-cached" } else { "cmm" },
                &[
                    "lane_wait_us", "rtt_us", "arm_service_us", "arm_encode_us", "arm_encode_cpu_us",
                    "staging_us", "writeback_us", "bytes_in", "bytes_out", "hits", "misses",
                    "reused_tokens", "h_total_ns", "h_setup_ns", "h_input_put_ns", "h_publish_ns",
                    "h_wait_ns", "h_result_get_ns", "h_validate_ns", "h_output_get_ns", "h_crc_ns",
                    "h_finish_ns", "h_polls", "h_sleeps", "h_poll_read_ns", "h_sleep_ns",
                    "timing_bad", "acct_bad",
                ],
                &["cache_bytes", "entries", "evictions", "drops"],
            ),
        })
    }
}

impl Encoder for CmmTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let c = self.client;
        if input.len() > c.max_text {
            bail!(
                "prompt of {} B exceeds DYN_TOKENIZER_CMM_MAX_TEXT={} (no silent host fallback)",
                input.len(),
                c.max_text
            );
        }
        let t0 = Instant::now();
        let lane = {
            let mut free = c.free.lock().unwrap();
            loop {
                if let Some(l) = free.pop() {
                    break l;
                }
                free = c.ready.wait(free).unwrap();
            }
        };
        let t1 = Instant::now();
        let mut res = StreamRes::default();
        let mut ht = HostTiming::default();
        let mut buf = c.bufs[lane as usize].lock().unwrap();
        let st = unsafe {
            (c.encode)(
                c.client,
                lane,
                input.as_ptr(),
                input.len(),
                self.cached,
                buf.as_mut_ptr(),
                c.max_ids,
                &mut res,
                &mut ht,
            )
        };
        let t2 = Instant::now();
        let ids = if st == 0 && res.status == 0 {
            Some(buf[..res.total_tokens as usize].to_vec())
        } else {
            None
        };
        drop(buf);
        c.free.lock().unwrap().push(lane);
        c.ready.notify_one();
        let Some(ids) = ids else {
            bail!(
                "CMM lane {lane}: k1ts_stream_encode_timed = {st}, ARM status {} (tokens {}, max_ids {})",
                res.status,
                res.total_tokens,
                c.max_ids
            );
        };
        if self.cached == 1 {
            if res.cache.hits > 0 {
                (self.on_hit)()
            } else if res.cache.misses > 0 {
                (self.on_miss)()
            }
        }
        let timing_ok = ht.valid();
        let acct_ok =
            cache_accounting_ok(self.cached == 1, &res.cache, input.len() as u64, ids.len() as u64);
        if !timing_ok || !acct_ok {
            tracing::warn!(lane, timing_ok, acct_ok, ?ht, cache = ?res.cache, "CMM request telemetry violates C23/T3");
        }
        let p = ht.phases();
        self.stats.add_last(
            &[
                (t1 - t0).as_micros() as u64,
                (t2 - t1).as_micros() as u64,
                (res.batch_end_ns - res.batch_start_ns) / 1000,
                res.encode_wall_ns / 1000,
                res.encode_cpu_ns / 1000,
                res.staging_wall_ns / 1000,
                res.writeback_wall_ns / 1000,
                input.len() as u64,
                4 * ids.len() as u64,
                res.cache.hits,
                res.cache.misses,
                res.cache.reused_tokens,
                ht.total_ns,
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
                p[6],
                p[7],
                p[8],
                ht.poll_count,
                ht.sleep_count,
                ht.poll_read_ns,
                ht.sleep_ns,
                !timing_ok as u64,
                !acct_ok as u64,
            ],
            &[
                res.cache.memory_bytes,
                res.cache.entries,
                res.cache.evictions,
                res.cache.admission_drops,
            ],
        );
        Ok(Encoding::Sp(ids))
    }
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|i| self.encode(i)).collect()
    }
}

impl Decoder for CmmTokenizer {
    fn decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.decoder.decode(ids, skip_special_tokens)
    }
}
impl Tokenizer for CmmTokenizer {}

// ---------------------------------------------------------------------------------------------
// C1 parity: every request's IDs vs an uncached HF encode on a separate instance, checked on one
// background thread ("dyn-parity") so the tokenize timer does not include it. Diagnostic runs only.
// ---------------------------------------------------------------------------------------------

pub struct ParityTokenizer {
    inner: Arc<dyn Tokenizer>,
    tx: Mutex<mpsc::Sender<(String, Vec<TokenIdType>)>>,
}

impl ParityTokenizer {
    pub fn new(inner: Arc<dyn Tokenizer>, reference: HuggingFaceTokenizer) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<(String, Vec<TokenIdType>)>();
        std::thread::Builder::new()
            .name("dyn-parity".into())
            .spawn(move || {
                let (mut n, mut bad) = (0u64, 0u64);
                for (text, ids) in rx {
                    n += 1;
                    match reference.encode(&text) {
                        Ok(r) if r.token_ids() == ids.as_slice() => {}
                        Ok(r) => {
                            bad += 1;
                            let at = r
                                .token_ids()
                                .iter()
                                .zip(&ids)
                                .position(|(a, b)| a != b)
                                .unwrap_or(r.token_ids().len().min(ids.len()));
                            tracing::error!(
                                "parity: MISMATCH request {n}: {} B, {} ids vs reference {}, first difference at id {at}",
                                text.len(),
                                ids.len(),
                                r.token_ids().len()
                            );
                        }
                        Err(e) => {
                            bad += 1;
                            tracing::error!("parity: reference encode failed on request {n}: {e}");
                        }
                    }
                    tracing::info!("parity: n={n} mismatches={bad}");
                }
            })?;
        tracing::warn!("parity check on (DYN_TOKENIZER_PARITY=1): diagnostic run, not a result");
        Ok(Self {
            inner,
            tx: Mutex::new(tx),
        })
    }
}

impl Encoder for ParityTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        let e = self.inner.encode(input)?;
        let _ = self
            .tx
            .lock()
            .unwrap()
            .send((input.to_owned(), e.token_ids().to_vec()));
        Ok(e)
    }
    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|i| self.encode(i)).collect()
    }
}

impl Decoder for ParityTokenizer {
    fn decode(&self, ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.inner.decode(ids, skip_special_tokens)
    }
}
impl Tokenizer for ParityTokenizer {}

// ---------------------------------------------------------------------------------------------
// A9 / M5: prompt encode off the tokio runtime, on a bounded pool of named threads ("dyn-tok")
// so their CPU is attributable (T2). The waiting task gives up its worker (block_in_place).
// ---------------------------------------------------------------------------------------------

type Job = Box<dyn FnOnce() + Send>;

pub struct OffloadPool {
    tx: Mutex<mpsc::Sender<Job>>,
}

/// `Some` when `DYN_TOKENIZER_OFFLOAD=1`; `DYN_TOKENIZER_THREADS` threads (default: the CPUs
/// this process may run on).
pub fn offload_pool() -> Option<&'static OffloadPool> {
    static POOL: OnceLock<Option<OffloadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        if !env_on("DYN_TOKENIZER_OFFLOAD") {
            return None;
        }
        let default = std::thread::available_parallelism().map_or(4, |n| n.get()) as u64;
        let n = env_u64("DYN_TOKENIZER_THREADS", default).max(1);
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..n {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("dyn-tok-{i}"))
                .spawn(move || {
                    loop {
                        let job = rx.lock().unwrap().recv();
                        match job {
                            Ok(job) => job(),
                            Err(_) => break,
                        }
                    }
                })
                .expect("spawn tokenizer pool thread");
        }
        tracing::info!(threads = n, "prompt encode off the runtime (DYN_TOKENIZER_OFFLOAD=1)");
        Some(OffloadPool { tx: Mutex::new(tx) })
    })
    .as_ref()
}

impl OffloadPool {
    pub fn encode(&self, tokenizer: &Arc<dyn Tokenizer>, text: &str) -> Result<Encoding> {
        let tok = tokenizer.clone();
        let text = text.to_owned();
        let (tx, rx) = mpsc::sync_channel(1);
        let job: Job = Box::new(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tok.encode(&text)));
            let _ = tx.send(r);
        });
        self.tx
            .lock()
            .unwrap()
            .send(job)
            .map_err(|_| anyhow::anyhow!("tokenizer pool is gone"))?;
        let wait = move || rx.recv();
        let r = match tokio::runtime::Handle::try_current() {
            Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(wait)
            }
            _ => wait(),
        };
        match r {
            Ok(Ok(result)) => result,
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            Err(_) => bail!("tokenizer pool dropped the request"),
        }
    }
}
