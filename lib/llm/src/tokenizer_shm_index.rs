// SPDX-License-Identifier: Apache-2.0
//! Token cache index in shared memory (RocketKV sc_demo.md §4.6, `dynamo-shm`): every frontend process maps one file
//! (`DYN_TOKENIZER_CACHE_SHM`, on /dev/shm = host DRAM), so a prefix one frontend encoded is read in place by the others.
//! No service, no socket: all code runs in the frontend, with loads and stores.
//!
//! The demo's subset of the pool in sc_demo.md §9, NOT that pool: no eviction, no leases, no crash reclaim. An object is written once and
//! never freed, so a reader needs no lease. When the budget is full, inserts are dropped and counted (a run sizes memory so that none is).
//!
//! Layout: header page | buckets (one u64 per slot: 0 = empty, else the object's arena offset) | arena.
//! Object: hash u64, key_len u64, n_ids u64, restart u64, ids [u32], key bytes; padded to 8. Keys are the exact prefix bytes, compared on
//! every hit: a hash collision cannot reuse ids. Publish = one Release CAS on a bucket, after the object is complete: a reader
//! sees nothing or the whole object. A writer killed mid-fill leaves unreachable arena bytes, never a visible half object.

use std::{
    collections::hash_map::DefaultHasher,
    fs::OpenOptions,
    hash::{Hash, Hasher},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

const MAGIC: u64 = u64::from_le_bytes(*b"RKTOKSH1");
const HEADER: usize = 4096;
const BUCKETS: usize = 1 << 20;
const PROBES: usize = 4096;
const OBJ_HEAD: usize = 32;

#[repr(C)]
struct Header {
    magic: AtomicU64,
    model_hash: u64,
    capacity: u64,
    bump: AtomicU64,
    entries: AtomicU64,
    drops: AtomicU64,
    duplicates: AtomicU64,
}

pub struct ShmIndex {
    base: *mut u8,
    len: usize,
    arena: *mut u8,
    capacity: u64,
}

// The mapping lives as long as the process (never unmapped while the index lives); every shared word is atomic and objects are immutable once published.
unsafe impl Send for ShmIndex {}
unsafe impl Sync for ShmIndex {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShmStats {
    pub entries: u64,
    pub used_bytes: u64,
    pub capacity_bytes: u64,
    pub drops: u64,
    pub duplicates: u64,
}

fn key_hash(key: &str) -> u64 {
    // SipHash 1-3, zero keys: Moka's hash (RandomState) with fixed keys, so the private and shared caches pay the same per byte,
    // and every process of one binary agrees.
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

impl ShmIndex {
    /// Create `path`, or join it. `capacity` = arena bytes (object heads included); `model_hash` separates tokenizers: a mismatch fails at start.
    pub fn open(path: &str, capacity: u64, model_hash: u64) -> io::Result<Self> {
        let len = HEADER + BUCKETS * 8 + capacity as usize;
        let created = OpenOptions::new().read(true).write(true).create_new(true).mode(0o600).open(path);
        let (file, creator) = match created {
            Ok(f) => (f, true),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => (OpenOptions::new().read(true).write(true).open(path)?, false),
            Err(e) => return Err(e),
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        if creator {
            file.set_len(len as u64)?;
        } else {
            while file.metadata()?.len() < len as u64 {
                if Instant::now() > deadline {
                    return Err(io::Error::other(format!("{path}: smaller than this frontend's budget (capacity {capacity})")));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        // SAFETY: a fresh shared mapping of a file at least `len` long; it is unmapped only in Drop.
        let base = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, file.as_raw_fd(), 0)
        };
        if base == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let base = base as *mut u8;
        let index = Self { base, len, arena: unsafe { base.add(HEADER + BUCKETS * 8) }, capacity };
        let h = index.header();
        if creator {
            // SAFETY: nobody reads these two until `magic` is set below.
            unsafe {
                let p = base as *mut Header;
                (&raw mut (*p).model_hash).write(model_hash);
                (&raw mut (*p).capacity).write(capacity);
            }
            h.bump.store(8, Ordering::Relaxed); // offset 0 means an empty bucket
            h.magic.store(MAGIC, Ordering::Release);
        } else {
            while h.magic.load(Ordering::Acquire) != MAGIC {
                if Instant::now() > deadline {
                    return Err(io::Error::other(format!("{path}: not a token cache, or its creator died")));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            if h.model_hash != model_hash || h.capacity != capacity {
                return Err(io::Error::other(format!(
                    "{path}: made for tokenizer {:x} with capacity {}, this frontend has {model_hash:x} and {capacity}",
                    h.model_hash, h.capacity
                )));
            }
        }
        Ok(index)
    }

    fn header(&self) -> &Header {
        // SAFETY: the mapping starts with the header page and outlives `self`.
        unsafe { &*(self.base as *const Header) }
    }

    fn bucket(&self, i: usize) -> &AtomicU64 {
        // SAFETY: i < BUCKETS, in the mapping, 8-aligned (a page boundary + 8 i).
        unsafe { &*(self.base.add(HEADER + 8 * (i & (BUCKETS - 1))) as *const AtomicU64) }
    }

    /// A published object's (hash, key, ids, restart). Offsets come only from a bucket, read with Acquire.
    fn object(&self, off: u64) -> (u64, &[u8], &[u32], usize) {
        // SAFETY: `off` was published after the object was written whole, and objects are immutable and never freed.
        unsafe {
            let p = self.arena.add(off as usize);
            let w = p as *const u64;
            let (hash, key_len, n_ids, restart) = (*w, *w.add(1) as usize, *w.add(2) as usize, *w.add(3) as usize);
            let ids = p.add(OBJ_HEAD) as *const u32;
            let key = p.add(OBJ_HEAD + 4 * n_ids);
            (hash, std::slice::from_raw_parts(key, key_len), std::slice::from_raw_parts(ids, n_ids), restart)
        }
    }

    /// The ids in place (no copy) and the restart point of exactly this prefix.
    pub fn get(&self, key: &str) -> Option<(&[u32], usize)> {
        let h = key_hash(key);
        for i in 0..PROBES {
            let off = self.bucket(h as usize + i).load(Ordering::Acquire);
            if off == 0 {
                return None;
            }
            let (oh, k, ids, restart) = self.object(off);
            if oh == h && k == key.as_bytes() {
                return Some((ids, restart));
            }
        }
        None
    }

    /// Writes the object, then publishes it. False = dropped (budget full or table full) or already there.
    pub fn insert(&self, key: &str, ids: &[u32], restart: usize) -> bool {
        let head = self.header();
        if self.get(key).is_some() {
            return false;
        }
        let size = ((OBJ_HEAD + 4 * ids.len() + key.len() + 7) & !7) as u64;
        let off = head.bump.fetch_add(size, Ordering::Relaxed);
        if off + size > self.capacity {
            head.bump.fetch_sub(size, Ordering::Relaxed);
            head.drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let h = key_hash(key);
        // SAFETY: fetch_add gave this writer [off, off + size) alone, inside the arena (checked above); 8-aligned since off and size are.
        unsafe {
            let p = self.arena.add(off as usize);
            let w = p as *mut u64;
            w.write(h);
            w.add(1).write(key.len() as u64);
            w.add(2).write(ids.len() as u64);
            w.add(3).write(restart as u64);
            std::ptr::copy_nonoverlapping(ids.as_ptr() as *const u8, p.add(OBJ_HEAD), 4 * ids.len());
            std::ptr::copy_nonoverlapping(key.as_ptr(), p.add(OBJ_HEAD + 4 * ids.len()), key.len());
        }
        for i in 0..PROBES {
            let b = self.bucket(h as usize + i);
            match b.compare_exchange(0, off, Ordering::Release, Ordering::Acquire) {
                Ok(_) => {
                    head.entries.fetch_add(1, Ordering::Relaxed);
                    return true;
                }
                Err(other) => {
                    let (oh, k, _, _) = self.object(other);
                    if oh == h && k == key.as_bytes() {
                        head.duplicates.fetch_add(1, Ordering::Relaxed); // two frontends filled one key: the first stays, these bytes stay unreachable
                        return false;
                    }
                }
            }
        }
        head.drops.fetch_add(1, Ordering::Relaxed);
        false
    }

    pub fn stats(&self) -> ShmStats {
        let h = self.header();
        ShmStats {
            entries: h.entries.load(Ordering::Relaxed),
            used_bytes: h.bump.load(Ordering::Relaxed),
            capacity_bytes: self.capacity,
            drops: h.drops.load(Ordering::Relaxed),
            duplicates: h.duplicates.load(Ordering::Relaxed),
        }
    }
}

impl Drop for ShmIndex {
    fn drop(&mut self) {
        // SAFETY: base/len are this mapping's; a slice from `get` borrows `self`, so none outlives it.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn path(name: &str) -> String {
        let dir = if std::path::Path::new("/dev/shm").is_dir() { "/dev/shm" } else { "/tmp" };
        let p = format!("{dir}/rktok-test-{}-{name}", std::process::id());
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn two_mappings_see_one_cache() {
        let p = path("two");
        let (a, b) = (ShmIndex::open(&p, 1 << 20, 7).unwrap(), ShmIndex::open(&p, 1 << 20, 7).unwrap());
        assert!(a.get("hello").is_none());
        assert!(a.insert("hello", &[1, 2, 3], 9));
        assert_eq!(b.get("hello"), Some((&[1u32, 2, 3][..], 9)));
        assert!(!b.insert("hello", &[1, 2, 3], 9), "already there");
        assert!(b.get("hell").is_none(), "exact bytes only");
        assert_eq!(a.stats().entries, 1);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn wrong_tokenizer_or_budget_fails_at_open() {
        let p = path("guard");
        let _a = ShmIndex::open(&p, 1 << 16, 7).unwrap();
        assert!(ShmIndex::open(&p, 1 << 16, 8).is_err());
        assert!(ShmIndex::open(&p, 1 << 15, 7).is_err());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn full_budget_drops_and_keeps_what_is_there() {
        let p = path("full");
        let a = ShmIndex::open(&p, 256, 7).unwrap();
        assert!(a.insert("first", &[1; 8], 0));
        assert!(!a.insert("too big", &[2; 100], 0));
        assert_eq!(a.get("first").unwrap().0, &[1; 8]);
        assert!(a.get("too big").is_none());
        let s = a.stats();
        assert_eq!(s.drops, 1);
        assert!(s.used_bytes <= s.capacity_bytes);
        std::fs::remove_file(&p).unwrap();
    }

    /// No torn reads: writers on two mappings race readers on one key set; a reader sees nothing or the whole object.
    #[test]
    fn racing_writers_and_readers() {
        let p = path("race");
        let a = Arc::new(ShmIndex::open(&p, 1 << 26, 7).unwrap());
        let b = Arc::new(ShmIndex::open(&p, 1 << 26, 7).unwrap());
        let value = |k: u32| -> Vec<u32> { (0..64 + k % 512).map(|i| k.wrapping_mul(31).wrapping_add(i)).collect() };
        let mut threads = vec![];
        for t in 0..8u32 {
            let m = if t % 2 == 0 { a.clone() } else { b.clone() };
            threads.push(std::thread::spawn(move || {
                let mut seen = 0;
                for r in 0..3 {
                    for k in 0..2000u32 {
                        let key = format!("key-{k}");
                        if t % 4 < 2 && r == 0 {
                            m.insert(&key, &value(k), k as usize);
                        }
                        if let Some((ids, restart)) = m.get(&key) {
                            assert_eq!(ids, value(k).as_slice());
                            assert_eq!(restart, k as usize);
                            seen += 1;
                        }
                    }
                }
                seen
            }));
        }
        assert!(threads.into_iter().map(|t| t.join().unwrap()).sum::<u32>() > 0);
        assert_eq!(a.stats().entries, 2000);
        std::fs::remove_file(&p).unwrap();
    }
}
