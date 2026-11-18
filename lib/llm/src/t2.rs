// SPDX-FileCopyrightText: Copyright (c) 2026 RocketKV. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RocketKV T2 (sc_demo.md): per-request timers inside the frontend, on with `DYN_T2=1`.
//!
//! Each stage adds its milliseconds under the request id (the HTTP context id = the client's
//! `chatcmpl` id); the HTTP handler emits one `t2` line once the worker stream is open:
//! `t2 rid=<id> t0_ms=<unix ms at handler entry> http=.. template=.. tokenize=.. mm=.. mm_route=..
//! route=.. forward=.. pipe=.. img_kb=.. msg_kb=..` (ms, 3 decimals; `pipe` = handler entry -> stream
//! open, so `pipe - sum(parts)` is what no timer covers; `img_kb` / `msg_kb` = KiB of inline `data:`
//! URLs sent to the worker in `multi_modal_data` / in `extra_args.messages`, B1). Off: nothing stored.

use std::fmt::Write as _;
use std::sync::LazyLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

pub const PARTS: [&str; 7] = [
    "http", "template", "tokenize", "mm", "mm_route", "route", "forward",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Http = 0,
    Template = 1,
    Tokenize = 2,
    Mm = 3,
    MmRoute = 4,
    Route = 5,
    Forward = 6,
}

#[derive(Debug, Default, Clone)]
pub struct Rec {
    pub ms: [f64; 7],
    pub t0_ms: f64,
    pub start: Option<Instant>,
    /// bytes of inline data: URLs in multi_modal_data, in extra_args.messages
    pub img_bytes: usize,
    pub msg_bytes: usize,
}

static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("DYN_T2").is_ok_and(|v| v == "1"));
static RECS: LazyLock<DashMap<String, Rec>> = LazyLock::new(DashMap::new);

tokio::task_local! {
    /// Request id for code that has no context in scope (the preprocessor's inner steps).
    pub static RID: String;
}

#[inline]
pub fn enabled() -> bool {
    *ON
}

/// Start a request's record: `t0` = handler entry (before the body is read), `http` = body read + JSON parse.
pub fn begin(rid: &str, start: Instant, t0_unix_ms: f64, http: f64) {
    if !enabled() {
        return;
    }
    let mut r = Rec {
        t0_ms: t0_unix_ms,
        start: Some(start),
        ..Default::default()
    };
    r.ms[Part::Http as usize] = http;
    RECS.insert(rid.to_string(), r);
}

/// Add `ms` to `part` of `rid` (creates the record when the handler did not, e.g. a non-HTTP entry).
pub fn add(rid: &str, part: Part, ms: f64) {
    if !enabled() {
        return;
    }
    RECS.entry(rid.to_string()).or_default().ms[part as usize] += ms;
}

/// Add under the task-local request id (no-op outside a `RID.scope`).
pub fn add_current(part: Part, since: Instant) {
    if !enabled() {
        return;
    }
    let ms = since.elapsed().as_secs_f64() * 1e3;
    let _ = RID.try_with(|rid| add(rid, part, ms));
}

/// Record the request's inline data-URL bytes under the task-local request id.
pub fn set_bytes_current(img_bytes: usize, msg_bytes: usize) {
    if !enabled() {
        return;
    }
    let _ = RID.try_with(|rid| {
        let mut r = RECS.entry(rid.to_string()).or_default();
        r.img_bytes = img_bytes;
        r.msg_bytes = msg_bytes;
    });
}

/// Add the time since `since` to `part` of `rid`.
pub fn add_since(rid: &str, part: Part, since: Instant) {
    add(rid, part, since.elapsed().as_secs_f64() * 1e3);
}

/// Remove the record and format its line (`pipe` from `begin`'s start, 0 without one); None when off or unknown.
pub fn take_line(rid: &str) -> Option<String> {
    if !enabled() {
        return None;
    }
    let (_, r) = RECS.remove(rid)?;
    let pipe = r.start.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1e3);
    Some(format_line(rid, &r, pipe))
}

pub fn format_line(rid: &str, r: &Rec, pipe_ms: f64) -> String {
    let mut s = format!("t2 rid={rid} t0_ms={:.3}", r.t0_ms);
    for (k, v) in PARTS.iter().zip(r.ms.iter()) {
        let _ = write!(s, " {k}={v:.3}");
    }
    let _ = write!(
        s,
        " pipe={pipe_ms:.3} img_kb={:.1} msg_kb={:.1}",
        r.img_bytes as f64 / 1024.0,
        r.msg_bytes as f64 / 1024.0
    );
    s
}

/// Emit the line for `rid` at INFO, target `t2`.
pub fn emit(rid: &str) {
    if let Some(line) = take_line(rid) {
        tracing::info!(target: "t2", "{line}");
    }
}

/// Drop a record without emitting (request failed before the stream opened).
pub fn drop_rec(rid: &str) {
    if enabled() {
        RECS.remove(rid);
    }
}

pub fn unix_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1e3)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t2_line_format() {
        let mut r = Rec {
            t0_ms: 1763500000123.5,
            ..Default::default()
        };
        r.ms[Part::Http as usize] = 12.0;
        r.ms[Part::Mm as usize] = 3.25;
        r.ms[Part::Forward as usize] = 140.0004;
        r.img_bytes = 2048;
        let l = format_line("abc-1", &r, 200.0);
        assert_eq!(
            l,
            "t2 rid=abc-1 t0_ms=1763500000123.500 http=12.000 template=0.000 tokenize=0.000 \
             mm=3.250 mm_route=0.000 route=0.000 forward=140.000 pipe=200.000 img_kb=2.0 msg_kb=0.0"
        );
        // every part present, in PARTS order, as key=value
        let keys: Vec<&str> = l
            .split(' ')
            .skip(3)
            .map(|kv| kv.split('=').next().unwrap())
            .collect();
        assert_eq!(&keys[..7], &PARTS);
        assert_eq!(keys[7], "pipe");
    }

    #[test]
    fn t2_add_accumulates_and_take_removes() {
        // ON is read once from the env; force it on for this test binary.
        unsafe { std::env::set_var("DYN_T2", "1") };
        if !enabled() {
            return; // another test read ON first with DYN_T2 unset
        }
        begin("rid-acc", Instant::now(), 1.0, 2.0);
        add("rid-acc", Part::Route, 1.5);
        add("rid-acc", Part::Route, 0.5);
        let l = take_line("rid-acc").unwrap();
        assert!(l.contains(" http=2.000 "), "{l}");
        assert!(l.contains(" route=2.000 "), "{l}");
        assert!(l.contains(" pipe="), "{l}");
        assert!(take_line("rid-acc").is_none());
    }
}
