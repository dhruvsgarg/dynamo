// SPDX-License-Identifier: Apache-2.0
//! RocketKV T8 (`tok_dynamo.md` §4.6): what one request costs on the request plane, frontend -> worker. With
//! `DYN_REQPLANE_STATS=1` the frontend stamps each request envelope with its wall-clock send time (Dynamo's own
//! `frontend_send_ts_ns` field, unset in stock v1.3.1) and both sides log running sums every 16 requests:
//!   `reqplane out n= bytes=`             frontend: envelope bytes sent (control header + serialized request)
//!   `reqplane in n= bytes= timed= transit_us=`   worker: envelope bytes received; send -> receive time (one host)
//! Unset, nothing changes on the wire or the path.
use std::sync::{Mutex, OnceLock};

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DYN_REQPLANE_STATS").is_ok_and(|v| v == "1"))
}

pub(crate) fn now_ns() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
}

/// n, bytes, timed, transit_ns: one lock per request, so a logged line never mixes half a request.
static OUT: Mutex<[u64; 2]> = Mutex::new([0; 2]);
static IN: Mutex<[u64; 4]> = Mutex::new([0; 4]);

pub(crate) fn sent(bytes: usize) {
    let mut s = OUT.lock().unwrap_or_else(|e| e.into_inner());
    s[0] += 1;
    s[1] += bytes as u64;
    if s[0] % 16 == 0 {
        tracing::info!("reqplane out n={} bytes={}", s[0], s[1]);
    }
}

pub(crate) fn received(bytes: usize, transit_ns: Option<u64>) {
    let mut s = IN.lock().unwrap_or_else(|e| e.into_inner());
    s[0] += 1;
    s[1] += bytes as u64;
    if let Some(t) = transit_ns {
        s[2] += 1;
        s[3] += t;
    }
    if s[0] % 16 == 0 {
        tracing::info!("reqplane in n={} bytes={} timed={} transit_us={}", s[0], s[1], s[2], s[3] / 1000);
    }
}
