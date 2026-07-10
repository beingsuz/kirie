//! PulseAudio capture thread: records the default sink's monitor (or an
//! explicit `--audio-device` source) as U8 / 44100 / mono and pushes raw bytes
//! into the SPSC ring for the FFT worker.
//!
//! Everything here is `!Send` (PulseAudio `Mainloop`/`Context`/`Stream` wrap Rc
//! internally), so the whole pipeline is constructed and driven *inside* the
//! capture thread; only the ring producer (Send) crosses the thread boundary.
//! No `unsafe` — libpulse-binding provides safe closure callbacks (V2).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Duration;

use libpulse_binding::context::{Context, FlagSet as ContextFlags, State as ContextState};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
use libpulse_binding::proplist::Proplist;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::{FlagSet as StreamFlags, PeekResult, State as StreamState, Stream};
use ringbuf::HeapProd;
use ringbuf::traits::Producer;

use crate::dsp::SAMPLE_RATE;
use crate::{AudioError, CaptureStatus};

/// Context name (cpp:197).
const CONTEXT_NAME: &str = "wallpaperengine-audioprocessing";
/// Record stream name (cpp:115).
const STREAM_NAME: &str = "output monitor";

/// Drive one `iterate` step, mapping quit/err to a typed error.
fn pump(mainloop: &mut Mainloop) -> Result<(), AudioError> {
    match mainloop.iterate(false) {
        IterateResult::Success(_) => Ok(()),
        IterateResult::Quit(_) | IterateResult::Err(_) => Err(AudioError::Mainloop),
    }
}

/// Blocking-poll wrapper: iterate until `check` returns `Some`, an error, or the
/// shutdown flag trips. A short sleep between non-blocking iterations keeps the
/// thread from busy-spinning while waiting for the server to answer.
fn iterate_until<T>(
    mainloop: &mut Mainloop,
    shutdown: &AtomicBool,
    mut check: impl FnMut() -> Option<Result<T, AudioError>>,
) -> Result<T, AudioError> {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return Err(AudioError::Mainloop);
        }
        pump(mainloop)?;
        if let Some(res) = check() {
            return res;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Connect the context, resolve the source, open the record stream and pump the
/// mainloop until shutdown. Errors here leave the spectrum silent (caller logs).
pub(crate) fn run(
    device: Option<String>,
    producer: HeapProd<u8>,
    status: &Arc<AtomicU8>,
    shutdown: &Arc<AtomicBool>,
) -> Result<(), AudioError> {
    let mut mainloop = Mainloop::new().ok_or_else(|| AudioError::Connect("no mainloop".into()))?;

    let proplist = Proplist::new().ok_or_else(|| AudioError::Connect("no proplist".into()))?;
    let mut context = Context::new_with_proplist(&mainloop, CONTEXT_NAME, &proplist)
        .ok_or_else(|| AudioError::Connect("no context".into()))?;
    context
        .connect(None, ContextFlags::NOFLAGS, None)
        .map_err(|e| AudioError::Connect(format!("{e:?}")))?;

    // Wait for PA_CONTEXT_READY (cpp:206-209).
    iterate_until(&mut mainloop, shutdown, || match context.get_state() {
        ContextState::Ready => Some(Ok(())),
        ContextState::Failed | ContextState::Terminated => {
            Some(Err(AudioError::Connect("context failed".into())))
        }
        _ => None,
    })?;

    // Resolve the capture source: explicit device, else "<default_sink>.monitor"
    // (cpp:121-128).
    let source = match device.filter(|d| !d.is_empty()) {
        Some(dev) => dev,
        None => {
            let default_sink: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
            {
                let slot = default_sink.clone();
                context.introspect().get_server_info(move |info| {
                    *slot.borrow_mut() = Some(
                        info.default_sink_name
                            .as_ref()
                            .map(|s| s.to_string())
                            .unwrap_or_default(),
                    );
                });
            }
            let sink = iterate_until(&mut mainloop, shutdown, || default_sink.borrow().clone().map(Ok))?;
            if sink.is_empty() {
                return Err(AudioError::NoMonitor);
            }
            format!("{sink}.monitor")
        }
    };

    // Record stream: U8 / 44100 / mono (cpp:107-110).
    let spec = Spec {
        format: Format::U8,
        channels: 1,
        rate: SAMPLE_RATE,
    };
    if !spec.is_valid() {
        return Err(AudioError::StreamConnect {
            source_name: source,
            reason: "invalid sample spec".into(),
        });
    }

    let stream = Rc::new(RefCell::new(
        Stream::new(&mut context, STREAM_NAME, &spec, None).ok_or_else(|| AudioError::StreamConnect {
            source_name: source.clone(),
            reason: "stream alloc failed".into(),
        })?,
    ));

    // Read callback: drain every peeked fragment into the ring; drop holes
    // (cpp:31-98). Bytes not fitting the ring are dropped (overflow tolerated).
    let producer = Rc::new(RefCell::new(producer));
    {
        let stream_cb = stream.clone();
        let producer_cb = producer.clone();
        stream
            .borrow_mut()
            .set_read_callback(Some(Box::new(move |_nbytes| {
                let mut s = stream_cb.borrow_mut();
                loop {
                    match s.peek() {
                        Ok(PeekResult::Data(data)) => {
                            producer_cb.borrow_mut().push_slice(data);
                            let _ = s.discard();
                        }
                        Ok(PeekResult::Hole(_)) => {
                            let _ = s.discard();
                        }
                        Ok(PeekResult::Empty) => break,
                        Err(_) => break,
                    }
                }
            })));
    }

    // Buffer attrs (cpp:130-137): U8/mono → bytes_per_sec == rate.
    let bytes_per_sec = SAMPLE_RATE;
    let fragsize = bytes_per_sec * 10 / 1000;
    let maxlength = fragsize + bytes_per_sec * 750 / 1000;
    let attr = BufferAttr {
        maxlength,
        tlength: u32::MAX,
        prebuf: u32::MAX,
        minreq: u32::MAX,
        fragsize,
    };

    stream
        .borrow_mut()
        .connect_record(Some(&source), Some(&attr), StreamFlags::ADJUST_LATENCY)
        .map_err(|e| AudioError::StreamConnect {
            source_name: source.clone(),
            reason: format!("{e:?}"),
        })?;

    // Wait for the stream to reach Ready before declaring success.
    iterate_until(&mut mainloop, shutdown, || match stream.borrow().get_state() {
        StreamState::Ready => Some(Ok(())),
        StreamState::Failed | StreamState::Terminated => Some(Err(AudioError::StreamConnect {
            source_name: source.clone(),
            reason: "stream failed".into(),
        })),
        _ => None,
    })?;

    status.store(CaptureStatus::Running.as_u8(), Ordering::Relaxed);
    tracing::info!(source = %source, "audio capture running");

    // Drive the mainloop; the read callback fires as fragments arrive.
    while !shutdown.load(Ordering::Relaxed) {
        pump(&mut mainloop)?;
        std::thread::sleep(Duration::from_millis(5));
    }

    // Clear the callback before teardown so the closure (holding an Rc to the
    // stream) is dropped, breaking the reference cycle.
    stream.borrow_mut().set_read_callback(None);
    let _ = stream.borrow_mut().disconnect();
    Ok(())
}
