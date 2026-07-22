//! Per-browser state registry for the shared CEF thread.
//!
//! CEF is initialized once per process, but one initialized context can host
//! **many** windowless browsers — the reference engine runs one browser per
//! output through a single `WebBrowserContext`. The CEF thread therefore keeps
//! a [`BrowserRegistry`]: each live [`crate::cef::CefBackend`] owns one entry,
//! addressed by the [`BrowserId`] handed back when its browser was created.
//!
//! The registry is generic over the browser handle type `B` so the bookkeeping
//! (id allocation, insert/remove, per-browser pointer edge derivation, and the
//! paint-gated property queue) is unit tested without constructing real CEF
//! objects — a real `cef::Browser` can only exist on an initialized CEF
//! thread.

use std::sync::Arc;

use crate::backend::{FrameSlot, PointerState};

use super::client::SharedSize;

/// Opaque handle addressing one browser in a [`BrowserRegistry`].
///
/// Allocated monotonically and never reused, so a stale command for a closed
/// browser can never alias a newer one.
pub type BrowserId = u64;

/// One live browser's thread-side state: its handle, the shared off-screen
/// size, its frame slot (read here to paint-gate property delivery), the
/// pointer state used to derive click edges, and the queued property batches.
#[derive(Debug)]
pub struct BrowserEntry<B> {
    /// The browser handle (a `cef::Browser` on the CEF thread; any stand-in in
    /// tests).
    pub browser: B,
    /// Off-screen size shared with the entry's render handler
    /// (`get_view_rect` reads it).
    pub size: Arc<SharedSize>,
    /// The frame slot this browser's render handler publishes into. The entry
    /// only *reads* it, to gate property delivery on the first paint.
    slot: FrameSlot,
    pointer: PointerState,
    last_left: bool,
    last_right: bool,
    /// Property batches waiting for the page (reference CWeb.cpp delivers the
    /// full set on the first rendered frame — the page may block its init on
    /// `applyUserProperties`; later singles are live property changes).
    pending_props: Vec<String>,
}

impl<B> BrowserEntry<B> {
    fn new(browser: B, size: Arc<SharedSize>, slot: FrameSlot) -> Self {
        Self {
            browser,
            size,
            slot,
            pointer: PointerState::default(),
            last_left: false,
            last_right: false,
            pending_props: Vec::new(),
        }
    }

    /// Store the latest pointer sample (absolute button state).
    pub fn set_pointer(&mut self, pointer: PointerState) {
        self.pointer = pointer;
    }

    /// The latest pointer sample.
    #[must_use]
    pub fn pointer(&self) -> PointerState {
        self.pointer
    }

    /// Left-button transition since the last call: `Some(pressed)` exactly
    /// once per state change, `None` while the state is unchanged. Mirrors the
    /// C++ `CWeb` input path (docs/subsystems-misc.md §3.5): buttons arrive as
    /// absolute state and the backend derives the click/release edges.
    pub fn left_edge(&mut self) -> Option<bool> {
        (self.pointer.left != self.last_left).then(|| {
            self.last_left = self.pointer.left;
            self.pointer.left
        })
    }

    /// Right-button transition since the last call; see [`Self::left_edge`].
    pub fn right_edge(&mut self) -> Option<bool> {
        (self.pointer.right != self.last_right).then(|| {
            self.last_right = self.pointer.right;
            self.pointer.right
        })
    }

    /// Queue a `__wpApplyProps` JSON batch for this browser's page.
    pub fn push_props(&mut self, json: String) {
        self.pending_props.push(json);
    }

    /// The queued property batches, released only once **this** browser has
    /// published its first paint (its own frame slot holds a frame); empty
    /// otherwise, keeping the batches queued.
    ///
    /// Executing earlier races the page's own scripts — the shim finds no
    /// `wallpaperPropertyListener` yet and silently drops the batch (the
    /// reference also delivers on the first rendered frame, CWeb.cpp). Order
    /// is preserved: the init batch first, live singles after.
    pub fn drain_props_if_painted(&mut self) -> Vec<String> {
        if self.pending_props.is_empty() || self.slot.load_full().is_none() {
            return Vec::new();
        }
        std::mem::take(&mut self.pending_props)
    }
}

/// The CEF thread's set of live browsers, keyed by [`BrowserId`].
///
/// Insertion order is preserved (a `Vec` — the population is one browser per
/// output, so linear lookup beats hashing) and ids increase monotonically.
#[derive(Debug)]
pub struct BrowserRegistry<B> {
    next_id: BrowserId,
    entries: Vec<(BrowserId, BrowserEntry<B>)>,
}

impl<B> BrowserRegistry<B> {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
        }
    }

    /// Register a freshly created browser (with the frame slot its render
    /// handler publishes into); returns its never-reused id.
    pub fn insert(&mut self, browser: B, size: Arc<SharedSize>, slot: FrameSlot) -> BrowserId {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push((id, BrowserEntry::new(browser, size, slot)));
        id
    }

    /// Remove and return the entry for `id`, or `None` if it was already
    /// removed (a stale command for a closed browser is a no-op).
    pub fn remove(&mut self, id: BrowserId) -> Option<BrowserEntry<B>> {
        let idx = self.entries.iter().position(|(i, _)| *i == id)?;
        Some(self.entries.remove(idx).1)
    }

    /// The entry for `id`, if still live.
    pub fn get_mut(&mut self, id: BrowserId) -> Option<&mut BrowserEntry<B>> {
        self.entries.iter_mut().find(|(i, _)| *i == id).map(|(_, e)| e)
    }

    /// `true` when no browser is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate every live entry in insertion order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (BrowserId, &mut BrowserEntry<B>)> {
        self.entries.iter_mut().map(|(id, e)| (*id, e))
    }

    /// Drain every entry (shutdown: close all remaining browsers).
    pub fn drain(&mut self) -> impl Iterator<Item = (BrowserId, BrowserEntry<B>)> + '_ {
        self.entries.drain(..)
    }
}

impl<B> Default for BrowserRegistry<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::{FrameBuffer, PixelFormat};

    fn size() -> Arc<SharedSize> {
        SharedSize::new(100, 100)
    }

    fn slot() -> FrameSlot {
        Arc::new(arc_swap::ArcSwapOption::empty())
    }

    /// Publish a 1x1 frame into `slot`, as a render handler's first paint would.
    fn paint(slot: &FrameSlot) {
        slot.store(Some(Arc::new(FrameBuffer {
            data: vec![0; 4],
            width: 1,
            height: 1,
            format: PixelFormat::Bgra8,
        })));
    }

    #[test]
    fn ids_are_distinct_and_monotonic() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(0, size(), slot());
        let b = reg.insert(1, size(), slot());
        let c = reg.insert(2, size(), slot());
        assert!(a < b && b < c);
    }

    #[test]
    fn ids_are_never_reused_after_removal() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(0, size(), slot());
        assert!(reg.remove(a).is_some());
        let b = reg.insert(1, size(), slot());
        assert_ne!(a, b, "a freed id must not be reallocated");
    }

    #[test]
    fn remove_targets_only_the_requested_entry() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(10, size(), slot());
        let b = reg.insert(20, size(), slot());
        let removed = reg.remove(a).expect("entry a");
        assert_eq!(removed.browser, 10);
        assert!(reg.get_mut(a).is_none());
        assert_eq!(reg.get_mut(b).map(|e| e.browser), Some(20));
        assert!(!reg.is_empty());
    }

    #[test]
    fn remove_twice_is_a_noop() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(0, size(), slot());
        assert!(reg.remove(a).is_some());
        assert!(reg.remove(a).is_none());
        assert!(reg.is_empty());
    }

    #[test]
    fn iteration_preserves_insertion_order() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(1, size(), slot());
        let b = reg.insert(2, size(), slot());
        let ids: Vec<BrowserId> = reg.iter_mut().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![a, b]);
    }

    #[test]
    fn per_entry_size_is_independent() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(0, SharedSize::new(640, 480), slot());
        let b = reg.insert(1, SharedSize::new(1920, 1080), slot());
        reg.get_mut(a).unwrap().size.set(800, 600);
        assert_eq!(reg.get_mut(a).unwrap().size.width(), 800);
        assert_eq!(reg.get_mut(b).unwrap().size.width(), 1920);
    }

    #[test]
    fn pointer_edges_fire_once_per_transition() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(0, size(), slot());
        let entry = reg.get_mut(a).unwrap();

        // No transition before any input.
        assert_eq!(entry.left_edge(), None);
        assert_eq!(entry.right_edge(), None);

        // Press left: one edge, then quiescent.
        entry.set_pointer(PointerState {
            x: 5,
            y: 6,
            left: true,
            right: false,
        });
        assert_eq!(entry.left_edge(), Some(true));
        assert_eq!(entry.left_edge(), None);
        assert_eq!(entry.right_edge(), None);

        // Release left, press right: one edge each.
        entry.set_pointer(PointerState {
            x: 5,
            y: 6,
            left: false,
            right: true,
        });
        assert_eq!(entry.left_edge(), Some(false));
        assert_eq!(entry.right_edge(), Some(true));
        assert_eq!(entry.left_edge(), None);
        assert_eq!(entry.right_edge(), None);
    }

    #[test]
    fn props_stay_queued_until_first_paint() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let s = slot();
        let a = reg.insert(0, size(), s.clone());
        let entry = reg.get_mut(a).unwrap();

        entry.push_props("{\"a\":{\"value\":1}}".to_owned());
        // No paint yet: nothing is released, the batch stays queued.
        assert!(entry.drain_props_if_painted().is_empty());
        assert!(entry.drain_props_if_painted().is_empty());

        paint(&s);
        assert_eq!(
            entry.drain_props_if_painted(),
            vec!["{\"a\":{\"value\":1}}".to_owned()]
        );
        // Delivered once: the queue is empty afterwards.
        assert!(entry.drain_props_if_painted().is_empty());
    }

    #[test]
    fn props_preserve_order_and_late_singles_flow_through() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let s = slot();
        let a = reg.insert(0, size(), s.clone());
        let entry = reg.get_mut(a).unwrap();

        entry.push_props("init".to_owned());
        entry.push_props("single-1".to_owned());
        paint(&s);
        assert_eq!(
            entry.drain_props_if_painted(),
            vec!["init".to_owned(), "single-1".to_owned()],
            "the init batch is delivered before later singles"
        );

        // A live single after the first paint is released immediately.
        entry.push_props("single-2".to_owned());
        assert_eq!(entry.drain_props_if_painted(), vec!["single-2".to_owned()]);
    }

    #[test]
    fn props_paint_gate_is_per_browser() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let slot_a = slot();
        let a = reg.insert(0, size(), slot_a.clone());
        let b = reg.insert(1, size(), slot());

        reg.get_mut(a).unwrap().push_props("for-a".to_owned());
        reg.get_mut(b).unwrap().push_props("for-b".to_owned());

        // Only browser A has painted: A's batch is released, B's stays queued.
        paint(&slot_a);
        assert_eq!(
            reg.get_mut(a).unwrap().drain_props_if_painted(),
            vec!["for-a".to_owned()]
        );
        assert!(reg.get_mut(b).unwrap().drain_props_if_painted().is_empty());
    }

    #[test]
    fn pointer_position_is_stored_verbatim() {
        let mut reg: BrowserRegistry<u8> = BrowserRegistry::new();
        let a = reg.insert(0, size(), slot());
        let entry = reg.get_mut(a).unwrap();
        entry.set_pointer(PointerState {
            x: -3,
            y: 99,
            left: false,
            right: false,
        });
        let p = entry.pointer();
        assert_eq!((p.x, p.y), (-3, 99));
    }
}
