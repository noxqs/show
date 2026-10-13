//! Results, kept against the film's own clock.
//!
//! Everything inference produces is about a moment, and the moment is media
//! time — the same clock as `Frame::at` and `Cue::start`, ultimately
//! `audio::Clock::position()`. Never the wall clock: a result stamped with
//! when it was *computed* would drift from the picture it describes as soon as
//! anything paused, seeked or ran slow.
//!
//! Deliberately the same shape as `Subtitles::at` and `Subtitles::retire`, so
//! the semantics are ones the rest of the program already uses.

use std::collections::VecDeque;
use std::time::Duration;

/// Something, and when it was true.
#[derive(Clone, Debug, PartialEq)]
pub struct Stamped<T> {
    pub at: Duration,
    /// The seek generation this belongs to. Results arriving after a seek
    /// describe a moment nobody is looking at any more.
    pub generation: u64,
    pub what: T,
}

/// How long a result stays worth keeping.
///
/// Long enough to cover a jump backwards within the frame cache, short enough
/// that a film left playing does not accumulate an afternoon of boxes.
const KEEP: Duration = Duration::from_secs(10);

pub struct Timeline<T> {
    entries: VecDeque<Stamped<T>>,
}

impl<T> Default for Timeline<T> {
    fn default() -> Self {
        Timeline::new()
    }
}

impl<T> Timeline<T> {
    pub fn new() -> Timeline<T> {
        Timeline { entries: VecDeque::new() }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Add a result, keeping the queue ordered by time.
    ///
    /// Usually a push at the back, since results arrive roughly in order; the
    /// search covers the case where two workers finish out of order.
    pub fn push(&mut self, entry: Stamped<T>) {
        let at = self
            .entries
            .iter()
            .rposition(|e| e.at <= entry.at)
            .map(|i| i + 1)
            .unwrap_or(0);
        self.entries.insert(at, entry);
    }

    /// The most recent result at or before `now`, if it is recent enough.
    ///
    /// `within` is what stops a stale answer being shown as a current one: a
    /// detection from four seconds ago describes a shot that has probably
    /// ended, and drawing it over what is on screen now would be a lie.
    pub fn at(&self, now: Duration, within: Duration) -> Option<&T> {
        self.entries
            .iter()
            .rev()
            .find(|e| e.at <= now)
            .filter(|e| now.saturating_sub(e.at) <= within)
            .map(|e| &e.what)
    }

    /// Forget everything well behind the play position, and everything from
    /// before the current seek.
    pub fn retire(&mut self, now: Duration, generation: u64) {
        self.entries.retain(|e| {
            e.generation == generation && e.at + KEEP >= now && e.at <= now + KEEP
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamped(secs: f32, what: &str) -> Stamped<String> {
        Stamped {
            at: Duration::from_secs_f32(secs),
            generation: 0,
            what: what.to_string(),
        }
    }

    #[test]
    fn the_most_recent_result_at_or_before_now_is_the_answer() {
        let mut line = Timeline::new();
        line.push(stamped(1.0, "one"));
        line.push(stamped(2.0, "two"));
        line.push(stamped(3.0, "three"));

        let within = Duration::from_secs(5);
        assert_eq!(line.at(Duration::from_secs_f32(2.5), within).map(String::as_str), Some("two"));
        assert_eq!(line.at(Duration::from_secs_f32(3.0), within).map(String::as_str), Some("three"));
        // Nothing has happened yet at half a second.
        assert!(line.at(Duration::from_secs_f32(0.5), within).is_none());
    }

    #[test]
    fn a_stale_result_is_not_offered_as_a_current_one() {
        let mut line = Timeline::new();
        line.push(stamped(1.0, "old"));
        // Four seconds later, that detection describes a shot that has
        // probably ended.
        assert!(line.at(Duration::from_secs(5), Duration::from_secs(2)).is_none());
        assert!(line.at(Duration::from_secs(2), Duration::from_secs(2)).is_some());
    }

    #[test]
    fn results_arriving_out_of_order_still_read_in_order() {
        let mut line = Timeline::new();
        line.push(stamped(3.0, "three"));
        line.push(stamped(1.0, "one"));
        line.push(stamped(2.0, "two"));
        let within = Duration::from_secs(5);
        assert_eq!(line.at(Duration::from_secs_f32(2.5), within).map(String::as_str), Some("two"));
    }

    #[test]
    fn a_seek_discards_what_described_the_old_position() {
        let mut line = Timeline::new();
        line.push(stamped(1.0, "before"));
        line.push(Stamped { generation: 1, ..stamped(1.5, "after") });
        line.retire(Duration::from_secs_f32(1.5), 1);
        assert_eq!(line.len(), 1);
        assert_eq!(line.at(Duration::from_secs(2), Duration::from_secs(5)).map(String::as_str), Some("after"));
    }

    #[test]
    fn retiring_keeps_what_is_near_the_play_head_in_both_directions() {
        let mut line = Timeline::new();
        for at in [1.0, 100.0, 105.0, 200.0] {
            line.push(stamped(at, "x"));
        }
        line.retire(Duration::from_secs(102), 0);
        // The two either side survive; the distant ones do not.
        assert_eq!(line.len(), 2, "kept {} entries", line.len());
    }
}
