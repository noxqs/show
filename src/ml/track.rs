//! Following things from one look to the next.
//!
//! The detector has no memory: each time it looks it reports a fresh set of
//! boxes and cannot say that this person is the one it saw a moment ago. That
//! matters for two reasons — a box that flickers off for one cycle should not
//! vanish from the screen, and clicking a thing to follow it needs the thing
//! to have an identity to follow.
//!
//! The method is the simple half of ByteTrack: match by overlap, keep what
//! matched, age out what did not. There is no motion model. At five looks a
//! second, over shots that last seconds, prediction buys very little and costs
//! a Kalman filter's worth of state and tuning — and a wrong prediction puts a
//! box where nothing is, which is worse than no box at all.
//!
//! Needs no model and no runtime, so it is arithmetic that can be tested.

use super::vision::Found;

/// How much two boxes must overlap to be called the same thing.
///
/// Generous, because five looks a second over a moving subject is a lot of
/// movement between looks, and splitting one object into two tracks is more
/// confusing on screen than briefly merging two.
const SAME_THING: f32 = 0.3;

/// How many looks a thing may go unseen before it is forgotten.
///
/// Two, so a single missed detection does not blink the box off, but a thing
/// that has genuinely left does not linger for a noticeable time.
const PATIENCE: u32 = 2;

/// A thing being followed.
#[derive(Clone, Debug)]
pub struct Track {
    /// Stable for as long as this thing is followed.
    pub id: u32,
    pub class: usize,
    pub confidence: f32,
    pub bounds: (f32, f32, f32, f32),
    /// Consecutive looks in which it was not found.
    missed: u32,
    /// How many looks it has been seen in, ever.
    pub seen: u32,
}

impl Track {
    /// Whether this has been seen enough to be worth drawing.
    ///
    /// One frame of a false positive is common; two in a row much less so.
    /// Waiting for the second costs a fifth of a second and removes most of
    /// the flicker.
    pub fn settled(&self) -> bool {
        self.seen >= 2
    }
}

/// How much two boxes overlap, as a fraction of the area they cover together.
pub fn overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    let shared = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    if shared <= 0.0 {
        return 0.0;
    }
    let area = |r: (f32, f32, f32, f32)| ((r.2 - r.0) * (r.3 - r.1)).max(0.0);
    let together = area(a) + area(b) - shared;
    if together <= 0.0 { 0.0 } else { shared / together }
}

#[derive(Default)]
pub struct Tracker {
    tracks: Vec<Track>,
    next_id: u32,
}

impl Tracker {
    pub fn new() -> Tracker {
        Tracker { tracks: Vec::new(), next_id: 1 }
    }

    pub fn clear(&mut self) {
        self.tracks.clear();
        // Ids are not reused: a stale reference to one should fail to find it
        // rather than quietly follow something else.
    }

    /// What is currently being followed and is worth drawing.
    ///
    /// Unused until clicking a thing follows it; kept because the arithmetic
    /// it needs is here and tested.
    #[allow(dead_code)]
    pub fn settled(&self) -> impl Iterator<Item = &Track> {
        self.tracks.iter().filter(|t| t.settled())
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Take a fresh set of detections and update what is being followed.
    ///
    /// Returns the id each detection belongs to, in the order given, so a
    /// caller can carry stable identity onto whatever it drew them from.
    pub fn update(&mut self, found: &[Found]) -> Vec<u32> {
        let mut claimed = vec![false; found.len()];

        // Greedy by overlap: every track takes its best unclaimed match, best
        // pairs first. Not optimal assignment, but the difference only shows
        // where two things of the same kind are nearly on top of each other,
        // and it costs a fraction of the code.
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (t, track) in self.tracks.iter().enumerate() {
            for (f, item) in found.iter().enumerate() {
                if item.class != track.class {
                    continue;
                }
                let score = overlap(track.bounds, item.bounds);
                if score >= SAME_THING {
                    pairs.push((t, f, score));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.total_cmp(&a.2));

        let mut ids = vec![0u32; found.len()];
        let mut matched = vec![false; self.tracks.len()];
        for (t, f, _) in pairs {
            if matched[t] || claimed[f] {
                continue;
            }
            matched[t] = true;
            claimed[f] = true;
            let track = &mut self.tracks[t];
            track.bounds = found[f].bounds;
            track.confidence = found[f].confidence;
            track.missed = 0;
            track.seen = track.seen.saturating_add(1);
            ids[f] = track.id;
        }

        // Anything not matched has gone unseen this time.
        for (t, track) in self.tracks.iter_mut().enumerate() {
            if !matched[t] {
                track.missed += 1;
            }
        }
        self.tracks.retain(|t| t.missed <= PATIENCE);

        // Whatever is left over is something new.
        for (f, item) in found.iter().enumerate() {
            if claimed[f] {
                continue;
            }
            ids[f] = self.next_id;
            self.tracks.push(Track {
                id: self.next_id,
                class: item.class,
                confidence: item.confidence,
                bounds: item.bounds,
                missed: 0,
                seen: 1,
            });
            self.next_id = self.next_id.wrapping_add(1).max(1);
        }

        ids
    }

    /// Whether a track is established enough to be worth drawing.
    ///
    /// A model fires once at a shadow now and then. Waiting for a second
    /// sighting costs one inference of delay and removes the flicker, which
    /// is the trade tracking exists to make.
    pub fn is_settled(&self, id: u32) -> bool {
        self.tracks.iter().any(|t| t.id == id && t.settled())
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(class: usize, bounds: (f32, f32, f32, f32)) -> Found {
        Found { class, confidence: 0.9, bounds }
    }

    #[test]
    fn overlapping_boxes_score_and_separate_ones_do_not() {
        let a = (0.0, 0.0, 10.0, 10.0);
        assert_eq!(overlap(a, a), 1.0, "a box is entirely itself");
        assert_eq!(overlap(a, (20.0, 20.0, 30.0, 30.0)), 0.0);
        // Half overlap: shared 50, together 150.
        let half = overlap(a, (5.0, 0.0, 15.0, 10.0));
        assert!((half - 1.0 / 3.0).abs() < 0.01, "{half}");
    }

    #[test]
    fn a_thing_that_moves_keeps_its_identity() {
        let mut tracker = Tracker::new();
        tracker.update(&[found(0, (0.0, 0.0, 100.0, 100.0))]);
        tracker.update(&[found(0, (10.0, 0.0, 110.0, 100.0))]);
        let ids: Vec<u32> = tracker.settled().map(|t| t.id).collect();
        assert_eq!(ids.len(), 1, "one thing became {}", ids.len());
        tracker.update(&[found(0, (20.0, 0.0, 120.0, 100.0))]);
        assert_eq!(tracker.settled().map(|t| t.id).collect::<Vec<_>>(), ids);
    }

    #[test]
    fn a_thing_that_jumps_across_the_frame_is_a_different_thing() {
        let mut tracker = Tracker::new();
        tracker.update(&[found(0, (0.0, 0.0, 100.0, 100.0))]);
        tracker.update(&[found(0, (500.0, 500.0, 600.0, 600.0))]);
        // The first is still being waited for; the second is new.
        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn two_kinds_in_the_same_place_are_not_confused() {
        let mut tracker = Tracker::new();
        let box_ = (0.0, 0.0, 100.0, 100.0);
        tracker.update(&[found(0, box_)]);
        tracker.update(&[found(2, box_)]);
        assert_eq!(tracker.len(), 2, "a person became a car");
    }

    #[test]
    fn one_missed_look_does_not_blink_the_box_off() {
        let mut tracker = Tracker::new();
        let box_ = (0.0, 0.0, 100.0, 100.0);
        tracker.update(&[found(0, box_)]);
        tracker.update(&[found(0, box_)]);
        assert_eq!(tracker.settled().count(), 1);

        // Missed once, and once more.
        tracker.update(&[]);
        assert_eq!(tracker.settled().count(), 1, "a single miss blinked it off");
        tracker.update(&[]);
        assert_eq!(tracker.settled().count(), 1);
        // Gone for good.
        tracker.update(&[]);
        assert_eq!(tracker.settled().count(), 0, "it never left");
    }

    #[test]
    fn a_single_sighting_is_not_drawn_yet() {
        // One frame of a false positive is common; drawing it is a flicker.
        let mut tracker = Tracker::new();
        tracker.update(&[found(0, (0.0, 0.0, 10.0, 10.0))]);
        assert_eq!(tracker.settled().count(), 0);
        tracker.update(&[found(0, (0.0, 0.0, 10.0, 10.0))]);
        assert_eq!(tracker.settled().count(), 1);
    }

    #[test]
    fn ids_are_not_reused_after_a_reset() {
        let mut tracker = Tracker::new();
        tracker.update(&[found(0, (0.0, 0.0, 10.0, 10.0))]);
        let first = tracker.settled().next().map(|t| t.id);
        tracker.clear();
        tracker.update(&[found(0, (0.0, 0.0, 10.0, 10.0))]);
        let second = tracker.tracks.first().map(|t| t.id);
        assert_ne!(first, second, "a stale reference would follow the wrong thing");
    }
}
