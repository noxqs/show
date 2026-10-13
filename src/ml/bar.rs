//! The bar along the top naming what is in view.
//!
//! It is also the legend, and since masks are coloured per object rather than
//! per kind, so is this: one tag per thing, drawn in that thing's own colour.
//! "Which colour is which" therefore needs no separate key — the answer is
//! already on screen, next to the name. Both sides take the colour from
//! `labels::colour_of`, so they cannot drift apart.
//!
//! Rebuilt only when the set of things changes, which is with the shot rather
//! than with the frame, so this goes through the ordinary overlay path where
//! re-uploading the whole thing is correct and cheap.

use crate::osd::{self, Overlay};

/// One thing in view: what it is, which one of its kind it is, and the palette
/// slot its mask is drawn in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tag {
    pub class: usize,
    /// One-based, within its own kind, so two people read as PERSON 1 and
    /// PERSON 2 rather than as the same word twice in different colours.
    pub ordinal: usize,
    /// The slot, not the colour: the mask is drawn from the slot too, and one
    /// of them converting to a colour early is how the two would stop agreeing.
    pub slot: u8,
}

/// Text size, and the padding around it.
const SCALE: f32 = 2.0;
const PAD: usize = 8;

/// What the bar is saying. Kept so it is rebuilt when it would *read*
/// differently, rather than on every frame or every new result.
#[derive(Clone, PartialEq, Eq)]
pub enum Legend {
    /// Everything in view, one entry per thing.
    Tags(Vec<Tag>),
    /// One thing, chosen by clicking it.
    Following { class: usize, id: u32 },
    /// One thing, cut out by pointing at it — it has no name to give.
    /// `following` while it is being kept up with through a moving film.
    CutOut { following: bool },
    /// The picture has been looked at, and there is nothing in it the model
    /// has a name for. Said outright, because a bar still reading LOOKING
    /// after the looking has finished is indistinguishable from a fault.
    Nothing,
}

/// Draw whatever the bar is currently saying.
pub fn draw(legend: &Legend, width: u32) -> Overlay {
    match legend {
        Legend::Tags(tags) => render(tags, width),
        Legend::Following { class, id } => following(*class, *id, width),
        Legend::CutOut { following } => cut_out(width, *following),
        Legend::Nothing => nothing(width),
    }
}

/// The bar when the picture holds nothing the model knows.
pub fn nothing(width: u32) -> Overlay {
    let width = width.max(1) as usize;
    let text_h = osd::line_height_at(SCALE);
    let height = text_h + PAD * 2;
    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&osd::CHROME);
    }
    osd::draw_text_into(&mut rgba, width, height, "NOTHING IT KNOWS", PAD, PAD, osd::DIM, SCALE);

    let hint = "IT KNOWS 80 EVERYDAY THINGS";
    let w = osd::text_width_at(hint, SCALE);
    if PAD * 4 + osd::text_width_at("NOTHING IT KNOWS", SCALE) + w < width {
        osd::draw_text_into(&mut rgba, width, height, hint, width - w - PAD, PAD, osd::DIM, SCALE);
    }
    Overlay::new(rgba, width as u32, height as u32)
}

/// The bar while a cut-out is on screen.
pub fn cut_out(width: u32, following: bool) -> Overlay {
    let width = width.max(1) as usize;
    let text_h = osd::line_height_at(SCALE);
    let height = text_h + PAD * 2;
    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&osd::CHROME);
    }
    // The slot the cut-out's own mask is drawn in, not its class's colour:
    // a cut-out has no tracker id, so it is drawn from a reserved slot.
    let colour = crate::ml::labels::colour_of(crate::ml::labels::CUT_OUT);
    let label = if following { "FOLLOWING WHAT I CUT OUT" } else { "CUT OUT" };
    osd::draw_text_into(&mut rgba, width, height, label, PAD, PAD, colour, SCALE);

    let hint = "CTRL CLICK IT AGAIN TO STOP";
    let w = osd::text_width_at(hint, SCALE);
    if PAD * 4 + osd::text_width_at(label, SCALE) + w < width {
        osd::draw_text_into(&mut rgba, width, height, hint, width - w - PAD, PAD, osd::DIM, SCALE);
    }
    Overlay::new(rgba, width as u32, height as u32)
}

/// The bar while one thing is being followed.
///
/// It takes the whole bar rather than sitting beside the tally: following is
/// a mode, and a legend for things that are no longer drawn would be a lie.
pub fn following(class: usize, id: u32, width: u32) -> Overlay {
    let width = width.max(1) as usize;
    let text_h = osd::line_height_at(SCALE);
    let height = text_h + PAD * 2;

    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&osd::CHROME);
    }

    let name = crate::ml::vision::CLASSES.get(class).copied().unwrap_or("thing");
    // The number is the point: two people in the shot are told apart by it,
    // and it is the same number for as long as the thing is followed.
    let label = format!("FOLLOWING {} #{id}", name.to_uppercase());
    osd::draw_text_into(&mut rgba, width, height, &label, PAD, PAD, crate::ml::colour(class), SCALE);

    let hint = "CLICK AGAIN TO STOP";
    let w = osd::text_width_at(hint, SCALE);
    if PAD + osd::text_width_at(&label, SCALE) + PAD * 3 + w < width {
        osd::draw_text_into(&mut rgba, width, height, hint, width - w - PAD, PAD, osd::DIM, SCALE);
    }
    Overlay::new(rgba, width as u32, height as u32)
}

/// Draw the tags across a bar `width` pixels wide.
pub fn render(tags: &[Tag], width: u32) -> Overlay {
    let width = width.max(1) as usize;
    let text_h = osd::line_height_at(SCALE);
    let height = text_h + PAD * 2;

    let mut rgba = Vec::with_capacity(width * height * 4);
    for _ in 0..width * height {
        rgba.extend_from_slice(&osd::CHROME);
    }

    if tags.is_empty() {
        // Said rather than left blank: an empty bar with nothing in it reads
        // as broken, and "nothing yet" is a real and temporary answer.
        osd::draw_text_into(&mut rgba, width, height, "LOOKING...", PAD, PAD, osd::TEXT, SCALE);
        return Overlay::new(rgba, width as u32, height as u32);
    }

    let mut x = PAD;
    for tag in tags {
        let name = crate::ml::vision::CLASSES.get(tag.class).copied().unwrap_or("thing");
        // The number is only worth saying when there is another of its kind to
        // tell it from. One dog on screen is THE dog.
        let alone = tags.iter().filter(|t| t.class == tag.class).count() == 1;
        let label = if alone {
            name.to_uppercase()
        } else {
            format!("{} {}", name.to_uppercase(), tag.ordinal)
        };
        let w = osd::text_width_at(&label, SCALE);
        // Stop rather than overflow: a tag half off the edge is worse than one
        // that is not shown.
        if x + w + PAD > width {
            break;
        }
        osd::draw_text_into(
            &mut rgba,
            width,
            height,
            &label,
            x,
            PAD,
            crate::ml::labels::colour_of(tag.slot),
            SCALE,
        );
        x += w + PAD * 2;
    }
    Overlay::new(rgba, width as u32, height as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(class: usize, ordinal: usize, slot: u8) -> Tag {
        Tag { class, ordinal, slot }
    }

    #[test]
    fn the_bar_spans_the_width_it_is_given() {
        let bar = render(&[tag(0, 1, 1)], 1920);
        assert_eq!(bar.width, 1920);
        assert!(bar.height > 0);
    }

    #[test]
    fn an_empty_bar_says_so_rather_than_looking_broken() {
        let empty: Overlay = render(&[], 800);
        let blank = {
            // A bar of nothing but chrome, for comparison.
            let mut rgba = Vec::new();
            for _ in 0..(800 * empty.height as usize) {
                rgba.extend_from_slice(&osd::CHROME);
            }
            rgba
        };
        assert_ne!(empty.rgba, blank, "an empty bar drew nothing at all");
    }

    #[test]
    fn two_of_a_kind_are_drawn_in_different_colours() {
        // The point of the whole change: two people are two colours, and the
        // bar says which is which. Same class, different slots.
        let a = tag(0, 1, crate::ml::labels::slot(1));
        let b = tag(0, 2, crate::ml::labels::slot(2));
        assert_ne!(
            crate::ml::labels::colour_of(a.slot),
            crate::ml::labels::colour_of(b.slot),
            "two people came out the same colour"
        );
    }

    #[test]
    fn a_tag_is_the_colour_its_mask_is() {
        // The bar is only a legend while both sides agree, so both read the
        // colour from the same place rather than each computing one.
        for id in [1u32, 2, 7, 14, 15, 40] {
            let slot = crate::ml::labels::slot(id);
            assert_eq!(crate::ml::labels::colour_of(slot), crate::ml::colour(slot as usize - 1));
        }
    }

    #[test]
    fn a_narrow_bar_stops_rather_than_overflowing() {
        // Far more tags than fit. The last must not be half off the edge.
        let many: Vec<Tag> = (0..40).map(|c| tag(c, 1, 1)).collect();
        let bar = render(&many, 300);
        assert_eq!(bar.width, 300);
    }
}
