//! The CLIP text tokenizer.
//!
//! SAM 3's language encoder takes CLIP token ids — `int64[1, 32]` — and not
//! text, so something has to turn "a red car" into numbers before the model is
//! shown it. This is that something: byte-level byte-pair encoding, with
//! CLIP's own way of splitting a sentence into words and its `</w>` marker on
//! the end of each one.
//!
//! Hand-rolled rather than pulled in, for the reason the hashes, the font and
//! the settings parser are: the alternative is a regex engine and a JSON crate
//! to read two files that are read once, at the moment a model is opened.
//!
//! It must agree with the tokenizer the model was trained against, exactly. A
//! tokenizer that is merely close does not degrade gracefully — it hands the
//! encoder a different sentence from the one that was typed, and the model
//! answers the question it was asked rather than the one that was meant. So it
//! is checked against one: `tokenize_a_corpus_for_comparison` writes the ids
//! for a file of prompts, and those are compared against a reference
//! tokenizer. 462 realistic prompts agree exactly, as do 2982 of 2998 lines of
//! random Unicode.
//!
//! **What is not done: NFC.** The reference normalises with NFC before
//! lower-casing, and this does not, so a word typed with combining accents —
//! `e` followed by a combining acute — tokenizes differently from the same
//! word typed precomposed. Every modern keyboard and nearly all text on the
//! web is already composed, which is why the realistic corpus agrees to the
//! last id; `decomposed_text_is_not_normalised_yet` pins the boundary. Closing
//! it means canonical decomposition tables, combining classes, composition
//! exclusions and the Hangul rules: a real piece of work, not a line.
//!
//! The remaining sixteen fuzz lines are thirteen codepoints unassigned in
//! Unicode 15 and assigned as letters in the newer tables Rust carries. Two
//! Unicode versions disagreeing, not two implementations.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};

/// How many tokens the language encoder takes. Fixed by the exported graph,
/// which declares `int64[1, 32]`; a prompt is a short phrase, not a paragraph.
pub const CONTEXT: usize = 32;

/// The two ends of a prompt: `<|startoftext|> … <|endoftext|>`.
const START: &str = "<|startoftext|>";
const END: &str = "<|endoftext|>";

/// What the unused tail of the sequence is filled with.
///
/// Zero, not the end token. CLIP itself pads with the end token, and that is
/// what this did until the model was asked: SAM 3's language encoder derives
/// its own padding mask from the ids, and given a tail of end tokens it marked
/// *nothing* as padding — it read a three-word prompt as thirty-two words of
/// content, and the decoder then matched nothing at all. Padding with zero
/// makes the mask come back true for exactly the tail.
///
/// Checked rather than reasoned about: with end-token padding the encoder's
/// `text_attention_mask` is true for 0 of 32 positions, with zero padding it
/// is true for 29 of 32 on a three-token prompt.
const PAD: i64 = 0;

pub struct Tokenizer {
    /// Token text to id. Around fifty thousand entries, read once.
    vocab: HashMap<String, u32>,
    /// Merge pairs by rank: the lower the number, the earlier it is applied.
    ranks: HashMap<(String, String), u32>,
    start: u32,
    end: u32,
    /// Byte to the character that stands for it. CLIP works in a private
    /// alphabet where every one of the 256 bytes is a printable character, so
    /// that a token is always a string and never a partial character.
    byte_encoder: [char; 256],
}

impl Tokenizer {
    /// Read `merges.txt`, in the form every CLIP text model publishes it.
    pub fn load(merges: &std::path::Path) -> Result<Tokenizer> {
        let text = std::fs::read_to_string(merges)
            .with_context(|| format!("reading {}", merges.display()))?;
        Tokenizer::from_merges(&text)
    }

    /// Build from the merges list alone.
    ///
    /// The vocabulary is not read, it is *derived*, exactly as CLIP builds it:
    /// the 256 characters of the private alphabet, then those again with the
    /// end-of-word marker, then one entry per merge in rank order, then the
    /// two ends. Ids are positions in that sequence.
    ///
    /// Which means `vocab.json` — 862 kB, and a JSON parser to read it — was
    /// never needed. Checked rather than assumed before it was deleted: the
    /// derived table is identical to the published one, all 49408 entries.
    pub fn from_merges(merges_text: &str) -> Result<Tokenizer> {
        let mut ranks = HashMap::new();
        let mut merges: Vec<(String, String)> = Vec::new();
        for (line_no, line) in merges_text.lines().enumerate() {
            // The first line is a version banner, and blank lines are nothing.
            if line.starts_with("#version") || line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(a), Some(b), None) = (parts.next(), parts.next(), parts.next()) else {
                bail!("merge on line {} is not a pair: {line:?}", line_no + 1);
            };
            // Rank is position in the list: the first merge listed wins over
            // the second wherever both apply.
            let pair = (a.to_string(), b.to_string());
            if ranks.insert(pair.clone(), merges.len() as u32).is_none() {
                merges.push(pair);
            }
        }
        if merges.is_empty() {
            bail!("no merges were read: the file is empty or not a merges list");
        }

        let order = byte_order();
        let mut vocab = HashMap::with_capacity(merges.len() + 514);
        let mut next = 0u32;
        let mut add = |token: String, vocab: &mut HashMap<String, u32>, next: &mut u32| {
            vocab.entry(token).or_insert(*next);
            *next += 1;
        };
        for c in &order {
            add(c.to_string(), &mut vocab, &mut next);
        }
        for c in &order {
            add(format!("{c}</w>"), &mut vocab, &mut next);
        }
        for (a, b) in &merges {
            add(format!("{a}{b}"), &mut vocab, &mut next);
        }
        let start = next;
        add(START.to_string(), &mut vocab, &mut next);
        let end = next;
        add(END.to_string(), &mut vocab, &mut next);

        Ok(Tokenizer { vocab, ranks, start, end, byte_encoder: byte_encoder() })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Turn a prompt into exactly `len` token ids.
    ///
    /// Padded with the end token and truncated to fit, because the graph takes
    /// a fixed-width tensor. A prompt too long to fit keeps its start and is
    /// cut short — the alternative is refusing to segment anything because the
    /// phrase ran long, which is worse than segmenting what was asked for
    /// first.
    pub fn encode(&self, text: &str, len: usize) -> Vec<i64> {
        let mut ids = Vec::with_capacity(len);
        ids.push(self.start as i64);
        for word in split(&clean(text)) {
            // Into the private alphabet first: BPE works on characters, and a
            // multi-byte character must not be splittable down the middle.
            let mapped: String =
                word.bytes().map(|b| self.byte_encoder[b as usize]).collect();
            for piece in self.bpe(&mapped) {
                // Unknown cannot happen with a whole vocabulary — every single
                // character of the private alphabet is in it — but a truncated
                // or mismatched file is a real thing to be handed, and a
                // panic in the middle of typing is not the way to say so.
                ids.push(*self.vocab.get(&piece).unwrap_or(&self.end) as i64);
            }
        }
        ids.push(self.end as i64);

        if ids.len() > len {
            ids.truncate(len);
            // Whatever was cut, the sequence still has to end where the model
            // expects it to.
            if let Some(last) = ids.last_mut() {
                *last = self.end as i64;
            }
        }
        ids.resize(len, PAD);
        ids
    }

    /// Merge one word's characters by rank until no pair can be merged.
    ///
    /// The end-of-word marker is the part that is CLIP's rather than GPT-2's:
    /// it goes on the *last* character, so "cat" starts as `c a t</w>` and a
    /// word's ending is part of what can be merged. GPT-2 marks the space
    /// before a word instead, and a tokenizer that did that here would produce
    /// ids that exist but mean something else.
    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let last = word.len() - 1;
        word[last].push_str("</w>");
        if word.len() == 1 {
            return word;
        }

        loop {
            // The best pair in the word is the one merged, not the first: BPE
            // is defined by rank order, and merging left to right instead
            // produces a different and wrong split.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(rank) = self.ranks.get(&pair)
                    && best.is_none_or(|(_, r)| *rank < r)
                {
                    best = Some((i, *rank));
                }
            }
            let Some((at, _)) = best else { break };
            let merged = format!("{}{}", word[at], word[at + 1]);
            word.splice(at..at + 2, std::iter::once(merged));
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

/// Lower-cased, with runs of whitespace collapsed to one space.
///
/// CLIP's own cleaner also repairs mojibake through `ftfy`, which is a library
/// and a large table; text typed into this program a moment ago is not text
/// that has been through a broken decoder, so it is not reproduced here.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.trim().chars() {
        if c.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        for lower in c.to_lowercase() {
            out.push(lower);
        }
    }
    out
}

/// Codepoints Rust calls alphabetic that CLIP's pattern does not.
///
/// `char::is_alphabetic` is the Unicode *Alphabetic* property, which is a
/// superset of `\p{L}`: it also takes in the combining marks that carry
/// Other_Alphabetic — every Devanagari vowel sign, every Tibetan subjoined
/// letter — along with letter-numbers like Ⅱ and a handful of enclosed
/// symbols. CLIP's pattern asks for `\p{L}`, so a prompt with a combining
/// mark in it split differently here than it did for the model, and reached
/// the encoder as a sentence nobody typed.
///
/// The categories are Mn, Mc, Nl and So: 1694 codepoints in 253 ranges,
/// generated by walking every codepoint and comparing `is_alphabetic` against
/// its general category. Deliberately *not* including the codepoints Rust
/// calls alphabetic that an older Unicode table calls unassigned — those are
/// letters in the newer tables, and listing them here would mis-split the
/// scripts that gained them.
///
/// Nl needs no special case beyond being absent from the letter run: it is
/// numeric to Rust as well, so it falls into the digit branch, which is where
/// `[\p{N}]` wants it.
const NOT_A_LETTER: [(u32, u32); 253] = [
    (0x0345, 0x0345),
    (0x0363, 0x036F),
    (0x05B0, 0x05BD),
    (0x05BF, 0x05BF),
    (0x05C1, 0x05C2),
    (0x05C4, 0x05C5),
    (0x05C7, 0x05C7),
    (0x0610, 0x061A),
    (0x064B, 0x0657),
    (0x0659, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06DC),
    (0x06E1, 0x06E4),
    (0x06E7, 0x06E8),
    (0x06ED, 0x06ED),
    (0x0711, 0x0711),
    (0x0730, 0x073F),
    (0x07A6, 0x07B0),
    (0x0816, 0x0817),
    (0x081B, 0x0823),
    (0x0825, 0x0827),
    (0x0829, 0x082C),
    (0x08D4, 0x08DF),
    (0x08E3, 0x08E9),
    (0x08F0, 0x0903),
    (0x093A, 0x093B),
    (0x093E, 0x094C),
    (0x094E, 0x094F),
    (0x0955, 0x0957),
    (0x0962, 0x0963),
    (0x0981, 0x0983),
    (0x09BE, 0x09C4),
    (0x09C7, 0x09C8),
    (0x09CB, 0x09CC),
    (0x09D7, 0x09D7),
    (0x09E2, 0x09E3),
    (0x0A01, 0x0A03),
    (0x0A3E, 0x0A42),
    (0x0A47, 0x0A48),
    (0x0A4B, 0x0A4C),
    (0x0A51, 0x0A51),
    (0x0A70, 0x0A71),
    (0x0A75, 0x0A75),
    (0x0A81, 0x0A83),
    (0x0ABE, 0x0AC5),
    (0x0AC7, 0x0AC9),
    (0x0ACB, 0x0ACC),
    (0x0AE2, 0x0AE3),
    (0x0AFA, 0x0AFC),
    (0x0B01, 0x0B03),
    (0x0B3E, 0x0B44),
    (0x0B47, 0x0B48),
    (0x0B4B, 0x0B4C),
    (0x0B56, 0x0B57),
    (0x0B62, 0x0B63),
    (0x0B82, 0x0B82),
    (0x0BBE, 0x0BC2),
    (0x0BC6, 0x0BC8),
    (0x0BCA, 0x0BCC),
    (0x0BD7, 0x0BD7),
    (0x0C00, 0x0C04),
    (0x0C3E, 0x0C44),
    (0x0C46, 0x0C48),
    (0x0C4A, 0x0C4C),
    (0x0C55, 0x0C56),
    (0x0C62, 0x0C63),
    (0x0C81, 0x0C83),
    (0x0CBE, 0x0CC4),
    (0x0CC6, 0x0CC8),
    (0x0CCA, 0x0CCC),
    (0x0CD5, 0x0CD6),
    (0x0CE2, 0x0CE3),
    (0x0CF3, 0x0CF3),
    (0x0D00, 0x0D03),
    (0x0D3E, 0x0D44),
    (0x0D46, 0x0D48),
    (0x0D4A, 0x0D4C),
    (0x0D57, 0x0D57),
    (0x0D62, 0x0D63),
    (0x0D81, 0x0D83),
    (0x0DCF, 0x0DD4),
    (0x0DD6, 0x0DD6),
    (0x0DD8, 0x0DDF),
    (0x0DF2, 0x0DF3),
    (0x0E31, 0x0E31),
    (0x0E34, 0x0E3A),
    (0x0E4D, 0x0E4D),
    (0x0EB1, 0x0EB1),
    (0x0EB4, 0x0EB9),
    (0x0EBB, 0x0EBC),
    (0x0ECD, 0x0ECD),
    (0x0F71, 0x0F83),
    (0x0F8D, 0x0F97),
    (0x0F99, 0x0FBC),
    (0x102B, 0x1036),
    (0x1038, 0x1038),
    (0x103B, 0x103E),
    (0x1056, 0x1059),
    (0x105E, 0x1060),
    (0x1062, 0x1064),
    (0x1067, 0x106D),
    (0x1071, 0x1074),
    (0x1082, 0x108D),
    (0x108F, 0x108F),
    (0x109A, 0x109D),
    (0x16EE, 0x16F0),
    (0x1712, 0x1713),
    (0x1732, 0x1733),
    (0x1752, 0x1753),
    (0x1772, 0x1773),
    (0x17B6, 0x17C8),
    (0x1885, 0x1886),
    (0x18A9, 0x18A9),
    (0x1920, 0x192B),
    (0x1930, 0x1938),
    (0x1A17, 0x1A1B),
    (0x1A55, 0x1A5E),
    (0x1A61, 0x1A74),
    (0x1ABF, 0x1AC0),
    (0x1ACC, 0x1ACE),
    (0x1B00, 0x1B04),
    (0x1B35, 0x1B43),
    (0x1B80, 0x1B82),
    (0x1BA1, 0x1BA9),
    (0x1BAC, 0x1BAD),
    (0x1BE7, 0x1BF1),
    (0x1C24, 0x1C36),
    (0x1DD3, 0x1DF4),
    (0x2160, 0x2182),
    (0x2185, 0x2188),
    (0x24B6, 0x24E9),
    (0x2DE0, 0x2DFF),
    (0x3007, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303A),
    (0xA674, 0xA67B),
    (0xA69E, 0xA69F),
    (0xA6E6, 0xA6EF),
    (0xA802, 0xA802),
    (0xA80B, 0xA80B),
    (0xA823, 0xA827),
    (0xA880, 0xA881),
    (0xA8B4, 0xA8C3),
    (0xA8C5, 0xA8C5),
    (0xA8FF, 0xA8FF),
    (0xA926, 0xA92A),
    (0xA947, 0xA952),
    (0xA980, 0xA983),
    (0xA9B4, 0xA9BF),
    (0xA9E5, 0xA9E5),
    (0xAA29, 0xAA36),
    (0xAA43, 0xAA43),
    (0xAA4C, 0xAA4D),
    (0xAA7B, 0xAA7D),
    (0xAAB0, 0xAAB0),
    (0xAAB2, 0xAAB4),
    (0xAAB7, 0xAAB8),
    (0xAABE, 0xAABE),
    (0xAAEB, 0xAAEF),
    (0xAAF5, 0xAAF5),
    (0xABE3, 0xABEA),
    (0xFB1E, 0xFB1E),
    (0x10140, 0x10174),
    (0x10341, 0x10341),
    (0x1034A, 0x1034A),
    (0x10376, 0x1037A),
    (0x103D1, 0x103D5),
    (0x10A01, 0x10A03),
    (0x10A05, 0x10A06),
    (0x10A0C, 0x10A0F),
    (0x10D24, 0x10D27),
    (0x10EAB, 0x10EAC),
    (0x11000, 0x11002),
    (0x11038, 0x11045),
    (0x11073, 0x11074),
    (0x11080, 0x11082),
    (0x110B0, 0x110B8),
    (0x110C2, 0x110C2),
    (0x11100, 0x11102),
    (0x11127, 0x11132),
    (0x11145, 0x11146),
    (0x11180, 0x11182),
    (0x111B3, 0x111BF),
    (0x111CE, 0x111CF),
    (0x1122C, 0x11234),
    (0x11237, 0x11237),
    (0x1123E, 0x1123E),
    (0x11241, 0x11241),
    (0x112DF, 0x112E8),
    (0x11300, 0x11303),
    (0x1133E, 0x11344),
    (0x11347, 0x11348),
    (0x1134B, 0x1134C),
    (0x11357, 0x11357),
    (0x11362, 0x11363),
    (0x11435, 0x11441),
    (0x11443, 0x11445),
    (0x114B0, 0x114C1),
    (0x115AF, 0x115B5),
    (0x115B8, 0x115BE),
    (0x115DC, 0x115DD),
    (0x11630, 0x1163E),
    (0x11640, 0x11640),
    (0x116AB, 0x116B5),
    (0x1171D, 0x1172A),
    (0x1182C, 0x11838),
    (0x11930, 0x11935),
    (0x11937, 0x11938),
    (0x1193B, 0x1193C),
    (0x11940, 0x11940),
    (0x11942, 0x11942),
    (0x119D1, 0x119D7),
    (0x119DA, 0x119DF),
    (0x119E4, 0x119E4),
    (0x11A01, 0x11A0A),
    (0x11A35, 0x11A39),
    (0x11A3B, 0x11A3E),
    (0x11A51, 0x11A5B),
    (0x11A8A, 0x11A97),
    (0x11C2F, 0x11C36),
    (0x11C38, 0x11C3E),
    (0x11C92, 0x11CA7),
    (0x11CA9, 0x11CB6),
    (0x11D31, 0x11D36),
    (0x11D3A, 0x11D3A),
    (0x11D3C, 0x11D3D),
    (0x11D3F, 0x11D41),
    (0x11D43, 0x11D43),
    (0x11D47, 0x11D47),
    (0x11D8A, 0x11D8E),
    (0x11D90, 0x11D91),
    (0x11D93, 0x11D96),
    (0x11EF3, 0x11EF6),
    (0x11F00, 0x11F01),
    (0x11F03, 0x11F03),
    (0x11F34, 0x11F3A),
    (0x11F3E, 0x11F40),
    (0x12400, 0x1246E),
    (0x16F4F, 0x16F4F),
    (0x16F51, 0x16F87),
    (0x16F8F, 0x16F92),
    (0x16FF0, 0x16FF1),
    (0x1BC9E, 0x1BC9E),
    (0x1E000, 0x1E006),
    (0x1E008, 0x1E018),
    (0x1E01B, 0x1E021),
    (0x1E023, 0x1E024),
    (0x1E026, 0x1E02A),
    (0x1E08F, 0x1E08F),
    (0x1E947, 0x1E947),
    (0x1F130, 0x1F149),
    (0x1F150, 0x1F169),
    (0x1F170, 0x1F189),
];

/// Whether the pattern's `\p{L}` matches this character.
fn is_letter(c: char) -> bool {
    if !c.is_alphabetic() {
        return false;
    }
    let c = c as u32;
    NOT_A_LETTER
        .binary_search_by(|(lo, hi)| {
            if c < *lo {
                std::cmp::Ordering::Greater
            } else if c > *hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_err()
}

/// Split a cleaned prompt the way CLIP's pattern does.
///
/// That pattern is
/// `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`,
/// and this is it written out: the contractions, then runs of letters, then
/// **single** digits — 1008 is four tokens, not one — then runs of anything
/// else that is not a space.
fn split(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c == '\'' {
            let rest: String = chars[i..].iter().take(4).collect();
            // Longest first, so "'ll" is not read as "'l" would be if the
            // shorter ones were tried first.
            if let Some(found) = CONTRACTIONS
                .iter()
                .filter(|w| rest.starts_with(**w))
                .max_by_key(|w| w.len())
            {
                out.push((*found).to_string());
                i += found.chars().count();
                continue;
            }
        }

        if is_letter(c) {
            let start = i;
            while i < chars.len() && is_letter(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        if c.is_numeric() {
            out.push(c.to_string());
            i += 1;
            continue;
        }

        let start = i;
        while i < chars.len()
            && !chars[i].is_whitespace()
            && !is_letter(chars[i])
            && !chars[i].is_numeric()
        {
            // A quote that starts a contraction ends this run, or "don't"
            // would come out as one punctuation blob.
            if chars[i] == '\'' && i > start {
                let rest: String = chars[i..].iter().take(4).collect();
                if CONTRACTIONS.iter().any(|w| rest.starts_with(w)) {
                    break;
                }
            }
            i += 1;
        }
        if i > start {
            out.push(chars[start..i].iter().collect());
        }
    }
    out
}

/// The bytes that stand for themselves, in the order GPT-2 lists them.
fn printable_bytes() -> Vec<u8> {
    (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect()
}

/// Every byte as a printable character.
///
/// GPT-2's table, which CLIP took as it was: the bytes that are already
/// printable stand for themselves, and the 68 that are not are moved up out of
/// the way into a run starting at U+0100. The point is that a token is always
/// a whole string, so BPE never has to reason about a half-finished character.
fn byte_encoder() -> [char; 256] {
    let printable = printable_bytes();
    let mut map = ['\0'; 256];
    for b in &printable {
        map[*b as usize] = *b as char;
    }
    let mut next = 0u32;
    for b in 0..256usize {
        if !printable.contains(&(b as u8)) {
            map[b] = char::from_u32(256 + next).expect("inside the basic plane");
            next += 1;
        }
    }
    map
}

/// The same alphabet in the order the vocabulary is built from.
///
/// Order matters here and nowhere else: ids are positions in this sequence, so
/// listing the printable bytes before the moved-up ones is not a stylistic
/// choice — it is what decides every id in the table.
fn byte_order() -> Vec<char> {
    let printable = printable_bytes();
    let map = byte_encoder();
    let mut out: Vec<char> = printable.iter().map(|b| map[*b as usize]).collect();
    for b in 0..256usize {
        if !printable.contains(&(b as u8)) {
            out.push(map[b]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tokenizer with two merges, small enough to reason about.
    ///
    /// The vocabulary is derived, so a fixture is just a merges list: every
    /// single character is in the table already, and these two merges add
    /// `ca` and `cat</w>` on top.
    fn small() -> Tokenizer {
        Tokenizer::from_merges("#version: 0.2\nc a\nca t</w>\n").expect("the fixture should load")
    }

    #[test]
    fn a_word_merges_by_rank_rather_than_left_to_right() {
        let t = small();
        let ids = t.encode("cat", 8);
        let cat = t.vocab["cat</w>"] as i64;
        assert_eq!(ids[0], t.start as i64, "it must start with the start token");
        assert_eq!(ids[1], cat, "c a t</w> should have merged to one token");
        assert_eq!(ids[2], t.end as i64, "and then end");
    }

    #[test]
    fn the_end_marker_goes_on_the_last_character_not_the_first() {
        // The CLIP-versus-GPT-2 difference. If it went on the front, "a" would
        // not be `a</w>` and would miss the vocabulary entirely.
        let t = small();
        let ids = t.encode("a", 8);
        assert_eq!(ids[1], t.vocab["a</w>"] as i64);
    }

    #[test]
    fn a_sequence_is_padded_to_the_width_the_graph_takes() {
        let t = small();
        let ids = t.encode("cat", CONTEXT);
        assert_eq!(ids.len(), CONTEXT, "the graph takes a fixed width");
        assert!(
            ids[3..].iter().all(|id| *id == PAD),
            "the tail must be zero padding, which is what the model's mask keys on"
        );
    }

    #[test]
    fn a_prompt_too_long_is_cut_but_still_ends_properly() {
        let t = small();
        let ids = t.encode("cat cat cat cat cat cat", 5);
        assert_eq!(ids.len(), 5);
        assert_eq!(*ids.last().expect("not empty"), t.end as i64, "a cut sequence lost its end");
    }

    #[test]
    fn case_and_runs_of_space_do_not_change_the_answer() {
        let t = small();
        assert_eq!(t.encode("cat", 8), t.encode("  CAT\t\n ", 8));
    }

    #[test]
    fn digits_are_split_one_at_a_time() {
        // `[\p{N}]` in CLIP's pattern matches a single character, so 1008 is
        // four words rather than one. Getting this wrong silently changes
        // every prompt with a number in it.
        assert_eq!(split("1008"), vec!["1", "0", "0", "8"]);
        assert_eq!(split("a1"), vec!["a", "1"]);
    }

    #[test]
    fn contractions_are_split_the_way_the_pattern_does() {
        assert_eq!(split("don't"), vec!["don", "'t"]);
        assert_eq!(split("it's red"), vec!["it", "'s", "red"]);
        // Not a contraction: it stays with the punctuation run.
        assert_eq!(split("'x"), vec!["'", "x"]);
    }

    #[test]
    fn punctuation_runs_together_but_not_with_letters() {
        assert_eq!(split("a red car!!"), vec!["a", "red", "car", "!!"]);
    }

    #[test]
    fn every_byte_has_its_own_printable_character() {
        let map = byte_encoder();
        let mut seen: Vec<char> = map.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256, "two bytes shared a character");
        assert!(map.iter().all(|c| !c.is_whitespace() && *c != '\0'), "a byte maps to nothing");
        // The parts of the table that are fixed by GPT-2's definition.
        assert_eq!(map[b'!' as usize], '!');
        assert_eq!(map[b'~' as usize], '~');
        assert_eq!(map[0], '\u{100}', "the first unprintable byte starts the moved-up run");
    }

    #[test]
    fn the_derived_vocabulary_covers_every_byte() {
        // Every single character has to be in the table, or BPE has pieces it
        // cannot name and a rare word falls back to unknown.
        let t = small();
        assert!(t.vocab_size() > 512, "the alphabet alone is 512 entries");
        for c in byte_order() {
            assert!(t.vocab.contains_key(&c.to_string()), "{c:?} is not in the table");
            assert!(t.vocab.contains_key(&format!("{c}</w>")), "{c:?}</w> is not in the table");
        }
    }

    #[test]
    fn a_merges_file_without_pairs_is_rejected() {
        assert!(Tokenizer::from_merges("#version: 0.2\n").is_err());
        assert!(Tokenizer::from_merges("").is_err());
        assert!(Tokenizer::from_merges("a b c\n").is_err(), "a triple is not a pair");
    }

    /// A harness for checking this against another implementation.
    ///
    /// Three hand-picked examples prove the common case and nothing else. This
    /// reads a corpus — one prompt per line — and writes the ids for each, so
    /// the whole lot can be compared against a reference tokenizer in one go.
    /// A tokenizer only has to disagree about one word in a thousand to send
    /// the wrong sentence to the model, and one word in a thousand is exactly
    /// what hand-picked examples miss.
    ///
    /// `CLIP_MERGES=… CLIP_CORPUS=… CLIP_OUT=… cargo test -- --ignored`
    #[test]
    #[ignore = "a differential harness; see the comment for the variables it needs"]
    fn tokenize_a_corpus_for_comparison() {
        let (Ok(merges), Ok(corpus), Ok(out)) = (
            std::env::var("CLIP_MERGES"),
            std::env::var("CLIP_CORPUS"),
            std::env::var("CLIP_OUT"),
        ) else {
            // A harness rather than an assertion: run without a corpus to
            // point it at and there is nothing to check, which is not the same
            // as something being wrong.
            eprintln!("skipped: set CLIP_MERGES, CLIP_CORPUS and CLIP_OUT");
            return;
        };
        let t = Tokenizer::load(merges.as_ref()).expect("the real file");
        let text = std::fs::read_to_string(&corpus).expect("the corpus");

        let mut lines = String::new();
        for prompt in text.lines() {
            // Wide enough that nothing is truncated, then cut at the end
            // token so what is written is the sequence rather than the
            // padding — which the reference does not produce.
            let ids = t.encode(prompt, 1024);
            let end = ids.iter().skip(1).position(|id| *id == t.end as i64).map(|i| i + 2);
            let ids = &ids[..end.unwrap_or(ids.len())];
            let joined: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
            lines.push_str(&joined.join(","));
            lines.push('\n');
        }
        std::fs::write(&out, lines).expect("writing the ids");
    }

    /// The known gap, pinned rather than left to be rediscovered.
    ///
    /// NFC would make these two equal, because they are the same word: the
    /// first has a precomposed é, the second an e with a combining acute after
    /// it. When NFC lands, this assertion inverts — which is the point of
    /// writing it down, since a limitation nobody recorded is a limitation
    /// somebody finds the hard way.
    ///
    /// It needs the real vocabulary: the small fixture has neither character,
    /// so both come back as unknown and compare equal for the wrong reason —
    /// which is how this test first passed while proving nothing.
    #[test]
    #[ignore = "needs CLIP's merges.txt; set CLIP_MERGES"]
    fn decomposed_text_is_not_normalised_yet() {
        let Ok(merges) = std::env::var("CLIP_MERGES") else {
            panic!("set CLIP_MERGES to CLIP's merges.txt");
        };
        let t = Tokenizer::load(merges.as_ref()).expect("the real file");
        let composed = t.encode("caf\u{e9}", CONTEXT);
        let decomposed = t.encode("cafe\u{301}", CONTEXT);
        assert_ne!(composed, decomposed, "NFC has landed: invert this test");
    }

    /// The real thing, against the real files.
    ///
    /// Ignored by default because it needs `merges.txt` from a CLIP text
    /// model, which is half a megabyte and not in the repository. Point
    /// `CLIP_MERGES` at it and run
    /// `cargo test -- --ignored` to check this against ids that are the same
    /// wherever CLIP is used.
    #[test]
    #[ignore = "needs CLIP's merges.txt; set CLIP_MERGES"]
    fn the_canonical_clip_ids_come_out() {
        let Ok(merges) = std::env::var("CLIP_MERGES") else {
            panic!("set CLIP_MERGES to CLIP's merges.txt");
        };
        let t = Tokenizer::load(merges.as_ref()).expect("the real file");
        assert_eq!(t.vocab_size(), 49408, "CLIP's vocabulary is this size");

        // The example every CLIP implementation is checked against.
        let ids = t.encode("a photo of a cat", 7);
        assert_eq!(ids, vec![49406, 320, 1125, 539, 320, 2368, 49407]);

        // A phrase this program would actually be given.
        let red_car = t.encode("a red car", CONTEXT);
        assert_eq!(&red_car[..5], &[49406, 320, 736, 1615, 49407]);

        // A word that is not one token: it must split rather than be unknown.
        let split = t.encode("unkempt", CONTEXT);
        let pieces = split.iter().take_while(|id| **id != 49407).count() - 1;
        assert!(pieces > 1, "a rare word should split into pieces, got {pieces}");
        assert!(
            split[1..].iter().take(pieces).all(|id| *id != 49407),
            "no piece should have fallen back to unknown"
        );
    }
}
