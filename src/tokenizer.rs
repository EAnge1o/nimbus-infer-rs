//! Byte-level BPE tokenizer, written from scratch (no `tokenizers`, no
//! `regex`). Loads HuggingFace-style `vocab.json` + `merges.txt` -- the same
//! files the Python engine reads -- and reproduces the GPT-2 pre-tokenizer
//! split pattern as a hand-rolled state machine:
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+
//! ```
//!
//! Everything is byte-level: any UTF-8 input round-trips losslessly and no
//! `<unk>` token is ever needed. Merges are applied greedily lowest-rank-first
//! (proven equivalent to the reference "replace all occurrences per pass"
//! strategy: a merge can only create *higher*-rank pairs, so the next
//! lowest-rank pair is unchanged until every occurrence of the current one is
//! consumed).

use std::collections::HashMap;
use std::path::Path;

use crate::json::Json;

/// The GPT-2 reversible byte <-> printable-unicode mapping.
#[derive(Debug, Clone)]
pub struct ByteAlphabet {
    pub byte_to_code: [u32; 256],
    pub code_to_byte: Vec<Option<u8>>, // indexed by code point (max 323)
}

pub fn byte_alphabet() -> ByteAlphabet {
    let mut byte_to_code = [0u32; 256];
    let mut code_to_byte = vec![None; 324];
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    for b in b'!'..=b'~' {
        bs.push(b);
    }
    for b in 0xA1u8..=0xAC {
        bs.push(b);
    }
    for b in 0xAEu8..=0xFF {
        bs.push(b);
    }
    let base = bs.len(); // 188 printable bytes keep their own value
    for b in 0u16..256 {
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
        }
    }
    for (offset, &b) in bs.iter().enumerate() {
        let code = if offset < base { b as u32 } else { 256 + (offset - base) as u32 };
        byte_to_code[b as usize] = code;
        code_to_byte[code as usize] = Some(b);
    }
    ByteAlphabet { byte_to_code, code_to_byte }
}

/// Unicode whitespace, mirroring the Python `re` module's `\s` for str
/// patterns as closely as the two standards allow.
fn is_py_space(c: char) -> bool {
    c.is_whitespace()
}

/// Split text into pre-tokenization chunks exactly like the GPT-2 regex.
pub fn split_chunks(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let n = chars.len();

    while i < n {
        let c = chars[i];
        // 1. Contractions: 's 't 're 've 'm 'll 'd (case sensitive).
        if c == '\'' {
            let rest: String = chars[(i + 1).min(n)..].iter().take(2).collect();
            let two_char = ["re", "ve", "ll"].iter().any(|s| rest.starts_with(s));
            let one_char = ["s", "t", "m", "d"].iter().any(|s| rest.starts_with(s));
            if two_char {
                out.push(chars[i..i + 3].iter().collect());
                i += 3;
                continue;
            }
            if one_char {
                out.push(chars[i..i + 2].iter().collect());
                i += 2;
                continue;
            }
        }
        // 2. Optional single literal space + a run of one class.
        let mut j = i;
        if c == ' ' {
            j = i + 1;
        }
        if j < n {
            let next = chars[j];
            if next.is_ascii_alphabetic() {
                while j < n && chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            if next.is_ascii_digit() {
                while j < n && chars[j].is_ascii_digit() {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            let is_other = |c: char| !c.is_ascii_alphanumeric() && !is_py_space(c);
            if is_other(next) {
                while j < n && is_other(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        // 3. Whitespace runs.
        if is_py_space(c) {
            let mut j = i;
            while j < n && is_py_space(chars[j]) {
                j += 1;
            }
            if j == n {
                // `\s+`: run extends to the end of the string.
                out.push(chars[i..j].iter().collect());
                i = j;
            } else if j - i > 1 {
                // `\s+(?!\S)`: the final whitespace char may be claimed by the
                // next word's optional-space branch; release only the prefix.
                out.push(chars[i..j - 1].iter().collect());
                i = j - 1;
            } else if c != ' ' {
                // A lone non-space whitespace char (e.g. "\tb"): `\s+` takes it.
                out.push(chars[i..j].iter().collect());
                i = j;
            } else {
                // A lone ' ' followed by a word: the optional-space branch of
                // the next iteration claims it. Emit nothing here.
            }
            continue;
        }
        // Unreachable: every char is contraction/letter/digit/other/space.
        out.push(c.to_string());
        i += 1;
    }
    out
}

#[derive(Debug, Clone)]
pub struct BPETokenizer {
    alphabet: ByteAlphabet,
    decoder: Vec<String>, // id -> original vocab string
    /// id of each single-byte token; `None` for control bytes the vocab
    /// omits (some modern tokenizers, e.g. SmolLM2, drop the rarest ones --
    /// HuggingFace skips them too).
    byte_token_id: [Option<u32>; 256],
    // Merge table in id space: (left_id, right_id) -> (rank, merged_id).
    merges: HashMap<(u32, u32), (u32, u32)>,
}

impl BPETokenizer {
    pub fn load(directory: &Path) -> Result<BPETokenizer, String> {
        let vocab_path = directory.join("vocab.json");
        let merges_path = directory.join("merges.txt");

        let vocab_text = std::fs::read_to_string(&vocab_path)
            .map_err(|e| format!("reading {}: {e}", vocab_path.display()))?;
        let vocab_json = crate::json::parse(&vocab_text).map_err(|e| e.to_string())?;
        let Json::Obj(entries) = &vocab_json else {
            return Err("vocab.json: expected an object".into());
        };

        let alphabet = byte_alphabet();
        let mut encoder = HashMap::with_capacity(entries.len());
        let mut decoder = Vec::with_capacity(entries.len());
        for (token, id) in entries {
            let id = id
                .as_usize()
                .ok_or_else(|| format!("vocab.json: bad id for {token:?}"))?;
            let key: Vec<u32> = token.chars().map(|c| c as u32).collect();
            if decoder.len() <= id {
                decoder.resize(id + 1, String::new());
            }
            decoder[id] = token.clone();
            encoder.insert(key, id as u32);
        }

        // Pre-resolve the id of every single-byte token (None if absent --
        // a handful of control bytes are missing from some modern vocabs).
        let mut byte_token_id = [None; 256];
        for (b, &code) in alphabet.byte_to_code.iter().enumerate() {
            byte_token_id[b] = encoder.get(&vec![code]).copied();
        }

        let merges_text = std::fs::read_to_string(&merges_path)
            .map_err(|e| format!("reading {}: {e}", merges_path.display()))?;
        let mut merges = HashMap::new();
        for (rank, line) in merges_text.lines().enumerate() {
            let line = line.trim_end_matches(['\r']);
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let Some((a, b)) = line.split_once(' ') else {
                continue;
            };
            let key_a: Vec<u32> = a.chars().map(|c| c as u32).collect();
            let key_b: Vec<u32> = b.chars().map(|c| c as u32).collect();
            let merged: Vec<u32> = key_a.iter().chain(key_b.iter()).copied().collect();
            if let (Some(&ia), Some(&ib), Some(&im)) =
                (encoder.get(&key_a), encoder.get(&key_b), encoder.get(&merged))
            {
                merges.insert((ia, ib), (rank as u32, im));
            }
        }

        Ok(BPETokenizer { alphabet, decoder, byte_token_id, merges })
    }

    /// Apply ranked merges greedily (lowest rank first), like GPT-2.
    fn bpe(&self, word: &mut Vec<u32>) {
        if word.len() < 2 {
            return;
        }
        loop {
            let mut best: Option<(u32, usize)> = None; // (rank, index)
            for i in 0..word.len() - 1 {
                if let Some(&(rank, _)) = self.merges.get(&(word[i], word[i + 1])) {
                    if best.map_or(true, |(best_rank, _)| rank < best_rank) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let (_, merged) = self.merges[&(word[at], word[at + 1])];
            word[at] = merged;
            word.remove(at + 1);
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in split_chunks(text) {
            let mut word: Vec<u32> = chunk
                .bytes()
                .filter_map(|b| self.byte_token_id[b as usize])
                .collect();
            self.bpe(&mut word);
            ids.extend(word);
        }
        ids
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(token) = self.decoder.get(id as usize) else {
            return Vec::new();
        };
        token
            .chars()
            .filter_map(|c| self.alphabet.code_to_byte.get(c as usize).copied().flatten())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend(self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Streaming decoder: feed token ids, receive only complete characters.
    pub fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder { tokenizer: self, stream: Utf8Stream::new() }
    }
}

/// Decode a token stream without ever splitting a character (a byte-level
/// token may carry only part of a multi-byte UTF-8 sequence).
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a BPETokenizer,
    stream: Utf8Stream,
}

impl<'a> IncrementalDecoder<'a> {
    pub fn push(&mut self, token_id: u32) -> String {
        self.stream.push_bytes(&self.tokenizer.token_bytes(token_id))
    }

    pub fn flush(&mut self) -> String {
        self.stream.flush()
    }
}

/// Incremental UTF-8: buffers a truncated trailing sequence until the next
/// token completes it; invalid bytes become U+FFFD (errors="replace").
#[derive(Debug, Default)]
pub struct Utf8Stream {
    buffer: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream { buffer: Vec::new() }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.buffer.extend_from_slice(bytes);
        self.take_complete()
    }

    pub fn flush(&mut self) -> String {
        let out = if self.buffer.is_empty() {
            String::new()
        } else {
            String::from_utf8_lossy(&self.buffer).into_owned()
        };
        self.buffer.clear();
        out
    }

    fn take_complete(&mut self) -> String {
        let mut out = String::new();
        loop {
            let Some(&first) = self.buffer.first() else { break };
            let needed = utf8_len(first);
            if needed == 0 {
                out.push('\u{FFFD}');
                self.buffer.remove(0);
                continue;
            }
            if self.buffer.len() < needed {
                break; // wait for the rest of the sequence
            }
            match std::str::from_utf8(&self.buffer[..needed]) {
                Ok(s) => out.push_str(s),
                Err(_) => out.push('\u{FFFD}'),
            }
            self.buffer.drain(..needed);
        }
        out
    }
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 0, // continuation byte or invalid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_matches_gpt2() {
        let a = byte_alphabet();
        assert_eq!(a.byte_to_code[b'!' as usize], 33);
        assert_eq!(a.byte_to_code[0], 256);
        assert_eq!(a.byte_to_code[32], 288);
        assert_eq!(a.byte_to_code[173], 323);
        assert_eq!(a.code_to_byte[256], Some(0));
        assert_eq!(a.code_to_byte[33], Some(b'!'));
    }

    #[test]
    fn splitter_matches_gpt2_regex() {
        assert_eq!(split_chunks("Hello world"), vec!["Hello", " world"]);
        assert_eq!(split_chunks("it's fine"), vec!["it", "'s", " fine"]);
        assert_eq!(split_chunks("don't"), vec!["don", "'t"]);
        assert_eq!(split_chunks("abc123def"), vec!["abc", "123", "def"]);
        assert_eq!(split_chunks("x  y"), vec!["x", " ", " y"]);
        assert_eq!(split_chunks("x\n\ny"), vec!["x", "\n", "\n", "y"]);
        assert_eq!(split_chunks("x  "), vec!["x", "  "]);
        assert_eq!(split_chunks("hello, world!"), vec!["hello", ",", " world", "!"]);
        assert_eq!(split_chunks("中 文"), vec!["中", " 文"]);
        assert_eq!(split_chunks("I'll go"), vec!["I", "'ll", " go"]);
        assert_eq!(split_chunks("a\tb"), vec!["a", "\t", "b"]);
        assert_eq!(split_chunks("a \t b"), vec!["a", " \t", " b"]);
        assert_eq!(split_chunks("  leading"), vec![" ", " leading"]);
    }

    #[test]
    fn utf8_stream_never_splits_a_char() {
        let mut stream = Utf8Stream::new();
        let bytes = "你".as_bytes(); // three bytes
        let mut got = String::new();
        got.push_str(&stream.push_bytes(&bytes[..1]));
        assert_eq!(got, ""); // incomplete
        got.push_str(&stream.push_bytes(&bytes[1..2]));
        assert_eq!(got, ""); // still incomplete
        got.push_str(&stream.push_bytes(&bytes[2..]));
        assert_eq!(got, "你"); // completes exactly now
    }

    #[test]
    fn utf8_stream_replaces_invalid_bytes() {
        let mut stream = Utf8Stream::new();
        let got = stream.push_bytes(&[0xFF, b'a']);
        assert_eq!(got, "\u{FFFD}a");
    }
}
