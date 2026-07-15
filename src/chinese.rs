//! Chinese character metadata and Unicode helpers.
//!
//! The attributed ChinesePuzzleTool assets under `data/chinese` are embedded
//! with `include_str!` and parsed lazily on first use.  No run-time data path or
//! network access is required.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::OnceLock;

/// Returns whether `ch` is a Chinese unified ideograph accepted by this
/// project.
///
/// The profile follows the Unicode `Unified_Ideograph` repertoire, including
/// extensions through the current allocated CJK extension blocks, and adds
/// U+3007 IDEOGRAPHIC NUMBER ZERO.  Radicals, stroke symbols, U+3006 and the
/// ideographic iteration mark are deliberately excluded.
pub fn is_han(ch: char) -> bool {
    let cp = ch as u32;
    ch == '\u{3007}'
        || matches!(
            cp,
            0x3400..=0x4DBF
                | 0x4E00..=0x9FFF
                | 0x20000..=0x2A6DF
                | 0x2A700..=0x2B73F
                | 0x2B740..=0x2B81F
                | 0x2B820..=0x2CEAF
                | 0x2CEB0..=0x2EBEF
                | 0x2EBF0..=0x2EE5F
                | 0x30000..=0x3134F
                | 0x31350..=0x323AF
                | 0x323B0..=0x3347F
                | 0xFA0E
                | 0xFA0F
                | 0xFA11
                | 0xFA13
                | 0xFA14
                | 0xFA1F
                | 0xFA21
                | 0xFA23
                | 0xFA24
                | 0xFA27
                | 0xFA28
                | 0xFA29
        )
}

/// Parsed ChinesePuzzleTool metadata, represented as deterministic inverted
/// indexes suitable for resolving query character classes.
#[derive(Clone, Debug, Default)]
pub struct ChineseData {
    components: BTreeMap<char, BTreeSet<char>>,
    strokes: BTreeMap<u16, BTreeSet<char>>,
    pinyin: BTreeMap<String, BTreeSet<char>>,
    words: Vec<Vec<char>>,
}

impl ChineseData {
    /// Returns the metadata bundled with this crate.
    ///
    /// The JavaScript data files remain separate, attributed source assets;
    /// `include_str!` only makes their parsing deterministic and removes the
    /// old run-time directory option.
    pub fn bundled() -> &'static Self {
        static DATA: OnceLock<ChineseData> = OnceLock::new();
        DATA.get_or_init(|| {
            Self::from_sources(
                include_str!("../data/chinese/factorHan.js"),
                include_str!("../data/chinese/spell.js"),
                include_str!("../data/chinese/out.js"),
            )
            .expect("bundled Chinese metadata must be valid")
        })
    }

    /// Parses in-memory ChinesePuzzleTool sources.  This is public so callers
    /// can load data from an archive and so the parser can be tested without
    /// filesystem fixtures.
    pub fn from_sources(
        factor_source: &str,
        spell_source: &str,
        word_source: &str,
    ) -> Result<Self, ChineseDataError> {
        let factors = parse_factor_han(factor_source)?;
        let mut components: BTreeMap<char, BTreeSet<char>> = BTreeMap::new();

        for &root in factors.keys().filter(|&&ch| is_han(ch)) {
            let mut closure = BTreeSet::new();
            let mut visited = HashSet::new();
            collect_components(root, &factors, &mut visited, &mut closure);
            for component in closure {
                components.entry(component).or_default().insert(root);
            }
        }

        let mut strokes: BTreeMap<u16, BTreeSet<char>> = BTreeMap::new();
        let mut memo = HashMap::new();
        for &root in factors.keys().filter(|&&ch| is_han(ch)) {
            let count = count_strokes(root, &factors, &mut memo, &mut HashSet::new())?;
            strokes.entry(count).or_default().insert(root);
        }

        let pinyin = parse_spell(spell_source)?;
        let words = parse_words(word_source)?;
        Ok(Self {
            components,
            strokes,
            pinyin,
            words,
        })
    }

    /// Characters which recursively contain `component`.
    ///
    /// As in ChinesePuzzleTool, a character is considered to contain itself.
    pub fn characters_with_component(&self, component: char) -> Option<&BTreeSet<char>> {
        self.components.get(&component)
    }

    /// Characters containing every requested component.
    pub fn characters_with_components(&self, components: &[char]) -> BTreeSet<char> {
        let mut components = components.iter();
        let Some(first) = components.next() else {
            return BTreeSet::new();
        };
        let Some(first_set) = self.components.get(first) else {
            return BTreeSet::new();
        };
        let mut result = first_set.clone();
        for component in components {
            let Some(next) = self.components.get(component) else {
                return BTreeSet::new();
            };
            result.retain(|ch| next.contains(ch));
        }
        result
    }

    /// Characters with the factor-derived stroke count `count`.
    ///
    /// The count follows ChinesePuzzleTool's first declared decomposition and
    /// counts terminal stroke/components.  It is provided for compatibility,
    /// not as an authoritative replacement for Unihan total-stroke data.
    pub fn characters_with_strokes(&self, count: u16) -> Option<&BTreeSet<char>> {
        self.strokes.get(&count)
    }

    /// Resolves a safe pinyin glob against keys such as `shan3` and returns the
    /// union of all matching characters.
    ///
    /// The glob consists of ASCII letters, `ü`/`v`, at most one `*`, and an
    /// optional final tone digit 0-4.  Omitting the tone matches every tone.
    pub fn characters_with_pinyin(
        &self,
        pattern: &str,
    ) -> Result<BTreeSet<char>, PinyinPatternError> {
        let pattern = PinyinGlob::parse(pattern)?;
        let mut result = BTreeSet::new();
        for (key, characters) in &self.pinyin {
            if pattern.matches_key(key) {
                result.extend(characters.iter().copied());
            }
        }
        Ok(result)
    }

    /// Resolves a word-completion template containing exactly one `.` slot.
    /// Every other scalar must be Han and the complete substituted word must
    /// occur in the bundled word list.
    pub fn characters_with_word_template(
        &self,
        template: &str,
    ) -> Result<BTreeSet<char>, WordTemplateError> {
        let mut slot = None;
        let mut expected = Vec::new();
        for (index, character) in template.chars().enumerate() {
            if character == '.' {
                if slot.replace(index).is_some() {
                    return Err(WordTemplateError::new("组词模板必须恰好包含一个 ."));
                }
                expected.push(None);
            } else if is_han(character) {
                expected.push(Some(character));
            } else {
                return Err(WordTemplateError::new(format!(
                    "组词模板只允许汉字和一个 .；不支持 {character:?}"
                )));
            }
        }
        let Some(slot) = slot else {
            return Err(WordTemplateError::new("组词模板必须恰好包含一个 ."));
        };

        let mut result = BTreeSet::new();
        for word in &self.words {
            if word.len() != expected.len() {
                continue;
            }
            if expected
                .iter()
                .zip(word)
                .all(|(expected, actual)| expected.is_none_or(|expected| expected == *actual))
            {
                result.insert(word[slot]);
            }
        }
        Ok(result)
    }

    pub fn component_class_count(&self) -> usize {
        self.components.len()
    }

    pub fn stroke_class_count(&self) -> usize {
        self.strokes.len()
    }

    pub fn pinyin_class_count(&self) -> usize {
        self.pinyin.len()
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }
}

#[derive(Clone, Debug)]
struct FactorEntry {
    alternatives: Vec<Vec<char>>,
}

fn parse_factor_han(source: &str) -> Result<BTreeMap<char, FactorEntry>, ChineseDataError> {
    let start = source
        .find('`')
        .ok_or_else(|| ChineseDataError::parse("factorHan.js", 1, "找不到模板字符串起始反引号"))?;
    let end = source
        .rfind('`')
        .filter(|&end| end > start)
        .ok_or_else(|| ChineseDataError::parse("factorHan.js", 1, "找不到模板字符串结束反引号"))?;

    let mut factors = BTreeMap::new();
    for (line_index, raw_line) in source[start + 1..end].lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        // The reference file currently contains one accidental empty line.
        if line.is_empty() {
            continue;
        }

        let has_alternatives = line.contains('|');
        let mut fields = line.split('|');
        let head = fields.next().unwrap_or_default();
        let mut head_chars = head.chars();
        let Some(character) = head_chars.next() else {
            return Err(ChineseDataError::parse(
                "factorHan.js",
                line_number,
                "缺少字头",
            ));
        };
        if head_chars.next().is_some() {
            return Err(ChineseDataError::parse(
                "factorHan.js",
                line_number,
                "字头必须恰好是一个 Unicode 标量",
            ));
        }

        let mut alternatives = Vec::new();
        for field in fields {
            let decomposition: Vec<char> = field.chars().collect();
            if decomposition.is_empty() {
                // The upstream file contains both a trailing separator and
                // one doubled separator between valid alternatives.
                continue;
            }
            alternatives.push(decomposition);
        }
        if has_alternatives && alternatives.is_empty() {
            return Err(ChineseDataError::parse(
                "factorHan.js",
                line_number,
                "拆分方案不能为空",
            ));
        }

        let entry = factors.entry(character).or_insert_with(|| FactorEntry {
            alternatives: Vec::new(),
        });
        for alternative in alternatives {
            if !entry.alternatives.contains(&alternative) {
                entry.alternatives.push(alternative);
            }
        }
    }

    if factors.is_empty() {
        return Err(ChineseDataError::parse(
            "factorHan.js",
            1,
            "模板字符串中没有拆字数据",
        ));
    }
    Ok(factors)
}

fn collect_components(
    character: char,
    factors: &BTreeMap<char, FactorEntry>,
    visited: &mut HashSet<char>,
    output: &mut BTreeSet<char>,
) {
    if !visited.insert(character) {
        return;
    }
    output.insert(character);
    if let Some(entry) = factors.get(&character) {
        // Component containment uses every declared alternative.  This fixes
        // the reference implementation's accidental `[0]` truncation.
        for alternative in &entry.alternatives {
            for &child in alternative {
                collect_components(child, factors, visited, output);
            }
        }
    }
}

fn count_strokes(
    character: char,
    factors: &BTreeMap<char, FactorEntry>,
    memo: &mut HashMap<char, u16>,
    visiting: &mut HashSet<char>,
) -> Result<u16, ChineseDataError> {
    if let Some(&count) = memo.get(&character) {
        return Ok(count);
    }
    if !visiting.insert(character) {
        return Err(ChineseDataError::parse(
            "factorHan.js",
            0,
            format!("拆字数据存在环：{character}"),
        ));
    }

    let count = match factors
        .get(&character)
        .and_then(|entry| entry.alternatives.first())
    {
        None => 1,
        Some(children) => {
            let mut total = 0_u16;
            for &child in children {
                total = total
                    .checked_add(count_strokes(child, factors, memo, visiting)?)
                    .ok_or_else(|| {
                        ChineseDataError::parse(
                            "factorHan.js",
                            0,
                            format!("{character} 的笔画计数溢出"),
                        )
                    })?;
            }
            total
        }
    };
    visiting.remove(&character);
    memo.insert(character, count);
    Ok(count)
}

fn parse_spell(source: &str) -> Result<BTreeMap<String, BTreeSet<char>>, ChineseDataError> {
    let mut pinyin: BTreeMap<String, BTreeSet<char>> = BTreeMap::new();
    let mut entry_count = 0_usize;

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if !line.starts_with('"') {
            continue;
        }

        let (base, mut position) = parse_js_string(line, 0)
            .map_err(|message| ChineseDataError::parse("spell.js", line_number, message))?;
        position = skip_ascii_whitespace(line, position);
        if line.as_bytes().get(position) != Some(&b':') {
            return Err(ChineseDataError::parse(
                "spell.js",
                line_number,
                "拼音键后缺少冒号",
            ));
        }
        position = skip_ascii_whitespace(line, position + 1);
        let (encoded, _) = parse_js_string(line, position)
            .map_err(|message| ChineseDataError::parse("spell.js", line_number, message))?;

        let Some((_, pairs)) = encoded.split_once(':') else {
            return Err(ChineseDataError::parse(
                "spell.js",
                line_number,
                "拼音值中缺少内部冒号",
            ));
        };
        if base.is_empty() {
            return Err(ChineseDataError::parse(
                "spell.js",
                line_number,
                "拼音键不能为空",
            ));
        }

        let mut pair_chars = pairs.chars();
        while let Some(character) = pair_chars.next() {
            let Some(raw_tone) = pair_chars.next() else {
                return Err(ChineseDataError::parse(
                    "spell.js",
                    line_number,
                    "拼音字码对缺少声调数字",
                ));
            };
            let Some(raw_tone) = raw_tone.to_digit(10) else {
                return Err(ChineseDataError::parse(
                    "spell.js",
                    line_number,
                    "拼音字码对的声调不是数字",
                ));
            };
            if is_han(character) {
                let key = format!("{base}{}", raw_tone % 5);
                pinyin.entry(key).or_default().insert(character);
            }
        }
        entry_count += 1;
    }

    if entry_count == 0 {
        return Err(ChineseDataError::parse("spell.js", 1, "没有找到拼音条目"));
    }
    Ok(pinyin)
}

fn parse_words(source: &str) -> Result<Vec<Vec<char>>, ChineseDataError> {
    let start = source
        .find('`')
        .ok_or_else(|| ChineseDataError::parse("out.js", 1, "找不到模板字符串起始反引号"))?;
    let end = source
        .rfind('`')
        .filter(|&end| end > start)
        .ok_or_else(|| ChineseDataError::parse("out.js", 1, "找不到模板字符串结束反引号"))?;

    let mut words = BTreeSet::new();
    for raw_line in source[start + 1..end].lines() {
        let word: Vec<char> = raw_line.trim().chars().collect();
        if !word.is_empty() && word.iter().copied().all(is_han) {
            words.insert(word);
        }
    }
    if words.is_empty() {
        return Err(ChineseDataError::parse("out.js", 1, "词表中没有纯汉字词条"));
    }
    Ok(words.into_iter().collect())
}

fn parse_js_string(input: &str, start: usize) -> Result<(String, usize), String> {
    if input.as_bytes().get(start) != Some(&b'"') {
        return Err("应为 JavaScript 双引号字符串".to_owned());
    }
    let mut output = String::new();
    let mut position = start + 1;

    while position < input.len() {
        let (character, next) = next_char(input, position).ok_or("字符串意外结束")?;
        position = next;
        match character {
            '"' => return Ok((output, position)),
            '\\' => {
                let (escaped, next) = next_char(input, position).ok_or("转义序列意外结束")?;
                position = next;
                match escaped {
                    '"' | '\\' | '/' => output.push(escaped),
                    'n' => output.push('\n'),
                    'r' => output.push('\r'),
                    't' => output.push('\t'),
                    'b' => output.push('\u{0008}'),
                    'f' => output.push('\u{000C}'),
                    'u' => {
                        if position + 4 > input.len() {
                            return Err("不完整的 \\u 转义".to_owned());
                        }
                        let digits = &input[position..position + 4];
                        if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                            return Err("无效的 \\u 转义".to_owned());
                        }
                        let value = u32::from_str_radix(digits, 16)
                            .map_err(|_| "无效的 \\u 转义".to_owned())?;
                        let Some(value) = char::from_u32(value) else {
                            return Err("\\u 转义不是 Unicode 标量".to_owned());
                        };
                        output.push(value);
                        position += 4;
                    }
                    other => return Err(format!("不支持的 JavaScript 转义：\\{other}")),
                }
            }
            other => output.push(other),
        }
    }
    Err("JavaScript 字符串缺少结束引号".to_owned())
}

fn skip_ascii_whitespace(input: &str, mut position: usize) -> usize {
    while input
        .as_bytes()
        .get(position)
        .is_some_and(u8::is_ascii_whitespace)
    {
        position += 1;
    }
    position
}

fn next_char(input: &str, position: usize) -> Option<(char, usize)> {
    let character = input.get(position..)?.chars().next()?;
    Some((character, position + character.len_utf8()))
}

/// Failure while parsing bundled Chinese metadata.
#[derive(Debug)]
pub enum ChineseDataError {
    Parse {
        source_name: &'static str,
        line: usize,
        message: String,
    },
}

impl ChineseDataError {
    fn parse(source_name: &'static str, line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            source_name,
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for ChineseDataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse {
                source_name,
                line,
                message,
            } if *line > 0 => write!(formatter, "{source_name}:{line}：{message}"),
            Self::Parse {
                source_name,
                message,
                ..
            } => write!(formatter, "{source_name}：{message}"),
        }
    }
}

impl Error for ChineseDataError {}

/// Error in the deliberately small, safe pinyin glob language.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PinyinPatternError {
    pub offset: usize,
    pub message: String,
}

impl PinyinPatternError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for PinyinPatternError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "拼音模式第 {} 个字符附近：{}",
            self.offset, self.message
        )
    }
}

impl Error for PinyinPatternError {}

#[derive(Clone, Debug)]
struct PinyinGlob {
    prefix: String,
    suffix: String,
    tone: Option<u8>,
    wildcard: bool,
}

impl PinyinGlob {
    fn parse(input: &str) -> Result<Self, PinyinPatternError> {
        if input.is_empty() {
            return Err(PinyinPatternError::new(0, "拼音不能为空"));
        }
        let mut characters: Vec<char> = input.chars().collect();
        let tone = match characters.last().copied() {
            Some(character) if character.is_ascii_digit() => {
                let tone = character.to_digit(10).unwrap() as u8;
                if tone > 4 {
                    return Err(PinyinPatternError::new(
                        characters.len() - 1,
                        "声调必须是 0 到 4",
                    ));
                }
                characters.pop();
                Some(tone)
            }
            _ => None,
        };
        if characters.is_empty() {
            return Err(PinyinPatternError::new(0, "缺少拼音音节"));
        }

        let mut normalized = String::new();
        let mut stars = 0;
        for (offset, character) in characters.into_iter().enumerate() {
            match character {
                '*' => {
                    stars += 1;
                    if stars > 1 {
                        return Err(PinyinPatternError::new(offset, "最多只能使用一个 *"));
                    }
                    normalized.push('*');
                }
                'ü' | 'Ü' => normalized.push('ü'),
                'v' | 'V' => normalized.push('ü'),
                character if character.is_ascii_alphabetic() => {
                    normalized.push(character.to_ascii_lowercase())
                }
                character if character.is_ascii_digit() => {
                    return Err(PinyinPatternError::new(offset, "声调数字只能放在末尾"));
                }
                other => {
                    return Err(PinyinPatternError::new(
                        offset,
                        format!("只允许字母、ü/v、* 和末尾声调；不支持 {other:?}"),
                    ));
                }
            }
        }

        let (prefix, suffix, wildcard) = if let Some((prefix, suffix)) = normalized.split_once('*')
        {
            (prefix.to_owned(), suffix.to_owned(), true)
        } else {
            (normalized, String::new(), false)
        };
        Ok(Self {
            prefix,
            suffix,
            tone,
            wildcard,
        })
    }

    fn matches_key(&self, key: &str) -> bool {
        if key.is_empty() {
            return false;
        }
        let (base, tone) = key.split_at(key.len() - 1);
        let Some(tone) = tone.chars().next().and_then(|tone| tone.to_digit(10)) else {
            return false;
        };
        if self
            .tone
            .is_some_and(|expected| u32::from(expected) != tone)
        {
            return false;
        }
        if !self.wildcard {
            return base == self.prefix;
        }
        base.starts_with(&self.prefix)
            && base.ends_with(&self.suffix)
            && base.chars().count() >= self.prefix.chars().count() + self.suffix.chars().count()
    }
}

/// Error in an `@z(...)` word-completion template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WordTemplateError {
    pub message: String,
}

impl WordTemplateError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for WordTemplateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for WordTemplateError {}

#[cfg(test)]
mod tests {
    use super::*;

    const FACTORS: &str = r#"const hanzi=`一
丨
亅
丁|一亅
木|一丨
林|木木
休|亻木|人木
𠀀|一丨
`"#;

    const SPELL: &str = r#"// From cnchar
const spell = {
    "ding": "1:丁6",
    "lin": "1:林2",
    "xiu": "1:休1",
};"#;

    const WORDS: &str = r#"const ws=`丁
林
森林
凡士林
黑魔
魔术
阿Ｑ`;
"#;

    #[test]
    fn han_profile_uses_unicode_scalars() {
        assert!(is_han('中'));
        assert!(is_han('〇'));
        assert!(is_han('𠀀'));
        assert!(is_han('﨑'));
        assert!(!is_han('A'));
        assert!(!is_han('々'));
        assert!(!is_han('⺄'));
    }

    #[test]
    fn parses_factor_data_without_utf16_assumptions() {
        let data = ChineseData::from_sources(FACTORS, SPELL, WORDS).unwrap();
        assert!(
            data.characters_with_component('木')
                .unwrap()
                .contains(&'林')
        );
        assert!(
            data.characters_with_component('人')
                .unwrap()
                .contains(&'休')
        );
        assert!(
            data.characters_with_component('丨')
                .unwrap()
                .contains(&'𠀀')
        );
        assert!(data.characters_with_strokes(2).unwrap().contains(&'丁'));
        assert!(data.characters_with_strokes(4).unwrap().contains(&'林'));
    }

    #[test]
    fn factor_parser_ignores_empty_upstream_alternatives() {
        let factors = parse_factor_han("const hanzi=`一\n巴|巳丨||𠃍丨一乚|\n`").unwrap();
        assert_eq!(factors[&'巴'].alternatives.len(), 2);
        assert!(parse_factor_han("const hanzi=`一\n巴|\n`").is_err());
        assert!(parse_factor_han("const hanzi=`一\n巴||\n`").is_err());
    }

    #[test]
    fn decodes_cnchar_tone_codes_modulo_five() {
        let data = ChineseData::from_sources(FACTORS, SPELL, WORDS).unwrap();
        assert!(
            data.characters_with_pinyin("ding1")
                .unwrap()
                .contains(&'丁')
        );
        assert!(data.characters_with_pinyin("*in").unwrap().contains(&'林'));
        assert!(
            !data
                .characters_with_pinyin("ding4")
                .unwrap()
                .contains(&'丁')
        );
    }

    #[test]
    fn pinyin_glob_is_safe_and_reports_errors() {
        let data = ChineseData::from_sources(FACTORS, SPELL, WORDS).unwrap();
        assert!(data.characters_with_pinyin("[").is_err());
        assert!(data.characters_with_pinyin("a**").is_err());
        assert!(data.characters_with_pinyin("lin9").is_err());
    }

    #[test]
    fn word_templates_have_one_han_slot() {
        let data = ChineseData::from_sources(FACTORS, SPELL, WORDS).unwrap();
        assert_eq!(
            data.characters_with_word_template(".士林").unwrap(),
            BTreeSet::from(['凡'])
        );
        assert_eq!(
            data.characters_with_word_template("黑.").unwrap(),
            BTreeSet::from(['魔'])
        );
        assert!(data.characters_with_word_template("黑魔").is_err());
        assert!(data.characters_with_word_template("..术").is_err());
    }

    #[test]
    fn bundled_data_is_available_without_runtime_paths() {
        let data = ChineseData::bundled();
        assert!(data.word_count() > 100_000);
        assert!(data.pinyin_class_count() > 1_000);
    }
}
