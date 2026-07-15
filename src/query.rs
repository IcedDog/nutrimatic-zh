//! Chinese query language parser and derivative-friendly matcher.

use crate::chinese::{ChineseData, is_han};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt;

const MAX_REPEAT: usize = 255;

/// A normalized inclusive Unicode scalar range.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CharRange {
    pub start: char,
    pub end: char,
}

/// A character class whose universe is the project's Han profile.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CharacterClass {
    /// Every character accepted by [`is_han`].
    Han,
    /// Explicit ranges, optionally complemented within the Han universe.
    Ranges {
        ranges: Vec<CharRange>,
        negated: bool,
    },
}

impl CharacterClass {
    pub fn han() -> Self {
        Self::Han
    }

    pub fn from_chars(characters: impl IntoIterator<Item = char>) -> Self {
        let mut characters: Vec<char> = characters.into_iter().filter(|&ch| is_han(ch)).collect();
        characters.sort_unstable();
        characters.dedup();
        let mut ranges: Vec<CharRange> = Vec::new();
        for character in characters {
            if let Some(last) = ranges.last_mut()
                && (last.end as u32).checked_add(1) == Some(character as u32)
            {
                last.end = character;
            } else {
                ranges.push(CharRange {
                    start: character,
                    end: character,
                });
            }
        }
        Self::Ranges {
            ranges,
            negated: false,
        }
    }

    pub fn from_ranges(ranges: Vec<CharRange>, negated: bool) -> Self {
        let mut ranges = ranges;
        ranges.sort_unstable();
        let mut normalized: Vec<CharRange> = Vec::new();
        for range in ranges {
            if let Some(last) = normalized.last_mut()
                && range.start as u32 <= (last.end as u32).saturating_add(1)
            {
                if range.end > last.end {
                    last.end = range.end;
                }
            } else {
                normalized.push(range);
            }
        }
        Self::Ranges {
            ranges: normalized,
            negated,
        }
    }

    pub fn contains(&self, character: char) -> bool {
        if !is_han(character) {
            return false;
        }
        match self {
            Self::Han => true,
            Self::Ranges { ranges, negated } => {
                let contained = ranges
                    .binary_search_by(|range| {
                        if character < range.start {
                            std::cmp::Ordering::Greater
                        } else if character > range.end {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .is_ok();
                contained != *negated
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(
            self,
            Self::Ranges {
                ranges,
                negated: false
            } if ranges.is_empty()
        )
    }

    pub fn ranges(&self) -> Option<&[CharRange]> {
        match self {
            Self::Han => None,
            Self::Ranges { ranges, .. } => Some(ranges),
        }
    }
}

/// Hashable regular-expression AST used directly as a derivative search state.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Expr {
    Empty,
    Epsilon,
    Class(u32),
    /// A single-character class carrying an equality binding.
    Capture {
        class: u32,
        group: u32,
    },
    Concat(Vec<Expr>),
    Alt(Vec<Expr>),
    Intersect(Vec<Expr>),
    Star(Box<Expr>),
    /// A permutation of the contained regex pieces, each used exactly once.
    Anagram(Vec<Expr>),
}

impl Expr {
    pub fn concat(parts: Vec<Self>) -> Self {
        let mut flattened = Vec::new();
        for part in parts {
            match part {
                Self::Empty => return Self::Empty,
                Self::Epsilon => {}
                Self::Concat(parts) => flattened.extend(parts),
                other => flattened.push(other),
            }
        }
        match flattened.len() {
            0 => Self::Epsilon,
            1 => flattened.pop().unwrap(),
            _ => Self::Concat(flattened),
        }
    }

    pub fn alt(parts: Vec<Self>) -> Self {
        let mut flattened = Vec::new();
        for part in parts {
            match part {
                Self::Empty => {}
                Self::Alt(parts) => flattened.extend(parts),
                other => flattened.push(other),
            }
        }
        flattened.sort_unstable();
        flattened.dedup();
        match flattened.len() {
            0 => Self::Empty,
            1 => flattened.pop().unwrap(),
            _ => Self::Alt(flattened),
        }
    }

    pub fn intersect(parts: Vec<Self>) -> Self {
        let mut flattened = Vec::new();
        for part in parts {
            match part {
                Self::Empty => return Self::Empty,
                Self::Intersect(parts) => flattened.extend(parts),
                other => flattened.push(other),
            }
        }
        flattened.sort_unstable();
        flattened.dedup();
        match flattened.len() {
            0 => Self::Epsilon,
            1 => flattened.pop().unwrap(),
            _ => Self::Intersect(flattened),
        }
    }

    pub fn star(inner: Self) -> Self {
        match inner {
            Self::Empty | Self::Epsilon => Self::Epsilon,
            Self::Star(_) => inner,
            other => Self::Star(Box::new(other)),
        }
    }

    pub fn anagram(parts: Vec<Self>) -> Self {
        let mut normalized = Vec::new();
        for part in parts {
            match part {
                Self::Empty => return Self::Empty,
                Self::Epsilon => {}
                other => normalized.push(other),
            }
        }
        normalized.sort_unstable();
        match normalized.len() {
            0 => Self::Epsilon,
            1 => normalized.pop().unwrap(),
            _ => Self::Anagram(normalized),
        }
    }

    pub fn nullable(&self) -> bool {
        match self {
            Self::Empty | Self::Class(_) | Self::Capture { .. } => false,
            Self::Epsilon => true,
            Self::Concat(parts) | Self::Intersect(parts) | Self::Anagram(parts) => {
                parts.iter().all(Self::nullable)
            }
            Self::Alt(parts) => parts.iter().any(Self::nullable),
            Self::Star(_) => true,
        }
    }

    pub fn is_dead(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Brzozowski derivative which deliberately treats captures as their
    /// underlying character class.  This is appropriate for index traversal;
    /// [`Program::matches`] performs the final equality check.
    pub fn derive_approx(&self, character: char, classes: &[CharacterClass]) -> Self {
        match self {
            Self::Empty | Self::Epsilon => Self::Empty,
            Self::Class(class) | Self::Capture { class, .. } => {
                if classes
                    .get(*class as usize)
                    .is_some_and(|class| class.contains(character))
                {
                    Self::Epsilon
                } else {
                    Self::Empty
                }
            }
            Self::Alt(parts) => Self::alt(
                parts
                    .iter()
                    .map(|part| part.derive_approx(character, classes))
                    .collect(),
            ),
            Self::Intersect(parts) => Self::intersect(
                parts
                    .iter()
                    .map(|part| part.derive_approx(character, classes))
                    .collect(),
            ),
            Self::Concat(parts) => derive_concat_approx(parts, character, classes),
            Self::Star(inner) => Self::concat(vec![
                inner.derive_approx(character, classes),
                Self::star((**inner).clone()),
            ]),
            Self::Anagram(parts) => {
                let mut alternatives = Vec::new();
                for index in 0..parts.len() {
                    let mut remaining = parts.clone();
                    let selected = remaining.remove(index);
                    alternatives.push(Self::concat(vec![
                        selected.derive_approx(character, classes),
                        Self::anagram(remaining),
                    ]));
                }
                Self::alt(alternatives)
            }
        }
    }
}

fn derive_concat_approx(parts: &[Expr], character: char, classes: &[CharacterClass]) -> Expr {
    let mut alternatives = Vec::new();
    for index in 0..parts.len() {
        if parts[..index].iter().all(Expr::nullable) {
            let mut result = Vec::with_capacity(parts.len() - index);
            result.push(parts[index].derive_approx(character, classes));
            result.extend_from_slice(&parts[index + 1..]);
            alternatives.push(Expr::concat(result));
        } else {
            break;
        }
    }
    Expr::alt(alternatives)
}

/// Parsed query plus its interned character classes and capture namespace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub expr: Expr,
    pub classes: Vec<CharacterClass>,
    pub capture_count: u32,
}

impl Program {
    pub fn parse(query: &str, data: Option<&ChineseData>) -> Result<Self, ParseError> {
        let mut parser = Parser::new(query, data);
        let expression = parser.parse_alt()?;
        if parser.position != query.len() {
            return Err(parser.error("无法解析的尾随内容"));
        }
        if expression == Expr::Epsilon && query.is_empty() {
            return Err(ParseError::new(0, "查询不能为空"));
        }
        Ok(Self {
            expr: expression,
            classes: parser.classes,
            capture_count: parser.capture_ids.len() as u32,
        })
    }

    pub fn root(&self) -> &Expr {
        &self.expr
    }

    pub fn nullable(&self, expression: &Expr) -> bool {
        expression.nullable()
    }

    pub fn is_dead(&self, expression: &Expr) -> bool {
        expression.is_dead()
    }

    pub fn derive_approx(&self, expression: &Expr, character: char) -> Expr {
        expression.derive_approx(character, &self.classes)
    }

    pub fn matches_approx(&self, input: &str) -> bool {
        let mut state = self.expr.clone();
        for character in input.chars() {
            state = self.derive_approx(&state, character);
            if state.is_dead() {
                return false;
            }
        }
        state.nullable()
    }

    /// Full-string match including `@t(ID)` equality bindings.
    pub fn matches(&self, input: &str) -> bool {
        if input.chars().any(|character| !is_han(character)) {
            return false;
        }
        let initial_captures = vec![None; self.capture_count as usize];
        let mut states = HashSet::from([(self.expr.clone(), initial_captures)]);

        for character in input.chars() {
            let mut next_states = HashSet::new();
            for (expression, captures) in states {
                for state in derive_bound(&expression, character, &captures, &self.classes) {
                    if !state.0.is_dead() {
                        next_states.insert(state);
                    }
                }
            }
            if next_states.is_empty() {
                return false;
            }
            states = next_states;
        }
        states
            .into_iter()
            .any(|(expression, _)| expression.nullable())
    }

    pub fn class(&self, id: u32) -> Option<&CharacterClass> {
        self.classes.get(id as usize)
    }
}

type Captures = Vec<Option<char>>;
type BoundState = (Expr, Captures);

fn derive_bound(
    expression: &Expr,
    character: char,
    captures: &Captures,
    classes: &[CharacterClass],
) -> Vec<BoundState> {
    match expression {
        Expr::Empty | Expr::Epsilon => Vec::new(),
        Expr::Class(class) => {
            if class_matches(classes, *class, character) {
                vec![(Expr::Epsilon, captures.clone())]
            } else {
                Vec::new()
            }
        }
        Expr::Capture { class, group } => {
            if !class_matches(classes, *class, character) {
                return Vec::new();
            }
            let Some(binding) = captures.get(*group as usize) else {
                return Vec::new();
            };
            if binding.is_some_and(|bound| bound != character) {
                return Vec::new();
            }
            let mut captures = captures.clone();
            captures[*group as usize] = Some(character);
            vec![(Expr::Epsilon, captures)]
        }
        Expr::Alt(parts) => {
            let mut output = Vec::new();
            for part in parts {
                output.extend(derive_bound(part, character, captures, classes));
            }
            deduplicate_bound(output)
        }
        Expr::Intersect(parts) => derive_intersection_bound(parts, character, captures, classes),
        Expr::Concat(parts) => derive_concat_bound(parts, character, captures, classes),
        Expr::Star(inner) => derive_bound(inner, character, captures, classes)
            .into_iter()
            .map(|(derived, captures)| {
                (
                    Expr::concat(vec![derived, Expr::star((**inner).clone())]),
                    captures,
                )
            })
            .collect(),
        Expr::Anagram(parts) => {
            let mut output = Vec::new();
            for index in 0..parts.len() {
                let mut remaining = parts.clone();
                let selected = remaining.remove(index);
                for (derived, captures) in derive_bound(&selected, character, captures, classes) {
                    output.push((
                        Expr::concat(vec![derived, Expr::anagram(remaining.clone())]),
                        captures,
                    ));
                }
            }
            deduplicate_bound(output)
        }
    }
}

fn derive_concat_bound(
    parts: &[Expr],
    character: char,
    captures: &Captures,
    classes: &[CharacterClass],
) -> Vec<BoundState> {
    let mut output = Vec::new();
    for index in 0..parts.len() {
        if !parts[..index].iter().all(Expr::nullable) {
            break;
        }
        for (derived, captures) in derive_bound(&parts[index], character, captures, classes) {
            let mut result = Vec::with_capacity(parts.len() - index);
            result.push(derived);
            result.extend_from_slice(&parts[index + 1..]);
            output.push((Expr::concat(result), captures));
        }
    }
    deduplicate_bound(output)
}

fn derive_intersection_bound(
    parts: &[Expr],
    character: char,
    captures: &Captures,
    classes: &[CharacterClass],
) -> Vec<BoundState> {
    let mut combined: Vec<(Vec<Expr>, Captures)> = vec![(Vec::new(), captures.clone())];
    for part in parts {
        let mut next = Vec::new();
        for (derived_parts, captures) in combined {
            for (derived, captures) in derive_bound(part, character, &captures, classes) {
                let mut parts = derived_parts.clone();
                parts.push(derived);
                next.push((parts, captures));
            }
        }
        if next.is_empty() {
            return Vec::new();
        }
        combined = next;
    }
    deduplicate_bound(
        combined
            .into_iter()
            .map(|(parts, captures)| (Expr::intersect(parts), captures))
            .collect(),
    )
}

fn class_matches(classes: &[CharacterClass], class: u32, character: char) -> bool {
    classes
        .get(class as usize)
        .is_some_and(|class| class.contains(character))
}

fn deduplicate_bound(states: Vec<BoundState>) -> Vec<BoundState> {
    states
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

struct Parser<'a> {
    input: &'a str,
    position: usize,
    data: Option<&'a ChineseData>,
    classes: Vec<CharacterClass>,
    class_ids: HashMap<CharacterClass, u32>,
    capture_ids: HashMap<String, u32>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, data: Option<&'a ChineseData>) -> Self {
        let han = CharacterClass::han();
        Self {
            input,
            position: 0,
            data,
            classes: vec![han.clone()],
            class_ids: HashMap::from([(han, 0)]),
            capture_ids: HashMap::new(),
        }
    }

    fn parse_alt(&mut self) -> Result<Expr, ParseError> {
        let mut alternatives = vec![self.parse_intersection()?];
        while self.consume('|') {
            alternatives.push(self.parse_intersection()?);
        }
        Ok(Expr::alt(alternatives))
    }

    fn parse_intersection(&mut self) -> Result<Expr, ParseError> {
        let mut parts = vec![self.parse_concat()?];
        while self.consume('&') {
            parts.push(self.parse_concat()?);
        }
        Ok(Expr::intersect(parts))
    }

    fn parse_concat(&mut self) -> Result<Expr, ParseError> {
        let mut parts = Vec::new();
        while let Some(character) = self.peek() {
            if matches!(character, ')' | '>' | '|' | '&') {
                break;
            }
            parts.push(self.parse_piece()?);
        }
        Ok(Expr::concat(parts))
    }

    fn parse_piece(&mut self) -> Result<Expr, ParseError> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some('?') => {
                self.bump();
                Ok(Expr::alt(vec![Expr::Epsilon, atom]))
            }
            Some('*') => {
                self.bump();
                Ok(Expr::star(atom))
            }
            Some('+') => {
                self.bump();
                Ok(Expr::concat(vec![atom.clone(), Expr::star(atom)]))
            }
            Some('{') if self.looks_like_repeat() => self.parse_repeat(atom),
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        let offset = self.position;
        let Some(character) = self.bump() else {
            return Err(ParseError::new(offset, "缺少匹配原子"));
        };
        match character {
            '.' => Ok(Expr::Class(0)),
            '#' => Err(ParseError::new(
                offset,
                "新版不再支持 #；一个汉字请写 .，任意多个请写 .*",
            )),
            '(' => {
                let expression = self.parse_alt()?;
                if !self.consume(')') {
                    return Err(self.error("分组缺少右括号 )"));
                }
                Ok(expression)
            }
            '[' => self.parse_character_class(offset),
            '<' => self.parse_anagram(offset),
            '@' => self.parse_property(offset),
            '{' => Err(ParseError::new(
                offset,
                "花括号只用于跟在原子后的重复次数，如 .{4}；属性请写 @p(...) 等",
            )),
            '"' => Err(ParseError::new(
                offset,
                "中文模式不使用双引号；请直接书写表达式或使用 (...) 分组",
            )),
            '\\' => {
                let escaped = self
                    .bump()
                    .ok_or_else(|| ParseError::new(offset, "转义序列不完整"))?;
                self.literal(escaped, offset)
            }
            '?' | '*' | '+' => Err(ParseError::new(
                offset,
                "量词必须跟在原子后；任意多个汉字请写 .*，至少一个请写 .+",
            )),
            ')' | ']' | '}' | '>' | '|' | '&' => {
                Err(ParseError::new(offset, format!("意外的符号：{character}")))
            }
            literal => self.literal(literal, offset),
        }
    }

    fn literal(&mut self, character: char, offset: usize) -> Result<Expr, ParseError> {
        if !is_han(character) {
            return Err(ParseError::new(
                offset,
                format!("仅允许汉字字面量；不支持 {character:?}"),
            ));
        }
        let class = CharacterClass::from_chars([character]);
        let id = self.intern_class(class);
        Ok(Expr::Class(id))
    }

    fn parse_character_class(&mut self, offset: usize) -> Result<Expr, ParseError> {
        let negated = self.consume('^');
        let mut ranges = Vec::new();
        while self.peek().is_some() && self.peek() != Some(']') {
            let start = self.class_character(offset)?;
            if self.peek() == Some('-') && self.peek_after_current() != Some(']') {
                self.bump();
                let end = self.class_character(offset)?;
                if start > end {
                    return Err(ParseError::new(offset, "字符类范围逆序"));
                }
                ranges.push(CharRange { start, end });
            } else {
                ranges.push(CharRange { start, end: start });
            }
        }
        if !self.consume(']') {
            return Err(ParseError::new(offset, "字符类缺少 ]"));
        }
        if ranges.is_empty() {
            return Err(ParseError::new(offset, "字符类不能为空"));
        }
        let class = CharacterClass::from_ranges(ranges, negated);
        let id = self.intern_class(class);
        Ok(Expr::Class(id))
    }

    fn class_character(&mut self, offset: usize) -> Result<char, ParseError> {
        let Some(mut character) = self.bump() else {
            return Err(ParseError::new(offset, "字符类意外结束"));
        };
        if character == '\\' {
            character = self
                .bump()
                .ok_or_else(|| ParseError::new(offset, "字符类转义不完整"))?;
        }
        if !is_han(character) {
            return Err(ParseError::new(
                offset,
                format!("字符类成员不是汉字：{character:?}"),
            ));
        }
        Ok(character)
    }

    fn parse_anagram(&mut self, offset: usize) -> Result<Expr, ParseError> {
        let mut parts = Vec::new();
        while self.peek().is_some() && self.peek() != Some('>') {
            parts.push(self.parse_piece()?);
        }
        if !self.consume('>') {
            return Err(ParseError::new(offset, "乱序表达式缺少 >"));
        }
        if parts.is_empty() {
            return Err(ParseError::new(offset, "乱序表达式不能为空"));
        }
        Ok(Expr::anagram(parts))
    }

    fn parse_property(&mut self, offset: usize) -> Result<Expr, ParseError> {
        let name_start = self.position;
        while self
            .peek()
            .is_some_and(|character| character.is_ascii_alphabetic())
        {
            self.bump();
        }
        let name = self.input[name_start..self.position].to_owned();
        if name.is_empty() || !self.consume('(') {
            return Err(ParseError::new(
                offset,
                "属性语法应为 @p(...)、@b(...)、@h(...)、@t(...) 或 @z(...)",
            ));
        }
        let content = self.read_property_content(offset)?;
        let content = content.trim();
        match name.as_str() {
            "b" => {
                let components: Vec<char> = content.chars().collect();
                if components.is_empty() {
                    return Err(ParseError::new(offset, "部件属性不能为空"));
                }
                if components
                    .iter()
                    .any(|character| character.is_ascii() || character.is_whitespace())
                {
                    return Err(ParseError::new(
                        offset,
                        "@b(...) 中只写连续部件；多个部件表示同时包含",
                    ));
                }
                let characters = self.metadata().characters_with_components(&components);
                Ok(self.expression_for_characters(characters))
            }
            "h" => {
                let (minimum, maximum) = parse_stroke_range(content).map_err(|message| {
                    ParseError::new(offset, format!("无效的笔画属性：{message}"))
                })?;
                let mut characters = BTreeSet::new();
                for count in minimum..=maximum {
                    if let Some(found) = self.metadata().characters_with_strokes(count) {
                        characters.extend(found.iter().copied());
                    }
                }
                Ok(self.expression_for_characters(characters))
            }
            "p" => {
                if content.is_empty() {
                    return Err(ParseError::new(offset, "拼音属性不能为空"));
                }
                let characters = self
                    .metadata()
                    .characters_with_pinyin(content)
                    .map_err(|error| ParseError::new(offset, format!("无效的拼音属性：{error}")))?;
                Ok(self.expression_for_characters(characters))
            }
            "t" => {
                if !valid_capture_id(content) {
                    return Err(ParseError::new(
                        offset,
                        "捕获 ID 必须以 ASCII 字母或下划线开头，且只含字母、数字、下划线",
                    ));
                }
                let group = if let Some(&group) = self.capture_ids.get(content) {
                    group
                } else {
                    let group = self.capture_ids.len() as u32;
                    self.capture_ids.insert(content.to_owned(), group);
                    group
                };
                Ok(Expr::Capture { class: 0, group })
            }
            "z" => {
                let characters = self
                    .metadata()
                    .characters_with_word_template(content)
                    .map_err(|error| ParseError::new(offset, format!("无效的组词模板：{error}")))?;
                Ok(self.expression_for_characters(characters))
            }
            other => Err(ParseError::new(offset, format!("未知属性：{other}"))),
        }
    }

    fn read_property_content(&mut self, offset: usize) -> Result<String, ParseError> {
        let mut content = String::new();
        while let Some(character) = self.bump() {
            match character {
                ')' => return Ok(content),
                '(' => return Err(ParseError::new(offset, "属性参数中不能嵌套括号")),
                other => content.push(other),
            }
        }
        Err(ParseError::new(offset, "属性缺少 )"))
    }

    fn metadata(&self) -> &ChineseData {
        match self.data {
            Some(data) => data,
            None => ChineseData::bundled(),
        }
    }

    fn expression_for_characters(&mut self, characters: BTreeSet<char>) -> Expr {
        let class = CharacterClass::from_chars(characters);
        if class.is_empty() {
            Expr::Empty
        } else {
            Expr::Class(self.intern_class(class))
        }
    }

    fn parse_repeat(&mut self, atom: Expr) -> Result<Expr, ParseError> {
        let offset = self.position;
        self.bump();
        let minimum = self.parse_decimal(offset)?;
        let maximum = if self.consume('}') {
            Some(minimum)
        } else if self.consume(',') {
            if self.consume('}') {
                None
            } else {
                let maximum = self.parse_decimal(offset)?;
                if !self.consume('}') {
                    return Err(ParseError::new(offset, "重复次数缺少 }"));
                }
                Some(maximum)
            }
        } else {
            return Err(ParseError::new(offset, "无效的重复次数"));
        };

        if minimum > MAX_REPEAT || maximum.is_some_and(|maximum| maximum > MAX_REPEAT) {
            return Err(ParseError::new(
                offset,
                format!("重复次数不能超过 {MAX_REPEAT}"),
            ));
        }
        if maximum.is_some_and(|maximum| maximum < minimum) {
            return Err(ParseError::new(offset, "重复次数上限小于下限"));
        }
        Ok(repeat_expression(atom, minimum, maximum))
    }

    fn parse_decimal(&mut self, offset: usize) -> Result<usize, ParseError> {
        let start = self.position;
        while self
            .peek()
            .is_some_and(|character| character.is_ascii_digit())
        {
            self.bump();
        }
        if start == self.position {
            return Err(ParseError::new(offset, "缺少重复次数"));
        }
        self.input[start..self.position]
            .parse()
            .map_err(|_| ParseError::new(offset, "重复次数无效"))
    }

    fn looks_like_repeat(&self) -> bool {
        self.peek() == Some('{')
            && self
                .input
                .get(self.position + 1..)
                .and_then(|rest| rest.chars().next())
                .is_some_and(|character| character.is_ascii_digit())
    }

    fn intern_class(&mut self, class: CharacterClass) -> u32 {
        if let Some(&id) = self.class_ids.get(&class) {
            return id;
        }
        let id = self.classes.len() as u32;
        self.classes.push(class.clone());
        self.class_ids.insert(class, id);
        id
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.position..)?.chars().next()
    }

    fn peek_after_current(&self) -> Option<char> {
        let current = self.peek()?;
        self.input
            .get(self.position + current.len_utf8()..)?
            .chars()
            .next()
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.position += character.len_utf8();
        Some(character)
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(self.position, message)
    }
}

fn repeat_expression(atom: Expr, minimum: usize, maximum: Option<usize>) -> Expr {
    let mut parts = Vec::new();
    for _ in 0..minimum {
        parts.push(atom.clone());
    }
    match maximum {
        None => parts.push(Expr::star(atom)),
        Some(maximum) => {
            for _ in minimum..maximum {
                parts.push(Expr::alt(vec![Expr::Epsilon, atom.clone()]));
            }
        }
    }
    Expr::concat(parts)
}

fn parse_stroke_range(input: &str) -> Result<(u16, u16), &'static str> {
    if input.is_empty() {
        return Err("笔画数不能为空");
    }
    let (minimum, maximum) = if let Some((minimum, maximum)) = input.split_once('-') {
        if maximum.contains('-') {
            return Err("范围只能包含一个 -");
        }
        let minimum = minimum.parse::<u16>().map_err(|_| "下限不是整数")?;
        let maximum = maximum.parse::<u16>().map_err(|_| "上限不是整数")?;
        (minimum, maximum)
    } else {
        let count = input.parse::<u16>().map_err(|_| "笔画数不是整数")?;
        (count, count)
    };
    if minimum > maximum {
        return Err("范围上限小于下限");
    }
    Ok((minimum, maximum))
}

fn valid_capture_id(id: &str) -> bool {
    let mut characters = id.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Query syntax error.  `offset` is a UTF-8 byte offset into the original
/// query, which makes it directly usable for source highlighting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    pub offset: usize,
    pub message: String,
}

impl ParseError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "查询字节偏移 {}：{}", self.offset, self.message)
    }
}

impl Error for ParseError {}

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
`"#;

    const SPELL: &str = r#"const spell = {
    "ding": "1:丁6",
    "lin": "1:林2",
    "xiu": "1:休1",
};"#;

    const WORDS: &str = r#"const ws=`丁
林
森林
凡士林
黑魔
魔术`;
"#;

    fn data() -> ChineseData {
        ChineseData::from_sources(FACTORS, SPELL, WORDS).unwrap()
    }

    #[test]
    fn literals_wildcards_and_derivatives_use_scalars() {
        let program = Program::parse("中..", None).unwrap();
        assert!(program.matches("中国人"));
        assert!(!program.matches("中国"));
        let state = program.derive_approx(program.root(), '中');
        assert!(!program.is_dead(&state));
    }

    #[test]
    fn character_classes_alternation_and_intersection_work() {
        assert!(Program::parse("[中人]", None).unwrap().matches("中"));
        assert!(!Program::parse("[^中]", None).unwrap().matches("中"));
        assert!(Program::parse("(中|国)&中", None).unwrap().matches("中"));
        assert!(!Program::parse("(中|国)&中", None).unwrap().matches("国"));
    }

    #[test]
    fn removed_aliases_have_migration_errors() {
        let hash = Program::parse("#", None).unwrap_err();
        assert!(hash.message.contains("请写 ."));
        let quote = Program::parse("\"中国\"", None).unwrap_err();
        assert!(quote.message.contains("使用 (...)"));
        for old_property in ["{plin2}", "{b氵青}", "{h7-9}", "{tA}", "{z#士林}"] {
            let error = Program::parse(old_property, Some(&data())).unwrap_err();
            assert!(error.message.contains("@p(...)"));
        }
    }

    #[test]
    fn quantifiers_keep_standard_postfix_semantics() {
        let program = Program::parse("中{2,3}", None).unwrap();
        assert!(program.matches("中中"));
        assert!(program.matches("中中中"));
        assert!(!program.matches("中"));
        assert!(Program::parse("中+", None).unwrap().matches("中中"));
        assert!(Program::parse("中?", None).unwrap().matches(""));
        let bare_star = Program::parse("*", None).unwrap_err();
        assert!(bare_star.message.contains("至少一个请写 .+"));
    }

    #[test]
    fn anagram_derivative_matches_permutations() {
        let program = Program::parse("<中国>", None).unwrap();
        assert!(program.matches("中国"));
        assert!(program.matches("国中"));
        assert!(!program.matches("中中"));
    }

    #[test]
    fn captures_are_approximate_during_search_and_exact_at_the_end() {
        let program = Program::parse("@t(A)@t(A)", None).unwrap();
        assert!(program.matches_approx("人民"));
        assert!(program.matches("人人"));
        assert!(!program.matches("人民"));
    }

    #[test]
    fn metadata_features_become_explicit_classes() {
        let data = data();
        assert!(Program::parse("@b(木)", Some(&data)).unwrap().matches("林"));
        assert!(
            Program::parse("@h(3-4)", Some(&data))
                .unwrap()
                .matches("林")
        );
        assert!(
            Program::parse("@p(*in2)", Some(&data))
                .unwrap()
                .matches("林")
        );
        assert!(
            Program::parse("@p(ding1)", Some(&data))
                .unwrap()
                .matches("丁")
        );
    }

    #[test]
    fn z_property_resolves_one_dictionary_slot() {
        let data = data();
        assert!(
            Program::parse("@z(.士林)", Some(&data))
                .unwrap()
                .matches("凡")
        );
        assert!(
            Program::parse("@z(黑.)&@z(.术)", Some(&data))
                .unwrap()
                .matches("魔")
        );
        assert!(Program::parse("@z(..术)", Some(&data)).is_err());
    }

    #[test]
    fn malformed_queries_report_errors() {
        assert!(Program::parse("[中", None).is_err());
        assert!(Program::parse("(中国", None).is_err());
        assert!(Program::parse("abc", None).is_err());
        assert!(Program::parse("", None).is_err());
    }
}
