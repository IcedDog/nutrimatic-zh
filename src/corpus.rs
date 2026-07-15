//! Dedicated preparation pipelines for the known Chinese corpus sources.
//!
//! This module intentionally does not provide a generic document cleaner.
//! Every accepted field is tied to a known source schema, so timestamps,
//! usernames, URLs, system prompts, markup annotations and other metadata do
//! not silently become index records.

use crate::chinese::is_han;

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::str::FromStr;

const ERROR_SAMPLE_LIMIT: usize = 20;
const INPUT_BUFFER_SIZE: usize = 1024 * 1024;
const TIEBA_INPUT_LIMIT: u64 = 64 * 1024 * 1024;
const XML_PENDING_LIMIT: usize = 1024 * 1024;
const WIKI_LINK_LIMIT: usize = 1024 * 1024;
const POETRY_TITLE_WEIGHT: u64 = 20;
const POETRY_BODY_WEIGHT: u64 = 10;

const POETRY_COLLECTIONS: &[&str] = &[
    "曹操诗集",
    "楚辞",
    "论语",
    "蒙学",
    "纳兰性德",
    "全唐诗",
    "诗经",
    "水墨唐诗",
    "四书五经",
    "宋词",
    "五代诗词",
    "幽梦影",
    "御定全唐詩",
    "元曲",
];

/// The fixed corpus layouts accepted by [`prepare_source`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    /// ChinesePuzzleTool's `out.js` template-literal word list.
    Wordlist,
    /// `563w_baidubaike.json`, one Baidu Baike object per line.
    BaiduBaike,
    /// A WikiExtractor directory containing extensionless JSONL shards.
    Wikipedia,
    /// A MediaWiki XML export from Moegirl/Mengniang Baike.
    Moegirl,
    /// Bilibili comments in `message,time,timestamp` CSV form.
    Bilibili,
    /// The small `original.json` Tieba/thread array.
    Tieba,
    /// ChatML training objects in `train.jsonl`.
    Train,
    /// The curated JSON collections in the chinese-poetry repository.
    Poetry,
}

impl fmt::Display for SourceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Wordlist => "wordlist",
            Self::BaiduBaike => "baidu-baike",
            Self::Wikipedia => "wikipedia",
            Self::Moegirl => "moegirl",
            Self::Bilibili => "bilibili",
            Self::Tieba => "tieba",
            Self::Train => "train",
            Self::Poetry => "poetry",
        })
    }
}

impl FromStr for SourceKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "wordlist" | "words" | "out.js" => Ok(Self::Wordlist),
            "baidu" | "baidu-baike" | "baidubaike" => Ok(Self::BaiduBaike),
            "wikipedia" | "wiki" | "wiki-zh" => Ok(Self::Wikipedia),
            "moegirl" | "mengniang" => Ok(Self::Moegirl),
            "bilibili" | "bili" => Ok(Self::Bilibili),
            "tieba" | "original" => Ok(Self::Tieba),
            "train" | "chatml" => Ok(Self::Train),
            "poetry" | "chinese-poetry" | "poems" => Ok(Self::Poetry),
            _ => Err(format!("未知语料来源：{value}")),
        }
    }
}

/// A normalized Han-only record and its source-specific frequency weight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightedRecord {
    pub text: String,
    pub weight: u64,
}

/// A bounded parse/encoding error sample for human audit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrepareErrorSample {
    pub source: PathBuf,
    pub line: u64,
    pub message: String,
    pub preview: String,
}

/// Byte-level progress for one fixed-source preparation run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrepareProgress {
    /// Current source file, useful for multi-shard Wikipedia inputs.
    pub source: PathBuf,
    /// One-based current file number.
    pub file_index: usize,
    pub file_count: usize,
    pub file_bytes_read: u64,
    pub file_bytes_total: u64,
    pub bytes_read: u64,
    pub bytes_total: u64,
}

/// Streaming preparation counters. General-source duplicates are aggregated
/// by the external index builder; Poetry retains only compact fingerprints to
/// suppress duplicate editions without retaining full records.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrepareStats {
    pub files_seen: u64,
    pub bytes_read: u64,
    pub physical_lines: u64,
    pub logical_records: u64,
    pub selected_fields: u64,
    pub fields_without_han: u64,
    pub emitted_records: u64,
    pub emitted_han_chars: u64,
    pub emitted_weight: u64,
    pub skipped_records: u64,
    pub parse_errors: u64,
    pub invalid_utf8_lines: u64,
}

/// Audit report for one invocation of [`prepare_source`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrepareReport {
    pub kind: SourceKind,
    pub stats: PrepareStats,
    pub records_by_weight: BTreeMap<u64, u64>,
    pub record_samples: Vec<WeightedRecord>,
    pub error_samples: Vec<PrepareErrorSample>,
}

impl PrepareReport {
    fn new(kind: SourceKind) -> Self {
        Self {
            kind,
            stats: PrepareStats::default(),
            records_by_weight: BTreeMap::new(),
            record_samples: Vec::new(),
            error_samples: Vec::new(),
        }
    }
}

/// Fatal filesystem or output failure.  Malformed source records are recorded
/// in [`PrepareReport`] and skipped so one bad line does not discard a 16 GiB
/// source.
#[derive(Debug)]
pub enum PrepareError {
    InvalidInput(String),
    Io { path: PathBuf, source: io::Error },
    Output(io::Error),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PoetryFingerprint {
    hash_a: u64,
    hash_b: u64,
    han_chars: u64,
}

impl fmt::Display for PrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Io { path, source } => {
                write!(formatter, "无法读取 {}：{source}", path.display())
            }
            Self::Output(source) => write!(formatter, "无法写出语料记录：{source}"),
        }
    }
}

impl Error for PrepareError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::Output(source) => Some(source),
            Self::InvalidInput(_) => None,
        }
    }
}

struct Preparer<F>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    report: PrepareReport,
    emit: F,
    poetry_bodies: HashSet<PoetryFingerprint>,
    poetry_titles: HashSet<(PoetryFingerprint, PoetryFingerprint)>,
}

impl<F> Preparer<F>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    fn new(kind: SourceKind, emit: F) -> Self {
        Self {
            report: PrepareReport::new(kind),
            emit,
            poetry_bodies: HashSet::new(),
            poetry_titles: HashSet::new(),
        }
    }

    fn error(&mut self, source: &Path, line: u64, message: impl Into<String>, preview: &str) {
        self.report.stats.parse_errors += 1;
        if self.report.error_samples.len() < ERROR_SAMPLE_LIMIT {
            self.report.error_samples.push(PrepareErrorSample {
                source: source.to_path_buf(),
                line,
                message: message.into(),
                preview: short_preview(preview, 120),
            });
        }
    }

    fn invalid_utf8(&mut self, source: &Path, line: u64, bytes: &[u8]) {
        self.report.stats.invalid_utf8_lines += 1;
        self.error(
            source,
            line,
            "该行不是严格 UTF-8，已跳过",
            &hex_preview(bytes, 24),
        );
    }

    fn selected_text(&mut self, text: &str, weight: u64) -> Result<(), PrepareError> {
        self.selected_text_inner(text, weight, false)
    }

    fn selected_social_text(&mut self, text: &str, weight: u64) -> Result<(), PrepareError> {
        self.selected_text_inner(text, weight, true)
    }

    fn selected_text_inner(
        &mut self,
        text: &str,
        weight: u64,
        social_annotations: bool,
    ) -> Result<(), PrepareError> {
        self.report.stats.selected_fields += 1;
        let text = if social_annotations {
            strip_reply_prefix(text)
        } else {
            text
        };
        let mut segment = String::new();
        let mut bracket_depth = 0usize;
        let mut found_han = false;

        for character in text.chars() {
            if social_annotations && character == '[' {
                if bracket_depth == 0 {
                    self.finish_segment(&mut segment, weight)?;
                }
                bracket_depth += 1;
                continue;
            }
            if social_annotations && character == ']' && bracket_depth != 0 {
                bracket_depth -= 1;
                continue;
            }
            if social_annotations && bracket_depth != 0 {
                continue;
            }

            if is_han(character) || character == '〇' {
                found_han = true;
                segment.push(character);
            } else {
                self.finish_segment(&mut segment, weight)?;
            }
        }
        self.finish_segment(&mut segment, weight)?;
        if !found_han {
            self.report.stats.fields_without_han += 1;
        }
        Ok(())
    }

    fn finish_segment(&mut self, segment: &mut String, weight: u64) -> Result<(), PrepareError> {
        if segment.is_empty() {
            return Ok(());
        }
        let text = std::mem::take(segment);
        let chars = text.chars().count() as u64;
        let record = WeightedRecord { text, weight };
        if self.report.record_samples.len() < ERROR_SAMPLE_LIMIT {
            self.report.record_samples.push(record.clone());
        }
        (self.emit)(record).map_err(PrepareError::Output)?;
        self.report.stats.emitted_records += 1;
        self.report.stats.emitted_han_chars += chars;
        self.report.stats.emitted_weight = self.report.stats.emitted_weight.saturating_add(weight);
        *self.report.records_by_weight.entry(weight).or_default() += 1;
        Ok(())
    }
}

/// Prepares one known source path. `Wikipedia` accepts a shard or directory,
/// and `Poetry` accepts a chinese-poetry repository root. Other kinds require
/// a regular file. Directory inputs are processed in sorted path order.
pub fn prepare_source<P, F>(
    kind: SourceKind,
    input: P,
    emit: F,
) -> Result<PrepareReport, PrepareError>
where
    P: AsRef<Path>,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    prepare_source_with_progress(kind, input, emit, |_| {})
}

/// Prepares one source while reporting byte progress. Progress callbacks are
/// observational only and do not change parsing or audit results.
pub fn prepare_source_with_progress<P, F, G>(
    kind: SourceKind,
    input: P,
    emit: F,
    mut progress: G,
) -> Result<PrepareReport, PrepareError>
where
    P: AsRef<Path>,
    F: FnMut(WeightedRecord) -> io::Result<()>,
    G: FnMut(PrepareProgress),
{
    let input = input.as_ref();
    let files = source_files(kind, input)?;
    let file_sizes = files
        .iter()
        .map(|path| {
            fs::metadata(path)
                .map(|metadata| metadata.len())
                .map_err(|source| PrepareError::Io {
                    path: path.clone(),
                    source,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total_bytes = file_sizes
        .iter()
        .try_fold(0_u64, |total, &size| total.checked_add(size))
        .ok_or_else(|| PrepareError::InvalidInput("语料总大小超过 u64".to_owned()))?;
    let file_count = files.len();
    let mut preparer = Preparer::new(kind, emit);
    let mut completed_bytes = 0_u64;
    for (file_offset, (path, file_bytes_total)) in files.into_iter().zip(file_sizes).enumerate() {
        let file_index = file_offset + 1;
        preparer.report.stats.files_seen += 1;
        let file = File::open(&path).map_err(|source| PrepareError::Io {
            path: path.clone(),
            source,
        })?;
        progress(PrepareProgress {
            source: path.clone(),
            file_index,
            file_count,
            file_bytes_read: 0,
            file_bytes_total,
            bytes_read: completed_bytes,
            bytes_total: total_bytes,
        });
        let mut reader = ProgressReader {
            inner: file,
            source: &path,
            file_index,
            file_count,
            file_bytes_read: 0,
            file_bytes_total,
            completed_bytes,
            total_bytes,
            progress: &mut progress,
        };
        process_reader(kind, &path, &mut reader, &mut preparer)?;
        completed_bytes = completed_bytes.saturating_add(file_bytes_total);
        progress(PrepareProgress {
            source: path,
            file_index,
            file_count,
            file_bytes_read: file_bytes_total,
            file_bytes_total,
            bytes_read: completed_bytes,
            bytes_total: total_bytes,
        });
    }
    Ok(preparer.report)
}

struct ProgressReader<'a, R, G> {
    inner: R,
    source: &'a Path,
    file_index: usize,
    file_count: usize,
    file_bytes_read: u64,
    file_bytes_total: u64,
    completed_bytes: u64,
    total_bytes: u64,
    progress: &'a mut G,
}

impl<R, G> Read for ProgressReader<'_, R, G>
where
    R: Read,
    G: FnMut(PrepareProgress),
{
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        if read != 0 {
            self.file_bytes_read = self.file_bytes_read.saturating_add(read as u64);
            (self.progress)(PrepareProgress {
                source: self.source.to_path_buf(),
                file_index: self.file_index,
                file_count: self.file_count,
                file_bytes_read: self.file_bytes_read,
                file_bytes_total: self.file_bytes_total,
                bytes_read: self.completed_bytes.saturating_add(self.file_bytes_read),
                bytes_total: self.total_bytes,
            });
        }
        Ok(read)
    }
}

fn source_files(kind: SourceKind, input: &Path) -> Result<Vec<PathBuf>, PrepareError> {
    let metadata = fs::metadata(input).map_err(|source| PrepareError::Io {
        path: input.to_path_buf(),
        source,
    })?;
    if metadata.is_file() {
        if kind == SourceKind::Poetry {
            return Err(PrepareError::InvalidInput(format!(
                "poetry 来源必须传 chinese-poetry 仓库根目录：{}",
                input.display()
            )));
        }
        return Ok(vec![input.to_path_buf()]);
    }
    if !metadata.is_dir() || !matches!(kind, SourceKind::Wikipedia | SourceKind::Poetry) {
        return Err(PrepareError::InvalidInput(format!(
            "{} 来源需要常规文件：{}",
            kind,
            input.display()
        )));
    }

    if kind == SourceKind::Poetry {
        return poetry_source_files(input);
    }

    let mut files = Vec::new();
    collect_files(input, &mut files)?;
    files.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("wiki_"))
    });
    files.sort();
    if files.is_empty() {
        return Err(PrepareError::InvalidInput(format!(
            "Wikipedia 目录中没有 wiki_* 分片：{}",
            input.display()
        )));
    }
    Ok(files)
}

fn poetry_source_files(root: &Path) -> Result<Vec<PathBuf>, PrepareError> {
    let mut files = Vec::new();
    for collection in POETRY_COLLECTIONS {
        let directory = root.join(collection);
        let metadata = fs::metadata(&directory).map_err(|source| PrepareError::Io {
            path: directory.clone(),
            source,
        })?;
        if !metadata.is_dir() {
            return Err(PrepareError::InvalidInput(format!(
                "chinese-poetry 缺少必需目录：{}",
                directory.display()
            )));
        }
        collect_files(&directory, &mut files)?;
    }
    files.retain(|path| is_poetry_source_path(root, path));
    files.sort();
    if files.is_empty() {
        return Err(PrepareError::InvalidInput(format!(
            "chinese-poetry 主数据目录中没有 JSON：{}",
            root.display()
        )));
    }
    Ok(files)
}

fn is_poetry_source_path(root: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    let mut components = relative.components();
    let Some(collection) = components.next() else {
        return false;
    };
    let collection = collection.as_os_str().to_string_lossy();
    if !POETRY_COLLECTIONS.contains(&collection.as_ref()) {
        return false;
    }
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("json"))
    {
        return false;
    }
    if relative.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|name| name.eq_ignore_ascii_case("error"))
    }) {
        return false;
    }
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    !matches!(
        file_name.to_ascii_lowercase().as_str(),
        "authors.tang.json" | "authors.song.json" | "author.song.json" | "authors.json"
    ) && file_name != "表面结构字.json"
}

fn collect_files(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), PrepareError> {
    let mut entries = fs::read_dir(directory)
        .map_err(|source| PrepareError::Io {
            path: directory.to_path_buf(),
            source,
        })?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|source| PrepareError::Io {
                    path: directory.to_path_buf(),
                    source,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    for path in entries {
        let metadata = fs::symlink_metadata(&path).map_err(|source| PrepareError::Io {
            path: path.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            collect_files(&path, output)?;
        } else if metadata.is_file() {
            output.push(path);
        }
    }
    Ok(())
}

fn process_reader<R, F>(
    kind: SourceKind,
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    match kind {
        SourceKind::Wordlist => process_wordlist(source, reader, preparer),
        SourceKind::BaiduBaike => process_baidu(source, reader, preparer),
        SourceKind::Wikipedia => process_wikipedia(source, reader, preparer),
        SourceKind::Moegirl => process_moegirl(source, reader, preparer),
        SourceKind::Bilibili => process_bilibili(source, reader, preparer),
        SourceKind::Tieba => process_tieba(source, reader, preparer),
        SourceKind::Train => process_train(source, reader, preparer),
        SourceKind::Poetry => process_poetry(source, reader, preparer),
    }
}

struct StrictLines<R: BufRead> {
    reader: R,
    bytes: Vec<u8>,
    line: u64,
    bytes_read: u64,
}

enum StrictLine {
    Text { number: u64, text: String },
    Invalid { number: u64, bytes: Vec<u8> },
}

impl<R: BufRead> StrictLines<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            bytes: Vec::new(),
            line: 0,
            bytes_read: 0,
        }
    }

    fn next_line(&mut self) -> io::Result<Option<StrictLine>> {
        self.bytes.clear();
        let read = self.reader.read_until(b'\n', &mut self.bytes)?;
        self.bytes_read += read as u64;
        if read == 0 {
            return Ok(None);
        }
        self.line += 1;
        while matches!(self.bytes.last(), Some(b'\n' | b'\r')) {
            self.bytes.pop();
        }
        Ok(Some(match std::str::from_utf8(&self.bytes) {
            Ok(text) => StrictLine::Text {
                number: self.line,
                text: text.to_owned(),
            },
            Err(_) => StrictLine::Invalid {
                number: self.line,
                bytes: self.bytes.clone(),
            },
        }))
    }
}

fn line_io_error(path: &Path, source: io::Error) -> PrepareError {
    PrepareError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn short_preview(value: &str, limit: usize) -> String {
    let mut output = String::new();
    for character in value.chars().take(limit) {
        if character.is_whitespace() {
            if !output.ends_with(' ') {
                output.push(' ');
            }
        } else {
            output.push(character);
        }
    }
    if value.chars().count() > limit {
        output.push('…');
    }
    output
}

fn hex_preview(bytes: &[u8], limit: usize) -> String {
    bytes
        .iter()
        .take(limit)
        .map(|byte| format!("{byte:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_reply_prefix(mut text: &str) -> &str {
    text = text.trim_start();
    for _ in 0..2 {
        let candidate = if let Some(rest) = text.strip_prefix("回复") {
            let rest = rest.trim_start();
            rest.strip_prefix('@').map(str::trim_start)
        } else {
            text.strip_prefix('@').map(str::trim_start)
        };
        let Some(after_at) = candidate else {
            break;
        };

        if let Some((offset, separator)) = after_at
            .char_indices()
            .find(|(_, character)| matches!(character, ':' | '：'))
        {
            text = after_at[offset + separator.len_utf8()..].trim_start();
            continue;
        }
        if let Some((offset, separator)) = after_at
            .char_indices()
            .find(|(_, character)| character.is_whitespace())
        {
            text = after_at[offset + separator.len_utf8()..].trim_start();
            continue;
        }
        break;
    }
    text
}

#[derive(Clone, Debug)]
enum JsonValue {
    Null,
    Bool,
    Number,
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?
            .iter()
            .find_map(|(candidate, value)| (candidate == key).then_some(value))
    }

    fn get_any(&self, keys: &[&str]) -> Option<&JsonValue> {
        keys.iter().find_map(|key| self.get(key))
    }
}

#[derive(Clone, Debug)]
struct JsonError {
    offset: usize,
    message: String,
}

impl fmt::Display for JsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "JSON 字节 {}：{}", self.offset, self.message)
    }
}

struct JsonParser<'a> {
    input: &'a str,
    bytes: &'a [u8],
    position: usize,
}

impl<'a> JsonParser<'a> {
    fn parse(input: &'a str) -> Result<JsonValue, JsonError> {
        let mut parser = Self {
            input,
            bytes: input.as_bytes(),
            position: 0,
        };
        parser.skip_whitespace();
        let value = parser.parse_value(0)?;
        parser.skip_whitespace();
        if parser.position != parser.bytes.len() {
            return Err(parser.error("顶层 JSON 值后还有多余内容"));
        }
        Ok(value)
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        if depth > 128 {
            return Err(self.error("嵌套层级超过 128"));
        }
        self.skip_whitespace();
        match self.bytes.get(self.position).copied() {
            Some(b'n') => {
                self.literal(b"null")?;
                Ok(JsonValue::Null)
            }
            Some(b't') => {
                self.literal(b"true")?;
                Ok(JsonValue::Bool)
            }
            Some(b'f') => {
                self.literal(b"false")?;
                Ok(JsonValue::Bool)
            }
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'{') => self.parse_object(depth + 1),
            Some(b'-' | b'0'..=b'9') => {
                self.parse_number()?;
                Ok(JsonValue::Number)
            }
            Some(_) => Err(self.error("无法识别的 JSON 值")),
            None => Err(self.error("意外到达文件末尾")),
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.position += 1;
        self.skip_whitespace();
        let mut values = Vec::new();
        if self.consume(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            values.push(self.parse_value(depth)?);
            self.skip_whitespace();
            if self.consume(b']') {
                break;
            }
            if !self.consume(b',') {
                return Err(self.error("数组元素之间缺少逗号或结束方括号"));
            }
        }
        Ok(JsonValue::Array(values))
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.position += 1;
        self.skip_whitespace();
        let mut fields = Vec::new();
        if self.consume(b'}') {
            return Ok(JsonValue::Object(fields));
        }
        loop {
            self.skip_whitespace();
            if self.bytes.get(self.position) != Some(&b'"') {
                return Err(self.error("对象键必须是字符串"));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if !self.consume(b':') {
                return Err(self.error("对象键后缺少冒号"));
            }
            fields.push((key, self.parse_value(depth)?));
            self.skip_whitespace();
            if self.consume(b'}') {
                break;
            }
            if !self.consume(b',') {
                return Err(self.error("对象字段之间缺少逗号或结束花括号"));
            }
        }
        Ok(JsonValue::Object(fields))
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        debug_assert_eq!(self.bytes.get(self.position), Some(&b'"'));
        self.position += 1;
        let mut output = String::new();
        while self.position < self.bytes.len() {
            match self.bytes[self.position] {
                b'"' => {
                    self.position += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.position += 1;
                    let escape = self
                        .bytes
                        .get(self.position)
                        .copied()
                        .ok_or_else(|| self.error("字符串转义不完整"))?;
                    match escape {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{0008}'),
                        b'f' => output.push('\u{000C}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => {
                            let first = self.parse_hex_escape(self.position + 1)?;
                            self.position += 4;
                            if (0xD800..=0xDBFF).contains(&first) {
                                if self.bytes.get(self.position + 1) != Some(&b'\\')
                                    || self.bytes.get(self.position + 2) != Some(&b'u')
                                {
                                    return Err(self.error("高代理项后缺少低代理项"));
                                }
                                let low = self.parse_hex_escape(self.position + 3)?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(self.error("Unicode 低代理项无效"));
                                }
                                let scalar = 0x1_0000
                                    + (((first - 0xD800) as u32) << 10)
                                    + (low - 0xDC00) as u32;
                                output.push(
                                    char::from_u32(scalar)
                                        .ok_or_else(|| self.error("代理项不是 Unicode 标量"))?,
                                );
                                self.position += 6;
                            } else if (0xDC00..=0xDFFF).contains(&first) {
                                return Err(self.error("出现孤立低代理项"));
                            } else {
                                output.push(
                                    char::from_u32(first as u32)
                                        .ok_or_else(|| self.error("转义不是 Unicode 标量"))?,
                                );
                            }
                        }
                        _ => return Err(self.error("未知字符串转义")),
                    }
                    self.position += 1;
                }
                byte if byte < 0x20 => return Err(self.error("字符串含未转义控制字符")),
                byte if byte.is_ascii() => {
                    output.push(byte as char);
                    self.position += 1;
                }
                _ => {
                    let character = self.input[self.position..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.error("UTF-8 字符不完整"))?;
                    output.push(character);
                    self.position += character.len_utf8();
                }
            }
        }
        Err(self.error("字符串没有闭合"))
    }

    fn parse_hex_escape(&self, start: usize) -> Result<u16, JsonError> {
        let end = start
            .checked_add(4)
            .ok_or_else(|| self.error("Unicode 转义位置溢出"))?;
        let digits = self
            .bytes
            .get(start..end)
            .ok_or_else(|| self.error("Unicode 转义不完整"))?;
        let mut value = 0u16;
        for &digit in digits {
            let nibble = match digit {
                b'0'..=b'9' => digit - b'0',
                b'a'..=b'f' => digit - b'a' + 10,
                b'A'..=b'F' => digit - b'A' + 10,
                _ => return Err(self.error("Unicode 转义含非十六进制字符")),
            };
            value = (value << 4) | nibble as u16;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<(), JsonError> {
        if self.consume(b'-') && self.position == self.bytes.len() {
            return Err(self.error("负号后缺少数字"));
        }
        if self.consume(b'0') {
            if self
                .bytes
                .get(self.position)
                .is_some_and(u8::is_ascii_digit)
            {
                return Err(self.error("数字不能含前导零"));
            }
        } else {
            self.consume_digits("数字缺少整数部分")?;
        }
        if self.consume(b'.') {
            self.consume_digits("小数点后缺少数字")?;
        }
        if matches!(self.bytes.get(self.position), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.bytes.get(self.position), Some(b'+' | b'-')) {
                self.position += 1;
            }
            self.consume_digits("指数部分缺少数字")?;
        }
        Ok(())
    }

    fn consume_digits(&mut self, message: &'static str) -> Result<(), JsonError> {
        let start = self.position;
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_digit)
        {
            self.position += 1;
        }
        if self.position == start {
            Err(self.error(message))
        } else {
            Ok(())
        }
    }

    fn literal(&mut self, literal: &[u8]) -> Result<(), JsonError> {
        if self.bytes[self.position..].starts_with(literal) {
            self.position += literal.len();
            Ok(())
        } else {
            Err(self.error("JSON 字面量拼写错误"))
        }
    }

    fn skip_whitespace(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.bytes.get(self.position) == Some(&expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn error(&self, message: impl Into<String>) -> JsonError {
        JsonError {
            offset: self.position,
            message: message.into(),
        }
    }
}

fn process_wordlist<R, F>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    let mut lines = StrictLines::new(BufReader::with_capacity(INPUT_BUFFER_SIZE, reader));
    let mut opened = false;
    let mut closed = false;
    while let Some(line) = lines
        .next_line()
        .map_err(|error| line_io_error(source, error))?
    {
        preparer.report.stats.physical_lines += 1;
        match line {
            StrictLine::Invalid { number, bytes } => preparer.invalid_utf8(source, number, &bytes),
            StrictLine::Text { number, text } => {
                let mut content = text.as_str();
                if !opened {
                    let Some(backtick) = content.find('`') else {
                        if !content.trim().is_empty() {
                            preparer.error(source, number, "词表模板字符串前出现未知内容", content);
                        }
                        continue;
                    };
                    opened = true;
                    content = &content[backtick + 1..];
                }
                if let Some(backtick) = content.find('`') {
                    let trailing = content[backtick + 1..].trim();
                    content = &content[..backtick];
                    closed = true;
                    if !trailing.is_empty() && trailing != ";" {
                        preparer.error(source, number, "词表结束反引号后含未知内容", trailing);
                    }
                }
                if !content.trim().is_empty() {
                    preparer.report.stats.logical_records += 1;
                    preparer.selected_text(content.trim(), 20)?;
                }
                if closed {
                    break;
                }
            }
        }
    }
    preparer.report.stats.bytes_read += lines.bytes_read;
    if !opened {
        preparer.error(source, 0, "找不到 out.js 模板字符串起始反引号", "");
    } else if !closed {
        preparer.error(source, 0, "找不到 out.js 模板字符串结束反引号", "");
    }
    Ok(())
}

fn process_baidu<R, F>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    process_json_lines(source, reader, preparer, |value, preparer| {
        let Some(_) = value.as_object() else {
            return Err("百度百科 JSONL 顶层必须是对象".to_owned());
        };
        emit_optional_string(value.get("title"), 10, preparer)?;
        emit_optional_string(value.get("summary"), 1, preparer)?;

        if let Some(sections) = value.get("sections") {
            let Some(sections) = sections.as_array() else {
                return Err("sections 必须是数组或缺失".to_owned());
            };
            for section in sections {
                let Some(_) = section.as_object() else {
                    return Err("sections 元素必须是对象".to_owned());
                };
                emit_optional_string(section.get("title"), 5, preparer)?;
                emit_optional_string(section.get("content"), 1, preparer)?;
            }
        }

        if let Some(tags) = value.get("tags") {
            let Some(tags) = tags.as_array() else {
                return Err("tags 必须是数组或缺失".to_owned());
            };
            for tag in tags {
                let Some(tag) = tag.as_str() else {
                    return Err("tags 元素必须是字符串".to_owned());
                };
                preparer
                    .selected_text(tag, 5)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    })
}

fn process_wikipedia<R, F>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    process_json_lines(source, reader, preparer, |value, preparer| {
        let Some(_) = value.as_object() else {
            return Err("Wikipedia JSONL 顶层必须是对象".to_owned());
        };
        let title = value
            .get("title")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| "Wikipedia 记录缺少字符串 title".to_owned())?;
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| "Wikipedia 记录缺少字符串 text".to_owned())?;
        preparer
            .selected_text(title, 10)
            .map_err(|error| error.to_string())?;
        preparer
            .selected_text(text, 1)
            .map_err(|error| error.to_string())?;
        Ok(())
    })
}

fn process_train<R, F>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    process_json_lines(source, reader, preparer, |value, preparer| {
        let text = value
            .get("text")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| "训练记录缺少字符串 text".to_owned())?;
        let count = emit_chatml_turns(text, preparer).map_err(|error| error.to_string())?;
        if count == 0 {
            preparer.report.stats.skipped_records += 1;
        }
        Ok(())
    })
}

fn process_json_lines<R, F, H>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
    mut handle: H,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
    H: FnMut(&JsonValue, &mut Preparer<F>) -> Result<(), String>,
{
    let mut lines = StrictLines::new(BufReader::with_capacity(INPUT_BUFFER_SIZE, reader));
    while let Some(line) = lines
        .next_line()
        .map_err(|error| line_io_error(source, error))?
    {
        preparer.report.stats.physical_lines += 1;
        match line {
            StrictLine::Invalid { number, bytes } => preparer.invalid_utf8(source, number, &bytes),
            StrictLine::Text { number, text } => {
                if text.trim().is_empty() {
                    continue;
                }
                match JsonParser::parse(&text) {
                    Ok(value) => match handle(&value, preparer) {
                        Ok(()) => preparer.report.stats.logical_records += 1,
                        Err(message) if message.starts_with("无法写出语料记录：") => {
                            return Err(PrepareError::Output(io::Error::other(message)));
                        }
                        Err(message) => {
                            preparer.report.stats.skipped_records += 1;
                            preparer.error(source, number, message, &text);
                        }
                    },
                    Err(error) => {
                        preparer.report.stats.skipped_records += 1;
                        preparer.error(source, number, error.to_string(), &text);
                    }
                }
            }
        }
    }
    preparer.report.stats.bytes_read += lines.bytes_read;
    Ok(())
}

fn emit_optional_string<F>(
    value: Option<&JsonValue>,
    weight: u64,
    preparer: &mut Preparer<F>,
) -> Result<(), String>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    match value {
        None | Some(JsonValue::Null) => Ok(()),
        Some(JsonValue::String(value)) => preparer
            .selected_text(value, weight)
            .map_err(|error| error.to_string()),
        Some(_) => Err("已选字段必须是字符串、null 或缺失".to_owned()),
    }
}

fn emit_chatml_turns<F>(text: &str, preparer: &mut Preparer<F>) -> Result<u64, PrepareError>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    const START: &str = "<|im_start|>";
    const END: &str = "<|im_end|>";
    let mut remaining = text;
    let mut emitted_turns = 0u64;
    while let Some(start) = remaining.find(START) {
        remaining = &remaining[start + START.len()..];
        let Some(end) = remaining.find(END) else {
            break;
        };
        let block = &remaining[..end];
        remaining = &remaining[end + END.len()..];
        let Some((role, body)) = block.split_once('\n') else {
            continue;
        };
        if matches!(role.trim(), "user" | "assistant") {
            preparer.selected_text(body, 1)?;
            emitted_turns += 1;
        }
    }
    Ok(emitted_turns)
}

fn process_tieba<R, F>(
    source: &Path,
    mut reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    let mut bytes = Vec::new();
    reader
        .by_ref()
        .take(TIEBA_INPUT_LIMIT + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| line_io_error(source, error))?;
    preparer.report.stats.bytes_read += bytes.len() as u64;
    if bytes.len() as u64 > TIEBA_INPUT_LIMIT {
        preparer.error(
            source,
            0,
            format!(
                "Tieba JSON 超过 {} MiB 上限",
                TIEBA_INPUT_LIMIT / 1024 / 1024
            ),
            "",
        );
        preparer.report.stats.skipped_records += 1;
        return Ok(());
    }
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(_) => {
            preparer.invalid_utf8(source, 0, &bytes);
            preparer.report.stats.skipped_records += 1;
            return Ok(());
        }
    };
    let root = match JsonParser::parse(text) {
        Ok(value) => value,
        Err(error) => {
            preparer.error(source, 0, error.to_string(), text);
            preparer.report.stats.skipped_records += 1;
            return Ok(());
        }
    };
    let threads: &[JsonValue] = if let Some(values) = root.as_array() {
        values
    } else if let Some(values) = root
        .get_any(&["data", "threads"])
        .and_then(JsonValue::as_array)
    {
        values
    } else {
        preparer.error(
            source,
            0,
            "Tieba 顶层必须是数组或含 data/threads 数组",
            text,
        );
        preparer.report.stats.skipped_records += 1;
        return Ok(());
    };

    for thread in threads {
        if let Err(message) = emit_tieba_thread(thread, preparer) {
            preparer.report.stats.skipped_records += 1;
            preparer.error(source, 0, message, "");
        } else {
            preparer.report.stats.logical_records += 1;
        }
    }
    Ok(())
}

fn emit_tieba_thread<F>(thread: &JsonValue, preparer: &mut Preparer<F>) -> Result<(), String>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    let Some(_) = thread.as_object() else {
        return Err("Tieba 数组元素必须是对象".to_owned());
    };
    if let Some(title) = thread.get_any(&["标题", "title"]) {
        emit_text_value(title, 5, preparer)?;
    }
    if let Some(content) = thread.get_any(&["楼主内容", "content"]) {
        emit_social_text_value(content, 1, preparer)?;
    }
    if let Some(replies) = thread.get_any(&["回复列表", "replies"]) {
        emit_reply_value(replies, preparer)?;
    }
    Ok(())
}

fn emit_text_value<F>(
    value: &JsonValue,
    weight: u64,
    preparer: &mut Preparer<F>,
) -> Result<(), String>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    match value {
        JsonValue::Null => Ok(()),
        JsonValue::String(text) => preparer
            .selected_text(text, weight)
            .map_err(|error| error.to_string()),
        JsonValue::Array(values) => {
            for value in values {
                emit_text_value(value, weight, preparer)?;
            }
            Ok(())
        }
        _ => Err("文本字段必须是字符串、字符串数组或 null".to_owned()),
    }
}

fn process_poetry<R, F>(
    source: &Path,
    mut reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| line_io_error(source, error))?;
    preparer.report.stats.bytes_read += bytes.len() as u64;
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(_) => {
            preparer.invalid_utf8(source, 0, &bytes);
            preparer.report.stats.skipped_records += 1;
            return Ok(());
        }
    };
    preparer.report.stats.physical_lines += text.lines().count() as u64;
    let root = match JsonParser::parse(text) {
        Ok(value) => value,
        Err(error) => {
            preparer.error(source, 0, error.to_string(), text);
            preparer.report.stats.skipped_records += 1;
            return Ok(());
        }
    };
    emit_poetry_value(&root, preparer)
}

fn emit_poetry_value<F>(value: &JsonValue, preparer: &mut Preparer<F>) -> Result<(), PrepareError>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    match value {
        JsonValue::Array(values) => {
            for value in values {
                emit_poetry_value(value, preparer)?;
            }
        }
        JsonValue::Object(fields) => {
            let mut primary = Vec::new();
            let mut supplementary = Vec::new();
            for (key, value) in fields {
                if matches!(key.as_str(), "paragraphs" | "para" | "content") {
                    collect_direct_poetry_strings(value, &mut primary);
                } else if matches!(key.as_str(), "preface" | "biography") {
                    collect_direct_poetry_strings(value, &mut supplementary);
                }
            }

            let primary_fingerprint = poetry_fingerprint(&primary);
            let supplementary_fingerprint = poetry_fingerprint(&supplementary);
            if let Some(identity) = primary_fingerprint.or(supplementary_fingerprint) {
                preparer.report.stats.logical_records += 1;
                for (key, value) in fields {
                    if matches!(
                        key.as_str(),
                        "title" | "chapter" | "section" | "subchapter" | "rhythmic"
                    ) {
                        let mut titles = Vec::new();
                        collect_direct_poetry_strings(value, &mut titles);
                        for title in titles {
                            if let Some(title_fingerprint) = poetry_fingerprint(&[title])
                                && preparer.poetry_titles.insert((title_fingerprint, identity))
                            {
                                preparer.selected_text(title, POETRY_TITLE_WEIGHT)?;
                            }
                        }
                    }
                }
            }

            if let Some(fingerprint) = primary_fingerprint
                && preparer.poetry_bodies.insert(fingerprint)
            {
                for text in primary {
                    preparer.selected_text(text, POETRY_BODY_WEIGHT)?;
                }
            }
            if let Some(fingerprint) = supplementary_fingerprint
                && preparer.poetry_bodies.insert(fingerprint)
            {
                for text in supplementary {
                    preparer.selected_text(text, POETRY_BODY_WEIGHT)?;
                }
            }

            for (_, value) in fields {
                if matches!(value, JsonValue::Array(_) | JsonValue::Object(_)) {
                    emit_poetry_value(value, preparer)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn collect_direct_poetry_strings<'a>(value: &'a JsonValue, output: &mut Vec<&'a str>) {
    match value {
        JsonValue::String(text) => output.push(text),
        JsonValue::Array(values) => {
            for value in values {
                if !matches!(value, JsonValue::Object(_)) {
                    collect_direct_poetry_strings(value, output);
                }
            }
        }
        _ => {}
    }
}

fn poetry_fingerprint(texts: &[&str]) -> Option<PoetryFingerprint> {
    let mut hash_a = 0xcbf2_9ce4_8422_2325_u64;
    let mut hash_b = 0x9e37_79b9_7f4a_7c15_u64;
    let mut han_chars = 0_u64;
    for text in texts {
        for character in text
            .chars()
            .filter(|&character| is_han(character) || character == '〇')
        {
            let scalar = character as u32 as u64;
            hash_a ^= scalar;
            hash_a = hash_a.wrapping_mul(0x0000_0100_0000_01b3);
            hash_b ^= scalar.wrapping_add(0x517c_c1b7_2722_0a95);
            hash_b = hash_b.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
            han_chars += 1;
        }
    }
    (han_chars != 0).then_some(PoetryFingerprint {
        hash_a,
        hash_b,
        han_chars,
    })
}

fn emit_social_text_value<F>(
    value: &JsonValue,
    weight: u64,
    preparer: &mut Preparer<F>,
) -> Result<(), String>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    match value {
        JsonValue::Null => Ok(()),
        JsonValue::String(text) => preparer
            .selected_social_text(text, weight)
            .map_err(|error| error.to_string()),
        JsonValue::Array(values) => {
            for value in values {
                emit_social_text_value(value, weight, preparer)?;
            }
            Ok(())
        }
        _ => Err("社交文本字段必须是字符串、字符串数组或 null".to_owned()),
    }
}

fn emit_reply_value<F>(value: &JsonValue, preparer: &mut Preparer<F>) -> Result<(), String>
where
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    match value {
        JsonValue::Null => Ok(()),
        JsonValue::String(text) => preparer
            .selected_social_text(text, 1)
            .map_err(|error| error.to_string()),
        JsonValue::Array(values) => {
            for value in values {
                emit_reply_value(value, preparer)?;
            }
            Ok(())
        }
        JsonValue::Object(_) => {
            if let Some(content) = value.get_any(&["回复内容", "内容", "content", "text"]) {
                emit_social_text_value(content, 1, preparer)?;
            }
            if let Some(nested) = value.get_any(&["回复列表", "replies"]) {
                emit_reply_value(nested, preparer)?;
            }
            Ok(())
        }
        _ => Err("回复必须是字符串、对象、数组或 null".to_owned()),
    }
}

struct CsvParser {
    row: Vec<String>,
    field: String,
    in_quotes: bool,
    quote_closed: bool,
    at_field_start: bool,
    malformed: u64,
}

impl CsvParser {
    fn new() -> Self {
        Self {
            row: Vec::new(),
            field: String::new(),
            in_quotes: false,
            quote_closed: false,
            at_field_start: true,
            malformed: 0,
        }
    }

    fn feed<E>(
        &mut self,
        input: &str,
        mut emit: impl FnMut(Vec<String>) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut characters = input.chars().peekable();
        while let Some(character) = characters.next() {
            if self.in_quotes {
                if character == '"' {
                    if characters.peek() == Some(&'"') {
                        characters.next();
                        self.field.push('"');
                    } else {
                        self.in_quotes = false;
                        self.quote_closed = true;
                    }
                } else if character == '\r' && characters.peek() == Some(&'\n') {
                    // Normalize embedded CRLF to one newline.
                } else {
                    self.field.push(character);
                }
                continue;
            }

            match character {
                '"' if self.at_field_start => {
                    self.in_quotes = true;
                    self.at_field_start = false;
                }
                ',' => self.finish_field(),
                '\n' => self.finish_row(&mut emit)?,
                '\r' if characters.peek() == Some(&'\n') => {}
                '\r' => self.finish_row(&mut emit)?,
                whitespace if self.quote_closed && whitespace.is_whitespace() => {}
                other => {
                    if self.quote_closed || other == '"' {
                        self.malformed += 1;
                    }
                    self.field.push(other);
                    self.at_field_start = false;
                }
            }
        }
        Ok(())
    }

    fn finish<E>(&mut self, mut emit: impl FnMut(Vec<String>) -> Result<(), E>) -> Result<bool, E> {
        if self.in_quotes {
            self.in_quotes = false;
            self.malformed += 1;
        }
        if !self.field.is_empty() || !self.row.is_empty() {
            self.finish_row(&mut emit)?;
        }
        Ok(self.malformed != 0)
    }

    fn reset_record(&mut self) {
        self.row.clear();
        self.field.clear();
        self.in_quotes = false;
        self.quote_closed = false;
        self.at_field_start = true;
        self.malformed += 1;
    }

    fn finish_field(&mut self) {
        self.row.push(std::mem::take(&mut self.field));
        self.quote_closed = false;
        self.at_field_start = true;
    }

    fn finish_row<E>(
        &mut self,
        emit: &mut impl FnMut(Vec<String>) -> Result<(), E>,
    ) -> Result<(), E> {
        self.finish_field();
        let row = std::mem::take(&mut self.row);
        if row.iter().any(|field| !field.is_empty()) {
            emit(row)?;
        }
        Ok(())
    }
}

fn process_bilibili<R, F>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    let mut lines = StrictLines::new(BufReader::with_capacity(INPUT_BUFFER_SIZE, reader));
    let mut parser = CsvParser::new();
    let mut message_column = None;
    let mut logical_row = 0u64;

    while let Some(line) = lines
        .next_line()
        .map_err(|error| line_io_error(source, error))?
    {
        preparer.report.stats.physical_lines += 1;
        match line {
            StrictLine::Invalid { number, bytes } => {
                preparer.invalid_utf8(source, number, &bytes);
                parser.reset_record();
            }
            StrictLine::Text { number, mut text } => {
                text.push('\n');
                parser.feed(&text, |row| {
                    logical_row += 1;
                    if message_column.is_none() {
                        message_column = row.iter().position(|field| {
                            field
                                .trim_start_matches('\u{FEFF}')
                                .trim()
                                .eq_ignore_ascii_case("message")
                        });
                        if message_column.is_none() {
                            preparer.error(
                                source,
                                number,
                                "Bilibili CSV 表头缺少 message 列",
                                &row.join(","),
                            );
                        }
                        return Ok(());
                    }
                    let column = message_column.expect("checked above");
                    let Some(message) = row.get(column) else {
                        preparer.report.stats.skipped_records += 1;
                        preparer.error(
                            source,
                            number,
                            "Bilibili CSV 记录列数少于表头",
                            &row.join(","),
                        );
                        return Ok(());
                    };
                    preparer.report.stats.logical_records += 1;
                    preparer.selected_social_text(message, 1)
                })?;
            }
        }
    }
    let malformed = parser.finish(|row| {
        logical_row += 1;
        if let Some(column) = message_column
            && let Some(message) = row.get(column)
        {
            preparer.report.stats.logical_records += 1;
            preparer.selected_social_text(message, 1)?;
        }
        Ok(())
    })?;
    preparer.report.stats.bytes_read += lines.bytes_read;
    if malformed {
        preparer.error(
            source,
            preparer.report.stats.physical_lines,
            "Bilibili CSV 存在未闭合或位置错误的引号",
            "",
        );
    }
    if logical_row == 0 {
        preparer.error(source, 0, "Bilibili CSV 为空", "");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum XmlField {
    Title,
    Namespace,
    Text,
}

struct MoegirlXml {
    pending: String,
    active: Option<XmlField>,
    in_page: bool,
    title: String,
    namespace_text: String,
    namespace: Option<i32>,
    redirect: bool,
    title_emitted: bool,
    wiki: WikiStripper,
}

impl MoegirlXml {
    fn new() -> Self {
        Self {
            pending: String::new(),
            active: None,
            in_page: false,
            title: String::new(),
            namespace_text: String::new(),
            namespace: None,
            redirect: false,
            title_emitted: false,
            wiki: WikiStripper::new(),
        }
    }

    fn feed<F>(
        &mut self,
        source: &Path,
        line: u64,
        chunk: &str,
        preparer: &mut Preparer<F>,
    ) -> Result<(), PrepareError>
    where
        F: FnMut(WeightedRecord) -> io::Result<()>,
    {
        self.pending.push_str(chunk);
        let input = std::mem::take(&mut self.pending);
        let bytes = input.as_bytes();
        let mut position = 0usize;

        while position < bytes.len() {
            let Some(relative) = bytes[position..].iter().position(|&byte| byte == b'<') else {
                self.handle_text(source, &input[position..], preparer)?;
                position = bytes.len();
                break;
            };
            let tag_start = position + relative;
            self.handle_text(source, &input[position..tag_start], preparer)?;

            if bytes[tag_start..].starts_with(b"<!--") {
                let Some(end) = find_bytes(bytes, tag_start + 4, b"-->") else {
                    position = tag_start;
                    break;
                };
                position = end + 3;
                continue;
            }
            if bytes[tag_start..].starts_with(b"<![CDATA[") {
                let start = tag_start + 9;
                let Some(end) = find_bytes(bytes, start, b"]]>") else {
                    position = tag_start;
                    break;
                };
                self.handle_text(source, &input[start..end], preparer)?;
                position = end + 3;
                continue;
            }
            if bytes[tag_start..].starts_with(b"<?") {
                let Some(end) = find_bytes(bytes, tag_start + 2, b"?>") else {
                    position = tag_start;
                    break;
                };
                position = end + 2;
                continue;
            }

            let Some(tag_end) = find_tag_end(&input, tag_start) else {
                position = tag_start;
                break;
            };
            self.handle_tag(source, line, &input[tag_start + 1..tag_end], preparer)?;
            position = tag_end + 1;
        }

        if position < input.len() {
            self.pending.push_str(&input[position..]);
            if self.pending.len() > XML_PENDING_LIMIT {
                preparer.error(
                    source,
                    line,
                    "XML 未闭合标签/注释超过 1 MiB，已重置解析状态",
                    &self.pending,
                );
                self.pending.clear();
                self.active = None;
            }
        }
        Ok(())
    }

    fn handle_tag<F>(
        &mut self,
        source: &Path,
        line: u64,
        tag: &str,
        preparer: &mut Preparer<F>,
    ) -> Result<(), PrepareError>
    where
        F: FnMut(WeightedRecord) -> io::Result<()>,
    {
        let (closing, name, self_closing) = tag_descriptor(tag);
        if name.is_empty() || name.starts_with('!') {
            return Ok(());
        }
        if closing {
            match name.as_str() {
                "title" if self.active == Some(XmlField::Title) => self.active = None,
                "ns" if self.active == Some(XmlField::Namespace) => {
                    self.active = None;
                    self.namespace = match self.namespace_text.trim().parse::<i32>() {
                        Ok(namespace) => Some(namespace),
                        Err(_) => {
                            preparer.error(
                                source,
                                line,
                                "MediaWiki ns 不是整数",
                                &self.namespace_text,
                            );
                            None
                        }
                    };
                }
                "text" if self.active == Some(XmlField::Text) => self.active = None,
                "page" => {
                    self.emit_title(preparer)?;
                    if self.namespace.is_some() {
                        preparer.report.stats.logical_records += 1;
                    }
                    self.reset_page();
                }
                _ => {}
            }
            return Ok(());
        }

        match name.as_str() {
            "page" => {
                self.reset_page();
                self.in_page = true;
            }
            "title" if self.in_page => self.active = (!self_closing).then_some(XmlField::Title),
            "ns" if self.in_page => self.active = (!self_closing).then_some(XmlField::Namespace),
            "redirect" if self.in_page => self.redirect = true,
            "text" if self.in_page => {
                self.wiki.reset();
                self.active = (!self_closing).then_some(XmlField::Text);
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_text<F>(
        &mut self,
        source: &Path,
        raw: &str,
        preparer: &mut Preparer<F>,
    ) -> Result<(), PrepareError>
    where
        F: FnMut(WeightedRecord) -> io::Result<()>,
    {
        if raw.is_empty() {
            return Ok(());
        }
        match self.active {
            Some(XmlField::Title) => self.title.push_str(&decode_xml_entities(raw)),
            Some(XmlField::Namespace) => self.namespace_text.push_str(raw),
            Some(XmlField::Text) if self.namespace == Some(0) && !self.redirect => {
                self.emit_title(preparer)?;
                let decoded = decode_xml_entities(raw);
                let text = self.wiki.feed(&decoded);
                if !text.trim().is_empty() {
                    preparer.selected_text(&text, 1)?;
                }
                if self.wiki.take_overflowed() {
                    preparer.error(source, 0, "MediaWiki 内部链接超过 1 MiB，已丢弃", "");
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn emit_title<F>(&mut self, preparer: &mut Preparer<F>) -> Result<(), PrepareError>
    where
        F: FnMut(WeightedRecord) -> io::Result<()>,
    {
        if !self.title_emitted
            && self.namespace == Some(0)
            && !self.redirect
            && !self.title.trim().is_empty()
        {
            preparer.selected_text(&self.title, 10)?;
            self.title_emitted = true;
        }
        Ok(())
    }

    fn reset_page(&mut self) {
        self.active = None;
        self.in_page = false;
        self.title.clear();
        self.namespace_text.clear();
        self.namespace = None;
        self.redirect = false;
        self.title_emitted = false;
        self.wiki.reset();
    }

    fn finish<F>(
        &mut self,
        source: &Path,
        line: u64,
        preparer: &mut Preparer<F>,
    ) -> Result<(), PrepareError>
    where
        F: FnMut(WeightedRecord) -> io::Result<()>,
    {
        if !self.pending.trim().is_empty() {
            preparer.error(source, line, "XML 文件末尾存在未闭合结构", &self.pending);
        }
        if self.in_page {
            self.emit_title(preparer)?;
            preparer.error(source, line, "XML 文件末尾 page 未闭合", "");
        }
        Ok(())
    }
}

fn process_moegirl<R, F>(
    source: &Path,
    reader: R,
    preparer: &mut Preparer<F>,
) -> Result<(), PrepareError>
where
    R: Read,
    F: FnMut(WeightedRecord) -> io::Result<()>,
{
    let mut lines = StrictLines::new(BufReader::with_capacity(INPUT_BUFFER_SIZE, reader));
    let mut xml = MoegirlXml::new();
    while let Some(line) = lines
        .next_line()
        .map_err(|error| line_io_error(source, error))?
    {
        preparer.report.stats.physical_lines += 1;
        match line {
            StrictLine::Invalid { number, bytes } => {
                preparer.invalid_utf8(source, number, &bytes);
                xml.pending.clear();
                xml.active = None;
            }
            StrictLine::Text { number, mut text } => {
                text.push('\n');
                xml.feed(source, number, &text, preparer)?;
            }
        }
    }
    preparer.report.stats.bytes_read += lines.bytes_read;
    xml.finish(source, lines.line, preparer)
}

fn tag_descriptor(tag: &str) -> (bool, String, bool) {
    let trimmed = tag.trim();
    let closing = trimmed.starts_with('/');
    let self_closing = trimmed.ends_with('/');
    let name = trimmed
        .trim_start_matches('/')
        .split(|character: char| character.is_ascii_whitespace() || character == '/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    (closing, name, self_closing)
}

fn find_tag_end(input: &str, start: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut quote = None;
    for (position, &byte) in bytes.iter().enumerate().skip(start + 1) {
        if let Some(expected) = quote {
            if byte == expected {
                quote = None;
            }
        } else if matches!(byte, b'"' | b'\'') {
            quote = Some(byte);
        } else if byte == b'>' {
            return Some(position);
        }
    }
    None
}

fn find_bytes(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    haystack
        .get(start..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|relative| start + relative)
}

fn find_ascii_case_insensitive(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    haystack
        .get(start..)?
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
        .map(|relative| start + relative)
}

fn decode_xml_entities(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut output = String::with_capacity(input.len());
    let mut position = 0usize;
    while position < bytes.len() {
        let Some(relative) = bytes[position..].iter().position(|&byte| byte == b'&') else {
            output.push_str(&input[position..]);
            break;
        };
        let start = position + relative;
        output.push_str(&input[position..start]);
        let search_end = (start + 40).min(bytes.len());
        let Some(relative_end) = bytes[start + 1..search_end]
            .iter()
            .position(|&byte| byte == b';')
        else {
            output.push('&');
            position = start + 1;
            continue;
        };
        let end = start + 1 + relative_end;
        let body = &input[start + 1..end];
        let decoded = match body {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            _ => decode_numeric_entity(body),
        };
        if let Some(character) = decoded {
            output.push(character);
        } else {
            output.push_str(&input[start..=end]);
        }
        position = end + 1;
    }
    output
}

fn decode_numeric_entity(body: &str) -> Option<char> {
    let scalar = if let Some(hex) = body.strip_prefix("#x").or_else(|| body.strip_prefix("#X")) {
        u32::from_str_radix(hex, 16).ok()?
    } else if let Some(decimal) = body.strip_prefix('#') {
        decimal.parse::<u32>().ok()?
    } else {
        return None;
    };
    char::from_u32(scalar)
}

struct WikiStripper {
    template_depth: usize,
    table_depth: usize,
    in_comment: bool,
    link_buffer: Option<String>,
    suppressed_tag: Option<String>,
    overflowed: bool,
}

impl WikiStripper {
    fn new() -> Self {
        Self {
            template_depth: 0,
            table_depth: 0,
            in_comment: false,
            link_buffer: None,
            suppressed_tag: None,
            overflowed: false,
        }
    }

    fn reset(&mut self) {
        self.template_depth = 0;
        self.table_depth = 0;
        self.in_comment = false;
        self.link_buffer = None;
        self.suppressed_tag = None;
        self.overflowed = false;
    }

    fn feed(&mut self, input: &str) -> String {
        let bytes = input.as_bytes();
        let mut output = String::with_capacity(input.len());
        let mut position = 0usize;

        if let Some(mut buffered) = self.link_buffer.take() {
            if let Some(end) = find_bytes(bytes, 0, b"]]") {
                buffered.push_str(&input[..end]);
                if !append_wiki_link(&mut output, &buffered) {
                    output.push(' ');
                }
                position = end + 2;
            } else {
                buffered.push_str(input);
                if buffered.len() <= WIKI_LINK_LIMIT {
                    self.link_buffer = Some(buffered);
                } else {
                    self.overflowed = true;
                }
                return output;
            }
        }

        while position < bytes.len() {
            if self.in_comment {
                if let Some(end) = find_bytes(bytes, position, b"-->") {
                    self.in_comment = false;
                    position = end + 3;
                    output.push(' ');
                } else {
                    return output;
                }
            } else if let Some(tag_name) = self.suppressed_tag.clone() {
                let closing = format!("</{tag_name}");
                let Some(start) = find_ascii_case_insensitive(bytes, position, closing.as_bytes())
                else {
                    return output;
                };
                let Some(end) = find_tag_end(input, start) else {
                    return output;
                };
                self.suppressed_tag = None;
                position = end + 1;
                output.push(' ');
            } else if self.template_depth != 0 {
                if bytes[position..].starts_with(b"{{") {
                    self.template_depth += 1;
                    position += 2;
                } else if bytes[position..].starts_with(b"}}") {
                    self.template_depth -= 1;
                    position += 2;
                    if self.template_depth == 0 {
                        output.push(' ');
                    }
                } else {
                    position += 1;
                }
            } else if self.table_depth != 0 {
                if bytes[position..].starts_with(b"{|") {
                    self.table_depth += 1;
                    position += 2;
                } else if bytes[position..].starts_with(b"|}") {
                    self.table_depth -= 1;
                    position += 2;
                    if self.table_depth == 0 {
                        output.push(' ');
                    }
                } else {
                    position += 1;
                }
            } else if bytes[position..].starts_with(b"<!--") {
                self.in_comment = true;
                position += 4;
            } else if bytes[position..].starts_with(b"{{") {
                self.template_depth = 1;
                position += 2;
            } else if bytes[position..].starts_with(b"{|") {
                self.table_depth = 1;
                position += 2;
            } else if bytes[position..].starts_with(b"[[") {
                if let Some(end) = find_bytes(bytes, position + 2, b"]]") {
                    if !append_wiki_link(&mut output, &input[position + 2..end]) {
                        output.push(' ');
                    }
                    position = end + 2;
                } else {
                    self.link_buffer = Some(input[position + 2..].to_owned());
                    return output;
                }
            } else if bytes[position] == b'<' {
                let Some(end) = find_tag_end(input, position) else {
                    return output;
                };
                let tag = &input[position + 1..end];
                let (closing, name, self_closing) = tag_descriptor(tag);
                position = end + 1;
                let mut removed_content = false;
                if !closing
                    && !self_closing
                    && matches!(name.as_str(), "ref" | "gallery" | "imagemap")
                {
                    let closing = format!("</{name}");
                    if let Some(start) =
                        find_ascii_case_insensitive(bytes, position, closing.as_bytes())
                    {
                        position = find_tag_end(input, start)
                            .map_or(bytes.len(), |close_end| close_end + 1);
                        removed_content = true;
                    } else {
                        self.suppressed_tag = Some(name);
                        return output;
                    }
                }
                if removed_content
                    || matches!(
                        name.as_str(),
                        "br" | "p"
                            | "div"
                            | "li"
                            | "tr"
                            | "td"
                            | "th"
                            | "h1"
                            | "h2"
                            | "h3"
                            | "h4"
                            | "h5"
                            | "h6"
                    )
                {
                    output.push(' ');
                }
            } else {
                let character = input[position..]
                    .chars()
                    .next()
                    .expect("valid UTF-8 boundary");
                output.push(character);
                position += character.len_utf8();
            }
        }
        output
    }

    fn take_overflowed(&mut self) -> bool {
        std::mem::take(&mut self.overflowed)
    }
}

fn append_wiki_link(output: &mut String, content: &str) -> bool {
    let target = content.split('|').next().unwrap_or_default();
    let namespace = target.split(':').next().unwrap_or_default().trim();
    let suppressed = target.contains(':')
        && matches!(
            namespace.to_ascii_lowercase().as_str(),
            "file" | "image" | "category" | "文件" | "图像" | "分类"
        );
    if suppressed {
        false
    } else {
        output.push_str(content.rsplit('|').next().unwrap_or(content));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture(kind: SourceKind, input: &str) -> (PrepareReport, Vec<WeightedRecord>) {
        let mut records = Vec::new();
        let report;
        {
            let mut preparer = Preparer::new(kind, |record| {
                records.push(record);
                Ok(())
            });
            preparer.report.stats.files_seen = 1;
            process_reader(
                kind,
                Path::new("fixture"),
                Cursor::new(input.as_bytes()),
                &mut preparer,
            )
            .unwrap();
            report = std::mem::replace(&mut preparer.report, PrepareReport::new(kind));
        }
        (report, records)
    }

    fn has(records: &[WeightedRecord], text: &str, weight: u64) -> bool {
        records
            .iter()
            .any(|record| record.text == text && record.weight == weight)
    }

    #[test]
    fn progress_reader_reports_file_and_overall_bytes() {
        let mut events = Vec::new();
        {
            let mut progress = |event| events.push(event);
            let mut reader = ProgressReader {
                inner: Cursor::new("中文".as_bytes()),
                source: Path::new("wiki_00"),
                file_index: 2,
                file_count: 3,
                file_bytes_read: 0,
                file_bytes_total: 6,
                completed_bytes: 4,
                total_bytes: 10,
                progress: &mut progress,
            };
            let mut output = Vec::new();
            reader.read_to_end(&mut output).unwrap();
            assert_eq!(output, "中文".as_bytes());
        }
        let last = events.last().unwrap();
        assert_eq!(last.source, Path::new("wiki_00"));
        assert_eq!(last.file_index, 2);
        assert_eq!(last.file_count, 3);
        assert_eq!(last.file_bytes_read, 6);
        assert_eq!(last.bytes_read, 10);
        assert_eq!(last.bytes_total, 10);
    }

    #[test]
    fn prepares_wordlist_template_with_weight_twenty() {
        let (_, records) = fixture(SourceKind::Wordlist, "const ws=`中国\n𠀀〇\n`;\n");
        assert!(has(&records, "中国", 20));
        assert!(has(&records, "𠀀〇", 20));
    }

    #[test]
    fn prepares_baidu_schema_with_declared_weights() {
        let input = concat!(
            r#"{"title":"中国A文学","summary":"摘要","sections":[{"title":"历史","content":"正文"}],"tags":["标签"],"url":"ignored"}"#,
            "\n"
        );
        let (_, records) = fixture(SourceKind::BaiduBaike, input);
        assert!(has(&records, "中国", 10));
        assert!(has(&records, "文学", 10));
        assert!(has(&records, "摘要", 1));
        assert!(has(&records, "历史", 5));
        assert!(has(&records, "正文", 1));
        assert!(has(&records, "标签", 5));
    }

    #[test]
    fn prepares_wikipedia_jsonl_title_and_text_only() {
        let input = concat!(
            r#"{"id":"1","url":"https://example","title":"数学","text":"数学\nEnglish分隔中文[方括号保留]"}"#,
            "\n"
        );
        let (_, records) = fixture(SourceKind::Wikipedia, input);
        assert!(has(&records, "数学", 10));
        assert!(has(&records, "数学", 1));
        assert!(has(&records, "分隔中文", 1));
        assert!(has(&records, "方括号保留", 1));
        assert!(!records.iter().any(|record| record.text.contains("例")));
    }

    #[test]
    fn prepares_only_namespace_zero_moegirl_pages_and_strips_wiki_markup() {
        let input = concat!(
            "<mediawiki><page><title>条目</title><ns>0</ns><revision><text>",
            "正文{{模板|噪音}}[[中国|中华]]&lt;ref&gt;注释&lt;/ref&gt;",
            "</text></revision></page>",
            "<page><title>讨论页</title><ns>1</ns><revision><text>不要</text></revision></page>",
            "</mediawiki>\n"
        );
        let (_, records) = fixture(SourceKind::Moegirl, input);
        assert!(has(&records, "条目", 10));
        assert!(has(&records, "正文", 1));
        assert!(has(&records, "中华", 1));
        assert!(!records.iter().any(|record| {
            matches!(
                record.text.as_str(),
                "模板" | "噪音" | "注释" | "讨论页" | "不要"
            )
        }));
    }

    #[test]
    fn prepares_multiline_bilibili_message_and_removes_reply_annotations() {
        let input = concat!(
            "message,time,timestamp\r\n",
            "\"回复 @张三：你好[笑哭]\r\n世界\",2023-01-01,1\r\n"
        );
        let (_, records) = fixture(SourceKind::Bilibili, input);
        assert!(has(&records, "你好", 1));
        assert!(has(&records, "世界", 1));
        assert!(
            !records
                .iter()
                .any(|record| { matches!(record.text.as_str(), "回复" | "张三" | "笑哭") })
        );
    }

    #[test]
    fn prepares_tieba_chinese_and_compatibility_fields() {
        let input = r#"[
            {"标题":"主题","楼主内容":"正文","回复列表":[{"内容":"回复"}]},
            {"title":"兼容标题","content":"兼容内容"}
        ]"#;
        let (_, records) = fixture(SourceKind::Tieba, input);
        assert!(has(&records, "主题", 5));
        assert!(has(&records, "正文", 1));
        assert!(has(&records, "回复", 1));
        assert!(has(&records, "兼容标题", 5));
        assert!(has(&records, "兼容内容", 1));
    }

    #[test]
    fn prepares_only_user_and_assistant_chatml_turns() {
        let input = concat!(
            r#"{"text":"<|im_start|>system\n系统说明<|im_end|>\n<|im_start|>user\n[中文]问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>"}"#,
            "\n"
        );
        let (_, records) = fixture(SourceKind::Train, input);
        assert!(has(&records, "中文", 1));
        assert!(has(&records, "问题", 1));
        assert!(has(&records, "回答", 1));
        assert!(!records.iter().any(|record| record.text == "系统说明"));
    }

    #[test]
    fn prepares_common_poetry_schemas() {
        let input = r#"[
            {"author":"太宗皇帝","title":"帝京篇","paragraphs":["秦川雄帝宅，函谷壮皇居。"]},
            {"author":"和岘","rhythmic":"导引","paragraphs":["气和玉烛，睿化著鸿明。"]},
            {"title":"关雎","chapter":"国风","section":"周南","content":["关关雎鸠，在河之洲。"]},
            {"title":"长相思","para":["山一程，水一程。"]},
            {"content":"读经宜冬，其神专也。","comment":["后人评论不要收录。"]}
        ]"#;
        let (_, records) = fixture(SourceKind::Poetry, input);
        assert!(has(&records, "帝京篇", 20));
        assert!(has(&records, "秦川雄帝宅", 10));
        assert!(has(&records, "导引", 20));
        assert!(has(&records, "气和玉烛", 10));
        assert!(has(&records, "关雎", 20));
        assert!(has(&records, "国风", 20));
        assert!(has(&records, "周南", 20));
        assert!(has(&records, "关关雎鸠", 10));
        assert!(has(&records, "山一程", 10));
        assert!(has(&records, "读经宜冬", 10));
        assert!(
            !records
                .iter()
                .any(|record| record.text.contains("后人评论"))
        );
    }

    #[test]
    fn prepares_nested_mengxue_content() {
        let input = r#"{
            "title":"古文观止",
            "abstract":["现代内容简介不要收录。"],
            "content":[{"title":"卷一","content":[
                {"chapter":"郑伯克段于鄢","paragraphs":["初，郑武公娶于申。"]}
            ]}]
        }"#;
        let (_, records) = fixture(SourceKind::Poetry, input);
        assert!(has(&records, "郑伯克段于鄢", 20));
        assert!(has(&records, "郑武公娶于申", 10));
        assert!(
            !records
                .iter()
                .any(|record| record.text.contains("现代内容简介"))
        );
    }

    #[test]
    fn includes_ancient_prefaces_and_biographies_but_not_modern_guides() {
        let input = r#"[
            {"title":"文字蒙求","preface":["雪堂谓筠曰，人之不识字也。"],"abstract":"现代简介。"},
            {"title":"帝京篇","biography":"帝姓李氏，讳世民。","paragraphs":["秦川雄帝宅。"],
             "notes":["现代注释。"],"prologue":"现代赏析。"}
        ]"#;
        let (_, records) = fixture(SourceKind::Poetry, input);
        assert!(has(&records, "雪堂谓筠曰", 10));
        assert!(has(&records, "帝姓李氏", 10));
        assert!(!records.iter().any(|record| {
            matches!(record.text.as_str(), "现代简介" | "现代注释" | "现代赏析")
        }));
    }

    #[test]
    fn deduplicates_poetry_by_han_only_full_body() {
        let input = r#"[
            {"title":"春晓","paragraphs":["春眠不觉晓，处处闻啼鸟。"]},
            {"title":"春晓","paragraphs":["春眠不觉晓；处处闻啼鸟！"]}
        ]"#;
        let (_, records) = fixture(SourceKind::Poetry, input);
        assert_eq!(
            records
                .iter()
                .filter(|record| record.text == "春眠不觉晓" && record.weight == 10)
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.text == "春晓" && record.weight == 20)
                .count(),
            1
        );
    }

    #[test]
    fn poetry_path_filter_keeps_primary_json_only() {
        let root = Path::new("chinese-poetry");
        assert!(is_poetry_source_path(
            root,
            &root.join("全唐诗").join("poet.tang.0.json")
        ));
        assert!(is_poetry_source_path(
            root,
            &root.join("御定全唐詩").join("json").join("001.json")
        ));
        assert!(!is_poetry_source_path(
            root,
            &root.join("全唐诗").join("authors.tang.json")
        ));
        assert!(!is_poetry_source_path(
            root,
            &root.join("全唐诗").join("error").join("poet.tang.0.json")
        ));
        assert!(!is_poetry_source_path(
            root,
            &root.join("rank").join("poet").join("poet.tang.0.json")
        ));
    }
}
