//! External-sort index builder and compact Unicode trie reader.
//!
//! The v2 format stores Unicode scalar labels directly as unsigned varints.
//! Nodes are written children-first, so an ordered input stream can be turned
//! into an index without retaining the complete trie in memory.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const INDEX_MAGIC: [u8; 8] = *b"NUTRIZH2";
const INDEX_VERSION: u16 = 2;
const HEADER_LEN: usize = 64;
const FLAG_IMPLICIT_PREFIXES: u32 = 1;
const KNOWN_FLAGS: u32 = FLAG_IMPLICIT_PREFIXES;

const SHARD_MAGIC: &str = "NUTRIMATIC_ZH_TEXT_SHARD_V2";
const DEFAULT_MAX_ENTRIES: usize = 1_000_000;
const MERGE_PROGRESS_STEP: u64 = 1024 * 1024;

pub type Result<T> = std::result::Result<T, IndexError>;

#[derive(Debug)]
pub enum IndexError {
    Io {
        path: Option<PathBuf>,
        source: io::Error,
    },
    AlreadyExists(PathBuf),
    OutputNotEmpty,
    InvalidArgument(String),
    InvalidFormat {
        offset: usize,
        message: String,
    },
    InvalidShard {
        path: PathBuf,
        line: usize,
        message: String,
    },
    OutOfOrder,
    Overflow(&'static str),
}

impl IndexError {
    fn io_at(path: impl AsRef<Path>, source: io::Error) -> Self {
        Self::Io {
            path: Some(path.as_ref().to_path_buf()),
            source,
        }
    }

    fn format(offset: usize, message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            offset,
            message: message.into(),
        }
    }

    fn shard(path: &Path, line: usize, message: impl Into<String>) -> Self {
        Self::InvalidShard {
            path: path.to_path_buf(),
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                path: Some(path),
                source,
            } => write!(f, "I/O error for {}: {source}", path.display()),
            Self::Io { path: None, source } => write!(f, "I/O error: {source}"),
            Self::AlreadyExists(path) => {
                write!(f, "refusing to overwrite existing path: {}", path.display())
            }
            Self::OutputNotEmpty => write!(f, "refusing to overwrite a non-empty output"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::InvalidFormat { offset, message } => {
                write!(f, "invalid index at byte {offset}: {message}")
            }
            Self::InvalidShard {
                path,
                line,
                message,
            } => write!(
                f,
                "invalid shard {} at line {line}: {message}",
                path.display()
            ),
            Self::OutOfOrder => write!(f, "index keys must be supplied in sorted order"),
            Self::Overflow(what) => write!(f, "numeric overflow while computing {what}"),
        }
    }
}

impl Error for IndexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for IndexError {
    fn from(source: io::Error) -> Self {
        Self::Io { path: None, source }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexMode {
    /// Each input record is one complete key. Only terminal nodes are results.
    Records,
    /// Emit one maximal window for every scalar position. Every trie prefix is
    /// therefore an indexed n-gram whose frequency is its subtree count.
    Ngrams { window_codepoints: usize },
}

impl IndexMode {
    fn validate(self) -> Result<()> {
        match self {
            Self::Records => Ok(()),
            Self::Ngrams {
                window_codepoints: 0,
            } => Err(IndexError::InvalidArgument(
                "the n-gram window must be greater than zero".to_string(),
            )),
            Self::Ngrams { window_codepoints } if window_codepoints > u32::MAX as usize => {
                Err(IndexError::InvalidArgument(
                    "the n-gram window does not fit in the index header".to_string(),
                ))
            }
            Self::Ngrams { .. } => Ok(()),
        }
    }

    pub fn implicit_prefixes(self) -> bool {
        matches!(self, Self::Ngrams { .. })
    }

    pub fn window_codepoints(self) -> Option<usize> {
        match self {
            Self::Records => None,
            Self::Ngrams { window_codepoints } => Some(window_codepoints),
        }
    }

    fn shard_tag(self) -> String {
        match self {
            Self::Records => "records".to_string(),
            Self::Ngrams { window_codepoints } => format!("ngrams:{window_codepoints}"),
        }
    }

    fn parse_shard_tag(tag: &str) -> Option<Self> {
        if tag == "records" {
            Some(Self::Records)
        } else if let Some(window) = tag.strip_prefix("ngrams:") {
            let window_codepoints = window.parse().ok()?;
            let mode = Self::Ngrams { window_codepoints };
            mode.validate().ok()?;
            Some(mode)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Edge {
    pub label: char,
    pub child_offset: u64,
    pub subtree: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    pub offset: u64,
    /// Exact terminal frequency for record indexes. In pruned n-gram indexes,
    /// this is also the residual frequency of omitted descendants; the root's
    /// residual bucket never represents an empty-string search result.
    pub terminal: u64,
    pub subtree: u64,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WalkEntry {
    pub key: String,
    pub terminal: u64,
    pub subtree: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IndexWriteStats {
    pub node_count: u64,
    /// Number of logical trie nodes omitted by streaming subtree pruning.
    pub pruned_node_count: u64,
    /// Number of distinct terminal keys supplied before subtree pruning.
    pub key_count: u64,
    pub root_subtree: u64,
    pub bytes_written: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ValidationStats {
    pub node_count: u64,
    pub edge_count: u64,
}

fn checked_add(a: u64, b: u64, what: &'static str) -> Result<u64> {
    a.checked_add(b).ok_or(IndexError::Overflow(what))
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(left, right)| left == right)
        .count()
}

fn write_varint<W: Write>(writer: &mut W, mut value: u64) -> io::Result<usize> {
    let mut bytes = [0u8; 10];
    let mut len = 0;
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        bytes[len] = byte;
        len += 1;
        if value == 0 {
            writer.write_all(&bytes[..len])?;
            return Ok(len);
        }
    }
}

fn read_varint(data: &[u8], position: &mut usize, limit: usize) -> Result<u64> {
    let start = *position;
    let mut value = 0u64;

    for index in 0..10 {
        if *position >= limit || *position >= data.len() {
            return Err(IndexError::format(start, "truncated varint"));
        }
        let byte = data[*position];
        *position += 1;

        if index == 9 && byte > 1 {
            return Err(IndexError::format(start, "varint overflows u64"));
        }
        value |= u64::from(byte & 0x7f) << (index * 7);

        if byte & 0x80 == 0 {
            if index > 0 && byte == 0 {
                return Err(IndexError::format(start, "non-canonical varint"));
            }
            return Ok(value);
        }
    }

    Err(IndexError::format(start, "varint is longer than ten bytes"))
}

fn put_u16(header: &mut [u8], offset: usize, value: u16) {
    header[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(header: &mut [u8], offset: usize, value: u32) {
    header[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(header: &mut [u8], offset: usize, value: u64) {
    header[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn get_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        data[offset..offset + 4]
            .try_into()
            .expect("fixed header slice"),
    )
}

fn get_u64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        data[offset..offset + 8]
            .try_into()
            .expect("fixed header slice"),
    )
}

#[derive(Debug)]
struct WrittenChild {
    label: u32,
    offset: u64,
    subtree: u64,
}

#[derive(Debug)]
struct PendingNode {
    label_from_parent: Option<u32>,
    // In pruned n-gram indexes this also holds residual frequency folded from
    // omitted descendants. Search uses `subtree`, never this terminal bucket.
    terminal: u64,
    children: Vec<WrittenChild>,
}

impl PendingNode {
    fn root() -> Self {
        Self {
            label_from_parent: None,
            terminal: 0,
            children: Vec::new(),
        }
    }

    fn child(label: u32) -> Self {
        Self {
            label_from_parent: Some(label),
            terminal: 0,
            children: Vec::new(),
        }
    }

    fn subtree_frequency(&self) -> Result<u64> {
        let mut subtree = self.terminal;
        for child in &self.children {
            subtree = checked_add(subtree, child.subtree, "subtree frequency")?;
        }
        if subtree > i64::MAX as u64 {
            return Err(IndexError::Overflow("subtree frequency"));
        }
        Ok(subtree)
    }
}

/// Streaming writer for lexicographically sorted Unicode-scalar keys.
pub struct TrieWriter<W: Write + Seek> {
    output: BufWriter<W>,
    mode: IndexMode,
    min_subtree_count: u64,
    position: u64,
    stack: Vec<PendingNode>,
    previous_key: Vec<u32>,
    node_count: u64,
    pruned_node_count: u64,
    key_count: u64,
    finished: bool,
}

impl<W: Write + Seek> TrieWriter<W> {
    pub fn new(output: W, mode: IndexMode) -> Result<Self> {
        Self::with_min_subtree_count(output, mode, 1)
    }

    /// Creates a writer which omits n-gram branches whose total corpus
    /// frequency is below `min_subtree_count`.
    ///
    /// Pruned frequency is folded into the nearest retained ancestor, so the
    /// subtree count used to rank every retained prefix remains exact.  This
    /// is only valid for implicit-prefix n-gram indexes; record indexes keep
    /// their exact-terminal semantics and therefore require a threshold of 1.
    pub fn with_min_subtree_count(
        mut output: W,
        mode: IndexMode,
        min_subtree_count: u64,
    ) -> Result<Self> {
        mode.validate()?;
        if min_subtree_count == 0 || min_subtree_count > i64::MAX as u64 {
            return Err(IndexError::InvalidArgument(
                "min_subtree_count must be between 1 and i64::MAX".to_string(),
            ));
        }
        if min_subtree_count != 1 && !mode.implicit_prefixes() {
            return Err(IndexError::InvalidArgument(
                "subtree pruning is only supported for n-gram indexes".to_string(),
            ));
        }
        let length = output.seek(SeekFrom::End(0))?;
        if length != 0 {
            return Err(IndexError::OutputNotEmpty);
        }
        output.seek(SeekFrom::Start(0))?;

        let mut output = BufWriter::new(output);
        output.write_all(&[0u8; HEADER_LEN])?;

        Ok(Self {
            output,
            mode,
            min_subtree_count,
            position: HEADER_LEN as u64,
            stack: vec![PendingNode::root()],
            previous_key: Vec::new(),
            node_count: 0,
            pruned_node_count: 0,
            key_count: 0,
            finished: false,
        })
    }

    pub fn push_str_sorted(&mut self, key: &str, count: u64) -> Result<()> {
        let scalars: Vec<u32> = key.chars().map(u32::from).collect();
        self.push_sorted(&scalars, count)
    }

    pub fn push_sorted(&mut self, key: &[u32], count: u64) -> Result<()> {
        if self.finished {
            return Err(IndexError::InvalidArgument(
                "cannot push keys after finishing the writer".to_string(),
            ));
        }
        if key.is_empty() {
            return Err(IndexError::InvalidArgument(
                "empty index keys are not supported".to_string(),
            ));
        }
        if count == 0 {
            return Err(IndexError::InvalidArgument(
                "index counts must be greater than zero".to_string(),
            ));
        }
        if count > i64::MAX as u64 {
            return Err(IndexError::InvalidArgument(
                "index counts must fit in a signed 64-bit integer".to_string(),
            ));
        }
        for &scalar in key {
            if char::from_u32(scalar).is_none() {
                return Err(IndexError::InvalidArgument(format!(
                    "U+{scalar:04X} is not a Unicode scalar value"
                )));
            }
        }
        if !self.previous_key.is_empty() && key < self.previous_key.as_slice() {
            return Err(IndexError::OutOfOrder);
        }

        let same = common_prefix(&self.previous_key, key);
        while self.stack.len() - 1 > same {
            self.close_last_node()?;
        }
        for &label in &key[same..] {
            self.stack.push(PendingNode::child(label));
        }

        let terminal = &mut self
            .stack
            .last_mut()
            .expect("root is always present")
            .terminal;
        *terminal = checked_add(*terminal, count, "terminal frequency")?;

        if key != self.previous_key.as_slice() {
            self.key_count = checked_add(self.key_count, 1, "key count")?;
        }
        self.previous_key.clear();
        self.previous_key.extend_from_slice(key);
        Ok(())
    }

    fn close_last_node(&mut self) -> Result<()> {
        let node = self.stack.pop().expect("a child node must exist");
        let label = node
            .label_from_parent
            .expect("only non-root nodes are closed here");
        let subtree = node.subtree_frequency()?;
        if subtree < self.min_subtree_count {
            self.pruned_node_count = checked_add(self.pruned_node_count, 1, "pruned node count")?;
            let parent = self.stack.last_mut().expect("the parent node must exist");
            parent.terminal = checked_add(
                parent.terminal,
                subtree,
                "frequency folded from a pruned subtree",
            )?;
            return Ok(());
        }
        let (offset, subtree) = self.write_node(&node)?;
        self.stack
            .last_mut()
            .expect("the parent node must exist")
            .children
            .push(WrittenChild {
                label,
                offset,
                subtree,
            });
        Ok(())
    }

    fn write_node(&mut self, node: &PendingNode) -> Result<(u64, u64)> {
        let subtree = node.subtree_frequency()?;
        let mut previous_label = None;
        for child in &node.children {
            if previous_label.is_some_and(|label| child.label <= label) {
                return Err(IndexError::OutOfOrder);
            }
            previous_label = Some(child.label);
        }

        let offset = self.position;
        self.position = checked_add(
            self.position,
            write_varint(&mut self.output, node.terminal)? as u64,
            "index position",
        )?;
        self.position = checked_add(
            self.position,
            write_varint(&mut self.output, subtree)? as u64,
            "index position",
        )?;
        self.position = checked_add(
            self.position,
            write_varint(&mut self.output, node.children.len() as u64)? as u64,
            "index position",
        )?;

        for child in &node.children {
            let delta = offset
                .checked_sub(child.offset)
                .ok_or(IndexError::Overflow("child back-reference"))?;
            if delta == 0 {
                return Err(IndexError::InvalidFormat {
                    offset: offset as usize,
                    message: "child node must precede its parent".to_string(),
                });
            }
            for value in [u64::from(child.label), delta, child.subtree] {
                self.position = checked_add(
                    self.position,
                    write_varint(&mut self.output, value)? as u64,
                    "index position",
                )?;
            }
        }

        self.node_count = checked_add(self.node_count, 1, "node count")?;
        Ok((offset, subtree))
    }

    pub fn finish(mut self) -> Result<(W, IndexWriteStats)> {
        while self.stack.len() > 1 {
            self.close_last_node()?;
        }
        let root = self.stack.pop().expect("root node must exist");
        let (root_offset, root_subtree) = self.write_node(&root)?;
        self.finished = true;

        let file_length = self.position;
        let flags = if self.mode.implicit_prefixes() {
            FLAG_IMPLICIT_PREFIXES
        } else {
            0
        };
        let window = self.mode.window_codepoints().unwrap_or(0) as u32;

        let mut header = [0u8; HEADER_LEN];
        header[..8].copy_from_slice(&INDEX_MAGIC);
        put_u16(&mut header, 8, INDEX_VERSION);
        put_u16(&mut header, 10, HEADER_LEN as u16);
        put_u32(&mut header, 12, flags);
        put_u64(&mut header, 16, file_length);
        put_u64(&mut header, 24, root_offset);
        put_u64(&mut header, 32, root_subtree);
        put_u64(&mut header, 40, self.node_count);
        put_u32(&mut header, 48, window);

        self.output.flush()?;
        self.output.seek(SeekFrom::Start(0))?;
        self.output.write_all(&header)?;
        self.output.flush()?;
        self.output.seek(SeekFrom::Start(file_length))?;

        let stats = IndexWriteStats {
            node_count: self.node_count,
            pruned_node_count: self.pruned_node_count,
            key_count: self.key_count,
            root_subtree,
            bytes_written: file_length,
        };
        let output = self
            .output
            .into_inner()
            .map_err(|error| IndexError::from(error.into_error()))?;
        Ok((output, stats))
    }
}

/// Owned, bounds-checked reader for a v2 index.
///
/// Pure std does not expose a portable memory map, so this implementation
/// reads the index into memory. All offsets remain file-relative u64 values.
/// open/from_bytes validate the fixed header and root node lazily; inspection
/// commands offering a --full mode should additionally call validate_full().
pub struct IndexReader {
    data: Box<[u8]>,
    mode: IndexMode,
    root_offset: u64,
    root_subtree: u64,
    node_count: u64,
}

impl IndexReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read(path).map_err(|error| IndexError::io_at(path, error))?;
        Self::from_bytes(data)
    }

    pub fn from_slice(data: &[u8]) -> Result<Self> {
        Self::from_bytes(data.to_vec())
    }

    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        if data.len() < HEADER_LEN {
            return Err(IndexError::format(
                0,
                "file is shorter than the fixed header",
            ));
        }
        if data[..8] != INDEX_MAGIC[..] {
            return Err(IndexError::format(0, "bad index magic"));
        }
        let version = get_u16(&data, 8);
        if version != INDEX_VERSION {
            return Err(IndexError::format(
                8,
                format!("unsupported index version {version}"),
            ));
        }
        let header_len = usize::from(get_u16(&data, 10));
        if header_len != HEADER_LEN {
            return Err(IndexError::format(10, "unexpected header length"));
        }

        let flags = get_u32(&data, 12);
        if flags & !KNOWN_FLAGS != 0 {
            return Err(IndexError::format(12, "unknown header flags are set"));
        }
        let declared_length = get_u64(&data, 16);
        if declared_length > i64::MAX as u64 {
            return Err(IndexError::format(
                16,
                "declared file length exceeds i64::MAX",
            ));
        }
        if declared_length != data.len() as u64 {
            return Err(IndexError::format(
                16,
                "declared file length does not match",
            ));
        }
        let root_offset = get_u64(&data, 24);
        if root_offset > i64::MAX as u64 {
            return Err(IndexError::format(24, "root offset exceeds i64::MAX"));
        }
        if root_offset < HEADER_LEN as u64 || root_offset >= declared_length {
            return Err(IndexError::format(24, "root offset is outside the payload"));
        }
        let root_subtree = get_u64(&data, 32);
        if root_subtree > i64::MAX as u64 {
            return Err(IndexError::format(
                32,
                "root subtree count exceeds i64::MAX",
            ));
        }
        let node_count = get_u64(&data, 40);
        if node_count == 0 {
            return Err(IndexError::format(40, "an index must contain a root node"));
        }
        if node_count > i64::MAX as u64 {
            return Err(IndexError::format(40, "node count exceeds i64::MAX"));
        }
        let window = get_u32(&data, 48) as usize;
        if data[52..HEADER_LEN].iter().any(|&byte| byte != 0) {
            return Err(IndexError::format(52, "reserved header bytes must be zero"));
        }

        let mode = if flags & FLAG_IMPLICIT_PREFIXES != 0 {
            if window == 0 {
                return Err(IndexError::format(
                    48,
                    "implicit-prefix indexes require a non-zero window",
                ));
            }
            IndexMode::Ngrams {
                window_codepoints: window,
            }
        } else {
            if window != 0 {
                return Err(IndexError::format(
                    48,
                    "record indexes must have a zero window",
                ));
            }
            IndexMode::Records
        };

        let reader = Self {
            data: data.into_boxed_slice(),
            mode,
            root_offset,
            root_subtree,
            node_count,
        };
        let (root, end) = reader.parse_node(root_offset)?;
        if end != reader.data.len() {
            return Err(IndexError::format(end, "root node is not the final object"));
        }
        if root.subtree != root_subtree {
            return Err(IndexError::format(
                root_offset as usize,
                "root subtree count disagrees with the header",
            ));
        }
        if root.terminal != 0 && !mode.implicit_prefixes() {
            return Err(IndexError::format(
                root_offset as usize,
                "a record-index root cannot contain an empty key",
            ));
        }
        Ok(reader)
    }

    pub fn mode(&self) -> IndexMode {
        self.mode
    }

    pub fn implicit_prefixes(&self) -> bool {
        self.mode.implicit_prefixes()
    }

    pub fn window_codepoints(&self) -> Option<usize> {
        self.mode.window_codepoints()
    }

    pub fn root_offset(&self) -> u64 {
        self.root_offset
    }

    pub fn root(&self) -> u64 {
        self.root_offset
    }

    pub fn root_subtree(&self) -> u64 {
        self.root_subtree
    }

    pub fn node_count(&self) -> u64 {
        self.node_count
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn node(&self, offset: u64) -> Result<Node> {
        self.parse_node(offset).map(|(node, _)| node)
    }

    fn parse_node(&self, offset: u64) -> Result<(Node, usize)> {
        if offset < HEADER_LEN as u64 || offset >= self.data.len() as u64 {
            return Err(IndexError::format(
                usize::try_from(offset).unwrap_or(usize::MAX),
                "node offset is outside the payload",
            ));
        }
        let node_start = usize::try_from(offset)
            .map_err(|_| IndexError::format(usize::MAX, "node offset does not fit usize"))?;
        let mut position = node_start;
        let terminal = read_varint(&self.data, &mut position, self.data.len())?;
        let subtree = read_varint(&self.data, &mut position, self.data.len())?;
        if terminal > i64::MAX as u64 || subtree > i64::MAX as u64 {
            return Err(IndexError::format(
                node_start,
                "node frequency exceeds i64::MAX",
            ));
        }
        let edge_count_u64 = read_varint(&self.data, &mut position, self.data.len())?;
        let edge_count = usize::try_from(edge_count_u64)
            .map_err(|_| IndexError::format(position, "edge count does not fit usize"))?;

        // Every edge occupies at least three one-byte varints.
        if edge_count > (self.data.len().saturating_sub(position)) / 3 {
            return Err(IndexError::format(position, "edge table is truncated"));
        }
        let mut edges = Vec::new();
        edges
            .try_reserve_exact(edge_count)
            .map_err(|_| IndexError::format(position, "edge table is too large"))?;

        let mut child_total = 0u64;
        let mut previous_label = None;
        for _ in 0..edge_count {
            let label_offset = position;
            let label_u64 = read_varint(&self.data, &mut position, self.data.len())?;
            let label_u32 = u32::try_from(label_u64)
                .map_err(|_| IndexError::format(label_offset, "label exceeds u32"))?;
            let label = char::from_u32(label_u32).ok_or_else(|| {
                IndexError::format(label_offset, "label is not a Unicode scalar value")
            })?;
            if previous_label.is_some_and(|previous| label_u32 <= previous) {
                return Err(IndexError::format(
                    label_offset,
                    "edge labels are not strictly increasing",
                ));
            }
            previous_label = Some(label_u32);

            let delta_offset = position;
            let delta = read_varint(&self.data, &mut position, self.data.len())?;
            if delta == 0 || delta > offset.saturating_sub(HEADER_LEN as u64) {
                return Err(IndexError::format(
                    delta_offset,
                    "child back-reference is outside the payload",
                ));
            }
            let child_offset = offset - delta;
            if child_offset < HEADER_LEN as u64 || child_offset >= offset {
                return Err(IndexError::format(
                    delta_offset,
                    "child node must precede its parent",
                ));
            }

            let child_subtree_offset = position;
            let child_subtree = read_varint(&self.data, &mut position, self.data.len())?;
            if child_subtree == 0 {
                return Err(IndexError::format(
                    child_subtree_offset,
                    "non-root child has an empty subtree",
                ));
            }
            if child_subtree > i64::MAX as u64 {
                return Err(IndexError::format(
                    child_subtree_offset,
                    "child subtree count exceeds i64::MAX",
                ));
            }
            child_total = checked_add(child_total, child_subtree, "decoded child counts")?;
            edges.push(Edge {
                label,
                child_offset,
                subtree: child_subtree,
            });
        }

        let computed_subtree = checked_add(terminal, child_total, "decoded subtree count")?;
        if computed_subtree != subtree {
            return Err(IndexError::format(
                node_start,
                "terminal and child counts do not equal the subtree count",
            ));
        }

        Ok((
            Node {
                offset,
                terminal,
                subtree,
                edges,
            },
            position,
        ))
    }

    pub fn find(&self, key: &str) -> Result<Option<Node>> {
        let mut node = self.node(self.root_offset)?;
        for label in key.chars() {
            let edge = match node.edges.binary_search_by_key(&label, |edge| edge.label) {
                Ok(index) => node.edges[index].clone(),
                Err(_) => return Ok(None),
            };
            node = self.node(edge.child_offset)?;
            if node.subtree != edge.subtree {
                return Err(IndexError::format(
                    edge.child_offset as usize,
                    "edge subtree count disagrees with its child node",
                ));
            }
        }
        Ok(Some(node))
    }

    /// Traverse and validate every reachable node. Search can use node() for
    /// lazy validation; index inspection and tests can request this full pass.
    pub fn validate_full(&self) -> Result<ValidationStats> {
        let mut stack = vec![(self.root_offset, self.root_subtree)];
        let mut seen = HashSet::new();
        let mut spans = Vec::new();
        let mut stats = ValidationStats::default();

        while let Some((offset, expected_subtree)) = stack.pop() {
            if !seen.insert(offset) {
                return Err(IndexError::format(
                    offset as usize,
                    "node is referenced more than once",
                ));
            }
            let (node, end) = self.parse_node(offset)?;
            if node.subtree != expected_subtree {
                return Err(IndexError::format(
                    offset as usize,
                    "parent edge has the wrong subtree count",
                ));
            }
            stats.node_count = checked_add(stats.node_count, 1, "validated node count")?;
            stats.edge_count = checked_add(
                stats.edge_count,
                node.edges.len() as u64,
                "validated edge count",
            )?;
            spans.push((offset, end as u64));
            for edge in node.edges.into_iter().rev() {
                stack.push((edge.child_offset, edge.subtree));
            }
        }

        if stats.node_count != self.node_count {
            return Err(IndexError::format(
                40,
                "reachable node count disagrees with the header",
            ));
        }
        spans.sort_unstable_by_key(|&(start, _)| start);
        let mut expected_start = HEADER_LEN as u64;
        for (start, end) in spans {
            if start != expected_start || end < start {
                return Err(IndexError::format(
                    start as usize,
                    "node payload contains a gap or overlapping node",
                ));
            }
            expected_start = end;
        }
        if expected_start != self.data.len() as u64 {
            return Err(IndexError::format(
                expected_start as usize,
                "node payload does not cover the declared file length",
            ));
        }
        Ok(stats)
    }

    pub fn walk(&self) -> Result<IndexWalker<'_>> {
        IndexWalker::new(self)
    }
}

struct WalkFrame {
    node: Node,
    next_edge: usize,
    yielded: bool,
}

pub struct IndexWalker<'a> {
    reader: &'a IndexReader,
    frames: Vec<WalkFrame>,
    key: Vec<char>,
    finished: bool,
}

impl<'a> IndexWalker<'a> {
    pub fn new(reader: &'a IndexReader) -> Result<Self> {
        Ok(Self {
            reader,
            frames: vec![WalkFrame {
                node: reader.node(reader.root_offset)?,
                next_edge: 0,
                yielded: false,
            }],
            key: Vec::new(),
            finished: false,
        })
    }
}

impl Iterator for IndexWalker<'_> {
    type Item = Result<WalkEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            let frame_index = match self.frames.len().checked_sub(1) {
                Some(index) => index,
                None => {
                    self.finished = true;
                    return None;
                }
            };

            if !self.frames[frame_index].yielded {
                self.frames[frame_index].yielded = true;
                let node = &self.frames[frame_index].node;
                if !self.key.is_empty() && (node.terminal > 0 || self.reader.implicit_prefixes()) {
                    return Some(Ok(WalkEntry {
                        key: self.key.iter().collect(),
                        terminal: node.terminal,
                        subtree: node.subtree,
                    }));
                }
            }

            let edge = {
                let frame = &mut self.frames[frame_index];
                if frame.next_edge < frame.node.edges.len() {
                    let edge = frame.node.edges[frame.next_edge].clone();
                    frame.next_edge += 1;
                    Some(edge)
                } else {
                    None
                }
            };

            if let Some(edge) = edge {
                self.key.push(edge.label);
                match self.reader.node(edge.child_offset) {
                    Ok(node) if node.subtree == edge.subtree => {
                        self.frames.push(WalkFrame {
                            node,
                            next_edge: 0,
                            yielded: false,
                        });
                    }
                    Ok(_) => {
                        self.finished = true;
                        return Some(Err(IndexError::format(
                            edge.child_offset as usize,
                            "edge subtree count disagrees with its child node",
                        )));
                    }
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
            } else {
                self.frames.pop();
                if !self.frames.is_empty() {
                    self.key.pop();
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuildOptions {
    pub output_path: PathBuf,
    pub shard_dir: PathBuf,
    pub shard_prefix: String,
    pub mode: IndexMode,
    /// Number of raw keys retained before a sorted text shard is written.
    pub max_entries_in_memory: usize,
    /// Retain only n-gram prefixes with at least this weighted corpus count.
    /// Text shards still contain all raw chains so cross-shard counts remain
    /// exact; pruning happens during the final merge.
    pub min_subtree_count: u64,
}

impl BuildOptions {
    pub fn new(
        output_path: impl Into<PathBuf>,
        shard_dir: impl Into<PathBuf>,
        shard_prefix: impl Into<String>,
        mode: IndexMode,
    ) -> Self {
        Self {
            output_path: output_path.into(),
            shard_dir: shard_dir.into(),
            shard_prefix: shard_prefix.into(),
            mode,
            max_entries_in_memory: DEFAULT_MAX_ENTRIES,
            min_subtree_count: 1,
        }
    }

    fn validate(&self) -> Result<()> {
        self.mode.validate()?;
        if self.max_entries_in_memory == 0 {
            return Err(IndexError::InvalidArgument(
                "max_entries_in_memory must be greater than zero".to_string(),
            ));
        }
        if self.min_subtree_count == 0 || self.min_subtree_count > i64::MAX as u64 {
            return Err(IndexError::InvalidArgument(
                "min_subtree_count must be between 1 and i64::MAX".to_string(),
            ));
        }
        if self.min_subtree_count != 1 && !self.mode.implicit_prefixes() {
            return Err(IndexError::InvalidArgument(
                "subtree pruning is only supported for n-gram indexes".to_string(),
            ));
        }
        validate_shard_prefix(&self.shard_prefix)?;
        if self.output_path.as_os_str().is_empty() {
            return Err(IndexError::InvalidArgument(
                "an explicit output path is required".to_string(),
            ));
        }
        if self.shard_dir.as_os_str().is_empty() {
            return Err(IndexError::InvalidArgument(
                "an explicit shard directory is required".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BuildReport {
    pub output_path: PathBuf,
    pub shard_paths: Vec<PathBuf>,
    pub records_seen: u64,
    pub empty_records: u64,
    pub chains_emitted: u64,
    pub min_subtree_count: u64,
    pub index: IndexWriteStats,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MergeReport {
    pub output_path: PathBuf,
    pub shard_count: usize,
    pub min_subtree_count: u64,
    pub index: IndexWriteStats,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MergeProgress {
    pub bytes_read: u64,
    pub bytes_total: u64,
    pub shard_records_read: u64,
    pub unique_keys_written: u64,
    pub shard_count: usize,
    pub complete: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexRecord {
    pub text: String,
    pub count: u64,
}

impl IndexRecord {
    pub fn new(text: impl Into<String>, count: u64) -> Self {
        Self {
            text: text.into(),
            count,
        }
    }
}

/// Stateful external-sort builder. Shards are deliberately retained after a
/// successful or failed merge; callers may inspect or explicitly clean them.
pub struct ExternalIndexBuilder {
    options: BuildOptions,
    entries: Vec<(Vec<u32>, u64)>,
    shard_paths: Vec<PathBuf>,
    next_shard: u64,
    records_seen: u64,
    empty_records: u64,
    chains_emitted: u64,
}

impl ExternalIndexBuilder {
    pub fn new(options: BuildOptions) -> Result<Self> {
        options.validate()?;
        if options.output_path.exists() {
            return Err(IndexError::AlreadyExists(options.output_path.clone()));
        }
        fs::create_dir_all(&options.shard_dir)
            .map_err(|error| IndexError::io_at(&options.shard_dir, error))?;
        let existing = list_text_shards(&options.shard_dir, &options.shard_prefix)?;
        if let Some(path) = existing.into_iter().next() {
            return Err(IndexError::AlreadyExists(path));
        }

        Ok(Self {
            options,
            entries: Vec::new(),
            shard_paths: Vec::new(),
            next_shard: 0,
            records_seen: 0,
            empty_records: 0,
            chains_emitted: 0,
        })
    }

    pub fn options(&self) -> &BuildOptions {
        &self.options
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    pub fn pending_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn push_record(&mut self, text: &str, count: u64) -> Result<()> {
        if count == 0 {
            return Err(IndexError::InvalidArgument(
                "record counts must be greater than zero".to_string(),
            ));
        }
        if count > i64::MAX as u64 {
            return Err(IndexError::InvalidArgument(
                "record counts must fit in a signed 64-bit integer".to_string(),
            ));
        }
        self.records_seen = checked_add(self.records_seen, 1, "record count")?;

        let scalars: Vec<u32> = text.chars().map(u32::from).collect();
        if scalars.is_empty() {
            self.empty_records = checked_add(self.empty_records, 1, "empty record count")?;
            return Ok(());
        }

        match self.options.mode {
            IndexMode::Records => self.push_key(scalars, count)?,
            IndexMode::Ngrams { window_codepoints } => {
                for start in 0..scalars.len() {
                    let end = start.saturating_add(window_codepoints).min(scalars.len());
                    self.push_key(scalars[start..end].to_vec(), count)?;
                }
            }
        }
        Ok(())
    }

    fn push_key(&mut self, key: Vec<u32>, count: u64) -> Result<()> {
        self.entries.push((key, count));
        self.chains_emitted = checked_add(self.chains_emitted, 1, "emitted chain count")?;
        if self.entries.len() >= self.options.max_entries_in_memory {
            self.spill()?;
        }
        Ok(())
    }

    /// Force the current in-memory run to an explicit, derived shard path.
    /// Returns None when there is nothing pending.
    pub fn spill(&mut self) -> Result<Option<PathBuf>> {
        if self.entries.is_empty() {
            return Ok(None);
        }
        let path = shard_path(
            &self.options.shard_dir,
            &self.options.shard_prefix,
            self.next_shard,
        );
        if path.exists() {
            return Err(IndexError::AlreadyExists(path));
        }

        self.entries
            .sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let file = create_new_file(&path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{SHARD_MAGIC}")?;
        writeln!(writer, "mode\t{}", self.options.mode.shard_tag())?;

        let mut index = 0;
        while index < self.entries.len() {
            let key = &self.entries[index].0;
            let mut count = self.entries[index].1;
            index += 1;
            while index < self.entries.len() && self.entries[index].0.as_slice() == key.as_slice() {
                count = checked_add(count, self.entries[index].1, "shard frequency")?;
                if count > i64::MAX as u64 {
                    return Err(IndexError::Overflow("shard frequency"));
                }
                index += 1;
            }
            write!(writer, "{count}\t")?;
            write_escaped_key(&mut writer, key)?;
            writer.write_all(b"\n")?;
        }
        writer
            .flush()
            .map_err(|error| IndexError::io_at(&path, error))?;
        let file = writer
            .into_inner()
            .map_err(|error| IndexError::io_at(&path, error.into_error()))?;
        file.sync_all()
            .map_err(|error| IndexError::io_at(&path, error))?;

        self.entries.clear();
        self.shard_paths.push(path.clone());
        self.next_shard = checked_add(self.next_shard, 1, "shard ordinal")?;
        Ok(Some(path))
    }

    pub fn finish(self) -> Result<BuildReport> {
        self.finish_with_progress(|_| {})
    }

    pub fn finish_with_progress<F>(mut self, progress: F) -> Result<BuildReport>
    where
        F: FnMut(MergeProgress),
    {
        self.spill()?;
        let merge = merge_text_shards_with_min_count_and_progress(
            &self.shard_paths,
            &self.options.output_path,
            self.options.mode,
            self.options.min_subtree_count,
            progress,
        )?;
        Ok(BuildReport {
            output_path: merge.output_path,
            shard_paths: self.shard_paths,
            records_seen: self.records_seen,
            empty_records: self.empty_records,
            chains_emitted: self.chains_emitted,
            min_subtree_count: merge.min_subtree_count,
            index: merge.index,
        })
    }
}

pub fn build_index<I>(records: I, options: BuildOptions) -> Result<BuildReport>
where
    I: IntoIterator<Item = IndexRecord>,
{
    let mut builder = ExternalIndexBuilder::new(options)?;
    for record in records {
        builder.push_record(&record.text, record.count)?;
    }
    builder.finish()
}

fn validate_shard_prefix(prefix: &str) -> Result<()> {
    if prefix.is_empty()
        || !prefix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(IndexError::InvalidArgument(
            "shard_prefix must contain only ASCII letters, digits, '-' or '_'".to_string(),
        ));
    }
    Ok(())
}

pub fn shard_path(directory: &Path, prefix: &str, ordinal: u64) -> PathBuf {
    directory.join(format!("{prefix}.{ordinal:06}.shard.txt"))
}

/// List only shards matching the builder's exact numeric naming convention.
pub fn list_text_shards(directory: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    validate_shard_prefix(prefix)?;
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut found = Vec::new();
    let start = format!("{prefix}.");
    let end = ".shard.txt";
    let entries = fs::read_dir(directory).map_err(|error| IndexError::io_at(directory, error))?;
    for entry in entries {
        let entry = entry.map_err(|error| IndexError::io_at(directory, error))?;
        let file_type = entry
            .file_type()
            .map_err(|error| IndexError::io_at(entry.path(), error))?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(middle) = name
            .strip_prefix(&start)
            .and_then(|name| name.strip_suffix(end))
        else {
            continue;
        };
        let Ok(ordinal) = middle.parse::<u64>() else {
            continue;
        };
        found.push((ordinal, entry.path()));
    }
    found.sort_by_key(|(ordinal, _)| *ordinal);
    Ok(found.into_iter().map(|(_, path)| path).collect())
}

fn create_new_file(path: &Path) -> Result<File> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            Err(IndexError::AlreadyExists(path.to_path_buf()))
        }
        Err(error) => Err(IndexError::io_at(path, error)),
    }
}

fn write_escaped_key<W: Write>(writer: &mut W, key: &[u32]) -> Result<()> {
    for &scalar in key {
        let character = char::from_u32(scalar).ok_or_else(|| {
            IndexError::InvalidArgument(format!("U+{scalar:04X} is not a Unicode scalar"))
        })?;
        match character {
            '\\' => writer.write_all(b"\\\\")?,
            '\t' => writer.write_all(b"\\t")?,
            '\n' => writer.write_all(b"\\n")?,
            '\r' => writer.write_all(b"\\r")?,
            character if character.is_control() => write!(writer, "\\u{{{:X}}}", scalar)?,
            character => write!(writer, "{character}")?,
        }
    }
    Ok(())
}

fn unescape_key(path: &Path, line: usize, encoded: &str) -> Result<Vec<u32>> {
    let mut output = Vec::new();
    let mut chars = encoded.chars();
    while let Some(character) = chars.next() {
        if character != '\\' {
            output.push(u32::from(character));
            continue;
        }
        let escape = chars
            .next()
            .ok_or_else(|| IndexError::shard(path, line, "trailing backslash"))?;
        match escape {
            '\\' => output.push(u32::from('\\')),
            't' => output.push(u32::from('\t')),
            'n' => output.push(u32::from('\n')),
            'r' => output.push(u32::from('\r')),
            'u' => {
                if chars.next() != Some('{') {
                    return Err(IndexError::shard(path, line, "expected '{' after \\u"));
                }
                let mut digits = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(digit) if digit.is_ascii_hexdigit() && digits.len() < 6 => {
                            digits.push(digit)
                        }
                        Some(_) => {
                            return Err(IndexError::shard(
                                path,
                                line,
                                "invalid hexadecimal Unicode escape",
                            ));
                        }
                        None => {
                            return Err(IndexError::shard(
                                path,
                                line,
                                "unterminated Unicode escape",
                            ));
                        }
                    }
                }
                if digits.is_empty() {
                    return Err(IndexError::shard(path, line, "empty Unicode escape"));
                }
                let scalar = u32::from_str_radix(&digits, 16)
                    .map_err(|_| IndexError::shard(path, line, "invalid Unicode escape"))?;
                if char::from_u32(scalar).is_none() {
                    return Err(IndexError::shard(
                        path,
                        line,
                        "escape is not a Unicode scalar value",
                    ));
                }
                output.push(scalar);
            }
            _ => return Err(IndexError::shard(path, line, "unknown backslash escape")),
        }
    }
    if output.is_empty() {
        return Err(IndexError::shard(
            path,
            line,
            "empty keys are not supported",
        ));
    }
    Ok(output)
}

struct ShardRecord {
    key: Vec<u32>,
    count: u64,
}

struct ShardCursor {
    path: PathBuf,
    reader: BufReader<File>,
    line: usize,
    previous_key: Option<Vec<u32>>,
    current: Option<ShardRecord>,
    bytes_read: u64,
}

impl ShardCursor {
    fn open(path: &Path, expected_mode: IndexMode) -> Result<Self> {
        let file = File::open(path).map_err(|error| IndexError::io_at(path, error))?;
        let mut cursor = Self {
            path: path.to_path_buf(),
            reader: BufReader::new(file),
            line: 0,
            previous_key: None,
            current: None,
            bytes_read: 0,
        };
        let magic = cursor.read_required_line()?;
        if magic != SHARD_MAGIC {
            return Err(IndexError::shard(path, 1, "bad shard magic"));
        }
        let mode_line = cursor.read_required_line()?;
        let tag = mode_line
            .strip_prefix("mode\t")
            .ok_or_else(|| IndexError::shard(path, 2, "missing mode header"))?;
        let mode = IndexMode::parse_shard_tag(tag)
            .ok_or_else(|| IndexError::shard(path, 2, "invalid mode header"))?;
        if mode != expected_mode {
            return Err(IndexError::shard(
                path,
                2,
                "shard mode does not match the merge",
            ));
        }
        cursor.advance()?;
        Ok(cursor)
    }

    fn read_required_line(&mut self) -> Result<String> {
        let mut buffer = String::new();
        let read = self
            .reader
            .read_line(&mut buffer)
            .map_err(|error| IndexError::io_at(&self.path, error))?;
        self.line += 1;
        if read == 0 {
            return Err(IndexError::shard(
                &self.path,
                self.line,
                "unexpected end of shard",
            ));
        }
        self.bytes_read = checked_add(self.bytes_read, read as u64, "shard bytes read")?;
        trim_line_ending(&mut buffer);
        Ok(buffer)
    }

    fn advance(&mut self) -> Result<()> {
        let mut buffer = String::new();
        let read = self
            .reader
            .read_line(&mut buffer)
            .map_err(|error| IndexError::io_at(&self.path, error))?;
        if read == 0 {
            self.current = None;
            return Ok(());
        }
        self.bytes_read = checked_add(self.bytes_read, read as u64, "shard bytes read")?;
        self.line += 1;
        trim_line_ending(&mut buffer);
        let (count, encoded) = buffer.split_once('\t').ok_or_else(|| {
            IndexError::shard(
                &self.path,
                self.line,
                "expected count and tab-separated key",
            )
        })?;
        let count = count
            .parse::<u64>()
            .map_err(|_| IndexError::shard(&self.path, self.line, "invalid count"))?;
        if count == 0 || count > i64::MAX as u64 {
            return Err(IndexError::shard(
                &self.path,
                self.line,
                "count must be between 1 and i64::MAX",
            ));
        }
        let key = unescape_key(&self.path, self.line, encoded)?;
        if self
            .previous_key
            .as_ref()
            .is_some_and(|previous| key.as_slice() <= previous.as_slice())
        {
            return Err(IndexError::shard(
                &self.path,
                self.line,
                "keys are not strictly increasing",
            ));
        }
        self.previous_key = Some(key.clone());
        self.current = Some(ShardRecord { key, count });
        Ok(())
    }
}

fn trim_line_ending(line: &mut String) {
    if line.ends_with('\n') {
        line.pop();
        if line.ends_with('\r') {
            line.pop();
        }
    }
}

/// Merge already-sorted text shards into one v2 index. A same-directory
/// partial file is synced before it is renamed into place. Shards and a failed
/// partial file are never removed by this function.
pub fn merge_text_shards(
    shard_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    mode: IndexMode,
) -> Result<MergeReport> {
    merge_text_shards_with_min_count(shard_paths, output_path, mode, 1)
}

/// Merges sorted shards while pruning n-gram branches below a weighted
/// frequency threshold. Frequencies from omitted branches remain included in
/// every retained ancestor's subtree count.
pub fn merge_text_shards_with_min_count(
    shard_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    mode: IndexMode,
    min_subtree_count: u64,
) -> Result<MergeReport> {
    merge_text_shards_with_min_count_and_progress(
        shard_paths,
        output_path,
        mode,
        min_subtree_count,
        |_| {},
    )
}

pub fn merge_text_shards_with_min_count_and_progress<F>(
    shard_paths: &[PathBuf],
    output_path: impl AsRef<Path>,
    mode: IndexMode,
    min_subtree_count: u64,
    mut progress: F,
) -> Result<MergeReport>
where
    F: FnMut(MergeProgress),
{
    mode.validate()?;
    if min_subtree_count == 0 || min_subtree_count > i64::MAX as u64 {
        return Err(IndexError::InvalidArgument(
            "min_subtree_count must be between 1 and i64::MAX".to_string(),
        ));
    }
    if min_subtree_count != 1 && !mode.implicit_prefixes() {
        return Err(IndexError::InvalidArgument(
            "subtree pruning is only supported for n-gram indexes".to_string(),
        ));
    }
    let output_path = output_path.as_ref();
    if output_path.exists() {
        return Err(IndexError::AlreadyExists(output_path.to_path_buf()));
    }
    let partial_path = partial_output_path(output_path);
    if partial_path.exists() {
        return Err(IndexError::AlreadyExists(partial_path));
    }

    let bytes_total = shard_paths.iter().try_fold(0_u64, |total, path| {
        let size = fs::metadata(path)
            .map_err(|error| IndexError::io_at(path, error))?
            .len();
        checked_add(total, size, "total shard bytes")
    })?;
    let mut cursors = Vec::new();
    for path in shard_paths {
        cursors.push(ShardCursor::open(path, mode)?);
    }
    let mut bytes_read = cursors.iter().try_fold(0_u64, |total, cursor| {
        checked_add(total, cursor.bytes_read, "merged shard bytes")
    })?;
    let mut shard_records_read = 0_u64;
    let mut unique_keys_written = 0_u64;
    let mut last_progress_bytes = bytes_read;
    progress(MergeProgress {
        bytes_read,
        bytes_total,
        shard_records_read,
        unique_keys_written,
        shard_count: shard_paths.len(),
        complete: false,
    });

    let output = create_new_file(&partial_path)?;
    let mut writer = TrieWriter::with_min_subtree_count(output, mode, min_subtree_count)?;
    let mut heap: BinaryHeap<Reverse<(Vec<u32>, usize)>> = BinaryHeap::new();
    for (index, cursor) in cursors.iter().enumerate() {
        if let Some(record) = &cursor.current {
            heap.push(Reverse((record.key.clone(), index)));
        }
    }

    let mut pending_key: Option<Vec<u32>> = None;
    let mut pending_count = 0u64;
    while let Some(Reverse((heap_key, cursor_index))) = heap.pop() {
        let cursor = &mut cursors[cursor_index];
        let record = cursor.current.take().ok_or_else(|| {
            IndexError::shard(
                &cursor.path,
                cursor.line,
                "merge cursor lost its current key",
            )
        })?;
        if record.key != heap_key {
            return Err(IndexError::shard(
                &cursor.path,
                cursor.line,
                "merge heap key disagrees with the shard cursor",
            ));
        }

        let cursor_bytes_before = cursor.bytes_read;
        cursor.advance()?;
        bytes_read = checked_add(
            bytes_read,
            cursor.bytes_read.saturating_sub(cursor_bytes_before),
            "merged shard bytes",
        )?;
        shard_records_read = checked_add(shard_records_read, 1, "merged shard records")?;
        if let Some(next) = &cursor.current {
            heap.push(Reverse((next.key.clone(), cursor_index)));
        }

        match &pending_key {
            Some(key) if key.as_slice() == record.key.as_slice() => {
                pending_count = checked_add(pending_count, record.count, "merged frequency")?;
                if pending_count > i64::MAX as u64 {
                    return Err(IndexError::Overflow("merged frequency"));
                }
            }
            Some(_) => {
                writer.push_sorted(
                    pending_key.as_deref().expect("pending key is present"),
                    pending_count,
                )?;
                unique_keys_written = checked_add(unique_keys_written, 1, "merged unique keys")?;
                pending_key = Some(record.key);
                pending_count = record.count;
            }
            None => {
                pending_key = Some(record.key);
                pending_count = record.count;
            }
        }
        if bytes_read.saturating_sub(last_progress_bytes) >= MERGE_PROGRESS_STEP
            || bytes_read >= bytes_total
        {
            progress(MergeProgress {
                bytes_read,
                bytes_total,
                shard_records_read,
                unique_keys_written,
                shard_count: shard_paths.len(),
                complete: false,
            });
            last_progress_bytes = bytes_read;
        }
    }
    if let Some(key) = pending_key {
        writer.push_sorted(&key, pending_count)?;
        unique_keys_written = checked_add(unique_keys_written, 1, "merged unique keys")?;
    }

    progress(MergeProgress {
        bytes_read,
        bytes_total,
        shard_records_read,
        unique_keys_written,
        shard_count: shard_paths.len(),
        complete: false,
    });

    let (output, stats) = writer.finish()?;
    output
        .sync_all()
        .map_err(|error| IndexError::io_at(&partial_path, error))?;
    drop(output);
    if output_path.exists() {
        return Err(IndexError::AlreadyExists(output_path.to_path_buf()));
    }
    fs::rename(&partial_path, output_path)
        .map_err(|error| IndexError::io_at(output_path, error))?;
    progress(MergeProgress {
        bytes_read,
        bytes_total,
        shard_records_read,
        unique_keys_written,
        shard_count: shard_paths.len(),
        complete: true,
    });
    Ok(MergeReport {
        output_path: output_path.to_path_buf(),
        shard_count: shard_paths.len(),
        min_subtree_count,
        index: stats,
    })
}

fn partial_output_path(output_path: &Path) -> PathBuf {
    let mut value = output_path.as_os_str().to_os_string();
    value.push(format!(".partial.{}", std::process::id()));
    PathBuf::from(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn encode_records(entries: &[(&str, u64)]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::new(cursor, IndexMode::Records).unwrap();
        for &(key, count) in entries {
            writer.push_str_sorted(key, count).unwrap();
        }
        let (cursor, _) = writer.finish().unwrap();
        cursor.into_inner()
    }

    fn explicit_test_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nutrimatic_zh_index_{label}_{}_{}_{}",
            std::process::id(),
            nanos,
            counter
        ))
    }

    #[test]
    fn ordered_writer_and_reader_round_trip_unicode() {
        let data = encode_records(&[("中", 2), ("中国", 3), ("文", 1)]);
        let reader = IndexReader::from_bytes(data).unwrap();

        assert_eq!(reader.mode(), IndexMode::Records);
        assert!(!reader.implicit_prefixes());
        assert_eq!(reader.root_subtree(), 6);
        assert_eq!(reader.node_count(), 4);

        let middle = reader.find("中").unwrap().unwrap();
        assert_eq!(middle.terminal, 2);
        assert_eq!(middle.subtree, 5);
        let china = reader.find("中国").unwrap().unwrap();
        assert_eq!(china.terminal, 3);
        assert_eq!(china.subtree, 3);
        assert!(reader.find("中华").unwrap().is_none());

        let validation = reader.validate_full().unwrap();
        assert_eq!(validation.node_count, 4);
        assert_eq!(validation.edge_count, 3);
        let walked: Vec<_> = reader
            .walk()
            .unwrap()
            .map(|entry| entry.unwrap().key)
            .collect();
        assert_eq!(walked, ["中", "中国", "文"]);
    }

    #[test]
    fn duplicate_sorted_keys_are_aggregated_by_the_writer() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::new(cursor, IndexMode::Records).unwrap();
        writer.push_str_sorted("中文", 2).unwrap();
        writer.push_str_sorted("中文", 5).unwrap();
        let (cursor, stats) = writer.finish().unwrap();
        assert_eq!(stats.key_count, 1);
        let reader = IndexReader::from_bytes(cursor.into_inner()).unwrap();
        assert_eq!(reader.find("中文").unwrap().unwrap().terminal, 7);
    }

    #[test]
    fn ngram_pruning_keeps_exact_frequency_on_retained_prefixes() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::with_min_subtree_count(
            cursor,
            IndexMode::Ngrams {
                window_codepoints: 2,
            },
            2,
        )
        .unwrap();
        writer.push_str_sorted("中华", 1).unwrap();
        writer.push_str_sorted("中国", 1).unwrap();
        writer.push_str_sorted("人民", 1).unwrap();
        let (cursor, stats) = writer.finish().unwrap();

        let reader = IndexReader::from_bytes(cursor.into_inner()).unwrap();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.pruned_node_count, 4);
        assert_eq!(reader.root_subtree(), 3);
        assert_eq!(reader.find("中").unwrap().unwrap().subtree, 2);
        assert!(reader.find("中华").unwrap().is_none());
        assert!(reader.find("中国").unwrap().is_none());
        assert!(reader.find("人").unwrap().is_none());
        assert_eq!(
            reader
                .walk()
                .unwrap()
                .map(|entry| entry.unwrap().key)
                .collect::<Vec<_>>(),
            ["中"]
        );
        reader.validate_full().unwrap();
    }

    #[test]
    fn fully_pruned_ngram_index_keeps_only_root_residual_frequency() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::with_min_subtree_count(
            cursor,
            IndexMode::Ngrams {
                window_codepoints: 2,
            },
            3,
        )
        .unwrap();
        writer.push_str_sorted("中华", 1).unwrap();
        writer.push_str_sorted("人民", 1).unwrap();
        let (cursor, stats) = writer.finish().unwrap();

        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.pruned_node_count, 4);
        assert_eq!(stats.root_subtree, 2);

        let reader = IndexReader::from_bytes(cursor.into_inner()).unwrap();
        let root = reader.node(reader.root()).unwrap();
        assert_eq!(root.terminal, 2);
        assert_eq!(root.subtree, 2);
        assert!(root.edges.is_empty());
        assert!(reader.walk().unwrap().next().is_none());
        reader.validate_full().unwrap();
    }

    #[test]
    fn writer_rejects_unsorted_or_nonempty_output() {
        let cursor = Cursor::new(vec![1]);
        assert!(matches!(
            TrieWriter::new(cursor, IndexMode::Records),
            Err(IndexError::OutputNotEmpty)
        ));

        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::new(cursor, IndexMode::Records).unwrap();
        writer.push_str_sorted("文", 1).unwrap();
        assert!(matches!(
            writer.push_str_sorted("中", 1),
            Err(IndexError::OutOfOrder)
        ));
    }

    #[test]
    fn reader_rejects_bad_magic_and_malformed_varint() {
        let mut bad_magic = encode_records(&[("中", 1)]);
        bad_magic[0] ^= 0xff;
        assert!(matches!(
            IndexReader::from_bytes(bad_magic),
            Err(IndexError::InvalidFormat { .. })
        ));

        let mut malformed = encode_records(&[("中", 1)]);
        let root = get_u64(&malformed, 24) as usize;
        malformed[root..].fill(0x80);
        assert!(matches!(
            IndexReader::from_bytes(malformed),
            Err(IndexError::InvalidFormat { .. })
        ));
    }

    #[test]
    fn external_ngram_build_counts_unicode_scalars_and_keeps_shards() {
        let root = explicit_test_dir("ngrams");
        let shards = root.join("shards");
        let output = root.join("index.nutri2");
        let mut options = BuildOptions::new(
            &output,
            &shards,
            "run",
            IndexMode::Ngrams {
                window_codepoints: 2,
            },
        );
        options.max_entries_in_memory = 2;

        let mut builder = ExternalIndexBuilder::new(options.clone()).unwrap();
        builder.push_record("𠮷中国", 1).unwrap();
        let report = builder.finish().unwrap();

        assert_eq!(report.records_seen, 1);
        assert_eq!(report.chains_emitted, 3);
        assert_eq!(report.shard_paths.len(), 2);
        assert!(report.shard_paths.iter().all(|path| path.exists()));
        assert!(output.exists());

        let reader = IndexReader::open(&output).unwrap();
        assert!(reader.implicit_prefixes());
        assert_eq!(reader.window_codepoints(), Some(2));
        assert_eq!(reader.find("𠮷中").unwrap().unwrap().terminal, 1);
        let implicit = reader.find("中").unwrap().unwrap();
        assert_eq!(implicit.terminal, 0);
        assert_eq!(implicit.subtree, 1);
        assert_eq!(reader.find("中国").unwrap().unwrap().terminal, 1);
        assert_eq!(reader.find("国").unwrap().unwrap().terminal, 1);
        reader.validate_full().unwrap();

        assert!(matches!(
            ExternalIndexBuilder::new(options),
            Err(IndexError::AlreadyExists(path)) if path == output
        ));

        for path in &report.shard_paths {
            fs::remove_file(path).unwrap();
        }
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&shards).unwrap();
        fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn external_builder_reports_monotonic_merge_progress() {
        let root = explicit_test_dir("merge_progress");
        let shards = root.join("shards");
        let output = root.join("index.nutri2");
        let mut options = BuildOptions::new(&output, &shards, "progress", IndexMode::Records);
        options.max_entries_in_memory = 1;

        let mut builder = ExternalIndexBuilder::new(options).unwrap();
        builder.push_record("中国", 2).unwrap();
        builder.push_record("中华", 3).unwrap();
        let mut events = Vec::new();
        let report = builder
            .finish_with_progress(|event| events.push(event))
            .unwrap();

        assert!(!events.is_empty());
        assert!(
            events
                .windows(2)
                .all(|pair| pair[0].bytes_read <= pair[1].bytes_read)
        );
        let final_event = events.last().unwrap();
        assert!(final_event.complete);
        assert_eq!(final_event.bytes_read, final_event.bytes_total);
        assert_eq!(final_event.shard_count, 2);
        assert_eq!(final_event.shard_records_read, 2);
        assert_eq!(final_event.unique_keys_written, 2);

        for path in &report.shard_paths {
            fs::remove_file(path).unwrap();
        }
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&shards).unwrap();
        fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn external_pruning_uses_global_cross_shard_frequency() {
        let root = explicit_test_dir("pruned_ngrams");
        let shards = root.join("shards");
        let output = root.join("index.nutri2");
        let mut options = BuildOptions::new(
            &output,
            &shards,
            "pruned",
            IndexMode::Ngrams {
                window_codepoints: 2,
            },
        );
        options.max_entries_in_memory = 1;
        options.min_subtree_count = 2;

        let mut builder = ExternalIndexBuilder::new(options).unwrap();
        builder.push_record("中华", 1).unwrap();
        builder.push_record("中华", 1).unwrap();
        builder.push_record("中国", 1).unwrap();
        let report = builder.finish().unwrap();

        // Each raw chain was forced into its own shard. Only the final merge
        // can see that 中华 and 华 reach the global threshold.
        assert_eq!(report.shard_paths.len(), 6);
        assert_eq!(report.min_subtree_count, 2);
        assert_eq!(report.index.root_subtree, 6);
        assert_eq!(report.index.node_count, 4);
        assert_eq!(report.index.pruned_node_count, 2);

        let reader = IndexReader::open(&output).unwrap();
        assert_eq!(reader.find("中").unwrap().unwrap().subtree, 3);
        assert_eq!(reader.find("中华").unwrap().unwrap().subtree, 2);
        assert_eq!(reader.find("华").unwrap().unwrap().subtree, 2);
        assert!(reader.find("中国").unwrap().is_none());
        assert!(reader.find("国").unwrap().is_none());
        reader.validate_full().unwrap();

        for path in &report.shard_paths {
            fs::remove_file(path).unwrap();
        }
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&shards).unwrap();
        fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn subtree_pruning_is_rejected_for_record_indexes() {
        let cursor = Cursor::new(Vec::new());
        assert!(matches!(
            TrieWriter::with_min_subtree_count(cursor, IndexMode::Records, 2),
            Err(IndexError::InvalidArgument(_))
        ));

        let root = explicit_test_dir("invalid_record_pruning");
        let mut options = BuildOptions::new(
            root.join("index.nutri2"),
            root.join("shards"),
            "records",
            IndexMode::Records,
        );
        options.min_subtree_count = 2;
        assert!(matches!(
            ExternalIndexBuilder::new(options),
            Err(IndexError::InvalidArgument(_))
        ));
        assert!(!root.exists());
    }

    #[test]
    fn text_shards_round_trip_escaped_controls() {
        let root = explicit_test_dir("escapes");
        let shards = root.join("shards");
        let output = root.join("index.nutri2");
        let options = BuildOptions::new(&output, &shards, "escaped", IndexMode::Records);
        let report = build_index([IndexRecord::new("中\\\n\t文", 4)], options).unwrap();
        let reader = IndexReader::open(&output).unwrap();
        assert_eq!(reader.find("中\\\n\t文").unwrap().unwrap().terminal, 4);

        for path in &report.shard_paths {
            fs::remove_file(path).unwrap();
        }
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&shards).unwrap();
        fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn empty_build_still_has_a_valid_root() {
        let root = explicit_test_dir("empty");
        let shards = root.join("shards");
        let output = root.join("index.nutri2");
        let options = BuildOptions::new(&output, &shards, "empty", IndexMode::Records);
        let report = build_index(Vec::<IndexRecord>::new(), options).unwrap();
        assert!(report.shard_paths.is_empty());
        let reader = IndexReader::open(&output).unwrap();
        assert_eq!(reader.root_subtree(), 0);
        assert_eq!(reader.node_count(), 1);
        reader.validate_full().unwrap();

        fs::remove_file(&output).unwrap();
        fs::remove_dir(&shards).unwrap();
        fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn header_only_shard_merges_to_an_empty_index() {
        let root = explicit_test_dir("header_only_shard");
        fs::create_dir_all(&root).unwrap();
        let shard = root.join("empty.000000.shard.txt");
        let output = root.join("index.nutri2");

        let file = create_new_file(&shard).unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{SHARD_MAGIC}").unwrap();
        writeln!(writer, "mode\trecords").unwrap();
        writer.flush().unwrap();
        let file = writer.into_inner().unwrap();
        file.sync_all().unwrap();
        drop(file);

        let report =
            merge_text_shards(std::slice::from_ref(&shard), &output, IndexMode::Records).unwrap();
        assert_eq!(report.shard_count, 1);
        assert_eq!(report.index.root_subtree, 0);
        assert!(!partial_output_path(&output).exists());

        let reader = IndexReader::open(&output).unwrap();
        assert_eq!(reader.root_subtree(), 0);
        assert_eq!(reader.node_count(), 1);
        reader.validate_full().unwrap();

        fs::remove_file(&shard).unwrap();
        fs::remove_file(&output).unwrap();
        fs::remove_dir(&root).unwrap();
    }

    #[test]
    fn merge_refuses_a_preexisting_partial_file() {
        let root = explicit_test_dir("partial_exists");
        fs::create_dir_all(&root).unwrap();
        let output = root.join("index.nutri2");
        let partial = partial_output_path(&output);
        File::create(&partial).unwrap();

        assert!(matches!(
            merge_text_shards(&[], &output, IndexMode::Records),
            Err(IndexError::AlreadyExists(path)) if path == partial
        ));
        assert!(!output.exists());

        fs::remove_file(&partial).unwrap();
        fs::remove_dir(&root).unwrap();
    }
}
