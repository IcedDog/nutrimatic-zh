//! Best-first trie search driven by Brzozowski derivatives.

use crate::index::{IndexError, IndexReader};
use crate::query::{Expr, Program};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub limit: usize,
    pub max_nodes: u64,
    pub max_states: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 100,
            max_nodes: 1_000_000,
            max_states: 20_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchStopReason {
    NodeLimit,
    StateLimit,
}

impl SearchStopReason {
    pub fn code(self) -> &'static str {
        match self {
            Self::NodeLimit => "node_limit",
            Self::StateLimit => "state_limit",
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::NodeLimit => "达到节点检查上限",
            Self::StateLimit => "达到查询状态上限",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchHit {
    pub text: String,
    pub score: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SearchReport {
    pub results: Vec<SearchHit>,
    pub visited: u64,
    pub derivative_states: usize,
    pub truncated: bool,
    pub stop_reason: Option<SearchStopReason>,
}

#[derive(Debug)]
pub enum SearchError {
    Index(IndexError),
    InvalidOptions(String),
    StateLimit { limit: usize },
}

impl fmt::Display for SearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(error) => write!(formatter, "索引读取失败：{error}"),
            Self::InvalidOptions(message) => write!(formatter, "搜索参数无效：{message}"),
            Self::StateLimit { limit } => write!(
                formatter,
                "查询自动机超过 {limit} 个状态；请缩短乱序式或减少交集"
            ),
        }
    }
}

impl Error for SearchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Index(error) => Some(error),
            _ => None,
        }
    }
}

impl From<IndexError> for SearchError {
    fn from(error: IndexError) -> Self {
        Self::Index(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueueItem {
    bound: u64,
    serial: u64,
    node_offset: u64,
    state: u32,
    text: String,
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound
            .cmp(&other.bound)
            // Earlier insertions win deterministic ties.
            .then_with(|| other.serial.cmp(&self.serial))
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RankedHit(SearchHit);

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .score
            .cmp(&other.0.score)
            // For equal scores a lexicographically smaller result ranks first.
            .then_with(|| other.0.text.cmp(&self.0.text))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct DerivativeMachine<'a> {
    program: &'a Program,
    states: Vec<Expr>,
    ids: HashMap<Expr, u32>,
    transitions: HashMap<(u32, char), Option<u32>>,
    max_states: usize,
}

impl<'a> DerivativeMachine<'a> {
    fn new(program: &'a Program, max_states: usize) -> Result<Self, SearchError> {
        if max_states == 0 || max_states > u32::MAX as usize {
            return Err(SearchError::InvalidOptions(
                "max_states 必须在 1..=u32::MAX 范围内".to_owned(),
            ));
        }
        let root = program.root().clone();
        Ok(Self {
            program,
            states: vec![root.clone()],
            ids: HashMap::from([(root, 0)]),
            transitions: HashMap::new(),
            max_states,
        })
    }

    fn nullable(&self, state: u32) -> bool {
        self.program.nullable(&self.states[state as usize])
    }

    fn transition(&mut self, state: u32, character: char) -> Result<Option<u32>, SearchError> {
        if let Some(&cached) = self.transitions.get(&(state, character)) {
            return Ok(cached);
        }

        let next = self
            .program
            .derive_approx(&self.states[state as usize], character);
        let next_id = if self.program.is_dead(&next) {
            None
        } else if let Some(&id) = self.ids.get(&next) {
            Some(id)
        } else {
            if self.states.len() >= self.max_states {
                return Err(SearchError::StateLimit {
                    limit: self.max_states,
                });
            }
            let id = self.states.len() as u32;
            self.states.push(next.clone());
            self.ids.insert(next, id);
            Some(id)
        };
        self.transitions.insert((state, character), next_id);
        Ok(next_id)
    }
}

/// Searches an index in descending corpus-frequency order.
///
/// The priority of an unexplored trie node is its subtree frequency, an upper
/// bound on every result below it. Once the retained result set is full and
/// that upper bound falls below its worst score, the search is complete.
pub fn search(
    index: &IndexReader,
    program: &Program,
    options: &SearchOptions,
) -> Result<SearchReport, SearchError> {
    search_internal(index, program, options, Duration::MAX, None)
}

/// Searches an index and periodically reports the best results found so far.
///
/// Progress reports are provisional: later, higher-scoring hits may replace
/// entries until the search completes. Returning `false` from the callback
/// stops the search early, which is useful when a streaming client disconnects.
pub fn search_with_progress<F>(
    index: &IndexReader,
    program: &Program,
    options: &SearchOptions,
    interval: Duration,
    mut progress: F,
) -> Result<SearchReport, SearchError>
where
    F: FnMut(&SearchReport) -> bool,
{
    search_internal(index, program, options, interval, Some(&mut progress))
}

fn search_internal(
    index: &IndexReader,
    program: &Program,
    options: &SearchOptions,
    progress_interval: Duration,
    mut progress: Option<&mut dyn FnMut(&SearchReport) -> bool>,
) -> Result<SearchReport, SearchError> {
    if options.limit == 0 {
        return Err(SearchError::InvalidOptions("limit 必须大于零".to_owned()));
    }
    if options.max_nodes == 0 {
        return Err(SearchError::InvalidOptions(
            "max_nodes 必须大于零".to_owned(),
        ));
    }
    let mut machine = DerivativeMachine::new(program, options.max_states)?;
    let mut queue = BinaryHeap::new();
    let mut serial = 0_u64;
    queue.push(QueueItem {
        bound: index.root_subtree(),
        serial,
        node_offset: index.root_offset(),
        state: 0,
        text: String::new(),
    });
    // RankedHit orders better hits higher. Reverse therefore keeps the worst
    // retained hit at the top, making bounded top-N replacement O(log N).
    let mut retained: BinaryHeap<Reverse<RankedHit>> = BinaryHeap::new();
    let mut visited = 0_u64;
    let mut stop_reason = None;
    let mut last_progress = Instant::now();

    'search: loop {
        if stop_reason.is_some() {
            break;
        }
        let Some(item) = queue.pop() else {
            break;
        };
        if visited >= options.max_nodes {
            stop_reason = Some(SearchStopReason::NodeLimit);
            break;
        }

        if retained.len() == options.limit {
            let worst_score = retained.peek().expect("retained is non-empty").0.0.score;
            if item.bound < worst_score {
                break;
            }
        }

        visited += 1;
        let node = index.node(item.node_offset)?;
        if node.subtree != item.bound {
            return Err(SearchError::Index(IndexError::InvalidFormat {
                offset: usize::try_from(item.node_offset).unwrap_or(usize::MAX),
                message: "父边频次与子节点不一致".to_owned(),
            }));
        }

        if !item.text.is_empty() && machine.nullable(item.state) {
            let score = if index.implicit_prefixes() {
                node.subtree
            } else {
                node.terminal
            };
            if score > 0 && (program.capture_count == 0 || program.matches(&item.text)) {
                retain_hit(
                    &mut retained,
                    options.limit,
                    SearchHit {
                        text: item.text.clone(),
                        score,
                    },
                );
            }
        }

        for edge in node.edges {
            let next_state = match machine.transition(item.state, edge.label) {
                Ok(Some(next_state)) => next_state,
                Ok(None) => continue,
                Err(SearchError::StateLimit { .. }) => {
                    stop_reason = Some(SearchStopReason::StateLimit);
                    break 'search;
                }
                Err(error) => return Err(error),
            };
            serial = serial.wrapping_add(1);
            let mut text = item.text.clone();
            text.push(edge.label);
            queue.push(QueueItem {
                bound: edge.subtree,
                serial,
                node_offset: edge.child_offset,
                state: next_state,
                text,
            });
        }

        let should_check_progress = progress_interval.is_zero() || visited % 256 == 0;
        if should_check_progress
            && last_progress.elapsed() >= progress_interval
            && let Some(callback) = progress.as_mut()
        {
            let report =
                search_report_snapshot(&retained, visited, machine.states.len(), stop_reason);
            if !callback(&report) {
                break;
            }
            last_progress = Instant::now();
        }
    }

    let results = sorted_results(retained.into_iter().map(|Reverse(hit)| hit.0));

    Ok(SearchReport {
        results,
        visited,
        derivative_states: machine.states.len(),
        truncated: stop_reason.is_some(),
        stop_reason,
    })
}

fn search_report_snapshot(
    retained: &BinaryHeap<Reverse<RankedHit>>,
    visited: u64,
    derivative_states: usize,
    stop_reason: Option<SearchStopReason>,
) -> SearchReport {
    let results = sorted_results(retained.iter().map(|Reverse(hit)| hit.0.clone()));
    SearchReport {
        results,
        visited,
        derivative_states,
        truncated: stop_reason.is_some(),
        stop_reason,
    }
}

fn sorted_results(results: impl IntoIterator<Item = SearchHit>) -> Vec<SearchHit> {
    let mut results: Vec<SearchHit> = results.into_iter().collect();
    results.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.text.cmp(&right.text))
    });
    results
}

fn retain_hit(retained: &mut BinaryHeap<Reverse<RankedHit>>, limit: usize, candidate: SearchHit) {
    let candidate = RankedHit(candidate);
    if retained.len() < limit {
        retained.push(Reverse(candidate));
        return;
    }

    let should_replace = retained
        .peek()
        .is_some_and(|Reverse(worst)| candidate > *worst);
    if should_replace {
        retained.pop();
        retained.push(Reverse(candidate));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexMode, TrieWriter};
    use crate::query::Program;
    use std::io::Cursor;

    fn sample_index(mode: IndexMode) -> IndexReader {
        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::new(cursor, mode).unwrap();
        writer.push_str_sorted("中华", 5).unwrap();
        writer.push_str_sorted("中国", 8).unwrap();
        writer.push_str_sorted("中国人", 3).unwrap();
        writer.push_str_sorted("人民", 7).unwrap();
        let (cursor, _) = writer.finish().unwrap();
        IndexReader::from_bytes(cursor.into_inner()).unwrap()
    }

    #[test]
    fn record_results_use_terminal_frequency() {
        let index = sample_index(IndexMode::Records);
        let program = Program::parse("中.*", None).unwrap();
        let report = search(&index, &program, &SearchOptions::default()).unwrap();
        assert_eq!(
            report.results,
            vec![
                SearchHit {
                    text: "中国".to_owned(),
                    score: 8,
                },
                SearchHit {
                    text: "中华".to_owned(),
                    score: 5,
                },
                SearchHit {
                    text: "中国人".to_owned(),
                    score: 3,
                },
            ]
        );
    }

    #[test]
    fn capture_constraints_are_checked_before_returning() {
        let index = sample_index(IndexMode::Records);
        let program = Program::parse("@t(A)@t(A)", None).unwrap();
        let report = search(&index, &program, &SearchOptions::default()).unwrap();
        assert!(report.results.is_empty());
    }

    #[test]
    fn computation_limit_marks_report_truncated() {
        let index = sample_index(IndexMode::Records);
        let program = Program::parse(".*", None).unwrap();
        let options = SearchOptions {
            max_nodes: 1,
            ..SearchOptions::default()
        };
        let report = search(&index, &program, &options).unwrap();
        assert!(report.truncated);
        assert_eq!(report.stop_reason, Some(SearchStopReason::NodeLimit));
    }

    #[test]
    fn state_limit_returns_partial_report_instead_of_an_error() {
        let index = sample_index(IndexMode::Records);
        let program = Program::parse("中.*", None).unwrap();
        let options = SearchOptions {
            max_states: 1,
            ..SearchOptions::default()
        };
        let report = search(&index, &program, &options).unwrap();
        assert!(report.truncated);
        assert_eq!(report.stop_reason, Some(SearchStopReason::StateLimit));
    }

    #[test]
    fn progress_search_reports_provisional_results() {
        let index = sample_index(IndexMode::Records);
        let program = Program::parse("中.*", None).unwrap();
        let mut reports = Vec::new();
        let final_report = search_with_progress(
            &index,
            &program,
            &SearchOptions::default(),
            Duration::ZERO,
            |report| {
                reports.push(report.clone());
                true
            },
        )
        .unwrap();
        assert!(!reports.is_empty());
        assert_eq!(reports.last().unwrap().results, final_report.results);
    }
}
