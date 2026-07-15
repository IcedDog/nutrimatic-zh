//! Command-line entry point and the built-in HTTP interface.

use crate::AppResult;
use crate::chinese::is_han;
use crate::corpus::{PrepareProgress, PrepareReport, SourceKind, prepare_source_with_progress};
use crate::index::{BuildOptions, ExternalIndexBuilder, IndexMode, IndexReader, MergeProgress};
use crate::query::Program;
use crate::search::{SearchOptions, SearchReport, search, search_with_progress};

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const HOME_HTML: &str = include_str!("../web_static/index.html");
const USAGE_HTML: &str = include_str!("../web_static/usage.html");
const STYLE_CSS: &str = include_str!("../web_static/style.css");
const APP_JS: &str = include_str!("../web_static/app.js");
const ROBOTS_TXT: &str = include_str!("../web_static/robots.txt");
const FAVICON: &[u8] = include_bytes!("../web_static/favicon.ico");
const TEA_SMALL: &[u8] = include_bytes!("../web_static/nutritea-small.jpg");

pub fn run_cli<I>(arguments: I) -> AppResult<()>
where
    I: IntoIterator<Item = OsString>,
{
    let mut arguments = arguments.into_iter();
    let program = arguments
        .next()
        .unwrap_or_else(|| OsString::from("nutrimatic-zh"));
    let Some(command) = arguments.next() else {
        print_help(&program);
        return Ok(());
    };
    let command = command
        .to_str()
        .ok_or_else(|| invalid_input("子命令必须是 UTF-8/Unicode 文本"))?;
    let rest: Vec<OsString> = arguments.collect();

    match command {
        "help" | "--help" | "-h" => print_help(&program),
        "prepare" => command_prepare(rest)?,
        "index" => command_index(rest)?,
        "search" => command_search(rest)?,
        "serve" => command_serve(rest)?,
        "inspect" => command_inspect(rest)?,
        "version" | "--version" | "-V" => {
            println!("nutrimatic-zh {}", env!("CARGO_PKG_VERSION"));
        }
        other => {
            return Err(
                invalid_input(format!("未知子命令 {other:?}；请运行 help 查看用法")).into(),
            );
        }
    }
    Ok(())
}

fn print_help(program: &OsStr) {
    let program = Path::new(program)
        .file_name()
        .unwrap_or(program)
        .to_string_lossy();
    println!(
        r#"Nutrimatic 中文版（Rust / Unicode）

用法：
  {program} prepare --kind KIND --input SOURCE --output RECORDS.tsv --report AUDIT.json
  {program} index   --output INDEX.ntri --shard-dir DIR RECORDS.tsv...
  {program} search  --index INDEX.ntri QUERY
  {program} serve   --index INDEX.ntri [--bind 127.0.0.1:8080]
  {program} inspect --index INDEX.ntri [--full]

推荐流程：先解析语料并人工检查 audit.json，再单独执行 index。大型语料、构建产物、
临时文件和索引应保存在仓库之外。运行 `{program} <子命令> --help` 可查看参数。"#
    );
}

#[derive(Default)]
struct ParsedOptions {
    values: HashMap<String, OsString>,
    flags: HashSet<String>,
    positional: Vec<OsString>,
}

fn parse_options(
    arguments: Vec<OsString>,
    value_names: &[&str],
    flag_names: &[&str],
) -> io::Result<ParsedOptions> {
    let value_names: HashSet<&str> = value_names.iter().copied().collect();
    let flag_names: HashSet<&str> = flag_names.iter().copied().collect();
    let mut output = ParsedOptions::default();
    let mut arguments = arguments.into_iter().peekable();
    let mut positional_only = false;

    while let Some(argument) = arguments.next() {
        if positional_only {
            output.positional.push(argument);
            continue;
        }
        if argument == "--" {
            positional_only = true;
            continue;
        }
        let Some(text) = argument.to_str() else {
            output.positional.push(argument);
            continue;
        };
        if !text.starts_with("--") {
            output.positional.push(argument);
            continue;
        }

        let option = &text[2..];
        let (name, inline_value) = option
            .split_once('=')
            .map_or((option, None), |(name, value)| (name, Some(value)));
        if value_names.contains(name) {
            if output.values.contains_key(name) {
                return Err(invalid_input(format!("选项 --{name} 重复")));
            }
            let value = if let Some(value) = inline_value {
                OsString::from(value)
            } else {
                arguments
                    .next()
                    .ok_or_else(|| invalid_input(format!("选项 --{name} 缺少值")))?
            };
            output.values.insert(name.to_owned(), value);
        } else if flag_names.contains(name) && inline_value.is_none() {
            if !output.flags.insert(name.to_owned()) {
                return Err(invalid_input(format!("选项 --{name} 重复")));
            }
        } else {
            return Err(invalid_input(format!("未知选项 --{name}")));
        }
    }
    Ok(output)
}

impl ParsedOptions {
    fn path(&self, name: &str) -> Option<PathBuf> {
        self.values.get(name).map(PathBuf::from)
    }

    fn required_path(&self, name: &str) -> io::Result<PathBuf> {
        self.path(name)
            .ok_or_else(|| invalid_input(format!("缺少必需选项 --{name}")))
    }

    fn text(&self, name: &str) -> io::Result<Option<String>> {
        self.values
            .get(name)
            .map(|value| {
                value
                    .to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid_input(format!("--{name} 的值必须是 Unicode 文本")))
            })
            .transpose()
    }

    fn usize(&self, name: &str) -> io::Result<Option<usize>> {
        self.text(name)?
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| invalid_input(format!("--{name} 不是有效非负整数")))
            })
            .transpose()
    }

    fn u64(&self, name: &str) -> io::Result<Option<u64>> {
        self.text(name)?
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|_| invalid_input(format!("--{name} 不是有效非负整数")))
            })
            .transpose()
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.contains(name)
    }
}

fn command_prepare(arguments: Vec<OsString>) -> AppResult<()> {
    let options = parse_options(arguments, &["kind", "input", "output", "report"], &["help"])?;
    if options.flag("help") {
        println!(
            r#"prepare：解析语料并生成加权中文记录

必需：
  --kind KIND               语料解析器类型
  --input PATH              输入文件或目录
  --output PATH             输出：权重<TAB>连续汉字
  --report PATH             审计 JSON

非汉字是片段边界。交互式终端会按输入总字节显示整体进度。"#
        );
        return Ok(());
    }
    if !options.positional.is_empty() {
        return Err(invalid_input("prepare 不接受位置参数").into());
    }

    let kind = options
        .text("kind")?
        .ok_or_else(|| invalid_input("缺少必需选项 --kind"))?
        .parse::<SourceKind>()
        .map_err(invalid_input)?;
    let input_path = options.required_path("input")?;
    let output_path = options.required_path("output")?;
    let report_path = options.required_path("report")?;
    ensure_distinct(&output_path, &report_path, "语料记录和审计报告")?;
    ensure_absent(&output_path)?;
    ensure_absent(&report_path)?;
    ensure_parent(&output_path)?;
    ensure_parent(&report_path)?;

    let output_partial = append_suffix(&output_path, ".partial");
    let report_partial = append_suffix(&report_path, ".partial");
    ensure_absent(&output_partial)?;
    ensure_absent(&report_partial)?;

    let output_file = create_new(&output_partial)?;
    let mut output = BufWriter::new(output_file);
    let mut progress = PrepareProgressBar::new(kind);
    let result = prepare_source_with_progress(
        kind,
        &input_path,
        |record| writeln!(output, "{}\t{}", record.weight, record.text),
        |snapshot| progress.update(&snapshot),
    );
    progress.finish_line();
    let report = result?;
    output.flush()?;
    let output_file = output.into_inner().map_err(|error| error.into_error())?;
    output_file.sync_all()?;

    let report_file = create_new(&report_partial)?;
    let mut report_output = BufWriter::new(report_file);
    report_output.write_all(prepare_report_json(&report).as_bytes())?;
    report_output.flush()?;
    let report_file = report_output
        .into_inner()
        .map_err(|error| error.into_error())?;
    report_file.sync_all()?;

    fs::rename(&output_partial, &output_path)?;
    fs::rename(&report_partial, &report_path)?;

    println!("{} 语料准备完成：{}", kind, output_path.display());
    println!("审计报告：{}", report_path.display());
    println!(
        "处理 {} 个文件，输出 {} 条记录（{} 个汉字），解析错误 {} 个。",
        report.stats.files_seen,
        report.stats.emitted_records,
        report.stats.emitted_han_chars,
        report.stats.parse_errors
    );
    println!("请检查该来源的审计报告；全部来源确认后再执行 index。");
    Ok(())
}

struct PrepareProgressBar {
    label: String,
    enabled: bool,
    last_draw: Option<Instant>,
    last_percent: u64,
    line_open: bool,
    completed: bool,
}

struct IndexProgressBar {
    enabled: bool,
    last_draw: Option<Instant>,
    last_percent: u64,
    last_label: &'static str,
    line_open: bool,
}

impl IndexProgressBar {
    fn new() -> Self {
        Self {
            enabled: io::stderr().is_terminal(),
            last_draw: None,
            last_percent: 0,
            last_label: "",
            line_open: false,
        }
    }

    fn update_read(
        &mut self,
        bytes_read: u64,
        bytes_total: u64,
        file_index: usize,
        file_count: usize,
        records_seen: u64,
        shard_count: usize,
    ) {
        self.draw(
            "读取记录",
            bytes_read,
            bytes_total,
            format!("文件 {file_index}/{file_count}  记录 {records_seen}  分片 {shard_count}"),
            bytes_total == 0 || bytes_read >= bytes_total,
        );
    }

    fn update_merge(&mut self, progress: &MergeProgress) {
        self.draw(
            "归并分片",
            progress.bytes_read,
            progress.bytes_total,
            format!(
                "分片 {}  分片记录 {}  唯一键 {}",
                progress.shard_count, progress.shard_records_read, progress.unique_keys_written
            ),
            progress.complete,
        );
    }

    fn draw(
        &mut self,
        label: &'static str,
        current: u64,
        total: u64,
        suffix: String,
        complete: bool,
    ) {
        if !self.enabled {
            return;
        }
        let current = current.min(total);
        let percent = if total == 0 {
            100
        } else {
            ((u128::from(current) * 100) / u128::from(total)) as u64
        };
        let now = Instant::now();
        if !complete
            && label == self.last_label
            && percent == self.last_percent
            && self
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(100))
        {
            return;
        }

        const WIDTH: usize = 28;
        let filled = if total == 0 {
            WIDTH
        } else {
            ((u128::from(current) * WIDTH as u128) / u128::from(total)) as usize
        };
        let bar = format!("{}{}", "#".repeat(filled), "-".repeat(WIDTH - filled));
        let (scale, unit) = if total >= 1024 * 1024 * 1024 {
            (1024.0 * 1024.0 * 1024.0, "GiB")
        } else {
            (1024.0 * 1024.0, "MiB")
        };
        eprint!(
            "\r{label:<12} [{bar}] {percent:>3}% {:>8.1}/{:<8.1} {unit}  {suffix}",
            current as f64 / scale,
            total as f64 / scale,
        );
        let _ = io::stderr().flush();
        self.last_draw = Some(now);
        self.last_percent = percent;
        self.last_label = label;
        self.line_open = true;
        if complete {
            self.finish_phase();
        }
    }

    fn finish_phase(&mut self) {
        if self.enabled && self.line_open {
            eprintln!();
            self.line_open = false;
        }
        self.last_draw = None;
        self.last_percent = 0;
        self.last_label = "";
    }
}

impl Drop for IndexProgressBar {
    fn drop(&mut self) {
        self.finish_phase();
    }
}

impl PrepareProgressBar {
    fn new(kind: SourceKind) -> Self {
        Self {
            label: kind.to_string(),
            enabled: io::stderr().is_terminal(),
            last_draw: None,
            last_percent: 0,
            line_open: false,
            completed: false,
        }
    }

    fn update(&mut self, progress: &PrepareProgress) {
        if !self.enabled || self.completed {
            return;
        }

        let bytes_read = progress.bytes_read.min(progress.bytes_total);
        let complete = progress.bytes_total == 0 || bytes_read >= progress.bytes_total;
        let percent = if progress.bytes_total == 0 {
            100
        } else {
            ((u128::from(bytes_read) * 100) / u128::from(progress.bytes_total)) as u64
        };
        let now = Instant::now();
        if !complete
            && percent == self.last_percent
            && self
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(100))
        {
            return;
        }

        const WIDTH: usize = 28;
        let filled = if progress.bytes_total == 0 {
            WIDTH
        } else {
            ((u128::from(bytes_read) * WIDTH as u128) / u128::from(progress.bytes_total)) as usize
        };
        let bar = format!("{}{}", "#".repeat(filled), "-".repeat(WIDTH - filled));
        let (scale, unit) = if progress.bytes_total >= 1024 * 1024 * 1024 {
            (1024.0 * 1024.0 * 1024.0, "GiB")
        } else {
            (1024.0 * 1024.0, "MiB")
        };
        eprint!(
            "\r{:<12} [{}] {:>3}% {:>8.1}/{:<8.1} {}  文件 {}/{}",
            self.label,
            bar,
            percent,
            bytes_read as f64 / scale,
            progress.bytes_total as f64 / scale,
            unit,
            progress.file_index,
            progress.file_count,
        );
        let _ = io::stderr().flush();
        self.last_draw = Some(now);
        self.last_percent = percent;
        self.line_open = true;
        if complete {
            eprintln!();
            self.line_open = false;
            self.completed = true;
        }
    }

    fn finish_line(&mut self) {
        if self.enabled && self.line_open {
            eprintln!();
            self.line_open = false;
        }
    }
}

fn command_index(arguments: Vec<OsString>) -> AppResult<()> {
    let options = parse_options(
        arguments,
        &[
            "output",
            "shard-dir",
            "shard-prefix",
            "window",
            "min-count",
            "max-entries",
            "report",
        ],
        &["help"],
    )?;
    if options.flag("help") {
        println!(
            r#"index：从一个或多个 prepare 输出建立中文连续片段索引

必需：
  --output PATH             最终 .ntri 索引（拒绝覆盖）
  --shard-dir DIR           外排分片目录（必须显式放到空间充足磁盘）
  RECORDS.tsv...            按顺序提供一个或多个 prepare 输出

可选：
  --window N                连续片段最大窗口，默认 20 个汉字
  --min-count N             最小加权频次，默认 5；低频分支在最终合并时裁剪
  --shard-prefix NAME       默认 nutrimatic-zh
  --max-entries N           每个内存排序批次的原始 key 数，默认 1000000
  --report PATH             另存建库统计 JSON

分片不会自动删除。每行必须是“权重<TAB>纯汉字”；索引会从每个汉字位置
生成最长 N 字窗口，所有窗口前缀都可作为结果。交互式终端会分别显示
读取记录和归并分片的进度条。"#
        );
        return Ok(());
    }
    if options.positional.is_empty() {
        return Err(invalid_input("index 至少需要一个 RECORDS.tsv").into());
    }

    let output_path = options.required_path("output")?;
    let shard_dir = options.required_path("shard-dir")?;
    ensure_distinct(&output_path, &shard_dir, "最终索引和分片目录")?;
    ensure_parent(&output_path)?;
    let report_path = options.path("report");
    if let Some(report_path) = &report_path {
        ensure_distinct(&output_path, report_path, "最终索引和统计报告")?;
        ensure_absent(report_path)?;
        ensure_parent(report_path)?;
    }
    let mode = IndexMode::Ngrams {
        window_codepoints: options.usize("window")?.unwrap_or(20),
    };
    let prefix = options
        .text("shard-prefix")?
        .unwrap_or_else(|| "nutrimatic-zh".to_owned());
    let mut build_options = BuildOptions::new(&output_path, &shard_dir, prefix, mode);
    build_options.min_subtree_count = options.u64("min-count")?.unwrap_or(5);
    if let Some(max_entries) = options.usize("max-entries")? {
        build_options.max_entries_in_memory = max_entries;
    }

    let input_paths = options
        .positional
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let input_sizes = input_paths
        .iter()
        .map(|path| fs::metadata(path).map(|metadata| metadata.len()))
        .collect::<io::Result<Vec<_>>>()?;
    let input_bytes_total = input_sizes
        .iter()
        .try_fold(0_u64, |total, &size| total.checked_add(size))
        .ok_or_else(|| invalid_input("索引输入总大小超过 u64"))?;

    let mut builder = ExternalIndexBuilder::new(build_options)?;
    let mut progress = IndexProgressBar::new();
    let mut completed_input_bytes = 0_u64;
    let mut records_seen = 0_u64;
    let input_count = input_paths.len();
    for (input_offset, (input_path, input_size)) in input_paths.iter().zip(input_sizes).enumerate()
    {
        let input_index = input_offset + 1;
        let input = File::open(input_path)?;
        let mut input = BufReader::with_capacity(1024 * 1024, input);
        let mut buffer = String::new();
        let mut line_number = 0_u64;
        let mut file_bytes_read = 0_u64;
        progress.update_read(
            completed_input_bytes,
            input_bytes_total,
            input_index,
            input_count,
            records_seen,
            builder.shard_paths().len(),
        );
        loop {
            buffer.clear();
            let read = input.read_line(&mut buffer)?;
            if read == 0 {
                break;
            }
            file_bytes_read = file_bytes_read.saturating_add(read as u64);
            line_number += 1;
            progress.update_read(
                completed_input_bytes.saturating_add(file_bytes_read),
                input_bytes_total,
                input_index,
                input_count,
                records_seen,
                builder.shard_paths().len(),
            );
            trim_line_ending(&mut buffer);
            if buffer.is_empty() {
                continue;
            }
            let (weight, text) = buffer.split_once('\t').ok_or_else(|| {
                invalid_input(format!(
                    "{} 第 {line_number} 行不是 权重<TAB>文本",
                    input_path.display()
                ))
            })?;
            let weight = weight.parse::<u64>().map_err(|_| {
                invalid_input(format!(
                    "{} 第 {line_number} 行权重无效",
                    input_path.display()
                ))
            })?;
            if weight == 0 || weight > i64::MAX as u64 {
                return Err(invalid_input(format!(
                    "{} 第 {line_number} 行权重必须在 1..=i64::MAX",
                    input_path.display()
                ))
                .into());
            }
            if text.is_empty() {
                continue;
            }
            if let Some(character) = text.chars().find(|&character| !is_han(character)) {
                return Err(invalid_input(format!(
                    "{} 第 {line_number} 行含非汉字 {character:?}；请重新 prepare",
                    input_path.display()
                ))
                .into());
            }
            builder.push_record(text, weight)?;
            records_seen = records_seen.saturating_add(1);
            progress.update_read(
                completed_input_bytes.saturating_add(file_bytes_read),
                input_bytes_total,
                input_index,
                input_count,
                records_seen,
                builder.shard_paths().len(),
            );
        }
        completed_input_bytes = completed_input_bytes.saturating_add(input_size);
        progress.update_read(
            completed_input_bytes,
            input_bytes_total,
            input_index,
            input_count,
            records_seen,
            builder.shard_paths().len(),
        );
    }
    progress.finish_phase();
    let report = builder.finish_with_progress(|snapshot| progress.update_merge(&snapshot))?;

    if let Some(report_path) = report_path {
        let mut output = BufWriter::new(create_new(&report_path)?);
        output.write_all(build_report_json(&report).as_bytes())?;
        output.flush()?;
        let file = output.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
    }

    println!("索引完成：{}", report.output_path.display());
    println!(
        "读取 {} 条记录，生成 {} 条链；保留加权频次至少 {} 的片段，写入 {} 个 trie 节点、{} 字节（裁剪 {} 个低频节点）。",
        report.records_seen,
        report.chains_emitted,
        report.min_subtree_count,
        report.index.node_count,
        report.index.bytes_written,
        report.index.pruned_node_count
    );
    println!("保留了 {} 个分片：", report.shard_paths.len());
    for path in &report.shard_paths {
        println!("  {}", path.display());
    }
    Ok(())
}

fn command_search(arguments: Vec<OsString>) -> AppResult<()> {
    let options = parse_options(
        arguments,
        &["index", "limit", "max-nodes", "max-states"],
        &["help"],
    )?;
    if options.flag("help") {
        println!("search --index INDEX [--limit 100] [--max-nodes 1000000] QUERY");
        return Ok(());
    }
    if options.positional.len() != 1 {
        return Err(invalid_input("search 必须且只能提供一个 QUERY 参数").into());
    }
    let index = IndexReader::open(options.required_path("index")?)?;
    let query = options.positional[0]
        .to_str()
        .ok_or_else(|| invalid_input("QUERY 必须是 Unicode 文本"))?;
    let program = Program::parse(query, None)?;
    let search_options = search_options_from_cli(&options)?;
    let report = search(&index, &program, &search_options)?;
    for result in &report.results {
        println!("{}\t{}", result.score, result.text);
    }
    eprintln!(
        "# 结果 {}，访问节点 {}，导数状态 {}{}",
        report.results.len(),
        report.visited,
        report.derivative_states,
        report
            .stop_reason
            .map(|reason| format!("，{}", reason.message()))
            .unwrap_or_default()
    );
    Ok(())
}

fn command_inspect(arguments: Vec<OsString>) -> AppResult<()> {
    let options = parse_options(arguments, &["index"], &["help", "full"])?;
    if options.flag("help") {
        println!("inspect --index INDEX [--full]");
        return Ok(());
    }
    if !options.positional.is_empty() {
        return Err(invalid_input("inspect 不接受位置参数").into());
    }
    let index_path = options.required_path("index")?;
    let index = IndexReader::open(&index_path)?;
    println!("文件：{}", index_path.display());
    println!("字节：{}", index.as_bytes().len());
    println!("模式：{:?}", index.mode());
    println!("节点（头部声明）：{}", index.node_count());
    println!("根频次：{}", index.root_subtree());
    if options.flag("full") {
        let validation = index.validate_full()?;
        println!(
            "完整校验通过：{} 个节点，{} 条边。",
            validation.node_count, validation.edge_count
        );
    }
    Ok(())
}

fn command_serve(arguments: Vec<OsString>) -> AppResult<()> {
    let options = parse_options(
        arguments,
        &["index", "bind", "limit", "max-nodes", "max-states"],
        &["help"],
    )?;
    if options.flag("help") {
        println!("serve --index INDEX [--bind 127.0.0.1:8080] [--limit 100]");
        return Ok(());
    }
    if !options.positional.is_empty() {
        return Err(invalid_input("serve 不接受位置参数").into());
    }
    let bind = options
        .text("bind")?
        .unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    let state = Arc::new(ServerState {
        index: IndexReader::open(options.required_path("index")?)?,
        search_options: search_options_from_cli(&options)?,
        search_gate: SearchGate::default(),
    });
    let listener = TcpListener::bind(&bind)?;
    println!("Nutrimatic 中文版正在监听 http://{bind}/");
    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    if let Err(error) = handle_connection(stream, &state) {
                        if !is_client_disconnect(error.as_ref()) {
                            eprintln!("[HTTP] 请求处理失败：{error}");
                        }
                    }
                });
            }
            Err(error) => eprintln!("[HTTP] 接受连接失败：{error}"),
        }
    }
    Ok(())
}

fn search_options_from_cli(options: &ParsedOptions) -> io::Result<SearchOptions> {
    let defaults = SearchOptions::default();
    Ok(SearchOptions {
        limit: options.usize("limit")?.unwrap_or(defaults.limit),
        max_nodes: options.u64("max-nodes")?.unwrap_or(defaults.max_nodes),
        max_states: options.usize("max-states")?.unwrap_or(defaults.max_states),
    })
}

struct ServerState {
    index: IndexReader,
    search_options: SearchOptions,
    search_gate: SearchGate,
}

const MAX_QUEUED_SEARCHES: usize = 32;

#[derive(Default)]
struct SearchGate {
    lock: Mutex<()>,
    in_system: AtomicUsize,
}

impl SearchGate {
    fn enqueue(&self) -> Option<SearchTicket<'_>> {
        let mut current = self.in_system.load(AtomicOrdering::Acquire);
        loop {
            if current >= MAX_QUEUED_SEARCHES + 1 {
                return None;
            }
            match self.in_system.compare_exchange_weak(
                current,
                current + 1,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            ) {
                Ok(_) => {
                    return Some(SearchTicket {
                        gate: self,
                        ahead: current,
                        counted: true,
                    });
                }
                Err(actual) => current = actual,
            }
        }
    }
}

struct SearchTicket<'a> {
    gate: &'a SearchGate,
    ahead: usize,
    counted: bool,
}

impl<'a> SearchTicket<'a> {
    fn ahead(&self) -> usize {
        self.ahead
    }

    fn wait(mut self) -> SearchPermit<'a> {
        let guard = self
            .gate
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.counted = false;
        SearchPermit {
            gate: self.gate,
            _guard: guard,
        }
    }
}

impl Drop for SearchTicket<'_> {
    fn drop(&mut self) {
        if self.counted {
            self.gate.in_system.fetch_sub(1, AtomicOrdering::AcqRel);
        }
    }
}

struct SearchPermit<'a> {
    gate: &'a SearchGate,
    _guard: MutexGuard<'a, ()>,
}

impl Drop for SearchPermit<'_> {
    fn drop(&mut self) {
        self.gate.in_system.fetch_sub(1, AtomicOrdering::AcqRel);
    }
}

fn handle_connection(mut stream: TcpStream, state: &ServerState) -> AppResult<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let request = read_http_request(&mut stream)?;
    let mut request_line = request
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace();
    let method = request_line.next().unwrap_or_default();
    let target = request_line.next().unwrap_or_default();
    let version = request_line.next().unwrap_or_default();
    if !matches!(method, "GET" | "HEAD") || !version.starts_with("HTTP/1.") {
        return write_response(
            &mut stream,
            method == "HEAD",
            405,
            "text/plain; charset=utf-8",
            b"Method Not Allowed\n",
            "no-store",
        )
        .map_err(Into::into);
    }

    let (path, query_string) = target.split_once('?').unwrap_or((target, ""));
    if path == "/api/search/stream" {
        return api_search_stream(&mut stream, method == "HEAD", query_string, state);
    }
    let response = match path {
        "/" | "/index.html" => StaticResponse::ok("text/html; charset=utf-8", HOME_HTML.as_bytes()),
        "/usage.html" => StaticResponse::ok("text/html; charset=utf-8", USAGE_HTML.as_bytes()),
        "/style.css" => StaticResponse::ok("text/css; charset=utf-8", STYLE_CSS.as_bytes()),
        "/app.js" => StaticResponse::ok("text/javascript; charset=utf-8", APP_JS.as_bytes()),
        "/robots.txt" => StaticResponse::ok("text/plain; charset=utf-8", ROBOTS_TXT.as_bytes()),
        "/favicon.ico" => StaticResponse::ok("image/x-icon", FAVICON),
        "/nutritea-small.jpg" => StaticResponse::ok("image/jpg", TEA_SMALL),
        "/healthz" => StaticResponse::json(200, b"{\"ok\":true}".to_vec()),
        "/api/search" => api_search(query_string, state),
        _ => StaticResponse {
            status: 404,
            content_type: "text/plain; charset=utf-8",
            body: b"Not Found\n".to_vec(),
            cache_control: "no-store",
        },
    };
    write_response(
        &mut stream,
        method == "HEAD",
        response.status,
        response.content_type,
        &response.body,
        response.cache_control,
    )?;
    Ok(())
}

#[derive(Debug)]
struct StaticResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
    cache_control: &'static str,
}

impl StaticResponse {
    fn ok(content_type: &'static str, body: &[u8]) -> Self {
        Self {
            status: 200,
            content_type,
            body: body.to_vec(),
            cache_control: "public, max-age=300",
        }
    }

    fn json(status: u16, body: Vec<u8>) -> Self {
        Self {
            status,
            content_type: "application/json; charset=utf-8",
            body,
            cache_control: "no-store",
        }
    }
}

fn api_search(query_string: &str, state: &ServerState) -> StaticResponse {
    let request = match parse_search_request(query_string, state) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let query = log_text(&request.query);
    let Some(ticket) = state.search_gate.enqueue() else {
        eprintln!("[搜索] 队列已满：{query}");
        return json_error(429, "搜索队列已满，请稍后重试");
    };
    if ticket.ahead() > 0 {
        eprintln!("[搜索] 排队：{query}（前面 {} 个任务）", ticket.ahead());
    }
    let _permit = ticket.wait();
    let started = Instant::now();
    eprintln!("[搜索] 开始：{query}（上限 {}）", request.options.limit);
    match search(&state.index, &request.program, &request.options) {
        Ok(report) => {
            log_search_complete(&query, &report, started.elapsed());
            StaticResponse::json(200, search_report_json(&report).into_bytes())
        }
        Err(error) => {
            eprintln!("[搜索] 失败：{query}；{error}");
            json_error(422, &error.to_string())
        }
    }
}

struct SearchRequest {
    query: String,
    program: Program,
    options: SearchOptions,
}

fn parse_search_request(
    query_string: &str,
    state: &ServerState,
) -> Result<SearchRequest, StaticResponse> {
    let parameters = match parse_query_string(query_string) {
        Ok(parameters) => parameters,
        Err(error) => return Err(log_bad_search_request(None, &error.to_string())),
    };
    let Some(query) = parameters.get("q") else {
        return Err(log_bad_search_request(None, "缺少查询参数 q"));
    };
    if query.is_empty() || query.len() > 2048 {
        return Err(log_bad_search_request(
            Some(query),
            "查询不能为空且不得超过 2048 个 UTF-8 字节",
        ));
    }

    let mut options = state.search_options.clone();
    if let Some(limit) = parameters.get("limit") {
        match limit.parse::<usize>() {
            Ok(limit @ 1..=500) => options.limit = limit,
            _ => {
                return Err(log_bad_search_request(
                    Some(query),
                    "limit 必须在 1..=500 范围内",
                ));
            }
        }
    }
    let program = match Program::parse(query, None) {
        Ok(program) => program,
        Err(error) => return Err(log_bad_search_request(Some(query), &error.to_string())),
    };
    Ok(SearchRequest {
        query: query.clone(),
        program,
        options,
    })
}

fn api_search_stream(
    stream: &mut TcpStream,
    head_only: bool,
    query_string: &str,
    state: &ServerState,
) -> AppResult<()> {
    let request = match parse_search_request(query_string, state) {
        Ok(request) => request,
        Err(response) => {
            write_response(
                stream,
                head_only,
                response.status,
                response.content_type,
                &response.body,
                response.cache_control,
            )?;
            return Ok(());
        }
    };
    let query = log_text(&request.query);

    write_stream_headers(stream)?;
    if head_only {
        return Ok(());
    }

    let Some(ticket) = state.search_gate.enqueue() else {
        eprintln!("[搜索] 队列已满：{query}");
        let line = format!(
            "{{\"type\":\"error\",\"error\":{}}}\n",
            json_string("搜索队列已满，请稍后重试")
        );
        write_http_chunk(stream, line.as_bytes())?;
        finish_http_chunks(stream)?;
        return Ok(());
    };
    if ticket.ahead() > 0 {
        eprintln!("[搜索] 排队：{query}（前面 {} 个任务）", ticket.ahead());
        let line = format!("{{\"type\":\"queued\",\"ahead\":{}}}\n", ticket.ahead());
        if let Err(error) = write_http_chunk(stream, line.as_bytes()) {
            if is_disconnect_io_error(&error) {
                return Ok(());
            }
            return Err(error.into());
        }
    }
    let _permit = ticket.wait();
    let started = Instant::now();
    eprintln!(
        "[搜索] 开始（实时）：{query}（上限 {}）",
        request.options.limit
    );

    let mut stream_error = None;
    let result = search_with_progress(
        &state.index,
        &request.program,
        &request.options,
        Duration::from_millis(300),
        |report| {
            let line = format!("{}\n", search_event_json("progress", report));
            match write_http_chunk(stream, line.as_bytes()) {
                Ok(()) => true,
                Err(error) => {
                    stream_error = Some(error);
                    false
                }
            }
        },
    );

    if let Some(error) = stream_error {
        if is_disconnect_io_error(&error) {
            eprintln!("[搜索] 客户端已断开：{query}");
            return Ok(());
        }
        return Err(error.into());
    }
    match result {
        Ok(report) => {
            log_search_complete(&query, &report, started.elapsed());
            let line = format!("{}\n", search_event_json("complete", &report));
            write_http_chunk(stream, line.as_bytes())?;
        }
        Err(error) => {
            eprintln!("[搜索] 失败：{query}；{error}");
            let line = format!(
                "{{\"type\":\"error\",\"error\":{}}}\n",
                json_string(&error.to_string())
            );
            write_http_chunk(stream, line.as_bytes())?;
        }
    }
    finish_http_chunks(stream)?;
    Ok(())
}

fn write_stream_headers(stream: &mut TcpStream) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/x-ndjson; charset=utf-8\r\n\
         Transfer-Encoding: chunked\r\n\
         Cache-Control: no-store, no-transform\r\n\
         X-Accel-Buffering: no\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Referrer-Policy: no-referrer\r\n\
         Content-Security-Policy: default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'\r\n\
         Connection: close\r\n\r\n"
    )?;
    stream.flush()
}

fn write_http_chunk(stream: &mut TcpStream, body: &[u8]) -> io::Result<()> {
    write!(stream, "{:X}\r\n", body.len())?;
    stream.write_all(body)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn finish_http_chunks(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn log_bad_search_request(query: Option<&str>, message: &str) -> StaticResponse {
    if let Some(query) = query {
        eprintln!("[搜索] 参数错误：{}；{message}", log_text(query));
    } else {
        eprintln!("[搜索] 参数错误：{message}");
    }
    json_error(400, message)
}

fn log_search_complete(query: &str, report: &SearchReport, elapsed: Duration) {
    let stop = report
        .stop_reason
        .map(|reason| format!("，{}", reason.message()))
        .unwrap_or_default();
    eprintln!(
        "[搜索] 完成：{query}；结果 {}，节点 {}，状态 {}，耗时 {:.3} 秒{}",
        report.results.len(),
        report.visited,
        report.derivative_states,
        elapsed.as_secs_f64(),
        stop
    );
}

fn log_text(value: &str) -> String {
    const LIMIT: usize = 160;
    let mut output = String::new();
    let mut characters = value.chars();
    for character in characters.by_ref().take(LIMIT) {
        match character {
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(output, "\\u{{{:X}}}", character as u32);
            }
            character => output.push(character),
        }
    }
    if characters.next().is_some() {
        output.push('…');
    }
    output
}

fn is_client_disconnect(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<io::Error>()
            && is_disconnect_io_error(error)
        {
            return true;
        }
        current = error.source();
    }
    false
}

fn is_disconnect_io_error(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::BrokenPipe
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::NotConnected
            | io::ErrorKind::UnexpectedEof
    )
}

fn read_http_request(stream: &mut TcpStream) -> io::Result<String> {
    const LIMIT: usize = 16 * 1024;
    let mut data = Vec::new();
    let mut buffer = [0_u8; 2048];
    loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..read]);
        if data.windows(4).any(|window| window == b"\r\n\r\n")
            || data.windows(2).any(|window| window == b"\n\n")
        {
            break;
        }
        if data.len() > LIMIT {
            return Err(invalid_input("HTTP 请求头超过 16 KiB"));
        }
    }
    String::from_utf8(data).map_err(|_| invalid_input("HTTP 请求头不是 UTF-8"))
}

fn write_response(
    stream: &mut TcpStream,
    head_only: bool,
    status: u16,
    content_type: &str,
    body: &[u8],
    cache_control: &str,
) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        429 => "Too Many Requests",
        422 => "Unprocessable Content",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: {cache_control}\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Referrer-Policy: no-referrer\r\n\
         Content-Security-Policy: default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'\r\n\
         Connection: close\r\n\r\n",
        body.len()
    )?;
    if !head_only {
        stream.write_all(body)?;
    }
    stream.flush()
}

fn parse_query_string(input: &str) -> io::Result<HashMap<String, String>> {
    let mut output = HashMap::new();
    if input.is_empty() {
        return Ok(output);
    }
    for field in input.split('&') {
        let (name, value) = field.split_once('=').unwrap_or((field, ""));
        output.insert(percent_decode(name)?, percent_decode(value)?);
    }
    Ok(output)
}

fn percent_decode(input: &str) -> io::Result<String> {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut position = 0;
    while position < bytes.len() {
        match bytes[position] {
            b'+' => {
                output.push(b' ');
                position += 1;
            }
            b'%' => {
                let digits = bytes
                    .get(position + 1..position + 3)
                    .ok_or_else(|| invalid_input("不完整的 URL 百分号转义"))?;
                let high = hex_value(digits[0])?;
                let low = hex_value(digits[1])?;
                output.push((high << 4) | low);
                position += 3;
            }
            byte => {
                output.push(byte);
                position += 1;
            }
        }
    }
    String::from_utf8(output).map_err(|_| invalid_input("URL 参数不是 UTF-8"))
}

fn hex_value(value: u8) -> io::Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err(invalid_input("URL 百分号转义含非十六进制字符")),
    }
}

fn json_error(status: u16, message: &str) -> StaticResponse {
    StaticResponse::json(
        status,
        format!("{{\"error\":{}}}", json_string(message)).into_bytes(),
    )
}

fn search_report_json(report: &SearchReport) -> String {
    search_report_json_with_type(None, report)
}

fn search_event_json(event_type: &str, report: &SearchReport) -> String {
    search_report_json_with_type(Some(event_type), report)
}

fn search_report_json_with_type(event_type: Option<&str>, report: &SearchReport) -> String {
    let results = report
        .results
        .iter()
        .map(|result| {
            format!(
                "{{\"text\":{},\"score\":{}}}",
                json_string(&result.text),
                result.score
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let event_type = event_type
        .map(|value| format!("\"type\":{},", json_string(value)))
        .unwrap_or_default();
    let stop_reason = report
        .stop_reason
        .map(|reason| json_string(reason.code()))
        .unwrap_or_else(|| "null".to_owned());
    format!(
        "{{{event_type}\"results\":[{results}],\"visited\":{},\"states\":{},\"truncated\":{},\"stop_reason\":{stop_reason}}}",
        report.visited, report.derivative_states, report.truncated
    )
}

fn prepare_report_json(report: &PrepareReport) -> String {
    let stats = &report.stats;
    let weights = report
        .records_by_weight
        .iter()
        .map(|(weight, count)| format!("{}:{}", json_string(&weight.to_string()), count))
        .collect::<Vec<_>>()
        .join(",");
    let records = report
        .record_samples
        .iter()
        .map(|record| {
            format!(
                "{{\"weight\":{},\"text\":{}}}",
                record.weight,
                json_string(&record.text)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let errors = report
        .error_samples
        .iter()
        .map(|sample| {
            format!(
                "{{\"source\":{},\"line\":{},\"message\":{},\"preview\":{}}}",
                json_string(&sample.source.to_string_lossy()),
                sample.line,
                json_string(&sample.message),
                json_string(&sample.preview)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\n  \"kind\":{},\n  \"stats\":{{",
            "\"files_seen\":{},\"bytes_read\":{},\"physical_lines\":{},",
            "\"logical_records\":{},\"selected_fields\":{},",
            "\"fields_without_han\":{},\"emitted_records\":{},",
            "\"emitted_han_chars\":{},\"emitted_weight\":{},",
            "\"skipped_records\":{},\"parse_errors\":{},",
            "\"invalid_utf8_lines\":{}",
            "}},\n  \"records_by_weight\":{{{}}},",
            "\n  \"record_samples\":[{}],",
            "\n  \"error_samples\":[{}]\n}}\n"
        ),
        json_string(&report.kind.to_string()),
        stats.files_seen,
        stats.bytes_read,
        stats.physical_lines,
        stats.logical_records,
        stats.selected_fields,
        stats.fields_without_han,
        stats.emitted_records,
        stats.emitted_han_chars,
        stats.emitted_weight,
        stats.skipped_records,
        stats.parse_errors,
        stats.invalid_utf8_lines,
        weights,
        records,
        errors,
    )
}

fn build_report_json(report: &crate::index::BuildReport) -> String {
    let shards = report
        .shard_paths
        .iter()
        .map(|path| json_string(&path.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\n  \"output\":{},\n  \"records_seen\":{},",
            "\n  \"empty_records\":{},\n  \"chains_emitted\":{},",
            "\n  \"min_subtree_count\":{},",
            "\n  \"node_count\":{},\n  \"key_count\":{},",
            "\n  \"pruned_node_count\":{},",
            "\n  \"root_subtree\":{},\n  \"bytes_written\":{},",
            "\n  \"shards\":[{}]\n}}\n"
        ),
        json_string(&report.output_path.to_string_lossy()),
        report.records_seen,
        report.empty_records,
        report.chains_emitted,
        report.min_subtree_count,
        report.index.node_count,
        report.index.key_count,
        report.index.pruned_node_count,
        report.index.root_subtree,
        report.index.bytes_written,
        shards
    )
}

fn json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{001F}' => {
                use std::fmt::Write as _;
                let _ = write!(output, "\\u{:04X}", character as u32);
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn trim_line_ending(value: &mut String) {
    if value.ends_with('\n') {
        value.pop();
        if value.ends_with('\r') {
            value.pop();
        }
    }
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn ensure_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn ensure_absent(path: &Path) -> io::Result<()> {
    if path.exists() {
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("拒绝覆盖现有路径：{}", path.display()),
        ))
    } else {
        Ok(())
    }
}

fn ensure_distinct(left: &Path, right: &Path, description: &str) -> io::Result<()> {
    if left == right {
        Err(invalid_input(format!("{description}不能使用同一路径")))
    } else {
        Ok(())
    }
}

fn create_new(path: &Path) -> io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexMode, TrieWriter};
    use std::io::Cursor;
    use std::net::Shutdown;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn url_and_json_escaping_are_unicode_safe() {
        assert_eq!(percent_decode("%E4%B8%AD+%23").unwrap(), "中 #");
        assert_eq!(json_string("中\n\""), "\"中\\n\\\"\"");
        assert_eq!(log_text("中文\n查询"), "中文\\n查询");
    }

    #[test]
    fn option_parser_preserves_non_unicode_paths_as_positionals() {
        let parsed = parse_options(
            vec![
                OsString::from("--limit"),
                OsString::from("12"),
                OsString::from("中文"),
            ],
            &["limit"],
            &[],
        )
        .unwrap();
        assert_eq!(parsed.usize("limit").unwrap(), Some(12));
        assert_eq!(parsed.positional, [OsString::from("中文")]);
    }

    #[test]
    fn prepare_report_includes_emitted_record_samples() {
        let report = PrepareReport {
            kind: SourceKind::Wordlist,
            stats: Default::default(),
            records_by_weight: Default::default(),
            record_samples: vec![crate::corpus::WeightedRecord {
                text: "中文".to_owned(),
                weight: 20,
            }],
            error_samples: Vec::new(),
        };
        let json = prepare_report_json(&report);
        assert!(json.contains(r#""record_samples":[{"weight":20,"text":"中文"}]"#));
    }

    #[test]
    fn streaming_search_uses_chunked_ndjson_and_completes() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = TrieWriter::new(cursor, IndexMode::Records).unwrap();
        writer.push_str_sorted("中华", 5).unwrap();
        writer.push_str_sorted("中国", 8).unwrap();
        let (cursor, _) = writer.finish().unwrap();
        let state = ServerState {
            index: IndexReader::from_bytes(cursor.into_inner()).unwrap(),
            search_options: SearchOptions::default(),
            search_gate: SearchGate::default(),
        };

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(
                    b"GET /api/search/stream?q=%E4%B8%AD.* HTTP/1.1\r\nHost: localhost\r\n\r\n",
                )
                .unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            response
        });

        let (stream, _) = listener.accept().unwrap();
        handle_connection(stream, &state).unwrap();
        let response = client.join().unwrap();
        assert!(response.contains("Transfer-Encoding: chunked"));
        assert!(response.contains("application/x-ndjson"));
        assert!(response.contains(r#""type":"complete""#));
        assert!(response.ends_with("0\r\n\r\n"));
    }

    #[test]
    fn search_request_keeps_decoded_chinese_for_logging() {
        let cursor = Cursor::new(Vec::new());
        let writer = TrieWriter::new(cursor, IndexMode::Records).unwrap();
        let (cursor, _) = writer.finish().unwrap();
        let state = ServerState {
            index: IndexReader::from_bytes(cursor.into_inner()).unwrap(),
            search_options: SearchOptions::default(),
            search_gate: SearchGate::default(),
        };
        let request = parse_search_request("q=%E4%B8%AD.*&limit=12", &state).unwrap();
        assert_eq!(request.query, "中.*");
        assert_eq!(request.options.limit, 12);
    }

    #[test]
    fn invalid_search_request_returns_readable_json_error() {
        let cursor = Cursor::new(Vec::new());
        let writer = TrieWriter::new(cursor, IndexMode::Records).unwrap();
        let (cursor, _) = writer.finish().unwrap();
        let state = ServerState {
            index: IndexReader::from_bytes(cursor.into_inner()).unwrap(),
            search_options: SearchOptions::default(),
            search_gate: SearchGate::default(),
        };
        let response = match parse_search_request("q=%E4%B8%AD%5B", &state) {
            Ok(_) => panic!("invalid query unexpectedly parsed"),
            Err(response) => response,
        };
        assert_eq!(response.status, 400);
        let body = String::from_utf8(response.body).unwrap();
        assert!(body.contains("字符类缺少 ]"));
        assert!(!body.contains("%E4%B8%AD"));
    }

    #[test]
    fn expected_client_disconnects_are_not_server_errors() {
        let error = io::Error::new(io::ErrorKind::BrokenPipe, "client closed");
        assert!(is_client_disconnect(&error));
        let error = io::Error::new(io::ErrorKind::InvalidData, "bad response");
        assert!(!is_client_disconnect(&error));
    }

    #[test]
    fn search_gate_runs_only_one_search_at_a_time() {
        let gate = Arc::new(SearchGate::default());
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let first_gate = Arc::clone(&gate);
        let first_tx = acquired_tx.clone();
        let first = thread::spawn(move || {
            let permit = first_gate.enqueue().unwrap().wait();
            first_tx.send(1).unwrap();
            release_rx.recv().unwrap();
            drop(permit);
        });
        assert_eq!(acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 1);

        let second_gate = Arc::clone(&gate);
        let second = thread::spawn(move || {
            let permit = second_gate.enqueue().unwrap().wait();
            acquired_tx.send(2).unwrap();
            drop(permit);
        });
        assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());
        release_tx.send(()).unwrap();
        assert_eq!(acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap(), 2);
        first.join().unwrap();
        second.join().unwrap();
    }
}
