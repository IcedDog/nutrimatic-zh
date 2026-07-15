fn main() {
    if let Err(error) = nutrimatic_zh::web::run_cli(std::env::args_os()) {
        eprintln!("错误：{error}");
        std::process::exit(1);
    }
}
