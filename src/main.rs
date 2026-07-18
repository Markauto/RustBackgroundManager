use std::process::ExitCode;

fn main() -> ExitCode {
    match bgm::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.chain().any(is_broken_pipe) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn is_broken_pipe(error: &(dyn std::error::Error + 'static)) -> bool {
    error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.kind() == std::io::ErrorKind::BrokenPipe)
}
