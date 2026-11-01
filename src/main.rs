use std::io::{self, Read, Write};
use std::process::ExitCode;

use ingle::cli::{Command, PromptInput, parse_args};
use ingle::http::serve_router;
use ingle::run::{ContinuationError, RunError, generate_from_command, open_serve};

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();
    let args: Vec<String> = std::env::args().collect();
    let infer_model = std::env::var("INFER_MODEL").ok();
    match parse_args(args, infer_model.as_deref()) {
        Ok(Command::Generate(command)) => generate(command),
        Ok(Command::Serve(command)) => serve(command),
        Err(error) => {
            eprint!("{}", error.message);
            ExitCode::from(2)
        }
    }
}

fn generate(command: ingle::cli::GenerateCommand) -> ExitCode {
    let stdin = if matches!(command.prompt, PromptInput::Stdin) {
        let mut buffer = String::new();
        if io::stdin().read_to_string(&mut buffer).is_err() {
            eprintln!("failed to read stdin");
            return ExitCode::from(1);
        }
        buffer
    } else {
        String::new()
    };
    match generate_from_command(command, &stdin) {
        Ok(text) => {
            print!("{text}");
            let _ = io::stdout().flush();
            ExitCode::SUCCESS
        }
        Err(error) => exit_run(error),
    }
}

fn serve(command: ingle::cli::ServeCommand) -> ExitCode {
    let listening = match open_serve(&command) {
        Ok(listening) => listening,
        Err(error) => return exit_run(error),
    };
    let model_name = listening.card.name;
    let address = match listening.listener.local_addr() {
        Ok(address) => address.to_string(),
        Err(_) => format!("{}:{}", command.host, command.port),
    };
    let _model = listening.model;
    if let Err(_error) = listening.listener.set_nonblocking(true) {
        eprintln!("failed to bind {address}");
        return ExitCode::from(1);
    }
    let listener = match tokio::net::TcpListener::from_std(listening.listener) {
        Ok(listener) => listener,
        Err(_) => {
            eprintln!("failed to bind {address}");
            return ExitCode::from(1);
        }
    };
    tracing::info!("listening on {address}");
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => {
            eprintln!("failed to start the runtime");
            return ExitCode::from(1);
        }
    };
    match runtime.block_on(async { axum::serve(listener, serve_router(&model_name)).await }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(_) => {
            eprintln!("the server stopped");
            ExitCode::from(1)
        }
    }
}

fn exit_run(error: RunError) -> ExitCode {
    match error {
        RunError::EmptyPrompt => ExitCode::from(1),
        RunError::Startup(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
        RunError::Continuation(ContinuationError::Encode(_)) => {
            eprintln!("the prompt cannot be tokenized");
            ExitCode::from(1)
        }
        RunError::Continuation(_) => {
            eprintln!("generation failed");
            ExitCode::from(1)
        }
        RunError::Usage(error) => {
            eprint!("{}", error.message);
            ExitCode::from(2)
        }
    }
}
