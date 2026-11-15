use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use ingle::cli::{Command, PromptInput, ServeEnv, apply_serve_env, parse_args};
use ingle::complete::Loaded;
use ingle::http::serve_attached;
use ingle::run::{ContinuationError, RunError, generate_from_command, open_serve};

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();
    let args: Vec<String> = std::env::args().collect();
    let infer_model = std::env::var("INFER_MODEL").ok();
    match parse_args(&args, infer_model.as_deref()) {
        Ok(Command::Generate(command)) => generate(command),
        Ok(Command::Serve(command)) => serve(command, &args),
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

fn serve(mut command: ingle::cli::ServeCommand, args: &[String]) -> ExitCode {
    let api_key = std::env::var("INFER_API_KEY").ok();
    let max_body_bytes = std::env::var("INFER_MAX_BODY_BYTES").ok();
    let queue_capacity = std::env::var("INFER_QUEUE_CAPACITY").ok();
    let shutdown_timeout_ms = std::env::var("INFER_SHUTDOWN_TIMEOUT_MS").ok();
    if let Err(error) = apply_serve_env(
        &mut command,
        args,
        ServeEnv {
            api_key: api_key.as_deref(),
            max_body_bytes: max_body_bytes.as_deref(),
            queue_capacity: queue_capacity.as_deref(),
            shutdown_timeout_ms: shutdown_timeout_ms.as_deref(),
        },
    ) {
        eprint!("{}", error.message);
        return ExitCode::from(2);
    }
    let listening = match open_serve(&command) {
        Ok(listening) => listening,
        Err(error) => return exit_run(error),
    };
    let model_name = listening.card.name.clone();
    let loaded = Loaded {
        vocab: listening.card.tokens,
        eos_id: listening.card.eos_id,
        add_bos: listening.card.add_bos,
        bos_id: listening.card.bos_id,
        working_context: listening.card.working_context,
        template: listening.card.chat_template,
    };
    let address = match listening.listener.local_addr() {
        Ok(address) => address.to_string(),
        Err(_) => format!("{}:{}", command.host, command.port),
    };
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
    let app = serve_attached(
        &model_name,
        loaded,
        Box::new(listening.model),
        command.api_key.filter(|key| !key.is_empty()),
        command.max_body_bytes,
        command.queue_capacity,
    );
    let timeout = Duration::from_millis(command.shutdown_timeout_ms);
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
    match runtime.block_on(async move {
        let notify = Arc::new(tokio::sync::Notify::new());
        let signal = Arc::clone(&notify);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                signal.notify_waiters();
            }
        });
        let stop = Arc::clone(&notify);
        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            stop.notified().await;
        });
        let deadline = Arc::clone(&notify);
        tokio::select! {
            result = server => result,
            _ = async move {
                deadline.notified().await;
                tokio::time::sleep(timeout).await;
            } => Ok(()),
        }
    }) {
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
