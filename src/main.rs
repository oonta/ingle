use std::io::{self, Read, Write};
use std::process::ExitCode;

use ingle::cli::{parse_args, Command, PromptInput};
use ingle::run::{generate_from_command, ContinuationError, RunError};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let infer_model = std::env::var("INFER_MODEL").ok();
    let command = match parse_args(args, infer_model.as_deref()) {
        Ok(Command::Generate(command)) => command,
        Err(error) => {
            eprint!("{}", error.message);
            return ExitCode::from(2);
        }
    };
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
        Err(RunError::EmptyPrompt) => ExitCode::from(1),
        Err(RunError::Startup(error)) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
        Err(RunError::Continuation(ContinuationError::Encode(_))) => {
            eprintln!("the prompt cannot be tokenized");
            ExitCode::from(1)
        }
        Err(RunError::Continuation(_)) => {
            eprintln!("generation failed");
            ExitCode::from(1)
        }
        Err(RunError::Usage(error)) => {
            eprint!("{}", error.message);
            ExitCode::from(2)
        }
    }
}
