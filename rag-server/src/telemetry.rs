use crate::config::{LogFormat, LoggingConfig};
use std::fs::OpenOptions;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
#[must_use = "WorkerGuard must be held for the lifetime of the application to prevent log flushing thread from dropping"]
pub fn init_telemetry(config: &LoggingConfig) -> Option<WorkerGuard> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));
    let (file_writer, guard) = if let Some(path) = &config.file_path {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(parent);
        }
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => {
                let (non_blocking, guard) = tracing_appender::non_blocking(file);
                (Some(non_blocking), Some(guard))
            }
            Err(e) => {
                eprintln!("Failed to open log file {path:?}: {e}");
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let registry = tracing_subscriber::registry().with(filter);
    match config.format {
        LogFormat::Pretty => {
            let stdout_layer = fmt::layer().with_target(true).with_thread_ids(true);
            let file_layer = file_writer.map(|w| {
                fmt::layer()
                    .with_writer(w)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_ansi(false)
            });
            let _ = registry.with(stdout_layer).with(file_layer).try_init();
        }
        LogFormat::Json => {
            let stdout_layer = fmt::layer().json().flatten_event(true);
            let file_layer =
                file_writer.map(|w| fmt::layer().json().flatten_event(true).with_writer(w));
            let _ = registry.with(stdout_layer).with(file_layer).try_init();
        }
    }
    guard
}
