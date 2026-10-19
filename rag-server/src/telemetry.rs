use crate::config::{LogFormat, LoggingConfig};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

pub struct SizeRotatingFile {
    path: PathBuf,
    max_bytes: u64,
    max_files: usize,
    file: Option<File>,
    written: u64,
}

impl SizeRotatingFile {
    pub fn new(path: PathBuf, max_bytes: u64, max_files: usize) -> io::Result<Self> {
        let file = open_append(&path)?;
        let written = file.metadata()?.len();
        Ok(Self {
            path,
            max_bytes,
            max_files,
            file: Some(file),
            written,
        })
    }

    fn numbered(&self, index: usize) -> PathBuf {
        let mut name = self.path.clone().into_os_string();
        name.push(format!(".{index}"));
        PathBuf::from(name)
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file = None;

        let oldest = self.numbered(self.max_files);
        if oldest.exists() {
            fs::remove_file(&oldest)?;
        }
        for index in (1..self.max_files).rev() {
            let from = self.numbered(index);
            if from.exists() {
                fs::rename(&from, self.numbered(index + 1))?;
            }
        }
        fs::rename(&self.path, self.numbered(1))?;

        self.written = 0;
        Ok(())
    }

    fn file_mut(&mut self) -> io::Result<&mut File> {
        if self.file.is_none() {
            self.file = Some(open_append(&self.path)?);
        }
        self.file
            .as_mut()
            .ok_or_else(|| io::Error::other("log file is not open"))
    }
}

impl Write for SizeRotatingFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.written > 0 && self.written + buf.len() as u64 > self.max_bytes {
            self.rotate()?;
        }
        let file = self.file_mut()?;
        let n = file.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

fn open_append(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

#[must_use = "WorkerGuard must be held for the lifetime of the application to prevent log flushing thread from dropping"]
pub fn init_telemetry(config: &LoggingConfig) -> Option<WorkerGuard> {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    let (file_writer, guard) = if let Some(path) = &config.file_path {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(parent);
        }

        let max_bytes = config.max_file_size_mb * 1024 * 1024;
        match SizeRotatingFile::new(path.clone(), max_bytes, config.max_files) {
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
