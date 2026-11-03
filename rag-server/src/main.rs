use rag_server::{config::AppConfig, server::run, telemetry::init_telemetry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::load()?;
    let _log_guard = init_telemetry(&config.logging);

    run(config).await
}
