use crate::error::ConfigError;
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_body_limit_mb: usize,
    pub snapshot_path: PathBuf,
    pub snapshot_interval_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_body_limit_mb: 32,
            snapshot_path: PathBuf::from("data/vector_store.json"),
            snapshot_interval_secs: 300,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub text_model_path: PathBuf,
    pub vision_model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_concurrency: usize,
    pub dimension: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            text_model_path: PathBuf::from("models/text_model.onnx"),
            vision_model_path: PathBuf::from("models/vision_model.onnx"),
            tokenizer_path: PathBuf::from("models/tokenizer.json"),
            max_concurrency: 4,
            dimension: 512,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:11434/v1".to_string(),
            model: "qwen2.5:7b".to_string(),
            api_key: None,
            timeout_secs: 45,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub file_path: Option<PathBuf>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Pretty,
            file_path: Some(PathBuf::from("data/server.log")),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub llm: LlmConfig,
    pub logging: LoggingConfig,
}

impl AppConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.server.port == 0 {
            return Err(ConfigError::InvalidParameter {
                field: "server.port",
                reason: "Port must be greater than 0".to_string(),
            });
        }

        if self.server.max_body_limit_mb == 0 {
            return Err(ConfigError::InvalidParameter {
                field: "server.max_body_limit_mb",
                reason: "Maximum body limit must be greater than 0 MB".to_string(),
            });
        }

        if self.server.snapshot_path.as_os_str().is_empty() {
            return Err(ConfigError::InvalidParameter {
                field: "server.snapshot_path",
                reason: "Snapshot path cannot be empty".to_string(),
            });
        }

        if self.model.max_concurrency == 0 {
            return Err(ConfigError::InvalidParameter {
                field: "model.max_concurrency",
                reason: "Max concurrency must be at least 1".to_string(),
            });
        }

        if self.model.dimension == 0 {
            return Err(ConfigError::InvalidParameter {
                field: "model.dimension",
                reason: "Model dimension must be greater than 0".to_string(),
            });
        }

        if self.llm.endpoint.trim().is_empty() {
            return Err(ConfigError::InvalidParameter {
                field: "llm.endpoint",
                reason: "LLM endpoint cannot be empty".to_string(),
            });
        }

        if self.llm.timeout_secs == 0 {
            return Err(ConfigError::InvalidParameter {
                field: "llm.timeout_secs",
                reason: "LLM timeout must be greater than 0 seconds".to_string(),
            });
        }
        Ok(())
    }

    pub fn load() -> Result<Self, ConfigError> {
        let mut figment = Figment::new().merge(Serialized::defaults(AppConfig::default()));
        let default_config_path = Path::new("config/default.toml");
        if default_config_path.exists() {
            figment = figment.merge(Toml::file(default_config_path).nested());
        }

        let local_config_path = Path::new("config/local.toml");
        if local_config_path.exists() {
            figment = figment.merge(Toml::file(local_config_path).nested());
        }

        let config: AppConfig = figment
            .merge(Env::prefixed("RAG_").split("__"))
            .extract()
            .map_err(|e| ConfigError::Load(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }
}
