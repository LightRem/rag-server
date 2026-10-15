use crate::error::GenerationError;
use async_trait::async_trait;
use reqwest::StatusCode;
use serde_json::json;
use std::time::Duration;
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        temperature: Option<f32>,
        max_tokens: Option<usize>,
    ) -> Result<String, GenerationError>;
}
pub struct OpenAiCompatibleLlmClient {
    endpoint: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    max_tokens: usize,
    client: reqwest::Client,
}
impl OpenAiCompatibleLlmClient {
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
    ) -> Self {
        let raw_endpoint = endpoint.into().trim().trim_end_matches('/').to_string();
        let normalized_endpoint = if raw_endpoint.ends_with("/chat/completions") {
            raw_endpoint
        } else if raw_endpoint.ends_with("/v1") {
            format!("{raw_endpoint}/chat/completions")
        } else {
            format!("{raw_endpoint}/v1/chat/completions")
        };
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(45))
            .build()
            .unwrap_or_default();
        Self {
            endpoint: normalized_endpoint,
            api_key,
            model: model.into(),
            temperature: 0.2,
            max_tokens: 1024,
            client,
        }
    }
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_default();
        self
    }
}
#[async_trait]
impl LlmClient for OpenAiCompatibleLlmClient {
    async fn generate(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        temperature: Option<f32>,
        max_tokens: Option<usize>,
    ) -> Result<String, GenerationError> {
        let mut req = self.client.post(&self.endpoint);
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }
        let body = json!({
            "model": self.model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ],
            "temperature": temperature.unwrap_or(self.temperature),
            "max_tokens": max_tokens.unwrap_or(self.max_tokens)
        });
        let resp = req.json(&body).send().await.map_err(|e| {
            if e.is_timeout() {
                GenerationError::Timeout(e.to_string())
            } else {
                GenerationError::RequestFailed(e.to_string())
            }
        })?;
        let status = resp.status();
        if status == StatusCode::GATEWAY_TIMEOUT || status == StatusCode::REQUEST_TIMEOUT {
            let err_text = resp.text().await.unwrap_or_default();
            return Err(GenerationError::Timeout(format!(
                "HTTP {status}: {err_text}"
            )));
        }
        if !status.is_success() {
            let err_text = resp.text().await.unwrap_or_default();
            return Err(GenerationError::RequestFailed(format!(
                "HTTP {status}: {err_text}"
            )));
        }
        let json_val: serde_json::Value = resp.json().await.map_err(|e| {
            if e.is_timeout() {
                GenerationError::Timeout(e.to_string())
            } else {
                GenerationError::ParseError(e.to_string())
            }
        })?;
        if let Some(err_obj) = json_val.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Upstream API error");
            return Err(GenerationError::RequestFailed(format!(
                "LLM provider error: {msg}"
            )));
        }
        let choices = json_val.get("choices").and_then(|c| c.as_array());
        let first_choice = choices.and_then(|arr| arr.first());
        let answer = first_choice
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str());
        let answer = match answer {
            Some(ans) if !ans.trim().is_empty() => ans.trim().to_string(),
            _ => {
                if let Some(finish_reason) = first_choice
                    .and_then(|c| c.get("finish_reason"))
                    .and_then(|f| f.as_str())
                    .filter(|&f| f != "stop")
                {
                    return Err(GenerationError::RequestFailed(format!(
                        "LLM execution terminated prematurely with finish_reason: '{finish_reason}'"
                    )));
                }
                return Err(GenerationError::EmptyResponse);
            }
        };
        Ok(answer)
    }
}
pub struct MockLlmClient {
    fixed_response: Option<String>,
}
impl MockLlmClient {
    pub fn new() -> Self {
        Self {
            fixed_response: None,
        }
    }
    pub fn with_response(response: impl Into<String>) -> Self {
        Self {
            fixed_response: Some(response.into()),
        }
    }
}
impl Default for MockLlmClient {
    fn default() -> Self {
        Self::new()
    }
}
#[async_trait]
impl LlmClient for MockLlmClient {
    async fn generate(
        &self,
        _system_prompt: &str,
        user_prompt: &str,
        _temperature: Option<f32>,
        _max_tokens: Option<usize>,
    ) -> Result<String, GenerationError> {
        if let Some(ref resp) = self.fixed_response {
            return Ok(resp.clone());
        }
        Ok(format!(
            "Grounded response synthesized from retrieved context (prompt chars: {}).",
            user_prompt.chars().count()
        ))
    }
}
