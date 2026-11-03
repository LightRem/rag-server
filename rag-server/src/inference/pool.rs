use crate::error::InferenceError;
use ort::{environment::Environment, session::Session};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

struct SessionPoolInner {
    sessions: Mutex<Vec<Session>>,
    semaphore: Arc<Semaphore>,
    environment: Arc<Environment>,
    model_path: PathBuf,
}

#[derive(Clone)]
pub struct SessionPool {
    inner: Arc<SessionPoolInner>,
}

impl SessionPool {
    pub fn new(
        environment: &Arc<Environment>,
        model_path: &Path,
        concurrency: usize,
    ) -> Result<Self, InferenceError> {
        let concurrency = concurrency.max(1);
        let mut sessions = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let session = Self::create_session(environment, model_path)?;
            sessions.push(session);
        }
        Ok(Self {
            inner: Arc::new(SessionPoolInner {
                sessions: Mutex::new(sessions),
                semaphore: Arc::new(Semaphore::new(concurrency)),
                environment: environment.clone(),
                model_path: model_path.to_path_buf(),
            }),
        })
    }

    fn create_session(env: &Arc<Environment>, path: &Path) -> Result<Session, InferenceError> {
        Session::builder(env.as_ref())
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| InferenceError::ModelLoad(e.to_string()))
    }

    pub async fn acquire_permit(&self) -> Result<OwnedSemaphorePermit, InferenceError> {
        self.inner
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| InferenceError::Execution(e.to_string()))
    }

    pub fn pop_session(&self) -> Result<Session, InferenceError> {
        let session_opt = {
            let mut lock = match self.inner.sessions.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            lock.pop()
        };
        if let Some(session) = session_opt {
            Ok(session)
        } else {
            Self::create_session(&self.inner.environment, &self.inner.model_path)
        }
    }

    pub fn return_session(&self, session: Session) {
        let mut lock = match self.inner.sessions.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        lock.push(session);
    }

    pub fn replace_corrupted_session(&self) -> bool {
        match Self::create_session(&self.inner.environment, &self.inner.model_path) {
            Ok(new_session) => {
                self.return_session(new_session);
                true
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to recreate corrupted ONNX runtime session");
                false
            }
        }
    }
}

pub struct SessionGuard {
    pool: SessionPool,
    session: Option<Session>,
    _permit: OwnedSemaphorePermit,
}

impl SessionGuard {
    pub fn new(pool: SessionPool, session: Session, permit: OwnedSemaphorePermit) -> Self {
        Self {
            pool,
            session: Some(session),
            _permit: permit,
        }
    }

    pub fn session_mut(&mut self) -> &mut Session {
        self.session.as_mut().expect("Session is active")
    }

    pub fn success(mut self) {
        if let Some(session) = self.session.take() {
            self.pool.return_session(session);
        }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            drop(session);
            let _ = self.pool.replace_corrupted_session();
        }
    }
}
