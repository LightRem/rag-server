use crate::error::ApiError;
use axum::{
    async_trait,
    extract::{FromRequest, Request, rejection::JsonRejection},
    response::{IntoResponse, Response},
};
#[derive(Debug, Clone, Copy, Default)]
pub struct AppJson<T>(pub T);
#[async_trait]
impl<S, T> FromRequest<S> for AppJson<T>
where
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = ApiError;
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(AppJson(value)),
            Err(rejection) => Err(ApiError::ValidationError(rejection.body_text())),
        }
    }
}
impl<T> IntoResponse for AppJson<T>
where
    axum::Json<T>: IntoResponse,
{
    fn into_response(self) -> Response {
        axum::Json(self.0).into_response()
    }
}
