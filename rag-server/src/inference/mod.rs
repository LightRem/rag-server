#[cfg(any(test, feature = "test-utils"))]
pub mod mock;
pub mod onnx;
pub mod pool;
pub mod preprocessor;
pub mod traits;

#[cfg(any(test, feature = "test-utils"))]
pub use mock::MockEmbedder;
pub use onnx::OnnxClipEmbedder;
pub use traits::{Embedder, l2_normalize};
