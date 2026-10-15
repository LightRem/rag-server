pub mod mock;
pub mod onnx;
pub mod preprocessor;
pub mod traits;
pub use mock::MockEmbedder;
pub use onnx::OnnxClipEmbedder;
pub use traits::{Embedder, l2_normalize};
