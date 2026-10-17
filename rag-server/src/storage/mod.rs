pub mod hnsw;
pub mod mem_store;
pub mod models;
pub mod partition;
pub mod persistence;
pub mod scoring;
pub mod traits;

pub use hnsw::{HnswConfig, HnswIndex, VectorView};
pub use mem_store::{DEFAULT_MAX_CAPACITY, InMemoryVectorStore};
pub use models::{SearchFilter, SearchResult, VectorPoint};
pub use scoring::CalibrationConfig;
pub use traits::{DocumentSummary, VectorStore};

#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);

    for (ac, bc) in (&mut a_chunks).zip(&mut b_chunks) {
        sum0 += ac[0] * bc[0];
        sum1 += ac[1] * bc[1];
        sum2 += ac[2] * bc[2];
        sum3 += ac[3] * bc[3];
        sum4 += ac[4] * bc[4];
        sum5 += ac[5] * bc[5];
        sum6 += ac[6] * bc[6];
        sum7 += ac[7] * bc[7];
    }

    let rem_sum: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder().iter())
        .map(|(&x, &y)| x * y)
        .sum();

    ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7)) + rem_sum
}
