pub mod asymmetric;
pub mod distinguishers;
pub mod error;
pub mod leakage_detection;
pub mod leakage_model;
pub mod preprocessors;
pub mod processors;
#[cfg(feature = "quicklog")]
pub mod quicklog;
pub mod trace;
pub mod util;

use std::ops::{Add, AddAssign, Mul};

use num_traits::{AsPrimitive, Zero};

pub use crate::error::Error;

/// Sample type that can be processed by [`muscat`] processors.
///
/// # Dyn compatibility
/// This trait is not [dyn compatible](https://doc.rust-lang.org/nightly/reference/items/traits.html#dyn-compatibility).
///
/// # Limitations
/// We are assuming that the sum of [`Container`] types will not overflow.
pub trait Sample: Sized {
    /// Bigger container type to perform computations (such as sums) of [`Self`] types that could
    /// otherwise overflow.
    type Container: Zero
        + Add
        + AddAssign
        + Mul<Output = Self::Container>
        + AsPrimitive<f32>
        + Clone
        + Copy
        + From<Self>;
}

macro_rules! impl_sample {
    ($($t:ty),* => $c:ty) => {
        $(
            impl Sample for $t {
                type Container = $c;
            }
        )*
    };
}

impl_sample! { u8, u16, u32, u64 => u64 }
impl_sample! { i8, i16, i32, i64 => i64 }
impl_sample! { f32 => f32 }
