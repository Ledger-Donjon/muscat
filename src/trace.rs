//! Defines the [`Trace`] storage structure.
use ndarray::Array1;

/// A side channel leakage record associated to its leakage data.
///
/// Leakage is stored as an `Array1<T>`. The leakage data type is `U`.
pub struct Trace<T, U> {
    /// Leakage waveform
    pub leakage: Array1<T>,
    /// Associated leakage data
    pub value: U,
}

impl<T, U> Trace<T, U> {
    pub fn new(leakage: Array1<T>, value: U) -> Self {
        Self { leakage, value }
    }

    /// Returns the number of points in the leakage waveform.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.leakage.len()
    }
}
