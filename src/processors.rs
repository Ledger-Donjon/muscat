//! Traces processing algorithms
use ndarray::{Array1, ArrayView1};
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::{iter::zip, ops::Add};

use crate::Sample;

/// Processes traces to calculate mean and variance.
#[derive(Serialize, Deserialize)]
pub struct MeanVar<T>
where
    T: Sample,
{
    /// Sum of traces
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    sum: Array1<<T as Sample>::Container>,
    /// Sum of square of traces
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    sum_squares: Array1<<T as Sample>::Container>,
    /// Number of traces processed
    count: usize,
}

impl<T> MeanVar<T>
where
    T: Sample + Copy,
{
    /// Creates a new mean and variance processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples per trace
    pub fn new(size: usize) -> Self {
        Self {
            sum: Array1::zeros(size),
            sum_squares: Array1::zeros(size),
            count: 0,
        }
    }

    /// Processes an input trace to update internal accumulators.
    ///
    /// # Panics
    /// Panics in debug if the length of the trace is different form the size of [`MeanVar`].
    pub fn process(&mut self, trace: ArrayView1<T>) {
        debug_assert!(trace.shape()[0] == self.size());

        for i in 0..self.sum.shape()[0] {
            let x = trace[i].into();

            self.sum[i] += x;
            self.sum_squares[i] += x * x;
        }

        self.count += 1;
    }

    /// Returns trace mean.
    pub fn mean(&self) -> Array1<f32> {
        self.sum.mapv(|x| x.as_() / self.count as f32)
    }

    /// Calculates and returns traces variance.
    pub fn var(&self) -> Array1<f32> {
        let count = self.count as f32;

        zip(self.sum.iter(), self.sum_squares.iter())
            .map(|(&sum, &sum_squares)| (sum_squares.as_() / count) - (sum.as_() / count).powi(2))
            .collect()
    }

    /// Returns the trace size handled.
    pub fn size(&self) -> usize {
        self.sum.shape()[0]
    }

    /// Returns the number of traces processed.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Determine if two [`MeanVar`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.size() == other.size()
    }
}

impl<T> Add for MeanVar<T>
where
    T: Sample + Copy,
{
    type Output = Self;

    /// Merge computations of two [`MeanVar`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`MeanVar::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            sum: self.sum + rhs.sum,
            sum_squares: self.sum_squares + rhs.sum_squares,
            count: self.count + rhs.count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MeanVar;
    use ndarray::array;

    #[test]
    fn test_mean_var() {
        let mut processor = MeanVar::new(4);
        processor.process(array![28038i16, 22066i16, -20614i16, -9763i16].view());
        assert_eq!(
            processor.mean(),
            array![28038f32, 22066f32, -20614f32, -9763f32]
        );
        assert_eq!(processor.var(), array![0f32, 0f32, 0f32, 0f32]);
        processor.process(array![31377, -6950, -15666, 26773].view());
        processor.process(array![24737, -18311, 24742, 17207].view());
        processor.process(array![12974, -29255, -28798, 18988].view());
        assert_eq!(
            processor.mean(),
            array![24281.5f32, -8112.5f32, -10084f32, 13301.25f32]
        );
        assert_eq!(
            processor.var(),
            array![48131136.0, 365777020.0, 426275900.0, 190260430.0]
        );
    }
}
