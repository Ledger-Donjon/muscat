//! Traces processing algorithms
use ndarray::{Array1, ArrayView1};
use std::{iter::zip, ops::Add};

/// Processes traces to calculate mean and variance.
#[derive(Debug, Clone)]
pub struct MeanVar {
    /// Sum of traces
    sum: Array1<i64>,
    /// Sum of square of traces
    sum_squares: Array1<i64>,
    /// Number of traces processed
    count: usize,
}

impl MeanVar {
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
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>) {
        debug_assert!(trace.len() == self.size());

        for i in 0..self.sum.len() {
            let x = trace[i].into();

            self.sum[i] += x;
            self.sum_squares[i] += x * x;
        }

        self.count += 1;
    }

    /// Returns trace mean.
    pub fn mean(&self) -> Array1<f64> {
        let count = self.count as f64;

        self.sum.mapv(|x| x as f64 / count)
    }

    /// Calculates and returns traces variance.
    pub fn var(&self) -> Array1<f64> {
        let count = self.count as f64;

        zip(self.sum.iter(), self.sum_squares.iter())
            .map(|(&sum, &sum_squares)| (sum_squares as f64 / count) - (sum as f64 / count).powi(2))
            .collect()
    }

    /// Returns the trace size handled.
    pub fn size(&self) -> usize {
        self.sum.len()
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

impl Add for MeanVar {
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
            array![28038f64, 22066f64, -20614f64, -9763f64]
        );
        assert_eq!(processor.var(), array![0f64, 0f64, 0f64, 0f64]);
        processor.process(array![31377, -6950, -15666, 26773].view());
        processor.process(array![24737, -18311, 24742, 17207].view());
        processor.process(array![12974, -29255, -28798, 18988].view());
        assert_eq!(
            processor.mean(),
            array![24281.5f64, -8112.5f64, -10084f64, 13301.25f64]
        );
        assert_eq!(
            processor.var(),
            array![48131112.25, 365776994.25, 426275924.0, 190260421.1875]
        );
    }
}
