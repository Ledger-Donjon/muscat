//! Traces processing algorithms, such as T-Test, SNR, etc.
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, ops::Add};

/// Processes traces to calculate mean and variance.
#[derive(Clone)]
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
        debug_assert!(trace.len() == self.sum.len());

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

        self.sum.map(|&x| x as f64 / count)
    }

    /// Calculates and returns traces variance.
    pub fn var(&self) -> Array1<f64> {
        let count = self.count as f64;

        zip(self.sum.iter(), self.sum_squares.iter())
            .map(|(&sum, &sum_squares)| (sum_squares as f64 / count) - (sum as f64 / count).powi(2))
            .collect()
    }

    /// Returns the number of traces processed.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Add for MeanVar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum: self.sum + rhs.sum,
            sum_squares: self.sum_squares + rhs.sum_squares,
            count: self.count + rhs.count,
        }
    }
}

/// Computes the SNR of the given traces.
///
/// `get_class` is a function returning the class of the given trace by index.
///
/// # Panics
/// Panic if `chunk_size` is 0.
pub fn snr<T, F>(
    leakages: ArrayView2<T>,
    classes: usize,
    get_class: F,
    chunk_size: usize,
) -> Array1<f64>
where
    T: Into<i64> + Copy + Sync,
    F: Fn(usize) -> usize + Sync,
{
    assert!(chunk_size > 0);

    // From benchmarks fold + reduce_with is faster than map + reduce/reduce_with and fold + reduce
    leakages
        .axis_chunks_iter(Axis(0), chunk_size)
        .enumerate()
        .par_bridge()
        .fold(
            || Snr::new(leakages.shape()[1], classes),
            |mut snr, (chunk_idx, leakages_chunk)| {
                for i in 0..leakages_chunk.shape()[0] {
                    snr.process(leakages_chunk.row(i), get_class(chunk_idx + i));
                }
                snr
            },
        )
        .reduce_with(|a, b| a + b)
        .unwrap()
        .snr()
}

/// Processes traces to calculate the Signal-to-Noise Ratio.
#[derive(Clone)]
pub struct Snr {
    mean_var: MeanVar,
    /// Sum of traces per class
    classes_sum: Array2<i64>,
    /// Counts the number of traces per class
    classes_count: Array1<usize>,
}

impl Snr {
    /// Creates a new SNR processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the input traces
    /// * `classes` - Number of classes
    pub fn new(size: usize, classes: usize) -> Self {
        Self {
            mean_var: MeanVar::new(size),
            classes_sum: Array2::zeros((classes, size)),
            classes_count: Array1::zeros(classes),
        }
    }

    /// Processes an input trace to update internal accumulators.
    ///
    /// # Panics
    /// Panics in debug if the length of the trace is different from the size of [`Snr`].
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>, class: usize) {
        debug_assert!(trace.len() == self.classes_sum.shape()[1]);

        self.mean_var.process(trace);

        for i in 0..self.classes_sum.shape()[1] {
            self.classes_sum[[class, i]] += trace[i].into();
        }

        self.classes_count[class] += 1;
    }

    /// Returns Signal-to-Noise Ratio of the traces.
    /// SNR = V[E[L|X]] / E[V[L|X]]
    pub fn snr(&self) -> Array1<f64> {
        let size = self.classes_sum.shape()[1];
        let classes = self.classes_sum.shape()[0];

        let mut acc: Array1<f64> = Array1::zeros(size);
        for class in 0..classes {
            if self.classes_count[class] == 0 {
                continue;
            }

            let class_sum = self.classes_sum.slice(s![class, ..]);
            for i in 0..size {
                acc[i] += (class_sum[i] as f64).powi(2) / (self.classes_count[class] as f64);
            }
        }

        let var = self.mean_var.var();
        let mean = self.mean_var.mean();
        // V[E[L|X]]
        let velx = (acc / self.mean_var.count as f64) - mean.mapv(|x| x.powi(2));
        1f64 / (var / velx - 1f64)
    }
}

impl Add for Snr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mean_var: self.mean_var + rhs.mean_var,
            classes_sum: self.classes_sum + rhs.classes_sum,
            classes_count: self.classes_count + rhs.classes_count,
        }
    }
}

/// Processes traces to calculate Welch's T-Test.
pub struct TTest {
    mean_var_1: MeanVar,
    mean_var_2: MeanVar,
}

impl TTest {
    /// Creates a new Welch's T-Test processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples per trace
    pub fn new(size: usize) -> Self {
        Self {
            mean_var_1: MeanVar::new(size),
            mean_var_2: MeanVar::new(size),
        }
    }

    /// Processes an input trace to update internal accumulators.
    ///
    /// # Arguments
    ///
    /// * `trace` - Input trace.
    /// * `class` - Indicates to which of the two partitions the given trace belongs.
    ///
    /// # Panics
    /// Panics in debug if `trace.len() != self.mean_var_1.sum.len()`.
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>, class: bool) {
        debug_assert!(trace.len() == self.mean_var_1.sum.len());

        if class {
            self.mean_var_2.process(trace);
        } else {
            self.mean_var_1.process(trace);
        }
    }

    /// Calculate and returns Welch's T-Test result.
    pub fn ttest(&self) -> Array1<f64> {
        // E(X1) - E(X2)
        let q = self.mean_var_1.mean() - self.mean_var_2.mean();

        // √(σ1²/N1 + σ2²/N2)
        let d = ((self.mean_var_1.var() / self.mean_var_1.count() as f64)
            + (self.mean_var_2.var() / self.mean_var_2.count() as f64))
            .mapv(f64::sqrt);
        q / d
    }
}

impl Add for TTest {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mean_var_1: self.mean_var_1 + rhs.mean_var_1,
            mean_var_2: self.mean_var_2 + rhs.mean_var_2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MeanVar;
    use crate::processors::TTest;
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

    #[test]
    fn test_ttest() {
        let mut processor = TTest::new(4);
        let traces = [
            array![77, 137, 51, 91],
            array![72, 61, 91, 83],
            array![39, 49, 52, 23],
            array![26, 114, 63, 45],
            array![30, 8, 97, 91],
            array![13, 68, 7, 45],
            array![17, 181, 60, 34],
            array![43, 88, 76, 78],
            array![0, 36, 35, 0],
            array![93, 191, 49, 26],
        ];
        for (i, trace) in traces.iter().enumerate() {
            processor.process(trace.view(), i % 3 == 0);
        }
        assert_eq!(
            processor.ttest(),
            array![
                -1.0910344547297484,
                -5.524921845887032,
                0.29385284736362266,
                0.23308466737856662
            ]
        );
    }
}
