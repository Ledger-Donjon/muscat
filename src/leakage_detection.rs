//! Leakage detection methods
use crate::processors::MeanVar;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, ops::Add};

/// Compute the SNR of the given traces.
///
/// `get_class` is a function returning the class of the given trace by index.
///
/// # Panics
/// Panic if `batch_size` is 0.
pub fn snr<T, F>(
    leakages: ArrayView2<T>,
    classes: usize,
    get_class: F,
    batch_size: usize,
) -> Array1<f64>
where
    T: Into<i64> + Copy + Sync,
    F: Fn(usize) -> usize + Sync,
{
    assert!(batch_size > 0);

    // From benchmarks fold + reduce_with is faster than map + reduce/reduce_with and fold + reduce
    leakages
        .axis_chunks_iter(Axis(0), batch_size)
        .enumerate()
        .par_bridge()
        .fold(
            || Snr::new(leakages.shape()[1], classes),
            |mut snr, (batch_idx, leakage_batch)| {
                for i in 0..leakage_batch.shape()[0] {
                    snr.process(leakage_batch.row(i), get_class(batch_idx + i));
                }
                snr
            },
        )
        .reduce_with(|a, b| a + b)
        .unwrap()
        .snr()
}

/// Processes traces to calculate the Signal-to-Noise Ratio.
#[derive(Debug, Clone)]
pub struct Snr {
    mean_var: MeanVar,
    /// Sum of traces per class
    classes_sum: Array2<i64>,
    /// Counts the number of traces per class
    classes_count: Array1<usize>,
}

impl Snr {
    /// Create a new SNR processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the input traces
    /// * `num_classes` - Number of classes
    pub fn new(size: usize, num_classes: usize) -> Self {
        Self {
            mean_var: MeanVar::new(size),
            classes_sum: Array2::zeros((num_classes, size)),
            classes_count: Array1::zeros(num_classes),
        }
    }

    /// Process an input trace to update internal accumulators.
    ///
    /// # Panics
    /// Panics in debug if the length of the trace is different from the size of [`Snr`].
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>, class: usize) {
        debug_assert!(trace.len() == self.size());

        self.mean_var.process(trace);

        for i in 0..self.size() {
            self.classes_sum[[class, i]] += trace[i].into();
        }

        self.classes_count[class] += 1;
    }

    /// Returns the Signal-to-Noise Ratio of the traces.
    /// SNR = V[E[L|X]] / E[V[L|X]]
    pub fn snr(&self) -> Array1<f64> {
        let size = self.size();

        let mut acc: Array1<f64> = Array1::zeros(size);
        for class in 0..self.num_classes() {
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
        let velx = (acc / self.mean_var.count() as f64) - mean.mapv(|x| x.powi(2));
        1f64 / (var / velx - 1f64)
    }

    /// Return the trace size handled
    pub fn size(&self) -> usize {
        self.classes_sum.shape()[1]
    }

    /// Return the number of classes handled.
    pub fn num_classes(&self) -> usize {
        self.classes_count.len()
    }

    /// Determine if two [`Snr`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.size() == other.size() && self.num_classes() == other.num_classes()
    }
}

impl Add for Snr {
    type Output = Self;

    /// Merge computations of two [`Snr`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`Snr::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            mean_var: self.mean_var + rhs.mean_var,
            classes_sum: self.classes_sum + rhs.classes_sum,
            classes_count: self.classes_count + rhs.classes_count,
        }
    }
}

/// Compute the Welch's T-test of the given traces.
///
/// # Panics
/// - Panic if `traces.shape()[0] != trace_classes.shape()[0]`
/// - Panic if `batch_size` is 0.
pub fn ttest<T>(
    traces: ArrayView2<T>,
    trace_classes: ArrayView1<bool>,
    batch_size: usize,
) -> Array1<f64>
where
    T: Into<i64> + Copy + Sync,
{
    assert_eq!(traces.shape()[0], trace_classes.shape()[0]);
    assert!(batch_size > 0);

    zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        trace_classes.axis_chunks_iter(Axis(0), batch_size),
    )
    .par_bridge()
    .fold(
        || TTest::new(traces.shape()[1]),
        |mut ttest, (trace_batch, trace_classes_batch)| {
            for i in 0..trace_batch.shape()[0] {
                ttest.process(trace_batch.row(i), trace_classes_batch[i]);
            }
            ttest
        },
    )
    .reduce_with(|a, b| a + b)
    .unwrap()
    .ttest()
}

/// Process traces to calculate Welch's T-Test.
#[derive(Debug)]
pub struct TTest {
    mean_var_1: MeanVar,
    mean_var_2: MeanVar,
}

impl TTest {
    /// Create a new Welch's T-Test processor.
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

    /// Process an input trace to update internal accumulators.
    ///
    /// # Arguments
    ///
    /// * `trace` - Input trace.
    /// * `class` - Indicates to which of the two partitions the given trace belongs.
    ///
    /// # Panics
    /// Panics in debug if `trace.len() != self.size()`.
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>, class: bool) {
        debug_assert!(trace.len() == self.size());

        if class {
            self.mean_var_2.process(trace);
        } else {
            self.mean_var_1.process(trace);
        }
    }

    /// Calculate and return Welch's T-Test result.
    pub fn ttest(&self) -> Array1<f64> {
        // E(X1) - E(X2)
        let q = self.mean_var_1.mean() - self.mean_var_2.mean();

        // √(σ1²/N1 + σ2²/N2)
        let d = ((self.mean_var_1.var() / self.mean_var_1.count() as f64)
            + (self.mean_var_2.var() / self.mean_var_2.count() as f64))
            .mapv(f64::sqrt);
        q / d
    }

    /// Return the trace size handled.
    pub fn size(&self) -> usize {
        self.mean_var_1.size()
    }

    /// Determine if two [`TTest`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.size() == other.size()
    }
}

impl Add for TTest {
    type Output = Self;

    /// Merge computations of two [`TTest`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`TTest::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            mean_var_1: self.mean_var_1 + rhs.mean_var_1,
            mean_var_2: self.mean_var_2 + rhs.mean_var_2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ttest, TTest};
    use ndarray::array;

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

    #[test]
    fn test_ttest_helper() {
        let mut processor = TTest::new(4);
        let traces = array![
            [77, 137, 51, 91],
            [72, 61, 91, 83],
            [39, 49, 52, 23],
            [26, 114, 63, 45],
            [30, 8, 97, 91],
            [13, 68, 7, 45],
            [17, 181, 60, 34],
            [43, 88, 76, 78],
            [0, 36, 35, 0],
            [93, 191, 49, 26],
        ];
        let trace_classes =
            array![true, false, false, true, false, false, true, false, false, true];
        for (i, trace) in traces.rows().into_iter().enumerate() {
            processor.process(trace, trace_classes[i]);
        }

        assert_eq!(
            processor.ttest(),
            ttest(traces.view(), trace_classes.view(), 2,)
        );
    }
}
