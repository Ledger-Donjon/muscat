use itertools::Itertools;
use ndarray::{s, Array1, ArrayView1};
use num_traits::{AsPrimitive, One, Zero};
use std::{
    cmp::Ordering,
    ops::{Div, Range},
};

use crate::processors::MeanVar;

/// Computes the centered product of "order" leakage samples
/// Used particularly when performing high-order SCA
#[derive(Debug)]
pub struct CenteredProduct {
    /// Sum of traces
    acc: Array1<i64>,
    /// Number of traces processed
    count: usize,
    /// Mean of traces
    mean: Array1<f64>,
    /// Indices of samples to combine
    intervals: Vec<Range<i32>>,
    /// Boolean to ensure that finalize function happened before apply
    processed: bool,
}

impl CenteredProduct {
    /// Creates a new CenteredProduct processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples per trace
    /// * `intervals` - Intervals to combine
    pub fn new(size: usize, intervals: Vec<Range<i32>>) -> Self {
        Self {
            acc: Array1::zeros(size),
            count: 0,
            intervals,
            processed: false,
            mean: Array1::zeros(size),
        }
    }

    /// Processes an input trace to update internal accumulators.
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>) {
        let size = self.acc.len();
        for i in 0..size {
            let x = trace[i].into();
            self.acc[i] += x;
        }
        self.count += 1
    }

    /// Compute the mean
    pub fn finalize(&mut self) {
        if self.count != 0 {
            self.mean = self.acc.mapv(|x| x as f64 / self.count as f64)
        }
        self.processed = true
    }

    /// Apply the processing to an input trace
    /// The centered product substract the mean of the traces and then perform products between every input time samples
    pub fn apply<T: Into<f64> + Copy>(&self, trace: ArrayView1<T>) -> Array1<f64> {
        // First we substract the mean trace
        let centered_trace: Array1<f64> = trace.mapv(|x| x.into()) - &self.mean;
        let length_out_trace: usize = self.intervals.iter().map(|x| x.len()).product();

        let mut centered_product_trace = Array1::ones(length_out_trace);

        // Then we do the products
        let multi_prod = (0..self.intervals.len())
            .map(|i| self.intervals[i].clone())
            .multi_cartesian_product(); //NOTE/TODO: maybe this can go in the struct parameters, which could improve performances

        for (idx, combination) in multi_prod.enumerate() {
            for i in combination {
                centered_product_trace[idx] *= centered_trace[i as usize] as f64;
            }
        }

        centered_product_trace
    }
}

/// Elevates parts of a trace to a certain power
#[derive(Debug)]
pub struct Power {
    intervals: Vec<Range<i32>>,
    power: i32,
}

impl Power {
    /// Creates a new Power processor.
    ///
    /// # Arguments
    ///
    /// * `intervals` - Intervals to elevate to the power
    /// * `power` - Power to elevate
    pub fn new(intervals: Vec<Range<i32>>, power: i32) -> Self {
        Self { intervals, power }
    }

    /// Processes an input trace
    pub fn process<T: Into<i64> + Copy>(&self, trace: ArrayView1<T>) -> Array1<f64> {
        // Concatenate the slices specified by the ranges
        let result: Array1<_> = self
            .intervals
            .iter()
            .flat_map(|range| trace.slice(s![range.clone()]).to_owned())
            .map(|val| val.into() as f64)
            .collect();

        result.mapv(|result| result.powi(self.power))
    }
}

/// Standardization of the traces by removing the mean and scaling to unit variance
#[derive(Debug)]
pub struct StandardScaler {
    /// meanVar processor
    meanvar: MeanVar,
    /// mean
    mean: Array1<f64>,
    /// std
    std: Array1<f64>,
}

impl StandardScaler {
    pub fn new(size: usize) -> Self {
        Self {
            meanvar: MeanVar::new(size),
            mean: Array1::zeros(size),
            std: Array1::zeros(size),
        }
    }

    /// Processes an input trace to update internal accumulators.
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: ArrayView1<T>) {
        self.meanvar.process(trace);
    }

    /// Compute mean and var
    pub fn finalize(&mut self) {
        self.mean = self.meanvar.mean();
        self.std = self.meanvar.var().mapv(f64::sqrt);
    }

    /// Apply the processing to an input trace
    pub fn apply<T: Into<f64> + Copy>(&self, trace: ArrayView1<T>) -> Array1<f64> {
        (trace.mapv(|x| x.into()) - &self.mean) / &self.std
    }
}

pub use dtw::dist;

/// Align traces using elastic alignment [1]. Elastic alignment is a dynamic alignment algorithm
/// based on FastDTW.
///
/// # Examples
/// ```rust
/// use muscat::preprocessors::{ElasticAlignment, dist::euclidean_distance};
/// use ndarray::array;
///
/// let reference_trace = array![77, 117, 5, 51, 91, -12, -33];
/// let trace_to_align = array![77, 117, 13, 15, 5, 51, 91];
/// let elastic_alignment = ElasticAlignment::new(reference_trace, 1, euclidean_distance);
///
/// let aligned_trace = elastic_alignment.align(trace_to_align.view());
/// ```
///
/// # References
/// [1] van Woudenberg, J.G.J., Witteman, M.F., Bakker, B. (2011). Improving Differential Power
/// Analysis by Elastic Alignment. In: Kiayias, A. (eds) Topics in Cryptology – CT-RSA 2011. CT-RSA
/// 2011. Lecture Notes in Computer Science, vol 6558. Springer, Berlin, Heidelberg.
/// https://doi.org/10.1007/978-3-642-19074-2_8
pub struct ElasticAlignment<T, D>
where
    D: Fn(T, T) -> T,
{
    reference_trace: Array1<T>,
    /// FastDTW radius
    radius: usize,
    /// Distance function
    dist: D,
}

impl<T, D> ElasticAlignment<T, D>
where
    T: dtw::Average + dtw::SumContainer + Copy + Zero + 'static,
    T::Container: Zero + One + Div<Output = T::Container> + AsPrimitive<T>,
    D: Fn(T, T) -> T,
{
    /// Creates a new [`ElasticAlignment`] with the given reference trace and distance function.
    pub fn new(reference_trace: Array1<T>, radius: usize, dist: D) -> Self {
        Self {
            reference_trace,
            radius,
            dist,
        }
    }

    /// Align given trace using elastic alignment (see [`ElasticAlignment`]).
    pub fn align_with_cmp<C>(&self, trace: ArrayView1<T>, cmp: &C) -> Array1<T>
    where
        C: Fn(&T::Container, &T::Container) -> Ordering,
    {
        let warp_path = dtw::fast_dtw_with_cmp(
            self.reference_trace.as_slice().unwrap(),
            trace.as_slice().unwrap(),
            self.radius,
            &self.dist,
            cmp,
        );

        let mut aligned_trace = Array1::zeros([trace.len()]);
        let mut k = 0;
        for j in 0..trace.len() {
            let mut count = T::Container::zero();
            let mut sum = T::Container::zero();

            while k < warp_path.len() && warp_path[k].0 == j {
                count = count + T::Container::one();
                sum = sum + T::Container::from(trace[warp_path[k].1]);
                k += 1;
            }

            aligned_trace[j] = (sum / count).as_();
        }

        aligned_trace
    }
}

impl<T, D> ElasticAlignment<T, D>
where
    T: dtw::Average + dtw::SumContainer + Copy + Zero + 'static,
    T::Container: Zero + One + Div<Output = T::Container> + AsPrimitive<T> + Ord,
    D: Fn(T, T) -> T,
{
    /// Align given trace using elastic alignment (see [`ElasticAlignment`]).
    ///
    /// NOTE: See [`ElasticAlignment::align_with_cmp`] for type than do not implement [`Ord`].
    pub fn align(&self, trace: ArrayView1<T>) -> Array1<T> {
        self.align_with_cmp(trace, &T::Container::cmp)
    }
}

#[cfg(test)]
mod tests {
    use crate::preprocessors::{dist, CenteredProduct, ElasticAlignment, Power, StandardScaler};
    use ndarray::array;

    fn round_to_2_digits(x: f64) -> f64 {
        (x * 100f64).round() / 100f64
    }

    #[test]
    fn test_centered_product() {
        let mut processor = CenteredProduct::new(5, vec![0..2, 3..5]);
        processor.process(array![0i16, 1i16, 2i16, -3i16, -4i16].view());
        processor.finalize();
        assert_eq!(
            processor.apply(array![0i16, 1i16, 2i16, -3i16, -4i16].view()),
            array![0f64, 0f64, 0f64, 0f64]
        );
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

        let mut processor2 = CenteredProduct::new(4, vec![0..1, 1..2, 2..4]);
        for t in traces.iter() {
            processor2.process(t.view());
        }
        processor2.finalize();

        let expected_results = [
            array![-11169.72, 61984.08],
            array![-32942.77, -31440.82],
            array![-540.46, -2533.96],
            array![-1521.45, 2049.30],
            array![36499.87, 36969.02],
            array![-36199.24, -4675.44],
            array![-3999.12, 37044.48],
            array![-189.74, -279.84],
            array![-54268.83, -121223.88],
            array![-46231.64, -130058.24],
        ];

        for (i, t) in traces.iter().enumerate() {
            assert_eq!(
                processor2.apply(t.view()).mapv(round_to_2_digits),
                expected_results[i]
            );
        }
    }

    #[test]
    fn test_power() {
        let t = array![-1, 2, -3, 4, -5, 6];
        let processor1 = Power {
            intervals: vec![0..2, 4..6],
            power: 1,
        };
        let processor2 = Power {
            intervals: vec![0..2, 4..6],
            power: 2,
        };
        let processor3 = Power {
            intervals: vec![0..2, 4..6],
            power: 3,
        };
        let expected_results = [
            array![-1.0, 2.0, -5.0, 6.0],
            array![1.0, 4.0, 25.0, 36.0],
            array![-1.0, 8.0, -125.0, 216.0],
        ];

        assert_eq!(
            processor1.process(t.view()).mapv(round_to_2_digits),
            expected_results[0]
        );
        assert_eq!(
            processor2.process(t.view()).mapv(round_to_2_digits),
            expected_results[1]
        );
        assert_eq!(
            processor3.process(t.view()).mapv(round_to_2_digits),
            expected_results[2]
        );
    }

    #[test]
    fn test_standard_scaler() {
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

        let mut processor = StandardScaler::new(4);
        for t in traces.iter() {
            processor.process(t.view());
        }
        processor.finalize();

        let expected_results = [
            array![1.25, 0.75, -0.28, 1.29],
            array![1.07, -0.56, 1.32, 1.03],
            array![-0.07, -0.76, -0.24, -0.94],
            array![-0.52, 0.36, 0.20, -0.22],
            array![-0.38, -1.47, 1.55, 1.29],
            array![-0.97, -0.44, -2.04, -0.22],
            array![-0.83, 1.51, 0.08, -0.58],
            array![0.07, -0.09, 0.72, 0.86],
            array![-1.42, -0.99, -0.92, -1.69],
            array![1.80, 1.68, -0.36, -0.84],
        ];

        for (i, t) in traces.iter().enumerate() {
            assert_eq!(
                processor.apply(t.view()).mapv(round_to_2_digits),
                expected_results[i]
            );
        }
    }

    #[test]
    fn test_elastic_align() {
        let reference_trace = array![77, 117, 5, 51, 91, -12, -33];
        let trace = array![77, 117, 13, 15, 5, 51, 91];

        let elastic_alignment =
            ElasticAlignment::new(reference_trace.clone(), 1, dist::euclidean_distance);
        assert_eq!(
            elastic_alignment.align(trace.view()),
            array![77, 117, 11, 51, 51, 51, 91]
        );
    }

    #[test]
    fn test_elastic_align_same_trace() {
        let reference_trace = array![77, 117, 5, 51, 91, -12, -33];

        let elastic_alignment =
            ElasticAlignment::new(reference_trace.clone(), 1, dist::euclidean_distance);
        assert_eq!(
            elastic_alignment.align(reference_trace.view()),
            reference_trace
        );
    }
}
