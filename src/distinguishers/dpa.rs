use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::{iter::zip, marker::PhantomData, ops::Add};

use crate::util::{argmax_by, argsort_by, max_per_row};

/// Compute the [`Dpa`] of the given traces using [`DpaProcessor`].
///
/// # Examples
/// ```
/// use muscat::distinguishers::dpa::dpa;
/// use muscat::leakage::sbox;
/// use ndarray::{array, Array1};
///
/// let traces = array![
///     [77, 137, 51, 91],
///     [72, 61, 91, 83],
///     [39, 49, 52, 23],
///     [26, 114, 63, 45],
///     [30, 8, 97, 91],
///     [13, 68, 7, 45],
///     [17, 181, 60, 34],
///     [43, 88, 76, 78],
///     [0, 36, 35, 0],
///     [93, 191, 49, 26],
/// ];
/// let plaintexts = array![
///     [1, 2],
///     [2, 1],
///     [1, 2],
///     [1, 2],
///     [2, 1],
///     [2, 1],
///     [1, 2],
///     [1, 2],
///     [2, 1],
///     [2, 1],
/// ];
/// let dpa = dpa(
///     traces.map(|&x| x as f32).view(),
///     plaintexts
///         .rows()
///         .into_iter()
///         .map(|x| x.to_owned())
///         .collect::<Array1<Array1<u8>>>()
///         .view(),
///     256,
///     |key: Array1<u8>, guess| sbox(key[0] ^ guess as u8) & 1 == 1,
///     2
/// );
/// ```
///
/// # Panics
/// Panic if `batch_size` is not strictly positive.
pub fn dpa<M, T, F>(
    traces: ArrayView2<T>,
    metadata: ArrayView1<M>,
    guess_range: usize,
    selection_function: F,
    batch_size: usize,
) -> Dpa
where
    T: Into<f32> + Copy + Sync,
    M: Clone + Send + Sync,
    F: Fn(M, usize) -> bool + Send + Sync + Copy,
{
    assert!(batch_size > 0);

    zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        metadata.axis_chunks_iter(Axis(0), batch_size),
    )
    .par_bridge()
    .fold(
        || DpaProcessor::new(traces.shape()[1], guess_range, selection_function),
        |mut dpa, (trace_batch, metadata_batch)| {
            for i in 0..trace_batch.shape()[0] {
                dpa.update(trace_batch.row(i), metadata_batch[i].clone());
            }

            dpa
        },
    )
    .reduce_with(|a, b| a + b)
    .unwrap()
    .finalize()
}

/// Result of the DPA[^1] on some traces.
///
/// [^1]: <https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf>
#[derive(Debug)]
pub struct Dpa {
    differential_curves: Array2<f32>,
}

impl Dpa {
    /// Return the rank of guesses
    pub fn rank(&self) -> Array1<usize> {
        let rank = argsort_by(&self.max_differential_curves().to_vec()[..], f32::total_cmp);

        Array1::from_vec(rank)
    }

    /// Return the differential curves
    pub fn differential_curves(&self) -> ArrayView2<f32> {
        self.differential_curves.view()
    }

    /// Return the guess with the highest differential peak.
    pub fn best_guess(&self) -> usize {
        argmax_by(self.max_differential_curves().view(), f32::total_cmp)
    }

    /// Return the maximum differential peak for each guess.
    pub fn max_differential_curves(&self) -> Array1<f32> {
        max_per_row(self.differential_curves.view())
    }
}

/// A processor that computes the [`Dpa`] of the given traces.
///
/// [^1]: <https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf>
/// [^2]: <https://web.mit.edu/6.857/OldStuff/Fall03/ref/kocher-DPATechInfo.pdf>
pub struct DpaProcessor<M, F>
where
    F: Fn(M, usize) -> bool,
{
    /// Number of samples per trace
    num_samples: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces for which the selection function equals false
    sum_0: Array2<f32>,
    /// Sum of traces for which the selection function equals true
    sum_1: Array2<f32>,
    /// Number of traces processed for which the selection function equals false
    count_0: Array1<usize>,
    /// Number of traces processed for which the selection function equals true
    count_1: Array1<usize>,
    selection_function: F,
    /// Number of traces processed
    num_traces: usize,
    _metadata: PhantomData<M>,
}

impl<M, F> DpaProcessor<M, F>
where
    M: Clone,
    F: Fn(M, usize) -> bool,
{
    pub fn new(num_samples: usize, guess_range: usize, selection_function: F) -> Self {
        Self {
            num_samples,
            guess_range,
            sum_0: Array2::zeros((guess_range, num_samples)),
            sum_1: Array2::zeros((guess_range, num_samples)),
            count_0: Array1::zeros(guess_range),
            count_1: Array1::zeros(guess_range),
            selection_function,
            num_traces: 0,
            _metadata: PhantomData,
        }
    }

    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.num_samples`.
    pub fn update<T>(&mut self, trace: ArrayView1<T>, metadata: M)
    where
        T: Into<f32> + Copy,
    {
        debug_assert_eq!(trace.shape()[0], self.num_samples);

        for guess in 0..self.guess_range {
            if (self.selection_function)(metadata.clone(), guess) {
                for i in 0..self.num_samples {
                    self.sum_1[[guess, i]] += trace[i].into();
                }
                self.count_1[guess] += 1;
            } else {
                for i in 0..self.num_samples {
                    self.sum_0[[guess, i]] += trace[i].into();
                }
                self.count_0[guess] += 1;
            }
        }

        self.num_traces += 1;
    }

    /// Finalizes the calculation after feeding the overall traces.
    pub fn finalize(&self) -> Dpa {
        let mut differential_curves = Array2::zeros((self.guess_range, self.num_samples));
        for guess in 0..self.guess_range {
            for i in 0..self.num_samples {
                let mean_0 = self.sum_0[[guess, i]] / self.count_0[guess] as f32;
                let mean_1 = self.sum_1[[guess, i]] / self.count_1[guess] as f32;

                differential_curves[[guess, i]] = f32::abs(mean_0 - mean_1);
            }
        }

        Dpa {
            differential_curves,
        }
    }

    /// Determine if two [`DpaProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    ///
    /// Note: [`DpaProcessor::selection_function`] cannot be checked for equality, but they must
    /// have the same selection functions in order to be compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples && self.guess_range == other.guess_range
    }
}

impl<M, F> Add for DpaProcessor<M, F>
where
    F: Fn(M, usize) -> bool,
    M: Clone,
{
    type Output = Self;

    /// Merge computations of two [`DpaProcessor`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`DpaProcessor::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            num_samples: self.num_samples,
            guess_range: self.guess_range,
            sum_0: self.sum_0 + rhs.sum_0,
            sum_1: self.sum_1 + rhs.sum_1,
            count_0: self.count_0 + rhs.count_0,
            count_1: self.count_1 + rhs.count_1,
            selection_function: self.selection_function,
            num_traces: self.num_traces + rhs.num_traces,
            _metadata: PhantomData,
        }
    }
}
