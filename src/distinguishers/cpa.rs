use crate::util::{argmax_by, argsort_by, max_per_row};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelIterator, ParallelIterator},
};
use std::{iter::zip, ops::Add};

/// Compute the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Panics
/// - Panic if `traces.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `batch_size` is 0.
pub fn cpa<T, F>(
    traces: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: F,
    batch_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
    F: Fn(usize, usize) -> usize + Send + Sync + Copy,
{
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);
    assert!(batch_size > 0);

    // From benchmarks fold + reduce_with is faster than map + reduce/reduce_with and fold + reduce
    zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        plaintexts.axis_chunks_iter(Axis(0), batch_size),
    )
    .par_bridge()
    .fold(
        || CpaProcessor::new(traces.shape()[1], guess_range, target_byte, leakage_func),
        |mut cpa, (trace_batch, plaintext_batch)| {
            for i in 0..trace_batch.shape()[0] {
                cpa.update(trace_batch.row(i), plaintext_batch.row(i));
            }

            cpa
        },
    )
    .reduce_with(|a, b| a + b)
    .unwrap()
    .finalize()
}

/// Result of the CPA[^1] on some traces.
///
/// [^1]: <https://www.iacr.org/archive/ches2004/31560016/31560016.pdf>
#[derive(Debug)]
pub struct Cpa {
    /// Pearson correlation coefficients
    pub(crate) corr: Array2<f32>,
}

impl Cpa {
    /// Rank guesses.
    pub fn rank(&self) -> Array1<usize> {
        let rank = argsort_by(&self.max_corr().to_vec()[..], f32::total_cmp);

        Array1::from_vec(rank)
    }

    /// Return the Pearson correlation coefficients.
    pub fn corr(&self) -> ArrayView2<f32> {
        self.corr.view()
    }

    /// Return the guess with the highest Pearson correlation coefficient.
    pub fn best_guess(&self) -> usize {
        argmax_by(self.max_corr().view(), f32::total_cmp)
    }

    /// Return the maximum Pearson correlation coefficient for each guess.
    pub fn max_corr(&self) -> Array1<f32> {
        max_per_row(self.corr.view())
    }
}

/// A processor that computes the [`Cpa`] of the given traces.
///
/// It implements algorithm 4 from [^1].
///
/// [^1]: <https://eprint.iacr.org/2013/794.pdf>
pub struct CpaProcessor<F>
where
    F: Fn(usize, usize) -> usize,
{
    /// Number of samples per trace
    num_samples: usize,
    /// Target byte index in a block
    target_byte: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces
    sum_traces: Array1<usize>,
    /// Sum of square of traces
    sum_square_traces: Array1<usize>,
    /// Sum of traces per key guess
    guess_sum_traces: Array1<usize>,
    /// Sum of square of traces per key guess
    guess_sum_squares_traces: Array1<usize>,
    /// Sum of traces per plaintext used
    /// See 4.3 in <https://eprint.iacr.org/2013/794.pdf>
    plaintext_sum_traces: Array2<usize>,
    /// Leakage model
    leakage_func: F,
    /// Number of traces processed
    num_traces: usize,
}

impl<F> CpaProcessor<F>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    pub fn new(
        num_samples: usize,
        guess_range: usize,
        target_byte: usize,
        leakage_func: F,
    ) -> Self {
        Self {
            num_samples,
            target_byte,
            guess_range,
            sum_traces: Array1::zeros(num_samples),
            sum_square_traces: Array1::zeros(num_samples),
            guess_sum_traces: Array1::zeros(guess_range),
            guess_sum_squares_traces: Array1::zeros(guess_range),
            plaintext_sum_traces: Array2::zeros((guess_range, num_samples)),
            leakage_func,
            num_traces: 0,
        }
    }

    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.num_samples`.
    pub fn update<T>(&mut self, trace: ArrayView1<T>, plaintext: ArrayView1<T>)
    where
        T: Into<usize> + Copy,
    {
        debug_assert_eq!(trace.shape()[0], self.num_samples);

        let partition = plaintext[self.target_byte].into();
        for i in 0..self.num_samples {
            self.sum_traces[i] += trace[i].into();
            self.sum_square_traces[i] += trace[i].into() * trace[i].into();

            self.plaintext_sum_traces[[partition, i]] += trace[i].into();
        }

        for guess in 0..self.guess_range {
            let value = (self.leakage_func)(plaintext[self.target_byte].into(), guess);
            self.guess_sum_traces[guess] += value;
            self.guess_sum_squares_traces[guess] += value * value;
        }

        self.num_traces += 1;
    }

    /// Finalize the calculation after feeding the overall traces.
    pub fn finalize(&self) -> Cpa {
        let mut modeled_leakages = Array1::zeros(self.guess_range);

        let mut corr = Array2::zeros((self.guess_range, self.num_samples));
        for guess in 0..self.guess_range {
            for u in 0..self.guess_range {
                modeled_leakages[u] = (self.leakage_func)(u, guess);
            }

            let mean_key = self.guess_sum_traces[guess] as f32 / self.num_traces as f32;
            let mean_squares_key =
                self.guess_sum_squares_traces[guess] as f32 / self.num_traces as f32;
            let var_key = mean_squares_key - (mean_key * mean_key);

            let guess_corr: Vec<_> = (0..self.num_samples)
                .into_par_iter()
                .map(|u| {
                    let mean_traces = self.sum_traces[u] as f32 / self.num_traces as f32;

                    let cov = self.sum_mult(
                        self.plaintext_sum_traces.slice(s![.., u]),
                        modeled_leakages.view(),
                    );
                    let cov = cov as f32 / self.num_traces as f32 - (mean_key * mean_traces);

                    let mean_squares_traces =
                        self.sum_square_traces[u] as f32 / self.num_traces as f32;
                    let var_traces = mean_squares_traces - (mean_traces * mean_traces);
                    f32::abs(cov / f32::sqrt(var_key * var_traces))
                })
                .collect();

            #[allow(clippy::needless_range_loop)]
            for u in 0..self.num_samples {
                corr[[guess, u]] = guess_corr[u];
            }
        }

        Cpa { corr }
    }

    fn sum_mult(&self, a: ArrayView1<usize>, b: ArrayView1<usize>) -> usize {
        a.dot(&b)
    }

    /// Determine if two [`CpaProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    ///
    /// Note: [`CpaProcessor::leakage_func`] cannot be checked for equality, but they must have the
    /// same leakage functions in order to be compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples
            && self.target_byte == other.target_byte
            && self.guess_range == other.guess_range
    }
}

impl<F> Add for CpaProcessor<F>
where
    F: Fn(usize, usize) -> usize + Sync,
{
    type Output = Self;

    /// Merge computations of two [`CpaProcessor`]. Processors need to be compatible to be merged
    /// together, otherwise it can panic or yield incoherent result (see
    /// [`CpaProcessor::is_compatible_with`]).
    ///
    /// # Panics
    /// Panics in debug if the processors are not compatible.
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.is_compatible_with(&rhs));

        Self {
            num_samples: self.num_samples,
            target_byte: self.target_byte,
            guess_range: self.guess_range,
            sum_traces: self.sum_traces + rhs.sum_traces,
            sum_square_traces: self.sum_square_traces + rhs.sum_square_traces,
            guess_sum_traces: self.guess_sum_traces + rhs.guess_sum_traces,
            guess_sum_squares_traces: self.guess_sum_squares_traces + rhs.guess_sum_squares_traces,
            plaintext_sum_traces: self.plaintext_sum_traces + rhs.plaintext_sum_traces,
            leakage_func: self.leakage_func,
            num_traces: self.num_traces + rhs.num_traces,
        }
    }
}
