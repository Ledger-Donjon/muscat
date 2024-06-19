use crate::util::{argsort_by, max_per_row};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelIterator, ParallelIterator},
};
use std::{iter::zip, ops::Add};

/// Compute the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Panics
/// - Panic if `leakages.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `chunk_size` is 0.
pub fn cpa<T, F>(
    leakages: ArrayView2<T>,
    plaintexts: ArrayView2<T>,
    guess_range: usize,
    target_byte: usize,
    leakage_func: F,
    chunk_size: usize,
) -> Cpa
where
    T: Into<usize> + Copy + Sync,
    F: Fn(usize, usize) -> usize + Send + Sync + Copy,
{
    assert_eq!(leakages.shape()[0], plaintexts.shape()[0]);
    assert!(chunk_size > 0);

    // From benchmarks fold + reduce_with is faster than map + reduce/reduce_with and fold + reduce
    zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .fold(
        || CpaProcessor::new(leakages.shape()[1], guess_range, target_byte, leakage_func),
        |mut cpa, (leakages_chunk, plaintexts_chunk)| {
            for i in 0..leakages_chunk.shape()[0] {
                cpa.update(leakages_chunk.row(i), plaintexts_chunk.row(i));
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
    /// Guess range upper excluded bound
    pub(crate) guess_range: usize,
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
        let max_corr = self.max_corr();

        let mut best_guess_corr = 0.0;
        let mut best_guess = 0;
        for guess in 0..self.guess_range {
            if max_corr[guess] > best_guess_corr {
                best_guess_corr = max_corr[guess];
                best_guess = guess;
            }
        }

        best_guess
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
    sum_leakages: Array1<usize>,
    /// Sum of square of traces
    sum_squares_leakages: Array1<usize>,
    /// Sum of traces per key guess
    guess_sum_leakages: Array1<usize>,
    /// Sum of square of traces per key guess
    guess_sum_squares_leakages: Array1<usize>,
    /// Sum of traces per plaintext used
    /// See 4.3 in <https://eprint.iacr.org/2013/794.pdf>
    plaintext_sum_leakages: Array2<usize>,
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
            sum_leakages: Array1::zeros(num_samples),
            sum_squares_leakages: Array1::zeros(num_samples),
            guess_sum_leakages: Array1::zeros(guess_range),
            guess_sum_squares_leakages: Array1::zeros(guess_range),
            plaintext_sum_leakages: Array2::zeros((guess_range, num_samples)),
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
            self.sum_leakages[i] += trace[i].into();
            self.sum_squares_leakages[i] += trace[i].into() * trace[i].into();

            self.plaintext_sum_leakages[[partition, i]] += trace[i].into();
        }

        for guess in 0..self.guess_range {
            let value = (self.leakage_func)(plaintext[self.target_byte].into(), guess);
            self.guess_sum_leakages[guess] += value;
            self.guess_sum_squares_leakages[guess] += value * value;
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

            let mean_key = self.guess_sum_leakages[guess] as f32 / self.num_traces as f32;
            let mean_squares_key =
                self.guess_sum_squares_leakages[guess] as f32 / self.num_traces as f32;
            let var_key = mean_squares_key - (mean_key * mean_key);

            let guess_corr: Vec<_> = (0..self.num_samples)
                .into_par_iter()
                .map(|u| {
                    let mean_leakages = self.sum_leakages[u] as f32 / self.num_traces as f32;

                    let cov = self.sum_mult(
                        self.plaintext_sum_leakages.slice(s![.., u]),
                        modeled_leakages.view(),
                    );
                    let cov = cov as f32 / self.num_traces as f32 - (mean_key * mean_leakages);

                    let mean_squares_leakages =
                        self.sum_squares_leakages[u] as f32 / self.num_traces as f32;
                    let var_leakages = mean_squares_leakages - (mean_leakages * mean_leakages);
                    f32::abs(cov / f32::sqrt(var_key * var_leakages))
                })
                .collect();

            #[allow(clippy::needless_range_loop)]
            for u in 0..self.num_samples {
                corr[[guess, u]] = guess_corr[u];
            }
        }

        Cpa {
            guess_range: self.guess_range,
            corr,
        }
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
            sum_leakages: self.sum_leakages + rhs.sum_leakages,
            sum_squares_leakages: self.sum_squares_leakages + rhs.sum_squares_leakages,
            guess_sum_leakages: self.guess_sum_leakages + rhs.guess_sum_leakages,
            guess_sum_squares_leakages: self.guess_sum_squares_leakages
                + rhs.guess_sum_squares_leakages,
            plaintext_sum_leakages: self.plaintext_sum_leakages + rhs.plaintext_sum_leakages,
            leakage_func: self.leakage_func,
            num_traces: self.num_traces + rhs.num_traces,
        }
    }
}
