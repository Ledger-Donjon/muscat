use crate::{
    Error, Sample,
    util::{argmax_by, argsort_by, max_per_row},
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::AsPrimitive;
use rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelIterator, ParallelIterator},
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, fs::File, iter::zip, ops::Add, path::Path};

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

/// Compute the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Examples
/// ```
/// use muscat::distinguishers::cpa::cpa;
/// use muscat::leakage_model::aes::sbox;
/// use ndarray::array;
///
/// let traces = array![
///     [77u8, 137, 51, 91],
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
///     [1usize, 2],
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
/// let cpa = cpa(traces.view(), plaintexts.view(), 256, 0, |plaintext, guess| sbox((plaintext ^ guess) as u8) as usize, 2);
/// ```
///
/// # Panics
/// - Panic if `traces.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `batch_size` is 0.
pub fn cpa<T, P, F>(
    traces: ArrayView2<T>,
    plaintexts: ArrayView2<P>,
    guess_range: usize,
    target_byte: usize,
    leakage_model: F,
    batch_size: usize,
) -> Cpa
where
    T: Sample + Copy + Sync,
    <T as Sample>::Container: Send + Sync,
    P: Into<usize> + Copy + Sync,
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
        || CpaProcessor::new(traces.shape()[1], guess_range),
        |mut cpa, (trace_batch, plaintext_batch)| {
            for i in 0..trace_batch.shape()[0] {
                cpa.update(
                    trace_batch.row(i),
                    plaintext_batch.row(i)[target_byte],
                    leakage_model,
                );
            }

            cpa
        },
    )
    .reduce_with(|a, b| a + b)
    .unwrap()
    .finalize(leakage_model)
}

/// A processor that computes the [`Cpa`] of the given traces.
///
/// It implements algorithm 4 from [^1].
///
/// [^1]: <https://eprint.iacr.org/2013/794.pdf>
#[derive(Serialize, Deserialize)]
pub struct CpaProcessor<T>
where
    T: Sample,
{
    /// Number of samples per trace
    num_samples: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    sum_traces: Array1<<T as Sample>::Container>,
    /// Sum of square of traces
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    sum_square_traces: Array1<<T as Sample>::Container>,
    /// Sum of traces per key guess
    guess_sum_traces: Array1<usize>,
    /// Sum of square of traces per key guess
    guess_sum_squares_traces: Array1<usize>,
    /// Sum of traces per plaintext used
    /// See 4.3 in <https://eprint.iacr.org/2013/794.pdf>
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    plaintext_sum_traces: Array2<<T as Sample>::Container>,
    /// Number of traces processed
    num_traces: usize,
}

impl<T> CpaProcessor<T>
where
    T: Sample,
{
    pub fn new(num_samples: usize, guess_range: usize) -> Self {
        Self {
            num_samples,
            guess_range,
            sum_traces: Array1::zeros(num_samples),
            sum_square_traces: Array1::zeros(num_samples),
            guess_sum_traces: Array1::zeros(guess_range),
            guess_sum_squares_traces: Array1::zeros(guess_range),
            plaintext_sum_traces: Array2::zeros((guess_range, num_samples)),
            num_traces: 0,
        }
    }

    /// Determine if two [`CpaProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples && self.guess_range == other.guess_range
    }
}

impl<T> CpaProcessor<T>
where
    T: Sample + Copy,
    <T as Sample>::Container: Sync,
{
    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.num_samples`.
    pub fn update<P, F>(&mut self, trace: ArrayView1<T>, plaintext: P, leakage_model: F)
    where
        P: Into<usize> + Copy,
        F: Fn(usize, usize) -> usize,
    {
        debug_assert_eq!(trace.shape()[0], self.num_samples);

        let plaintext = plaintext.into();
        for i in 0..self.num_samples {
            let t = trace[i].into();

            self.sum_traces[i] += t;
            self.sum_square_traces[i] += t * t;

            self.plaintext_sum_traces[[plaintext, i]] += t;
        }

        for guess in 0..self.guess_range {
            let value = leakage_model(plaintext, guess);
            self.guess_sum_traces[guess] += value;
            self.guess_sum_squares_traces[guess] += value * value;
        }

        self.num_traces += 1;
    }

    /// Finalize the calculation after feeding the overall traces.
    pub fn finalize<F>(&self, leakage_model: F) -> Cpa
    where
        F: Fn(usize, usize) -> usize,
    {
        let mut modeled_leakages = Array1::zeros(self.guess_range);

        let mut corr = Array2::zeros((self.guess_range, self.num_samples));
        for guess in 0..self.guess_range {
            for u in 0..self.guess_range {
                modeled_leakages[u] = leakage_model(u, guess) as f32;
            }

            let mean_key = self.guess_sum_traces[guess] as f32 / self.num_traces as f32;
            let mean_squares_key =
                self.guess_sum_squares_traces[guess] as f32 / self.num_traces as f32;
            let var_key = mean_squares_key - (mean_key * mean_key);

            let guess_corr: Vec<_> = (0..self.num_samples)
                .into_par_iter()
                .map(|u| {
                    let mean_traces = self.sum_traces[u].as_() / self.num_traces as f32;

                    let cov = self
                        .plaintext_sum_traces
                        .column(u)
                        .mapv(|x| x.as_())
                        .dot(&modeled_leakages.view());
                    let cov = cov / self.num_traces as f32 - (mean_key * mean_traces);

                    let mean_squares_traces =
                        self.sum_square_traces[u].as_() / self.num_traces as f32;
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
}

impl<T> CpaProcessor<T>
where
    T: Sample,
    <T as Sample>::Container: Serialize,
{
    /// Save the [`CpaProcessor`] to a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let file = File::create(path)?;
        serde_json::to_writer(file, self)?;

        Ok(())
    }
}

impl<T> CpaProcessor<T>
where
    T: Sample,
    <T as Sample>::Container: for<'de> Deserialize<'de>,
{
    /// Load a [`CpaProcessor`] from a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let p: Self = serde_json::from_reader(file)?;

        Ok(p)
    }
}

impl<T> Add for CpaProcessor<T>
where
    T: Sample + Copy,
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
            guess_range: self.guess_range,
            sum_traces: self.sum_traces + rhs.sum_traces,
            sum_square_traces: self.sum_square_traces + rhs.sum_square_traces,
            guess_sum_traces: self.guess_sum_traces + rhs.guess_sum_traces,
            guess_sum_squares_traces: self.guess_sum_squares_traces + rhs.guess_sum_squares_traces,
            plaintext_sum_traces: self.plaintext_sum_traces + rhs.plaintext_sum_traces,
            num_traces: self.num_traces + rhs.num_traces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CpaProcessor, cpa};
    use ndarray::array;
    use serde::Deserialize;

    #[test]
    fn test_cpa_helper() {
        let traces = array![
            [77u8, 137, 51, 91],
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
        let plaintexts = array![[1usize], [3], [1], [2], [3], [2], [2], [1], [3], [1]];

        let leakage_model = |plaintext, guess| plaintext ^ guess;
        let mut processor = CpaProcessor::new(traces.shape()[1], 256);
        for i in 0..traces.shape()[0] {
            processor.update(traces.row(i), plaintexts.row(i)[0], leakage_model);
        }
        assert_eq!(
            processor.finalize(leakage_model).corr(),
            cpa(traces.view(), plaintexts.view(), 256, 0, leakage_model, 2).corr()
        );
    }

    #[test]
    fn test_serialize_deserialize_processor() {
        let traces = array![
            [77u8, 137, 51, 91],
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
        let plaintexts = array![[1usize], [3], [1], [2], [3], [2], [2], [1], [3], [1]];

        let leakage_model = |value, guess| value ^ guess;
        let mut processor = CpaProcessor::new(traces.shape()[1], 256);
        for i in 0..traces.shape()[0] {
            processor.update(traces.row(i), plaintexts.row(i)[0], leakage_model);
        }

        let serialized = serde_json::to_string(&processor).unwrap();
        let mut deserializer = serde_json::Deserializer::from_str(serialized.as_str());
        let restored_processor: CpaProcessor<u8> =
            CpaProcessor::deserialize(&mut deserializer).unwrap();

        assert_eq!(processor.num_samples, restored_processor.num_samples);
        assert_eq!(processor.guess_range, restored_processor.guess_range);
        assert_eq!(processor.sum_traces, restored_processor.sum_traces);
        assert_eq!(
            processor.sum_square_traces,
            restored_processor.sum_square_traces
        );
        assert_eq!(
            processor.guess_sum_traces,
            restored_processor.guess_sum_traces
        );
        assert_eq!(
            processor.guess_sum_squares_traces,
            restored_processor.guess_sum_squares_traces
        );
        assert_eq!(
            processor.plaintext_sum_traces,
            restored_processor.plaintext_sum_traces
        );
        assert_eq!(processor.num_traces, restored_processor.num_traces);
    }
}
