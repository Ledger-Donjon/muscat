use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::AsPrimitive;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fs::File, iter::zip, ops::Add, path::Path};

use crate::{Error, Sample, distinguishers::cpa::Cpa};

/// Compute the [`Cpa`] of the given traces using [`CpaProcessor`].
///
/// # Examples
/// ```
/// use muscat::distinguishers::cpa_normal::cpa;
/// use muscat::leakage_model::aes::sbox;
/// use ndarray::array;
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
/// let cpa = cpa(traces.map(|&x| x as f32).view(), plaintexts.view(), 256, |plaintext, guess| sbox((plaintext[0] ^ guess) as u8) as usize, 2);
/// ```
///
/// # Panics
/// - Panic if `traces.shape()[0] != plaintexts.shape()[0]`
/// - Panic if `batch_size` is 0.
pub fn cpa<T, P, F>(
    traces: ArrayView2<T>,
    plaintexts: ArrayView2<P>,
    guess_range: usize,
    leakage_model: F,
    batch_size: usize,
) -> Cpa
where
    T: Sample + Copy + Sync,
    <T as Sample>::Container: Send,
    P: Into<usize> + Copy + Sync,
    F: Fn(ArrayView1<usize>, usize) -> usize + Send + Sync + Copy,
{
    assert_eq!(traces.shape()[0], plaintexts.shape()[0]);
    assert!(batch_size > 0);

    zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        plaintexts.axis_chunks_iter(Axis(0), batch_size),
    )
    .par_bridge()
    .fold(
        || CpaProcessor::new(traces.shape()[1], batch_size, guess_range),
        |mut cpa, (trace_batch, plaintext_batch)| {
            cpa.update(trace_batch, plaintext_batch, leakage_model);

            cpa
        },
    )
    .reduce_with(|x, y| x + y)
    .unwrap()
    .finalize()
}

/// A processor that computes the [`Cpa`] of the given traces.
///
/// [^1]: <https://www.iacr.org/archive/ches2004/31560016/31560016.pdf>
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
    sum_traces2: Array1<<T as Sample>::Container>,
    /// Sum of traces per key guess
    guess_sum_traces: Array1<f32>,
    /// Sum of square of traces per key guess
    guess_sum_traces2: Array1<f32>,
    values: Array2<f32>,
    cov: Array2<f32>,
    /// Batch size
    batch_size: usize,
    /// Number of traces processed
    num_traces: usize,
}

impl<T> CpaProcessor<T>
where
    T: Sample + Copy,
{
    pub fn new(num_samples: usize, batch_size: usize, guess_range: usize) -> Self {
        Self {
            num_samples,
            guess_range,
            sum_traces: Array1::zeros(num_samples),
            sum_traces2: Array1::zeros(num_samples),
            guess_sum_traces: Array1::zeros(guess_range),
            guess_sum_traces2: Array1::zeros(guess_range),
            values: Array2::zeros((batch_size, guess_range)),
            cov: Array2::zeros((guess_range, num_samples)),
            batch_size,
            num_traces: 0,
        }
    }

    /// # Panics
    /// - Panic in debug if `trace_batch.shape()[0] != plaintext_batch.shape()[0]`.
    /// - Panic in debug if `trace_batch.shape()[1] != self.num_samples`.
    pub fn update<P, F>(
        &mut self,
        trace_batch: ArrayView2<T>,
        plaintext_batch: ArrayView2<P>,
        leakage_model: F,
    ) where
        P: Into<usize> + Copy,
        F: Fn(ArrayView1<usize>, usize) -> usize,
    {
        debug_assert_eq!(trace_batch.shape()[0], plaintext_batch.shape()[0]);
        debug_assert_eq!(trace_batch.shape()[1], self.num_samples);

        /* This function updates the internal arrays of the CPA
        It accepts trace_batch and plaintext_batch to update them*/
        let plaintext_batch = plaintext_batch.mapv(|m| m.into());

        // Update values
        for row in 0..self.batch_size {
            for guess in 0..self.guess_range {
                let pass_to_leakage = plaintext_batch.row(row);
                self.values[[row, guess]] = leakage_model(pass_to_leakage, guess) as f32;
            }
        }

        self.cov = self.cov.clone()
            + self
                .values
                .t()
                .dot(&trace_batch.mapv(|x| <T as Sample>::Container::from(x).as_()));

        // Update key leakages
        for i in 0..self.num_samples {
            self.sum_traces[i] += trace_batch
                .column(i)
                .mapv(|x| <T as Sample>::Container::from(x))
                .sum(); // trace[i] as usize;
            self.sum_traces2[i] += trace_batch
                .column(i)
                .mapv(|x| <T as Sample>::Container::from(x))
                .mapv(|x| x * x)
                .sum();
            // (trace[i] * trace[i]) as usize;
        }

        for guess in 0..self.guess_range {
            self.guess_sum_traces[guess] += self.values.column(guess).sum(); //self.values[guess] as usize;
            self.guess_sum_traces2[guess] +=
                self.values.column(guess).dot(&self.values.column(guess));
            // (self.values[guess] * self.values[guess]) as usize;
        }

        self.num_traces += self.batch_size;
    }

    /// Finalize the calculation after feeding the overall traces.
    pub fn finalize(&self) -> Cpa {
        let cov_n = self.cov.clone() / self.num_traces as f32;
        let avg_keys = self.guess_sum_traces.clone() / self.num_traces as f32;
        let std_key = self.guess_sum_traces2.clone() / self.num_traces as f32;
        let avg_traces = self.sum_traces.mapv(|x| x.as_()) / self.num_traces as f32;
        let std_traces = self.sum_traces2.mapv(|x| x.as_()) / self.num_traces as f32;

        let mut corr = Array2::zeros((self.guess_range, self.num_samples));
        for i in 0..self.guess_range {
            for x in 0..self.num_samples {
                let numerator = cov_n[[i, x]] - (avg_keys[i] * avg_traces[x]);

                let denominator_1 = std_key[i] - (avg_keys[i] * avg_keys[i]);

                let denominator_2 = std_traces[x] - (avg_traces[x] * avg_traces[x]);
                if numerator != 0.0 {
                    corr[[i, x]] = f32::abs(numerator / f32::sqrt(denominator_1 * denominator_2));
                }
            }
        }

        Cpa { corr }
    }

    /// Determine if two [`CpaProcessor`] are compatible for addition.
    ///
    /// If they were created with the same parameters, they are compatible.
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples
            && self.batch_size == other.batch_size
            && self.guess_range == other.guess_range
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
        let p: CpaProcessor<T> = serde_json::from_reader(file)?;

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
            sum_traces2: self.sum_traces2 + rhs.sum_traces2,
            guess_sum_traces: self.guess_sum_traces + rhs.guess_sum_traces,
            guess_sum_traces2: self.guess_sum_traces2 + rhs.guess_sum_traces2,
            values: self.values + rhs.values,
            cov: self.cov + rhs.cov,
            batch_size: self.batch_size,
            num_traces: self.num_traces + rhs.num_traces,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use super::{CpaProcessor, cpa};
    use ndarray::{ArrayView1, Axis, array};
    use serde::Deserialize;

    #[test]
    fn test_cpa_helper() {
        let traces = array![
            [77usize, 137, 51, 91],
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

        let leakage_model = |plaintext: ArrayView1<usize>, guess| plaintext[0] ^ guess;
        let mut processor = CpaProcessor::new(traces.shape()[1], 1, 256);
        for (trace, plaintext) in zip(
            traces.axis_chunks_iter(Axis(0), 1),
            plaintexts.axis_chunks_iter(Axis(0), 1),
        ) {
            processor.update(
                trace.map(|&x| x as f32).view(),
                plaintext.view(),
                leakage_model,
            );
        }
        assert_eq!(
            processor.finalize().corr(),
            cpa(
                traces.map(|&x| x as f32).view(),
                plaintexts.view(),
                256,
                leakage_model,
                2
            )
            .corr()
        );
    }

    #[test]
    fn test_serialize_deserialize_processor() {
        let traces = array![
            [77usize, 137, 51, 91],
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

        let leakage_model = |plaintext: ArrayView1<usize>, guess| plaintext[0] ^ guess;
        let mut processor = CpaProcessor::new(traces.shape()[1], 1, 256);
        for (trace, plaintext) in zip(
            traces.axis_chunks_iter(Axis(0), 1),
            plaintexts.axis_chunks_iter(Axis(0), 1),
        ) {
            processor.update(
                trace.map(|&x| x as f32).view(),
                plaintext.view(),
                leakage_model,
            );
        }

        let serialized = serde_json::to_string(&processor).unwrap();
        let mut deserializer: serde_json::Deserializer<serde_json::de::StrRead<'_>> =
            serde_json::Deserializer::from_str(serialized.as_str());
        let restored_processor: CpaProcessor<f32> =
            CpaProcessor::deserialize(&mut deserializer).unwrap();

        assert_eq!(processor.num_samples, restored_processor.num_samples);
        assert_eq!(processor.guess_range, restored_processor.guess_range);
        assert_eq!(processor.sum_traces, restored_processor.sum_traces);
        assert_eq!(processor.sum_traces2, restored_processor.sum_traces2);
        assert_eq!(
            processor.guess_sum_traces,
            restored_processor.guess_sum_traces
        );
        assert_eq!(
            processor.guess_sum_traces2,
            restored_processor.guess_sum_traces2
        );
        assert_eq!(processor.values, restored_processor.values);
        assert_eq!(processor.cov, restored_processor.cov);
        assert_eq!(processor.batch_size, restored_processor.batch_size);
        assert_eq!(processor.num_traces, restored_processor.num_traces);
    }
}
