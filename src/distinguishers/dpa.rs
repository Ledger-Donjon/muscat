use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::AsPrimitive;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fs::File, iter::zip, marker::PhantomData, ops::Add, path::Path};

use crate::{
    util::{argmax_by, argsort_by, max_per_row},
    Error, Sample,
};

/// Compute the [`Dpa`] of the given traces using [`DpaProcessor`].
///
/// # Examples
/// ```
/// use muscat::distinguishers::dpa::dpa;
/// use muscat::leakage_model::aes::sbox;
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
///     |plaintext: Array1<u8>, guess| sbox(plaintext[0] ^ guess as u8) & 1 == 1,
///     2
/// );
/// ```
///
/// # Panics
/// Panic if `batch_size` is not strictly positive.
pub fn dpa<T, M, F>(
    traces: ArrayView2<T>,
    metadata: ArrayView1<M>,
    guess_range: usize,
    selection_function: F,
    batch_size: usize,
) -> Dpa
where
    T: Sample + Copy + Sync,
    <T as Sample>::Container: Send,
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
        || DpaProcessor::new(traces.shape()[1], guess_range),
        |mut dpa, (trace_batch, metadata_batch)| {
            for i in 0..trace_batch.shape()[0] {
                dpa.update(
                    trace_batch.row(i),
                    metadata_batch[i].clone(),
                    selection_function,
                );
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
#[derive(Serialize, Deserialize)]
pub struct DpaProcessor<T, M>
where
    T: Sample,
{
    /// Number of samples per trace
    num_samples: usize,
    /// Guess range upper excluded bound
    guess_range: usize,
    /// Sum of traces for which the selection function equals false
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    sum_0: Array2<<T as Sample>::Container>,
    /// Sum of traces for which the selection function equals true
    #[serde(bound(serialize = "<T as Sample>::Container: Serialize"))]
    #[serde(bound(deserialize = "<T as Sample>::Container: Deserialize<'de>"))]
    sum_1: Array2<<T as Sample>::Container>,
    /// Number of traces processed for which the selection function equals false
    count_0: Array1<usize>,
    /// Number of traces processed for which the selection function equals true
    count_1: Array1<usize>,
    /// Number of traces processed
    num_traces: usize,
    _metadata: PhantomData<M>,
}

impl<T, M> DpaProcessor<T, M>
where
    T: Sample + Copy,
    M: Clone,
{
    pub fn new(num_samples: usize, guess_range: usize) -> Self {
        Self {
            num_samples,
            guess_range,
            sum_0: Array2::zeros((guess_range, num_samples)),
            sum_1: Array2::zeros((guess_range, num_samples)),
            count_0: Array1::zeros(guess_range),
            count_1: Array1::zeros(guess_range),
            num_traces: 0,
            _metadata: PhantomData,
        }
    }

    /// # Panics
    /// Panic in debug if `trace.shape()[0] != self.num_samples`.
    pub fn update<F>(&mut self, trace: ArrayView1<T>, metadata: M, selection_function: F)
    where
        F: Fn(M, usize) -> bool,
    {
        debug_assert_eq!(trace.shape()[0], self.num_samples);

        for guess in 0..self.guess_range {
            if selection_function(metadata.clone(), guess) {
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
                let mean_0 = self.sum_0[[guess, i]].as_() / self.count_0[guess] as f32;
                let mean_1 = self.sum_1[[guess, i]].as_() / self.count_1[guess] as f32;

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
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_samples == other.num_samples && self.guess_range == other.guess_range
    }
}

impl<T, M> DpaProcessor<T, M>
where
    T: Sample,
    <T as Sample>::Container: Serialize,
{
    /// Save the [`DpaProcessor`] to a file.
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

impl<T, M> DpaProcessor<T, M>
where
    T: Sample,
    <T as Sample>::Container: for<'de> Deserialize<'de>,
{
    /// Load a [`DpaProcessor`] from a file.
    ///
    /// # Warning
    /// The file format is not stable as muscat is active development. Thus, the format might
    /// change between versions.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let p: DpaProcessor<T, M> = serde_json::from_reader(file)?;

        Ok(p)
    }
}

impl<T, M> Add for DpaProcessor<T, M>
where
    T: Sample + Copy,
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
            num_traces: self.num_traces + rhs.num_traces,
            _metadata: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{dpa, DpaProcessor};
    use ndarray::{array, Array1, ArrayView1};
    use serde::Deserialize;

    #[test]
    fn test_dpa_helper() {
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
        let plaintexts = array![[1], [3], [1], [2], [3], [2], [2], [1], [3], [1]];

        let selection_function =
            |plaintext: ArrayView1<u8>, guess| (plaintext[0] as usize ^ guess) & 1 == 1;
        let mut processor = DpaProcessor::new(traces.shape()[1], 256);
        for i in 0..traces.shape()[0] {
            processor.update(
                traces.row(i).map(|&x| x as f32).view(),
                plaintexts.row(i),
                selection_function,
            );
        }
        assert_eq!(
            processor.finalize().differential_curves(),
            dpa(
                traces.view().map(|&x| x as f32).view(),
                plaintexts
                    .rows()
                    .into_iter()
                    .collect::<Array1<ArrayView1<u8>>>()
                    .view(),
                256,
                selection_function,
                2
            )
            .differential_curves()
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
        let plaintexts = array![[1], [3], [1], [2], [3], [2], [2], [1], [3], [1]];

        let selection_function =
            |plaintext: ArrayView1<u8>, guess| (plaintext[0] as usize ^ guess) & 1 == 1;
        let mut processor = DpaProcessor::new(traces.shape()[1], 256);
        for i in 0..traces.shape()[0] {
            processor.update(
                traces.row(i).map(|&x| x as f32).view(),
                plaintexts.row(i),
                selection_function,
            );
        }

        let serialized = serde_json::to_string(&processor).unwrap();
        let mut deserializer = serde_json::Deserializer::from_str(serialized.as_str());
        let restored_processor =
            DpaProcessor::<f32, ArrayView1<u8>>::deserialize(&mut deserializer).unwrap();

        assert_eq!(processor.num_samples, restored_processor.num_samples);
        assert_eq!(processor.guess_range, restored_processor.guess_range);
        assert_eq!(processor.sum_0, restored_processor.sum_0);
        assert_eq!(processor.sum_1, restored_processor.sum_1);
        assert_eq!(processor.count_0, restored_processor.count_0);
        assert_eq!(processor.count_1, restored_processor.count_1);
        assert_eq!(processor.num_traces, restored_processor.num_traces);
    }
}
