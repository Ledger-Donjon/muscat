use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use muscat::Sample;
use muscat::distinguishers::cpa::{self, Cpa, CpaProcessor};
use muscat::distinguishers::cpa_normal;
use muscat::leakage_model::{aes::sbox, hw};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Uniform;
use num_traits::AsPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::iter::zip;

pub fn leakage_model(plaintext: usize, guess: usize) -> usize {
    hw(sbox((plaintext ^ guess) as u8) as usize)
}

fn cpa_sequential(traces: &Array2<u8>, plaintexts: &Array2<u8>) -> Cpa {
    let mut cpa = CpaProcessor::new(traces.shape()[1], 256);

    for i in 0..traces.shape()[0] {
        cpa.update(traces.row(i), plaintexts.row(i)[0] as usize, leakage_model);
    }

    cpa.finalize(leakage_model)
}

pub fn leakage_model_normal(plaintext: ArrayView1<usize>, guess: usize) -> usize {
    hw(sbox((plaintext[1] ^ guess) as u8) as usize)
}

fn cpa_normal_sequential(traces: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let batch_size = 500;

    let mut cpa = cpa_normal::CpaProcessor::new(traces.shape()[1], batch_size, 256);

    for (trace_batch, plaintext_batch) in zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        plaintexts.axis_chunks_iter(Axis(0), batch_size),
    ) {
        cpa.batch_update(
            trace_batch.map(|&x| x as f32).view(),
            plaintext_batch,
            leakage_model_normal,
        );
    }

    cpa.finalize()
}

fn bench_cpa(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("cpa");

    group.measurement_time(std::time::Duration::from_secs(60));

    for num_traces in [5000, 10000, 25000].into_iter() {
        let traces = Array2::random_using((num_traces, 5000), Uniform::new(-2., 2.), &mut rng);
        let plaintexts = Array2::random_using(
            (num_traces, 16),
            Uniform::new_inclusive(0u8, 255u8),
            &mut rng,
        );

        group.bench_with_input(
            BenchmarkId::new("cpa_sequential", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| {
                b.iter(|| {
                    cpa_sequential(&traces.map(|&x| ((x + 2.) * (256. / 4.)) as u8), plaintexts)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_parallel", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| {
                b.iter(|| {
                    cpa::cpa(
                        traces.map(|&x| ((x + 2.) * (256. / 4.)) as u8).view(),
                        plaintexts.map(|&x| x as usize).view(),
                        256,
                        0,
                        leakage_model,
                        500,
                    )
                })
            },
        );
        // For 25000 traces, 60s of measurement_time is too low
        if num_traces <= 10000 {
            group.bench_with_input(
                BenchmarkId::new("cpa_normal_sequential", num_traces),
                &(&traces, &plaintexts),
                |b, (traces, plaintexts)| b.iter(|| cpa_normal_sequential(traces, plaintexts)),
            );
        }
        group.bench_with_input(
            BenchmarkId::new("cpa_normal_parallel", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| {
                b.iter(|| {
                    cpa_normal::cpa(
                        traces.map(|&x| x as f32).view(),
                        plaintexts.view(),
                        256,
                        leakage_model_normal,
                        500,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_finalize(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("cpa_finalize");

    group.measurement_time(std::time::Duration::from_secs(60));

    for num_traces in [5000, 10000, 25000].into_iter() {
        let traces = Array2::random_using((num_traces, 5000), Uniform::new(-2., 2.), &mut rng);
        let plaintexts = Array2::random_using(
            (num_traces, 16),
            Uniform::new_inclusive(0u8, 255u8),
            &mut rng,
        );

        let mut cpa = CpaProcessorFinalize::new(traces.shape()[1], 256);

        for i in 0..traces.shape()[0] {
            cpa.update(traces.row(i), plaintexts.row(i)[0] as usize, leakage_model);
        }

        group.bench_function(BenchmarkId::new("finalize", num_traces), |b| {
            b.iter(|| cpa.finalize(leakage_model))
        });

        group.bench_function(BenchmarkId::new("finalize_nopar", num_traces), |b| {
            b.iter(|| cpa.finalize_nopar(leakage_model))
        });

        group.bench_function(BenchmarkId::new("finalize_par_guess", num_traces), |b| {
            b.iter(|| cpa.finalize_par_guess(leakage_model))
        });
    }

    group.finish();
}

#[derive(Serialize, Deserialize)]
pub struct CpaProcessorFinalize<T>
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

impl<T> CpaProcessorFinalize<T>
where
    T: Sample + Copy,
    <T as Sample>::Container: Sync,
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
    pub fn finalize<F>(&self, leakage_model: F) -> Array2<f32>
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

        corr
    }

    /// Finalize the calculation after feeding the overall traces.
    pub fn finalize_nopar<F>(&self, leakage_model: F) -> Array2<f32>
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

        corr
    }

    /// Finalize the calculation after feeding the overall traces.
    pub fn finalize_par_guess<F>(&self, leakage_model: F) -> Array2<f32>
    where
        F: Fn(usize, usize) -> usize + Sync,
    {
        let mut corr = Array2::zeros((self.guess_range, self.num_samples));
        corr.axis_iter_mut(Axis(0))
            .enumerate()
            .par_bridge()
            // WARN: Not sure about the mut here
            .for_each(|(guess, mut corr_row)| {
                let mut modeled_leakages = Array1::zeros(self.guess_range);

                for u in 0..self.guess_range {
                    modeled_leakages[u] = leakage_model(u, guess) as f32;
                }

                let mean_key = self.guess_sum_traces[guess] as f32 / self.num_traces as f32;
                let mean_squares_key =
                    self.guess_sum_squares_traces[guess] as f32 / self.num_traces as f32;
                let var_key = mean_squares_key - (mean_key * mean_key);

                let guess_corr: Vec<_> = (0..self.num_samples)
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
                    corr_row[u] = guess_corr[u];
                }
            });

        corr
    }
}

criterion_group!(benches, bench_cpa, bench_finalize);
criterion_main!(benches);
