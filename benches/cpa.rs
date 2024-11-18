use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::distinguishers::cpa::{self, Cpa, CpaProcessor};
use muscat::distinguishers::cpa_normal;
use muscat::leakage_model::{aes::sbox, hw};
use ndarray::{Array2, ArrayView1, Axis};
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::iter::zip;

pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn cpa_sequential(traces: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let mut cpa = CpaProcessor::new(traces.shape()[1], 256, 0, leakage_model);

    for i in 0..traces.shape()[0] {
        cpa.update(
            traces.row(i).map(|&x| x as usize).view(),
            plaintexts.row(i).map(|&y| y as usize).view(),
        );
    }

    cpa.finalize()
}

pub fn leakage_model_normal(value: ArrayView1<usize>, guess: usize) -> usize {
    hw(sbox((value[1] ^ guess) as u8) as usize)
}

fn cpa_normal_sequential(traces: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let batch_size = 500;

    let mut cpa =
        cpa_normal::CpaProcessor::new(traces.shape()[1], batch_size, 256, leakage_model_normal);

    for (trace_batch, plaintext_batch) in zip(
        traces.axis_chunks_iter(Axis(0), batch_size),
        plaintexts.axis_chunks_iter(Axis(0), batch_size),
    ) {
        cpa.update(trace_batch.map(|&x| x as f32).view(), plaintext_batch);
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
            |b, (traces, plaintexts)| b.iter(|| cpa_sequential(traces, plaintexts)),
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_parallel", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| {
                b.iter(|| {
                    cpa::cpa(
                        traces.map(|&x| x as usize).view(),
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

criterion_group!(benches, bench_cpa);
criterion_main!(benches);
