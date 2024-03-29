use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::cpa::Cpa;
use muscat::cpa_normal;
use muscat::leakage::{hw, sbox};
use ndarray::{Array2, ArrayView1, Axis};
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::iter::zip;

pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn cpa_sequential(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let mut cpa = Cpa::new(leakages.shape()[1], 256, 0, leakage_model);

    for i in 0..leakages.shape()[0] {
        cpa.update(
            leakages.row(i).map(|&x| x as usize),
            plaintexts.row(i).map(|&y| y as usize),
        );
    }

    cpa.finalize();

    cpa
}

fn cpa_parallel(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let chunk_size = 500;

    let mut cpa = zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, plaintexts_chunk)| {
        let mut cpa = Cpa::new(leakages.shape()[1], 256, 0, leakage_model);

        for i in 0..leakages_chunk.shape()[0] {
            cpa.update(
                leakages_chunk.row(i).map(|&x| x as usize),
                plaintexts_chunk.row(i).map(|&y| y as usize),
            );
        }

        cpa
    })
    .reduce(
        || Cpa::new(leakages.shape()[1], 256, 0, leakage_model),
        |a: Cpa, b| a + b,
    );

    cpa.finalize();

    cpa
}

pub fn leakage_model_normal(value: ArrayView1<usize>, guess: usize) -> usize {
    hw(sbox((value[1] ^ guess) as u8) as usize)
}

fn cpa_normal_sequential(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> cpa_normal::Cpa {
    let chunk_size = 500;

    let mut cpa = cpa_normal::Cpa::new(leakages.shape()[1], chunk_size, 256, leakage_model_normal);

    for (leakages_chunk, plaintexts_chunk) in zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    ) {
        cpa.update(
            leakages_chunk.map(|&x| x as f32),
            plaintexts_chunk.to_owned(),
        );
    }

    cpa.finalize();

    cpa
}

fn cpa_normal_parallel(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> cpa_normal::Cpa {
    let chunk_size = 500;

    let mut cpa = zip(
        leakages.axis_chunks_iter(Axis(0), chunk_size),
        plaintexts.axis_chunks_iter(Axis(0), chunk_size),
    )
    .par_bridge()
    .map(|(leakages_chunk, plaintexts_chunk)| {
        let mut cpa =
            cpa_normal::Cpa::new(leakages.shape()[1], chunk_size, 256, leakage_model_normal);
        cpa.update(
            leakages_chunk.map(|&x| x as f32),
            plaintexts_chunk.to_owned(),
        );
        cpa
    })
    .reduce(
        || cpa_normal::Cpa::new(leakages.shape()[1], chunk_size, 256, leakage_model_normal),
        |x, y| x + y,
    );

    cpa.finalize();

    cpa
}

fn bench_cpa(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("cpa");

    group.measurement_time(std::time::Duration::from_secs(60));

    for nb_traces in [5000, 10000, 25000].into_iter() {
        let leakages = Array2::random_using((nb_traces, 5000), Uniform::new(-2., 2.), &mut rng);
        let plaintexts = Array2::random_using(
            (nb_traces, 16),
            Uniform::new_inclusive(0u8, 255u8),
            &mut rng,
        );

        group.bench_with_input(
            BenchmarkId::new("cpa_sequential", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| cpa_sequential(leakages, plaintexts)),
        );
        group.bench_with_input(
            BenchmarkId::new("cpa_parallel", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| cpa_parallel(leakages, plaintexts)),
        );
        // For 25000 traces, 60s of measurement_time is too low
        if nb_traces <= 10000 {
            group.bench_with_input(
                BenchmarkId::new("cpa_normal_sequential", nb_traces),
                &(&leakages, &plaintexts),
                |b, (leakages, plaintexts)| b.iter(|| cpa_normal_sequential(leakages, plaintexts)),
            );
        }
        group.bench_with_input(
            BenchmarkId::new("cpa_normal_parallel", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| cpa_normal_parallel(leakages, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpa);
criterion_main!(benches);
