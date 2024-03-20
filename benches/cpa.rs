use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::cpa::Cpa;
use muscat::leakage::{hw, sbox};
use ndarray::Array2;
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn cpa_sequential(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let mut cpa = Cpa::new(leakages.shape()[1], 256, 0, leakage_model);

    for i in 0..leakages.shape()[0] {
        cpa.update(
            leakages.row(i).map(|x| *x as usize),
            plaintexts.row(i).map(|y| *y as usize),
        );
    }

    cpa.finalize();

    cpa
}

fn cpa_parallel(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
    let mut cpa = (0..8)
        .into_par_iter()
        .map(|chunk_num| {
            let mut cpa = Cpa::new(leakages.shape()[1], 256, 0, leakage_model);

            for i in
                (chunk_num * leakages.shape()[0] / 8)..((chunk_num + 1) * leakages.shape()[0] / 8)
            {
                cpa.update(
                    leakages.row(i).map(|x| *x as usize),
                    plaintexts.row(i).map(|y| *y as usize),
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

fn bench_cpa(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("cpa");

    group.measurement_time(std::time::Duration::from_secs(60));

    for nb_traces in [5000, 10000, 25000].into_iter() {
        let leakages = Array2::random_using((nb_traces, 5000), Uniform::new(-2f64, 2f64), &mut rng);
        let plaintexts = Array2::random_using(
            (nb_traces, 16),
            Uniform::new_inclusive(0u8, 255u8),
            &mut rng,
        );

        group.bench_with_input(
            BenchmarkId::new("sequential", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| cpa_sequential(leakages, plaintexts)),
        );
        group.bench_with_input(
            BenchmarkId::new("parallel", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| cpa_parallel(leakages, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpa);
criterion_main!(benches);
