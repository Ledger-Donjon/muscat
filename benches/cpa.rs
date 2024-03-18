use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::cpa::Cpa;
use muscat::leakage::{hw, sbox};
use ndarray::Array2;
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn compute_cpa(leakages: &Array2<f64>, plaintexts: &Array2<u8>) -> Cpa {
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
            BenchmarkId::from_parameter(nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| compute_cpa(leakages, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cpa);
criterion_main!(benches);
