use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::leakage::{hw, sbox};
use muscat::processors::Snr;
use ndarray::Array2;
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn leakage_model(value: usize, guess: usize) -> usize {
    hw(sbox((value ^ guess) as u8) as usize)
}

fn compute_snr(leakages: &Array2<i64>, plaintexts: &Array2<u8>) -> Snr {
    let mut snr = Snr::new(leakages.shape()[1], 256);

    for i in 0..leakages.shape()[0] {
        snr.process(&leakages.row(i), plaintexts.row(i)[0] as usize);
    }

    snr
}

fn bench_snr(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("snr");

    group.measurement_time(std::time::Duration::from_secs(60));

    for nb_traces in [5000, 10000, 25000].into_iter() {
        let leakages = Array2::random_using((nb_traces, 5000), Uniform::new(-200, 200), &mut rng);
        let plaintexts =
            Array2::random_using((nb_traces, 16), Uniform::new_inclusive(0, 255), &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| compute_snr(leakages, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_snr);
criterion_main!(benches);
