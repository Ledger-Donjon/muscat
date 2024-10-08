use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::leakage_detection::{snr, Snr};
use ndarray::{Array1, Array2};
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn snr_sequential(leakages: &Array2<i64>, plaintexts: &Array2<u8>) -> Array1<f64> {
    let mut snr = Snr::new(leakages.shape()[1], 256);

    for i in 0..leakages.shape()[0] {
        snr.process(leakages.row(i), plaintexts.row(i)[0] as usize);
    }

    snr.snr()
}

fn snr_parallel(leakages: &Array2<i64>, plaintexts: &Array2<u8>) -> Array1<f64> {
    snr(leakages.view(), 256, |i| plaintexts.row(i)[0].into(), 500)
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
            BenchmarkId::new("sequential", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| snr_sequential(leakages, plaintexts)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| snr_parallel(leakages, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_snr);
criterion_main!(benches);
