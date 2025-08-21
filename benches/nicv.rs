use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use muscat::leakage_detection::{NicvProcessor, nicv};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::Uniform;

fn nicv_sequential(traces: &Array2<i64>, plaintexts: &Array2<u8>) -> Array1<f32> {
    let mut nicv = NicvProcessor::new(traces.shape()[1], 256);

    for i in 0..traces.shape()[0] {
        nicv.process(traces.row(i), plaintexts.row(i)[0] as usize);
    }

    nicv.nicv()
}

fn nicv_parallel(traces: &Array2<i64>, plaintexts: &Array2<u8>) -> Array1<f32> {
    nicv(traces.view(), 256, |i| plaintexts.row(i)[0].into(), 500)
}

fn bench_nicv(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("nicv");

    group.measurement_time(std::time::Duration::from_secs(60));

    for num_traces in [5000, 10000, 25000].into_iter() {
        let traces = Array2::random_using((num_traces, 5000), Uniform::new(-200, 200), &mut rng);
        let plaintexts =
            Array2::random_using((num_traces, 16), Uniform::new_inclusive(0, 255), &mut rng);

        group.bench_with_input(
            BenchmarkId::new("sequential", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| b.iter(|| nicv_sequential(traces, plaintexts)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| b.iter(|| nicv_parallel(traces, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_nicv);
criterion_main!(benches);
