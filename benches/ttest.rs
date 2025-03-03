use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use muscat::leakage_detection::{TTestProcessor, ttest};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::{Standard, Uniform};

fn ttest_sequential(traces: &Array2<i64>, trace_classes: &Array1<bool>) -> Array1<f64> {
    let mut ttest = TTestProcessor::new(traces.shape()[1]);

    for i in 0..traces.shape()[0] {
        ttest.process(traces.row(i), trace_classes[i]);
    }

    ttest.ttest()
}

fn ttest_parallel(traces: &Array2<i64>, trace_classes: &Array1<bool>) -> Array1<f64> {
    ttest(traces.view(), trace_classes.view(), 500)
}

fn bench_ttest(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("ttest");

    group.measurement_time(std::time::Duration::from_secs(60));

    for num_traces in [5000, 10000, 25000].into_iter() {
        let traces = Array2::random_using((num_traces, 5000), Uniform::new(-200, 200), &mut rng);
        let plaintexts = Array1::random_using(num_traces, Standard, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("sequential", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| b.iter(|| ttest_sequential(traces, plaintexts)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| b.iter(|| ttest_parallel(traces, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_ttest);
criterion_main!(benches);
