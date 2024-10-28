use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::distinguishers::dpa::{dpa, Dpa, DpaProcessor};
use muscat::leakage::sbox;
use ndarray::{Array1, Array2};
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn selection_function(metadata: Array1<u8>, guess: usize) -> bool {
    usize::from(sbox(metadata[1] ^ guess as u8)) & 1 == 1
}

fn dpa_sequential(traces: &Array2<f32>, plaintexts: &Array2<u8>) -> Dpa {
    let mut dpa = DpaProcessor::new(traces.shape()[1], 256, selection_function);

    for i in 0..traces.shape()[0] {
        dpa.update(traces.row(i), plaintexts.row(i).to_owned());
    }

    dpa.finalize()
}

fn dpa_parallel(traces: &Array2<f32>, plaintexts: &Array2<u8>) -> Dpa {
    dpa(
        traces.view(),
        plaintexts
            .rows()
            .into_iter()
            .map(|x| x.to_owned())
            .collect::<Array1<Array1<u8>>>()
            .view(),
        256,
        selection_function,
        500,
    )
}

fn bench_dpa(c: &mut Criterion) {
    // Seed rng to get the same output each run
    let mut rng = StdRng::seed_from_u64(0);

    let mut group = c.benchmark_group("dpa");

    group.measurement_time(std::time::Duration::from_secs(60));

    for num_traces in [1000, 2000, 5000].into_iter() {
        let traces = Array2::random_using((num_traces, 5000), Uniform::new(-2., 2.), &mut rng);
        let plaintexts =
            Array2::random_using((num_traces, 16), Uniform::new_inclusive(0, 255), &mut rng);

        group.bench_with_input(
            BenchmarkId::new("sequential", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| b.iter(|| dpa_sequential(traces, plaintexts)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", num_traces),
            &(&traces, &plaintexts),
            |b, (traces, plaintexts)| b.iter(|| dpa_parallel(traces, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dpa);
criterion_main!(benches);
