use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use muscat::dpa::{dpa, Dpa, DpaProcessor};
use muscat::leakage::sbox;
use ndarray::{Array1, Array2};
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn selection_function(metadata: Array1<u8>, guess: usize) -> usize {
    sbox(metadata[1] ^ guess as u8).into()
}

fn dpa_sequential(leakages: &Array2<f32>, plaintexts: &Array2<u8>) -> Dpa {
    let mut dpa = DpaProcessor::new(leakages.shape()[1], 256, selection_function);

    for i in 0..leakages.shape()[0] {
        dpa.update(leakages.row(i), plaintexts.row(i).to_owned());
    }

    dpa.finalize()
}

fn dpa_parallel(leakages: &Array2<f32>, plaintexts: &Array2<u8>) -> Dpa {
    dpa(
        leakages.view(),
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

    for nb_traces in [1000, 2000, 5000].into_iter() {
        let leakages = Array2::random_using((nb_traces, 5000), Uniform::new(-2., 2.), &mut rng);
        let plaintexts =
            Array2::random_using((nb_traces, 16), Uniform::new_inclusive(0, 255), &mut rng);

        group.bench_with_input(
            BenchmarkId::new("sequential", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| dpa_sequential(leakages, plaintexts)),
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", nb_traces),
            &(&leakages, &plaintexts),
            |b, (leakages, plaintexts)| b.iter(|| dpa_parallel(leakages, plaintexts)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dpa);
criterion_main!(benches);
