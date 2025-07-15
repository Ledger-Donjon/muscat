use muscat::asymmetric::collision::Collision;
use ndarray::{Array2, s};
use ndarray_npy::{read_npy, write_npy};
use std::env;
use std::path::PathBuf;

fn main() {
    type TRACESFMT = i16; // traces format
    let traces_dir = PathBuf::from(env::var("PATH").expect("PATH is missed"));
    println!("{}", traces_dir.display());
    let traces: Array2<TRACESFMT> = read_npy(traces_dir.join("data.npy")).unwrap();
    let range_0 = 0..10_usize; // samples of first pattern
    let range_1 = 10..20_usize; // samples of second pattern
    let mut coll_attack = Collision::new(range_0.len(), range_1.len());
    let pattern_0: Array2<TRACESFMT> = traces.slice(s![.., range_0]).to_owned();
    let pattern_1: Array2<TRACESFMT> = traces.slice(s![.., range_1]).to_owned();
    coll_attack.update(pattern_0, pattern_1);
    let result: Array2<f32> = coll_attack.finalise();
    write_npy("data.npy", &result).unwrap(); //save result
}
