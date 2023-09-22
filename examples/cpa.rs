use simple_bar::ProgressBar;
use rayon::prelude::{ParallelIterator, ParallelBridge};
use std::time::Instant;
use ndarray::*;
use muscat::cpa::*;
use muscat::leakage::{hw, sbox};
use muscat::util::{read_array_2_from_npy_file, write_npy};


// traces format
type FormatTraces = i16;
type FormatMetadata = i32;

// leakage model
pub fn leakage_model(value: usize, guess: usize) -> usize{
    hw(sbox[(value ^ guess) as usize] as usize)
}


// multi-threading cpa
fn cpa()
{
    let size: usize = 5000;  // Number of samples 
    let guess_range = 256;  // 2**(key length)
    let target_byte = 1;    
    let folder = String::from("data"); // Directory of leakages and metadata
    let nfiles = 5; // Number of files in the directory. TBD: Automating this value
    let mut bar = ProgressBar::default(nfiles as u32, 50, false);
    
    /* Parallel operation using multi-threading on patches */
    let mut cpa: Cpa = (0..nfiles).into_iter().
    map(|n| {
        bar.update();
        let dir_l: String = format!("{}{}{}{}", folder, "/l", n.to_string(), ".npy" );
        let dir_p = format!("{}{}{}{}", folder, "/p", n.to_string(), ".npy");
        let leakages: Array2<FormatTraces>= read_array_2_from_npy_file::<FormatTraces>(&dir_l);
        let plaintext: Array2<FormatMetadata> = read_array_2_from_npy_file::<FormatMetadata>(&dir_p);
        (leakages, plaintext)
        
    }).into_iter().par_bridge().map(|patch|
    {   
        let mut c: Cpa = Cpa::new(size, guess_range, target_byte, leakage_model);
        let len_leakage = patch.0.shape()[0];       
        for i in 0..len_leakage{
            c.update(
                patch.0.row(i).map(|x| *x as usize), 
                patch.1.row(i).map(|y| *y as usize)
            );      
        }    
        c
    }).reduce(|| Cpa::new(size, guess_range, target_byte, leakage_model), |a: Cpa, b| a+b);
    cpa.finalize();
    println!("Guessed key = {}", cpa.pass_guess());
    // save corr key curves in npy
    write_npy("examples/results/corr.npy", cpa.pass_corr_array().view());
    
}




    
fn main(){
    cpa();
}


