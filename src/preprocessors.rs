use ndarray::{Array1, ArrayView1};
use std::ops::Range;
use itertools::Itertools;

// Computes the centered product of "order" leakage samples
// Used particularly when performing high-order SCA
struct CenteredProduct{
    /// Sum of traces
    acc: Array1<i64>,
    /// Number of traces processed
    count: usize,    
    /// Mean of traces
    mean: Array1<f64>,
    /// Indices of samples to combine
    intervals: Vec<Range<i32>>,
    /// Boolean to ensure that finalize function happened before apply
    processed: bool,
}

impl CenteredProduct{
    /// Creates a new CenteredProduct processor.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples per trace
    /// * `intervals` - Intervals to combine
    pub fn new(size: usize, intervals: Vec<Range<i32>>) -> Self {
        Self {
            acc: Array1::zeros(size),
            count: 0,
            intervals: intervals,
            processed: false,
            mean: Array1::zeros(size),
        }
    }

    /// Processes an input trace to update internal accumulators.
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: &ArrayView1<T>) {
        let size = self.acc.len();
        for i in 0..size {
            let x = trace[i].into();
            self.acc[i] += x;
        }
        self.count += 1
    }
    
    /// Compute the mean
    pub fn finalize(&mut self){
        if self.count != 0{
            self.mean = self.acc.map(|&x| x as f64 / self.count as f64)
        }
        self.processed = true
    }
    
    /// Apply the processing to an input trace 
    /// The centered product substract the mean of the traces and then perform products between every input time samples
    pub fn apply<T: Into<f64> + Copy>(&mut self, trace: &ArrayView1<T>) -> Array1<f64>{
        // First we substract the mean trace
        let centered_trace:Array1<f64> = trace.mapv(|x| f64::from(x.into())) - &self.mean;
        let length_out_trace:usize = self.intervals.iter().map(|x| x.len()).product();

        let mut centered_product_trace = Array1::ones(length_out_trace);
        
        // Then we do the products
        let mut multi_prod = (0..self.intervals.len()).map(|i| self.intervals[i].clone()).multi_cartesian_product(); //NOTE/TODO: maybe this can go in the struct parameters, which could improve parameters        
    
        for (idx,combination) in multi_prod.enumerate(){
            println!("{:?}",combination);
            for i in combination{
                 centered_product_trace[idx] *= centered_trace[i as usize] as f64;
            }
        }
        println!{"{:?}",centered_product_trace};
        return centered_product_trace;
    }
}

#[cfg(test)]
mod tests {
    use super::CenteredProduct;
    use ndarray::array;

    fn round_to_2_digits(x:f64)->f64{
        return (x * 100 as f64).round() / 100 as f64;
    }

    #[test]
    fn test_centered_product() {
        let mut processor = CenteredProduct::new(5,vec![0..2, 3..5]);
        processor.process(&array![0i16, 1i16, 2i16, -3i16, -4i16].view());
        processor.finalize();
        assert_eq!(
            processor.apply(&array![0i16, 1i16, 2i16, -3i16, -4i16].view()),
            array![0f64,0f64,0f64,0f64]
        );
        let traces = [
            array![77, 137, 51, 91],
            array![72, 61, 91, 83],
            array![39, 49, 52, 23],
            array![26, 114, 63, 45],
            array![30, 8, 97, 91],
            array![13, 68, 7, 45],
            array![17, 181, 60, 34],
            array![43, 88, 76, 78],
            array![0, 36, 35, 0],
            array![93, 191, 49, 26],
        ];

        let mut processor2 = CenteredProduct::new(4,vec![0..1,1..2,2..4]);
        for t in traces.iter(){
            processor2.process(&t.view());
        }
        processor2.finalize();

        let expected_results =[
            array![-11169.72, 61984.08],
            array![-32942.77, -31440.82],
            array![-540.46, -2533.96],
            array![-1521.45, 2049.30],
            array![36499.87, 36969.02],
            array![-36199.24, -4675.44],
            array![-3999.12, 37044.48],
            array![-189.74, -279.84],
            array![-54268.83, -121223.88],
            array![-46231.64, -130058.24],
        ];

        for (i,t) in traces.iter().enumerate(){
            assert_eq!(
                processor2.apply(&t.view()).map(|x| round_to_2_digits(*x)),
                expected_results[i]
            );
        }
    }

}

