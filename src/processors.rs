use std::ops::Add;

use ndarray::{Array1, Array2, s, ArrayView1};

/// Processes traces to calculate mean and variance
#[derive(Clone)]
pub struct MeanVar {
    /// Sum of traces
    acc_1: Array1<i64>,
    /// Sum of square of traces
    acc_2: Array1<i64>,
    /// Number of traces processed
    count: usize
}

impl MeanVar {
    /// Creates a new mean and variance processor
    /// 
    /// # Arguments
    /// 
    /// * `new` - Number of samples per trace
    pub fn new(size: usize) -> Self {
        Self {
            acc_1: Array1::zeros(size),
            acc_2: Array1::zeros(size),
            count: 0
        }
    }

    /// Processes an input trace to update internal accumulators
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: &ArrayView1<T>) {
        let size = self.acc_1.len();
        for i in 0..size {
            let x = trace[i].into();
            self.acc_1[i] += x;
            self.acc_2[i] += x * x;
        }
        self.count += 1
    }

    /// Returns trace mean
    pub fn mean(&self) -> Array1<f64> {
        self.acc_1.map(|&x| x as f64 / self.count as f64)
    }

    /// Calculates and returns traces variance
    pub fn var(&self) -> Array1<f64> {
        self.acc_1.iter().zip(self.acc_2.iter()).map(|(&acc_1, &acc_2)| {
            let acc_1 = acc_1 as f64;
            let acc_2 = acc_2 as f64;
            let count = self.count as f64;
            (acc_2 / count) - (acc_1 / count).powi(2)
        }).collect()
    }
}

impl Add for MeanVar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            acc_1: self.acc_1 + rhs.acc_1,
            acc_2: self.acc_2 + rhs.acc_2,
            count: self.count + rhs.count
        }
    }
}

/// Processes traces to calculate the Signal-to-Noise Ratio
#[derive(Clone)]
pub struct Snr {
    mean_var: MeanVar,
    part_acc_1: Array2<i64>,
    part_acc_2: Array2<i64>,
    counters: Array1<usize>
}

impl Snr {
    /// Creates a new SNR processor
    /// 
    /// # Arguments
    /// 
    /// * `size` - Size of the input traces
    pub fn new(size: usize, classes: usize) -> Self {
        Self {
            mean_var: MeanVar::new(size),
            part_acc_1: Array2::zeros((classes, size)),
            part_acc_2: Array2::zeros((classes, size)),
            counters: Array1::zeros(classes)
        }
    }

    /// Processes an input trace to update internal accumulators
    pub fn process<T: Into<i64> + Copy>(&mut self, trace: &ArrayView1<T>, class: usize) {
        self.mean_var.process(trace);
        let size = self.part_acc_1.shape()[1];
        self.counters[class] += 1;
        for i in 0..size {
            self.part_acc_1[[class, i]] += trace[i].into();
            self.part_acc_2[[class, i]] += (trace[i].into()).pow(2);
        }
    }

    /// Returns SNR of the trace
    pub fn snr(&self) -> Array1<f64> {
        let size = self.part_acc_1.shape()[1];
        let classes = self.part_acc_1.shape()[0];
        let mut acc: Array1<f64> = Array1::zeros(size);
        for class in 0..classes {
            let acc_1 = self.part_acc_1.slice(s![class, ..]);
            let count = self.counters[class] as f64;
            if self.counters[class] > 0 {
                for j in 0..size {
                    acc[j] += (acc_1[j] as f64).powf(2.0) / count;
                }
            }
        }
        let var = self.mean_var.var();
        let mean = self.mean_var.mean();
        // V[E[L|X]]
        let velx = (acc / self.mean_var.count as f64) - mean.mapv(|x| x.powf(2.0));
        1f64 / (var / velx - 1f64)
    }
}

impl Add for Snr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            mean_var: self.mean_var + rhs.mean_var,
            part_acc_1: self.part_acc_1 + rhs.part_acc_1,
            part_acc_2: self.part_acc_2 + rhs.part_acc_2,
            counters: self.counters + rhs.counters
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use super::MeanVar;

    #[test]
    fn test_mean_var() {
        let mut processor = MeanVar::new(4);
        processor.process(&array![28038i16, 22066i16, -20614i16, -9763i16].view());
        assert_eq!(processor.mean(), array![28038f64, 22066f64, -20614f64, -9763f64]);
        assert_eq!(processor.var(), array![0f64, 0f64, 0f64, 0f64]);
        processor.process(&array![31377, -6950, -15666, 26773].view());
        processor.process(&array![24737, -18311, 24742, 17207].view());
        processor.process(&array![12974, -29255, -28798, 18988].view());
        assert_eq!(processor.mean(), array![24281.5f64, -8112.5f64, -10084f64, 13301.25f64]);
        assert_eq!(processor.var(), array![48131112.25, 365776994.25, 426275924.0, 190260421.1875]);
    }
}