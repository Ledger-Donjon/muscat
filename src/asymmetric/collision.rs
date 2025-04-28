use std::ops::Add;

use ndarray::Array2;

pub struct Collision{
    sumx: Array2<f32>,
    sumx2: Array2<f32>,
    sumy: Array2<f32>,
    sumy2: Array2<f32>,
    sumxy: Array2<f32>,
    count: usize
}

impl Collision {
    pub fn new(size_0: usize, size_1: usize)->Self{
        Collision { sumx: Array2::zeros((size_1, size_0)),
                    sumx2: Array2::zeros((size_1, size_0)),
                    sumy: Array2::zeros((size_1, size_0)),
                    sumy2: Array2::zeros((size_1, size_0)),
                    sumxy: Array2::zeros((size_1, size_0)),
                    count: 0}
    }

    pub fn update<T: Copy + Into<f32> >(&mut self, pattern_0: Array2<T>, pattern_1: Array2<T> ){
        let x: Array2<f32> = pattern_0.map(|w: &T| (*w).into());
        let y: Array2<f32> = pattern_1.map(|v: &T| (*v).into());
        for index_x in 0..x.shape()[1]{
            let tmp_sumx = x.column(index_x).sum();
            let tmp_sumx2: f32 = x.column(index_x).iter().map(|x| (*x) * (*x)).sum();
            // println!("{}", tmp_sumx);
            for index_y in 0..y.shape()[1]{
                self.sumx[[index_y, index_x]] += tmp_sumx;
                self.sumx2[[index_y, index_x]] += tmp_sumx2;
                self.sumy[[index_y, index_x]] += y.column(index_y).sum();
                self.sumy2[[index_y, index_x]] += y.column(index_y).map(|tmp| (*tmp) * (*tmp)).sum();
                self.sumxy[[index_y, index_x]] += x.column(index_x).dot(&y.column(index_y));
            }
        }
        self.count += x.shape()[0];


    }
    
    pub fn finalise(&mut self) -> Array2<f32>{
        let num_0: Array2<f32> = self.sumxy.clone() / self.count as f32;
        let num_1: Array2<f32> = (self.sumx.clone()/self.count as f32) * (self.sumy.clone()/self.count as f32);
        
        let den_0: Array2<f32> = (self.sumx2.clone() / self.count as f32) - ((self.sumx.clone() / self.count as f32).pow2());
        let den_1: Array2<f32> = (self.sumy2.clone() / self.count as f32) - ((self.sumy.clone() / self.count as f32).pow2());
        let dec: Array2<f32> = den_0.sqrt() * den_1.sqrt();
        (num_0 - num_1) / dec
    }
        
    }


impl Add for Collision {
    type Output = Self;
    fn add(self, rhs: Self) -> Self{
        Self { sumx: self.sumx + rhs.sumx,
             sumx2: self.sumx2 + rhs.sumx2,
            sumy: self.sumy + rhs.sumy,
            sumy2: self.sumy2 + rhs.sumy2,
            sumxy: self.sumxy + rhs.sumxy,
            count: self.count + rhs.count }
    }
    
}

    
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn run_my_class() {
            use ndarray::array;
            let pattern_0: Array2<f32> = array![[4.0, 19.0, 6.0], [17.0, 8.0, 16.0], [8.0, 1.0, 12.0]];
            let pattern_1: Array2<f32> = array![[7.0, 2.0, 10.0,], [15.0, 9.0, 10.0], [3.0, 9.0, 11.0]];
            let mut c = Collision::new(pattern_0.shape()[1], pattern_1.shape()[1]);
            c.update(pattern_0, pattern_1);
            println!("{}", c.finalise());
        }
    }





