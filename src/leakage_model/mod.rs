pub mod aes;

pub fn hw(value: usize) -> usize {
    let mut tmp = 0;
    for i in 0..8 {
        if (value & (1 << i)) == (1 << i) {
            tmp += 1;
        }
    }
    tmp
}
