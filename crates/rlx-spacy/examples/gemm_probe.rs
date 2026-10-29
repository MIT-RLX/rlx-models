//! Which accumulation order reproduces BLIS's sgemm bits? (probe_*.f32 from dump_gemm.py)
fn f32s(p: &str) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
fn main() {
    let d = "/Users/Shared/tribev2-ref/spacy";
    let (x, w, y) = (
        f32s(&format!("{d}/probe_X.f32")),
        f32s(&format!("{d}/probe_W.f32")),
        f32s(&format!("{d}/probe_Y.f32")),
    );
    let (m, n, k) = (10usize, 288usize, 672usize);
    let variants: Vec<(&str, Box<dyn Fn(&[f32], &[f32]) -> f32>)> = vec![
        (
            "blis sup 4x4 tree",
            Box::new(|a: &[f32], b: &[f32]| {
                let n = a.len();
                let main = n - n % 16;
                let mut v = [[0f32; 4]; 4];
                let mut k = 0;
                while k < main {
                    for r in 0..4 {
                        for l in 0..4 {
                            v[r][l] = a[k + 4 * r + l].mul_add(b[k + 4 * r + l], v[r][l]);
                        }
                    }
                    k += 16;
                }
                let t: Vec<f32> = (0..4)
                    .map(|l| (v[1][l] + v[0][l]) + (v[2][l] + v[3][l]))
                    .collect();
                let mut s = (t[0] + t[1]) + (t[2] + t[3]);
                for i in main..n {
                    s = a[i].mul_add(b[i], s);
                }
                s
            }),
        ),
        (
            "seq fma",
            Box::new(|a: &[f32], b: &[f32]| {
                a.iter().zip(b).fold(0f32, |s, (p, q)| p.mul_add(*q, s))
            }),
        ),
        (
            "seq plain",
            Box::new(|a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |s, (p, q)| s + p * q)),
        ),
        (
            "4 lanes fma",
            Box::new(|a: &[f32], b: &[f32]| {
                let mut l = [0f32; 4];
                for i in 0..a.len() {
                    l[i % 4] = a[i].mul_add(b[i], l[i % 4]);
                }
                (l[0] + l[1]) + (l[2] + l[3])
            }),
        ),
        (
            "4 lanes fma seq-sum",
            Box::new(|a: &[f32], b: &[f32]| {
                let mut l = [0f32; 4];
                for i in 0..a.len() {
                    l[i % 4] = a[i].mul_add(b[i], l[i % 4]);
                }
                ((l[0] + l[1]) + l[2]) + l[3]
            }),
        ),
        (
            "8 lanes fma",
            Box::new(|a: &[f32], b: &[f32]| {
                let mut l = [0f32; 8];
                for i in 0..a.len() {
                    l[i % 8] = a[i].mul_add(b[i], l[i % 8]);
                }
                ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
            }),
        ),
    ];
    for kc in [k, 256, 240, 320, 336, 384, 512, 640] {
        for (name, f) in &variants {
            let mut same = 0;
            for i in 0..m {
                for j in 0..n {
                    let (a, b) = (&x[i * k..(i + 1) * k], &w[j * k..(j + 1) * k]);
                    let mut s = 0f32;
                    let mut at = 0;
                    while at < k {
                        let e = (at + kc).min(k);
                        let part = f(&a[at..e], &b[at..e]);
                        s = if at == 0 { part } else { s + part };
                        at = e;
                    }
                    same += (s.to_bits() == y[i * n + j].to_bits()) as usize;
                }
            }
            if true {
                println!("kc {kc:4} {name:22}: {same}/{} exact", m * n);
            }
        }
    }
}
