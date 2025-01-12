use std::fmt::LowerExp;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let x_shape = x.shape();
    let n = x_shape[x_shape.len() - 1];

    assert!(x.shape() == y.shape());
    assert!(w.shape()[0] == n);

    // 提前获取数据切片
    let x_data = x.data();
    let w_data = w.data();
    let mut y_data = unsafe { y.data_mut() };

    // 分块处理
    for chunk in x_data.chunks(n) {
        // 计算分母
        let denominator: f32 =
            (chunk.iter().map(|&x| x * x).sum::<f32>() / (n as f32) + epsilon).sqrt();

        // 计算归一化后的值
        for (i, &x_val) in chunk.iter().enumerate() {
            y_data[i] = w_data[i] * x_val / denominator;
        }

        // 更新 y_data 的偏移量
        y_data = &mut y_data[n..];
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    _y.iter_mut()
        .zip(_x)
        .for_each(|(yi, xi)| (*yi) *= (*xi) / (1.0 + (-*xi).exp()));
}

fn matmul_transb_2d(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // a: [m, k]
    // b: [n, k], 这里的 b 需要转置后才与 a 相乘，这样数据存储的目的是为了空间局部性
    // c: [m, n]
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape().clone();

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    assert!(a_shape[1] == b_shape[1] || a_shape[1] == 1 || b_shape[1] == 1);
    assert!(c_shape[0] == a_shape[0] && c_shape[1] == b_shape[0]);

    if a_shape[1] == b_shape[1] {
        // 无广播
        for i in 0..c_shape[0] {
            let slice_a = &a_data[i * a_shape[1]..(i + 1) * a_shape[1]]; // A 的第 i 行
            for j in 0..c_shape[1] {
                let slice_b = &b_data[j * a_shape[1]..(j + 1) * a_shape[1]]; // B 的第 j 行（实际上是 B^T 的第 j 列）

                // 计算点积
                let dot_product: f32 = slice_a
                    .iter()
                    .zip(slice_b.iter())
                    .map(|(&x, &y)| x * y)
                    .sum();

                // 更新 C 的值
                c_data[i * c_shape[1] + j] =
                    alpha * dot_product + beta * c_data[i * c_shape[1] + j];
            }
        }
    } else {
        if a_shape[1] == 1 {
            // 需要对 a 做广播
            for i in 0..c_shape[0] {
                let val_a = a_data[i]; // A 的第 i 行元素
                for j in 0..c_shape[1] {
                    let slice_b = &b_data[j * b_shape[1]..(j + 1) * b_shape[1]]; // B 的第 j 行（实际上是 B^T 的第 j 列）

                    // 计算点积
                    let dot_product: f32 = slice_b.iter().map(|&x| val_a * x).sum::<f32>();
                    c_data[i * c_shape[1] + j] =
                        alpha * dot_product + beta * c_data[i * c_shape[1] + j];
                }
            }
        } else {
            // 需要对 b 做广播
            for i in 0..c_shape[0] {
                let slice_a = &a_data[i * a_shape[1]..(i + 1) * a_shape[1]]; // A 的第 i 行
                for j in 0..c_shape[1] {
                    let val_b = b_data[j]; // B 的第 j 行元素

                    // 计算点积
                    let dot_product: f32 = slice_a.iter().map(|&x| val_b * x).sum::<f32>();
                    c_data[i * c_shape[1] + j] =
                        alpha * dot_product + beta * c_data[i * c_shape[1] + j];
                }
            }
        }
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // a: [..., m, k]
    // b: [..., n, k]
    // c: [..., m, n]
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();
    assert!(a_shape.len() == b_shape.len() && a_shape.len() == c_shape.len());

    if a_shape.len() == 2 {
        // 二维矩阵
        matmul_transb_2d(c, beta, a, b, alpha);
    } else {
        // 高维矩阵
        // 只需要计算当前最高维，低纬进行递归处理
        let a_sub_shape = &a_shape[1..];
        let b_sub_shape = &b_shape[1..];
        let c_sub_shape = &c_shape[1..];
        let a_diff: usize = a_sub_shape.iter().product();
        let b_diff: usize = b_sub_shape.iter().product();
        let c_diff: usize = c_sub_shape.iter().product();

        if a_shape[0] == b_shape[0] {
            // 不需要广播
            for i in 0..a_shape[0] {
                let a_slice = a.slice(a_diff * i, &a_sub_shape.to_vec());
                let b_slice = b.slice(b_diff * i, &b_sub_shape.to_vec());
                let mut c_slice = c.slice(c_diff * i, &c_sub_shape.to_vec());
                matmul_transb(&mut c_slice, beta, &a_slice, &b_slice, alpha);
            }
        } else {
            // 需要广播
            if a_shape[0] == 1 {
                // 需要对 a 做广播
                let a_slice = a.slice(0, &a_sub_shape.to_vec());
                for i in 0..b_shape[0] {
                    let b_slice = b.slice(b_diff * i, &b_sub_shape.to_vec());
                    let mut c_slice = c.slice(c_diff * i, &c_sub_shape.to_vec());
                    matmul_transb(&mut c_slice, beta, &a_slice, &b_slice, alpha);
                }
            } else {
                // 需要对 b 做广播
                let b_slice = b.slice(0, &b_sub_shape.to_vec());
                for i in 0..a_shape[0] {
                    let a_slice = a.slice(a_diff * i, &a_sub_shape.to_vec());
                    let mut c_slice = c.slice(c_diff * i, &c_sub_shape.to_vec());
                    matmul_transb(&mut c_slice, beta, &a_slice, &b_slice, alpha);
                }
            }
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_high_dim() {
    // 高维矩阵乘法
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], &vec![2, 2, 2]);
    let a = Tensor::<f32>::new(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &vec![2, 2, 3],
    );
    let b = Tensor::<f32>::new(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &vec![2, 2, 3],
    );
    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(
            vec![14., 32., 32., 77., 194., 266., 266., 365.],
            &vec![2, 2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_broadcast_first_dim() {
    // 广播第一个维度
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0., 0., 0., 0., 0.], &vec![2, 2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![1, 2, 2]); // [[[1, 2], [3, 4]]]
    let b = Tensor::<f32>::new(vec![5., 6., 7., 8., 9., 10., 11., 12.], &vec![2, 2, 2]); // [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![17., 23., 39., 53., 29., 35., 67., 81.], &vec![2, 2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_broadcast_middle_dim() {
    // 广播中间维度
    let mut c = Tensor::<f32>::new(
        vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
        &vec![2, 2, 2, 2],
    );
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], &vec![2, 1, 2, 2]);
    let b = Tensor::<f32>::new(
        vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ],
        &vec![2, 2, 2, 2],
    );
    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(
            vec![
                5., 11., 11., 25., 17., 23., 39., 53., 105., 127., 143., 173., 149., 171., 203.,
                233.
            ],
            &vec![2, 2, 2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_broadcast_last_dim() {
    // 广播最后一个维度
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0., 0., 0., 0., 0.], &vec![2, 2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2, 1]); // [[[1], [2]], [[3], [4]]]
    let b = Tensor::<f32>::new(vec![5., 6., 7., 8., 9., 10., 11., 12.], &vec![2, 2, 2]); // [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![11., 15., 22., 30., 57., 69., 76., 92.], &vec![2, 2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_broadcast_all_dims() {
    // 广播所有维度
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0.], &vec![2, 1, 2]);
    let a = Tensor::<f32>::new(vec![1., 2.], &vec![1, 1, 2]); // [[[1, 2]]]
    let b = Tensor::<f32>::new(vec![5., 6., 7., 8., 9., 10., 11., 12.], &vec![2, 2, 2]); // [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![17., 23., 29., 35.], &vec![2, 1, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_edge_case() {
    // 边界情况：某些维度为 1
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0.], &vec![2, 1, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 1, 2]); // [[[1, 2]], [[3, 4]]]
    let b = Tensor::<f32>::new(vec![5., 6., 7., 8.], &vec![1, 2, 2]); // [[[5, 6], [7, 8]]]
    matmul_transb(&mut c, 0., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![17., 23., 39., 53.], &vec![2, 1, 2]),
        1e-3
    ));
}
