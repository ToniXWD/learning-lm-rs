use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let data_ptr = tensor.data().as_ptr() as *const f32;
            let data_f32 =
                unsafe { std::slice::from_raw_parts(data_ptr, tensor.shape().iter().product()) }
                    .to_vec();
            let shape: Vec<usize> = tensor.shape().to_vec();
            Tensor::new(data_f32, &shape)
        };

        let layer_num = config.num_hidden_layers as usize;

        let get_tensors = |name: &str| {
            let mut res: Vec<Tensor<f32>> = vec![];
            for i in 0..layer_num {
                let name = format!("model.layers.{i}.{name}");
                res.push(get_tensor(&name));
            }
            res
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_tensors("input_layernorm.weight"),
            wq: get_tensors("self_attn.q_proj.weight"),
            wk: get_tensors("self_attn.k_proj.weight"),
            wv: get_tensors("self_attn.v_proj.weight"),
            wo: get_tensors("self_attn.o_proj.weight"),
            rms_ffn_w: get_tensors("post_attention_layernorm.weight"),
            w_up: get_tensors("mlp.up_proj.weight"),
            w_gate: get_tensors("mlp.gate_proj.weight"),
            w_down: get_tensors("mlp.down_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
