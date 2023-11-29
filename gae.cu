// #include <ATen/cuda/CUDAContext.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void gae_kernel(const float *rewards, const float *values, const int *cu_seqlens,
                           float *adv_out, float *ret_out, int batch_size, float gamma,
                           float lmbda) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size) { return; }
  // get the idx-th start index from cu_seqlens
  int rs_idx = cu_seqlens[idx];
  int re_idx = cu_seqlens[idx + 1];
  int vs_idx = rs_idx + idx;
  float lastgae = 0.0;
  for (int i = re_idx - rs_idx - 1; i >= 0; i--) {
    float delta = rewards[rs_idx + i] + gamma * values[vs_idx + i + 1] - values[vs_idx + i];
    lastgae = delta + gamma * lmbda * lastgae;
    adv_out[rs_idx + i] = lastgae;
    ret_out[rs_idx + i] = lastgae + values[vs_idx + i];
  }
  // int stride = blockDim.x * gridDim.x;
  // for (int i = idx; i < num_envs; i += stride) {
  //   float prev_return = 0;
  //   for (int j = num_steps - 1; j >= 0; j--) {
  //     float delta = rewards[i * num_steps + j]
  //                   + gamma * values[i * num_steps + j + 1] * masks[i * num_steps + j]
  //                   - values[i * num_steps + j];
  //     prev_return = delta + gamma * lmbda * masks[i * num_steps + j] * prev_return;
  //     out[i * num_steps + j] = prev_return + values[i * num_steps + j];
  //   }
  // }
}

template<int num_threads>
std::vector<at::Tensor> gae(at::Tensor &rewards, at::Tensor &values, at::Tensor &cu_seqlens,
                            float gamma, float lmbda) {
  int batch_size = cu_seqlens.numel() - 1;
  int total_seqlen = rewards.size(0);
  CHECK_DEVICE(rewards);
  CHECK_DEVICE(values);
  CHECK_DEVICE(cu_seqlens);
  CHECK_CONTIGUOUS(rewards);
  CHECK_CONTIGUOUS(values);
  CHECK_CONTIGUOUS(cu_seqlens);
  CHECK_SHAPE(values, total_seqlen + batch_size);
  TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
  TORCH_CHECK(cu_seqlens[0].item<int>() == 0, "cu_seqlens[0] must be 0");
  TORCH_CHECK(cu_seqlens[-1].item<int>() == total_seqlen, "cu_seqlens[-1] must be total_seqlen");
  TORCH_CHECK(rewards.dtype() == values.dtype(), "rewards and values must have the same dtype");

  int num_blocks = (batch_size + num_threads - 1) / num_threads;
  auto adv_out = at::zeros_like(rewards);
  auto ret_out = at::zeros_like(rewards);
  gae_kernel<<<num_blocks, num_threads>>>(rewards.data_ptr<float>(), values.data_ptr<float>(),
                                          cu_seqlens.data_ptr<int>(), adv_out.data_ptr<float>(),
                                          ret_out.data_ptr<float>(), batch_size, gamma, lmbda);
  return {adv_out, ret_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Generalized Advantage Estimation (CUDA)";
  m.def("gae", &gae<16>, "Generalized Advantage Estimation (CUDA)");
}