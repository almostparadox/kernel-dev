// 03-cuda-llm-kernels/paged_attention/paged_ops.cpp
#include <cstdint>
#include <cmath>

#if defined(__has_include)
  #if __has_include(<torch/extension.h>)
    #include <torch/extension.h>
    #define HAVE_TORCH_EXTENSION 1
  #endif
  #if __has_include(<cuda_runtime.h>)
    #include <cuda_runtime.h>
    #include <cuda_fp16.h>
    #define HAVE_CUDA_RUNTIME 1
  #endif
  #if __has_include(<c10/cuda/CUDAStream.h>)
    #include <c10/cuda/CUDAStream.h>
    #define HAVE_CUDA_STREAM 1
  #endif
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#if !defined(HAVE_CUDA_RUNTIME)
struct __half_raw {
    uint16_t x;
};
using half = __half_raw;
using cudaStream_t = void*;
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define PAGED_EXPORT __declspec(dllexport)
#else
#define PAGED_EXPORT __attribute__((visibility("default")))
#endif

// Forward declaration of CUDA launcher defined in paged_decode.cu
void paged_attention_v1_launcher(
    half* out,
    const half* q,
    const half* k_pool,
    const half* v_pool,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream
);

// Forward declaration of CUDA launcher defined in paged_split_kv.cu
void paged_attention_splitkv_launcher(
    half* out,
    float* tmp_out,
    float* tmp_metadata,
    const half* q,
    const half* k_pool,
    const half* v_pool,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_splits,
    float scale,
    cudaStream_t stream
);

// C-linkage export for ctypes integration
PAGED_EXPORT void launch_paged_attention_v1(
    half* out,
    const half* q,
    const half* k_pool,
    const half* v_pool,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads,
    int head_dim,
    float scale,
    cudaStream_t stream
) {
    paged_attention_v1_launcher(
        out,
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        max_blocks_per_seq,
        batch_size,
        num_heads,
        head_dim,
        scale,
        stream
    );
}

// C-linkage export for ctypes integration (Split-KV)
PAGED_EXPORT void launch_paged_attention_splitkv(
    half* out,
    float* tmp_out,
    float* tmp_metadata,
    const half* q,
    const half* k_pool,
    const half* v_pool,
    const int32_t* block_tables,
    const int32_t* context_lens,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_splits,
    float scale,
    cudaStream_t stream
) {
    paged_attention_splitkv_launcher(
        out,
        tmp_out,
        tmp_metadata,
        q,
        k_pool,
        v_pool,
        block_tables,
        context_lens,
        max_blocks_per_seq,
        batch_size,
        num_heads,
        head_dim,
        num_splits,
        scale,
        stream
    );
}

#ifdef __cplusplus
}
#endif

#if defined(HAVE_TORCH_EXTENSION)

static void check_paged_attention_v1_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const c10::optional<torch::Tensor>& out_opt
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_pool.is_cuda(), "k_pool must be a CUDA tensor");
    TORCH_CHECK(v_pool.is_cuda(), "v_pool must be a CUDA tensor");
    TORCH_CHECK(block_tables.is_cuda(), "block_tables must be a CUDA tensor");
    TORCH_CHECK(context_lens.is_cuda(), "context_lens must be a CUDA tensor");

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k_pool.is_contiguous(), "k_pool must be contiguous");
    TORCH_CHECK(v_pool.is_contiguous(), "v_pool must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");

    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "q must be float16");
    TORCH_CHECK(k_pool.scalar_type() == at::ScalarType::Half, "k_pool must be float16");
    TORCH_CHECK(v_pool.scalar_type() == at::ScalarType::Half, "v_pool must be float16");
    TORCH_CHECK(block_tables.scalar_type() == at::ScalarType::Int, "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::ScalarType::Int, "context_lens must be int32");

    auto device = q.device();
    TORCH_CHECK(k_pool.device() == device, "k_pool device mismatch with q");
    TORCH_CHECK(v_pool.device() == device, "v_pool device mismatch with q");
    TORCH_CHECK(block_tables.device() == device, "block_tables device mismatch with q");
    TORCH_CHECK(context_lens.device() == device, "context_lens device mismatch with q");

    torch::Tensor q_sq = (q.dim() == 4 && q.size(1) == 1) ? q.squeeze(1) : q;
    TORCH_CHECK(q_sq.dim() == 3, "q must have shape [batch_size, num_heads, head_dim]");

    const int64_t batch_size = q_sq.size(0);
    const int64_t num_heads = q_sq.size(1);
    const int64_t head_dim = q_sq.size(2);

    TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
                "head_dim must be 64, 128, or 256, got ", head_dim);

    TORCH_CHECK(k_pool.dim() == 4, "k_pool must be 4D: [num_blocks, num_heads, block_size, head_dim]");
    TORCH_CHECK(v_pool.dim() == 4, "v_pool must be 4D: [num_blocks, num_heads, block_size, head_dim]");
    TORCH_CHECK(k_pool.size(1) == num_heads, "k_pool num_heads mismatch with q");
    TORCH_CHECK(v_pool.size(1) == num_heads, "v_pool num_heads mismatch with q");
    TORCH_CHECK(k_pool.size(2) == BLOCK_SIZE, "k_pool block_size must match BLOCK_SIZE (16)");
    TORCH_CHECK(v_pool.size(2) == BLOCK_SIZE, "v_pool block_size must match BLOCK_SIZE (16)");
    TORCH_CHECK(k_pool.size(3) == head_dim, "k_pool head_dim mismatch with q");
    TORCH_CHECK(v_pool.size(3) == head_dim, "v_pool head_dim mismatch with q");

    TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2D: [batch_size, max_blocks_per_seq]");
    TORCH_CHECK(block_tables.size(0) == batch_size, "block_tables batch_size mismatch with q");

    TORCH_CHECK(context_lens.dim() == 1, "context_lens must be 1D: [batch_size]");
    TORCH_CHECK(context_lens.size(0) == batch_size, "context_lens batch_size mismatch with q");

    if (out_opt.has_value()) {
        const auto& out = out_opt.value();
        TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
        TORCH_CHECK(out.device() == device, "out device mismatch with q");
        TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
        TORCH_CHECK(out.scalar_type() == at::ScalarType::Half, "out must be float16");
        TORCH_CHECK(out.dim() == 3, "out must be 3D [batch_size, num_heads, head_dim]");
        TORCH_CHECK(out.size(0) == batch_size && out.size(1) == num_heads && out.size(2) == head_dim,
                    "out shape mismatch with [batch_size, num_heads, head_dim]");
    }
}

torch::Tensor paged_attention_v1(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    float scale = 0.0f,
    c10::optional<torch::Tensor> out_opt = c10::nullopt
) {
    check_paged_attention_v1_inputs(q, k_pool, v_pool, block_tables, context_lens, out_opt);

#if defined(HAVE_CUDA_RUNTIME) && defined(HAVE_CUDA_STREAM)
    torch::Tensor q_sq = (q.dim() == 4 && q.size(1) == 1) ? q.squeeze(1) : q;
    const int batch_size = static_cast<int>(q_sq.size(0));
    const int num_heads = static_cast<int>(q_sq.size(1));
    const int head_dim = static_cast<int>(q_sq.size(2));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    torch::Tensor out;
    if (out_opt.has_value()) {
        out = out_opt.value();
    } else {
        out = torch::empty({batch_size, num_heads, head_dim}, q_sq.options());
    }

    const float final_scale = (scale > 0.0f)
        ? scale
        : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    paged_attention_v1_launcher(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_sq.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_pool.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_pool.data_ptr<at::Half>()),
        block_tables.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(),
        max_blocks_per_seq,
        batch_size,
        num_heads,
        head_dim,
        final_scale,
        stream
    );

    return out;
#else
    TORCH_CHECK(false, "PagedAttention CUDA kernel is not available (CUDA stream/runtime not compiled)");
#endif
}

torch::Tensor paged_attention_splitkv(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    int64_t num_splits = 4,
    float scale = 0.0f,
    c10::optional<torch::Tensor> out_opt = c10::nullopt
) {
    TORCH_CHECK(num_splits >= 1, "num_splits must be at least 1, got ", num_splits);
    TORCH_CHECK(num_splits <= 128, "num_splits must be <= 128, got ", num_splits);
    check_paged_attention_v1_inputs(q, k_pool, v_pool, block_tables, context_lens, out_opt);

#if defined(HAVE_CUDA_RUNTIME) && defined(HAVE_CUDA_STREAM)
    torch::Tensor q_sq = (q.dim() == 4 && q.size(1) == 1) ? q.squeeze(1) : q;
    const int batch_size = static_cast<int>(q_sq.size(0));
    const int num_heads = static_cast<int>(q_sq.size(1));
    const int head_dim = static_cast<int>(q_sq.size(2));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));
    const int n_splits = static_cast<int>(num_splits);

    torch::Tensor out;
    if (out_opt.has_value()) {
        out = out_opt.value();
    } else {
        out = torch::empty({batch_size, num_heads, head_dim}, q_sq.options());
    }

    torch::Tensor tmp_out = torch::empty(
        {batch_size, num_heads, n_splits, head_dim},
        q_sq.options().dtype(torch::kFloat32)
    );
    torch::Tensor tmp_meta = torch::empty(
        {batch_size, num_heads, n_splits, 2},
        q_sq.options().dtype(torch::kFloat32)
    );

    const float final_scale = (scale > 0.0f)
        ? scale
        : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    paged_attention_splitkv_launcher(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        tmp_out.data_ptr<float>(),
        tmp_meta.data_ptr<float>(),
        reinterpret_cast<const half*>(q_sq.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_pool.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_pool.data_ptr<at::Half>()),
        block_tables.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(),
        max_blocks_per_seq,
        batch_size,
        num_heads,
        head_dim,
        n_splits,
        final_scale,
        stream
    );

    return out;
#else
    TORCH_CHECK(false, "PagedAttention Split-KV CUDA kernel is not available (CUDA stream/runtime not compiled)");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PagedAttention CUDA C++ Extension";
    m.def(
        "paged_attention_v1",
        &paged_attention_v1,
        "PagedAttention V1 single-pass decode kernel",
        py::arg("q"),
        py::arg("k_pool"),
        py::arg("v_pool"),
        py::arg("block_tables"),
        py::arg("context_lens"),
        py::arg("scale") = 0.0f,
        py::arg("out") = py::none()
    );
    m.def(
        "paged_attention_v1_out",
        [](torch::Tensor& out,
           const torch::Tensor& q,
           const torch::Tensor& k_pool,
           const torch::Tensor& v_pool,
           const torch::Tensor& block_tables,
           const torch::Tensor& context_lens,
           float scale) {
            return paged_attention_v1(q, k_pool, v_pool, block_tables, context_lens, scale, out);
        },
        "PagedAttention V1 single-pass decode kernel (out passed first)",
        py::arg("out"),
        py::arg("q"),
        py::arg("k_pool"),
        py::arg("v_pool"),
        py::arg("block_tables"),
        py::arg("context_lens"),
        py::arg("scale") = 0.0f
    );
    m.def(
        "paged_attention_splitkv",
        &paged_attention_splitkv,
        "FlashDecoding Split-KV long-context decode kernel",
        py::arg("q"),
        py::arg("k_pool"),
        py::arg("v_pool"),
        py::arg("block_tables"),
        py::arg("context_lens"),
        py::arg("num_splits") = 4,
        py::arg("scale") = 0.0f,
        py::arg("out") = py::none()
    );
    m.def(
        "paged_attention_splitkv_out",
        [](torch::Tensor& out,
           const torch::Tensor& q,
           const torch::Tensor& k_pool,
           const torch::Tensor& v_pool,
           const torch::Tensor& block_tables,
           const torch::Tensor& context_lens,
           int64_t num_splits,
           float scale) {
            return paged_attention_splitkv(q, k_pool, v_pool, block_tables, context_lens, num_splits, scale, out);
        },
        "FlashDecoding Split-KV long-context decode kernel (out passed first)",
        py::arg("out"),
        py::arg("q"),
        py::arg("k_pool"),
        py::arg("v_pool"),
        py::arg("block_tables"),
        py::arg("context_lens"),
        py::arg("num_splits") = 4,
        py::arg("scale") = 0.0f
    );
}

#endif // HAVE_TORCH_EXTENSION
