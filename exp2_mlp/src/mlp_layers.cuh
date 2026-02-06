#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

struct LayerShape {
    int batch;
    int in_dim;
    int out_dim;
};

inline double layer_flops(const LayerShape& shape) {
    return 2.0 * static_cast<double>(shape.batch) * shape.in_dim * shape.out_dim;
}

inline double mlp_gflops(const std::vector<int>& layers, int batch, double millis) {
    double total_flops = 0.0;
    for (size_t i = 0; i + 1 < layers.size(); ++i) {
        LayerShape shape{batch, layers[i], layers[i + 1]};
        total_flops += layer_flops(shape);
    }
    return total_flops / (millis * 1e6);
}

__global__ void bias_add_kernel(const float* __restrict__ bias,
                                float* __restrict__ activations,
                                LayerShape shape) {
    /* TODO(student): each thread should add the bias for its neuron across the batch. */
    (void)bias;
    (void)activations;
    (void)shape;
}

__global__ void relu_kernel(float* __restrict__ activations, size_t elements) {
    /* TODO(student): ReLU activation (set negative values to zero). */
    (void)activations;
    (void)elements;
}

__global__ void gelu_kernel(float* __restrict__ activations, size_t elements) {
    /* TODO(student): Approximate GELU, e.g., 0.5*x*(1+tanh(...)). */
    (void)activations;
    (void)elements;
}

inline void launch_bias_add(const float* bias, float* activations, const LayerShape& shape, cudaStream_t stream) {
    const int threads = 256;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    bias_add_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape);
    (void)elements;  // silence unused warnings until kernel implemented
}

inline void launch_activation(const std::string& activation,
                              float* activations,
                              const LayerShape& shape,
                              cudaStream_t stream) {
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (activation == "relu") {
        relu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else if (activation == "gelu") {
        gelu_kernel<<<blocks, threads, 0, stream>>>(activations, elements);
    } else {
        // TODO(student): add more activations as desired
    }
}

__global__ void fused_bias_activation_kernel(const float* __restrict__ bias,
                                             float* __restrict__ activations,
                                             LayerShape shape,
                                             int activation_type) {
    /* TODO(student): fuse bias add + activation.
       activation_type: 0=ReLU, 1=GELU, extend as needed. */
    (void)bias;
    (void)activations;
    (void)shape;
    (void)activation_type;
}

inline void launch_fused_bias_activation(const float* bias,
                                         const std::string& activation,
                                         float* activations,
                                         const LayerShape& shape,
                                         cudaStream_t stream) {
    int activation_type = (activation == "gelu") ? 1 : 0;
    const size_t elements = static_cast<size_t>(shape.batch) * shape.out_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    fused_bias_activation_kernel<<<blocks, threads, 0, stream>>>(bias, activations, shape, activation_type);
    (void)elements;
}

inline void run_gemm_layer(const float* input,
                           const float* weight,
                           float* output,
                           const LayerShape& shape,
                           cublasHandle_t handle) {
    /* TODO(student): call cublasSgemm (or StridedBatched) with the correct transpose options.
       Remember cuBLAS assumes column-major by default; consider using CUBLAS_OP_T to match row-major data. */
    (void)input;
    (void)weight;
    (void)output;
    (void)shape;
    (void)handle;
}
