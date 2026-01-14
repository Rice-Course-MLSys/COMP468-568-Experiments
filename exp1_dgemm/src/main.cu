#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernel.cuh"

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    std::string impl = "baseline";
    bool verify = true;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--m") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            opt.m = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--n") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            opt.n = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--k") == 0 || strcmp(argv[i], "-k") == 0) && i + 1 < argc) {
            opt.k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dgemm [--m int] [--n int] [--k int] [--impl baseline|naive|tiled|cublas] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

double gflops(int m, int n, int k, double millis) {
    double flops = 2.0 * m * n * k;
    return flops / (millis * 1e6);
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    const int m = opt.m, n = opt.n, k = opt.k;
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);

    std::vector<float> h_a(m * k), h_b(k * n), h_c(m * n, 0.0f), h_ref(m * n, 0.0f);

    /* TODO(student): initialize h_a, h_b with reproducible random data (e.g., std::sin / std::cos). */

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    /* TODO(student): allocate device buffers and copy host data over. */

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "create start");
    check_cuda(cudaEventCreate(&stop), "create stop");

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline" || opt.impl == "naive" || opt.impl == "tiled") {
        /* TODO(student): choose the right launch helper based on opt.impl and record elapsed_ms. */
        (void)elapsed_ms;  // remove once timing implemented
    } else if (opt.impl == "cublas") {
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        check_cuda(cudaEventRecord(start), "record start");
        check_cublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        d_b,
                        n,
                        d_a,
                        k,
                        &beta,
                        d_c,
                        n),
            "cublasSgemm");
        check_cuda(cudaEventRecord(stop), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
        check_cublas(cublasDestroy(handle), "cublasDestroy");
    } else {
        throw std::invalid_argument("Unknown implementation: " + opt.impl);
    }

    /* TODO(student): copy d_c back into h_c. */

    if (opt.verify) {
        /* TODO(student): run cuBLAS reference into h_ref (or reuse above) and compute max error. */
    }

    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " M=" << m << " N=" << n << " K=" << k
                  << " Time(ms)=" << elapsed_ms << " GFLOP/s=" << gflops(m, n, k, elapsed_ms)
                  << std::endl;
    }

    /* TODO(student): free device memory and destroy CUDA events. */
    return 0;
}

