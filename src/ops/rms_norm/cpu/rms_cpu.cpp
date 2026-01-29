#include "rms_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
static void rms_norm_impl(T *output, const T *input, const T *weight, size_t N, size_t M, float eps) {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (size_t j = 0; j < M; j++) {
            float val = casting(float, input[i * M + j]);
            sum += val * val;
        }
        float rms = std::sqrt(sum / static_cast<float>(M) + eps);
#pragma omp simd
        for (size_t j = 0; j < M; j++) {
            float x = casting(float, input[i * M + j]);
            float w = casting(float, weight[j]);
            output[i * M + j] = casting(T, w * x / rms);
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(std::byte *output,
              const std::byte *input,
              const std::byte *weight,
              size_t N,
              size_t M,
              float eps,
              llaisysDataType_t dtype) {
    using namespace llaisys;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl(recast(float *, output), recast(const float *, input), recast(const float *, weight), N, M,
                             eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl(recast(fp16_t *, output), recast(const fp16_t *, input), recast(const fp16_t *, weight), N,
                             M, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl(recast(bf16_t *, output), recast(const bf16_t *, input), recast(const bf16_t *, weight), N,
                             M, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu