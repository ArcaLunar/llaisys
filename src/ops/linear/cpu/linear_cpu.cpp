#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstring>
#include <omp.h>

template <typename T>
static void linear_impl(T *output, const T *input, const T *weight, const T *bias, size_t N, size_t M, size_t K) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            float sum = 0;

            #pragma omp simd reduction(+:sum)
            for (size_t m = 0; m < M; m++) sum += casting(float, input[n * M + m]) * casting(float, weight[k * M + m]);
            
            if (bias != nullptr)
                sum += casting(float, bias[k]);
            output[n * K + k] = casting(T, sum);
        }
    }
}

namespace llaisys::ops::cpu {

void linear(std::byte *output,
            const std::byte *input,
            const std::byte *weight,
            const std::byte *bias,
            size_t N,
            size_t M,
            size_t K,
            llaisysDataType_t dtype) {

    using namespace llaisys;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_impl(recast(float *, output), recast(const float *, input), recast(const float *, weight),
                           recast(const float *, bias), N, M, K);
    case LLAISYS_DTYPE_F16:
        return linear_impl(recast(fp16_t *, output), recast(const fp16_t *, input), recast(const fp16_t *, weight),
                           recast(const fp16_t *, bias), N, M, K);
    case LLAISYS_DTYPE_BF16:
        return linear_impl(recast(bf16_t *, output), recast(const bf16_t *, input), recast(const bf16_t *, weight),
                           recast(const bf16_t *, bias), N, M, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu