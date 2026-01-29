#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    ASSERT(max_idx->numel() == 1, "argmax(): max_idx must have exactly one element");
    ASSERT(max_val->numel() == 1, "argmax(): max_val must have exactly one element");
    ASSERT(vals->isContiguous(), "argmax(): input tensor must be contiguous");

    llaisysDeviceType_t device = vals->deviceType();
    if (device == LLAISYS_DEVICE_CPU)
        cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->numel(), vals->dtype());
    else
        EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
