#include "op.hpp"
#include "../../utils.hpp"
#include "cpu/rms_cpu.hpp"
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        llaisys::ops::cpu::rms_norm(out->data(), in->data(), weight->data(), out->shape()[0], out->shape()[1], eps,
                                    out->dtype());
    } else
        EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
