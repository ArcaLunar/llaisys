#include "op.hpp"
#include "cpu/selfattn_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    auto seqlen = q->shape()[0];
    auto num_head = q->shape()[1];
    auto head_dim = q->shape()[2];
    auto kvlen = k->shape()[0];
    auto num_kv_head = k->shape()[1];
    auto vdim = v->shape()[2];

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::self_attn(attn_val->data(), q->data(), k->data(), v->data(), seqlen, num_head, head_dim, kvlen,
                       num_kv_head, vdim, scale, attn_val->dtype());
    } else
        EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
