#include "simple.hpp"
#include <cstring>

namespace llaisys::kvcache::simple {

KVCache::KVCache(usize capacity,
                 usize num_kv_head,
                 usize head_dim,
                 usize vdim,
                 llaisysDataType_t dtype,
                 llaisysDeviceType_t device,
                 int device_id)
    : capacity(capacity), cache_size(0), num_kv_head(num_kv_head),
      head_dim(head_dim), vdim(vdim), dtype(dtype), device(device),
      device_id(device_id),
      keys(Tensor::create(
          {capacity, num_kv_head, head_dim}, dtype, device, device_id)),
      values(Tensor::create(
          {capacity, num_kv_head, vdim}, dtype, device, device_id)) {
    std::memset(static_cast<std::byte *>(keys->data()), 0,
                capacity * num_kv_head * head_dim * keys->elementSize());
    std::memset(static_cast<std::byte *>(values->data()), 0,
                capacity * num_kv_head * vdim * values->elementSize());
}

void KVCache::reset() { cache_size = 0; }

void KVCache::insert(const tensor &new_keys,
                     const tensor &new_values,
                     usize n_new) {
    if (cache_size + n_new > capacity)
        throw std::runtime_error("KVCache insert position exceeds capacity.");
    if (new_keys->shape()[0] != n_new || new_keys->shape()[1] != num_kv_head
        || new_keys->shape()[2] != head_dim)
        throw std::runtime_error("New keys tensor shape mismatch.");
    if (new_values->shape()[0] != n_new || new_values->shape()[1] != num_kv_head
        || new_values->shape()[2] != vdim)
        throw std::runtime_error("New values tensor shape mismatch.");
    if (new_keys->deviceType() != device || new_values->deviceType() != device)
        throw std::runtime_error("Device mismatch");
    if (!new_keys->isContiguous() || !new_values->isContiguous())
        throw std::runtime_error("New K/V must be contiguous");
    if (!keys->isContiguous() || !values->isContiguous())
        throw std::runtime_error("Cache storage must be contiguous");

    do {
        auto begin = static_cast<std::byte *>(keys->data())
                   + cache_size * num_kv_head * head_dim * keys->elementSize();
        auto numel = n_new * num_kv_head * head_dim;
        std::memcpy(begin, static_cast<const std::byte *>(new_keys->data()),
                    numel * new_keys->elementSize());
    } while (false);
    do {
        auto begin = static_cast<std::byte *>(values->data())
                   + cache_size * num_kv_head * vdim * values->elementSize();
        auto numel = n_new * num_kv_head * vdim;
        std::memcpy(begin, static_cast<const std::byte *>(new_values->data()),
                    numel * new_values->elementSize());
    } while (false);

    cache_size += n_new;
}

KVCache::tensor KVCache::getKeysSlice() {
    return keys->slice(0, 0, cache_size);
}

KVCache::tensor KVCache::getValuesSlice() {
    return values->slice(0, 0, cache_size);
}

} // namespace llaisys::kvcache::simple