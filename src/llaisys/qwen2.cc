#include "llaisys/models/qwen2.h"
#include "../kvcache/simple.hpp"
#include "../ops/ops.hpp"
#include "llaisys.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "llaisys_tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#define createTensor(...)                                                      \
    new LlaisysTensor { llaisys::Tensor::create(__VA_ARGS__) }

#define CASE(id, name, val)                                                    \
    case id:                                                                   \
        do {                                                                   \
            if (layer_id != -1) {                                              \
                std::cerr << "[qwen2.cc:setWeights()] " #name                  \
                             " should not have layer_id"                       \
                          << std::endl;                                        \
                exit(1);                                                       \
            }                                                                  \
            auto ts = val->tensor;                                             \
            model->weights.name = createTensor(                                \
                ts->shape(), ts->dtype(), model->device, model->device_id);    \
            model->weights.name->tensor = ts;                                  \
            std::cerr << "[qwen2.cc:setWeights()] Set " #name << std::endl;    \
        } while (0);                                                           \
        break;

#define CASE_ARRAY(id, name, val)                                              \
    case id:                                                                   \
        do {                                                                   \
            if (layer_id < 0 || layer_id >= static_cast<int>(nlayer)) {        \
                std::cerr << "[qwen2.cc:setWeights()] " #name                  \
                             " layer_id out of range"                          \
                          << std::endl;                                        \
                exit(1);                                                       \
            }                                                                  \
            auto ts = val->tensor;                                             \
            model->weights.name[layer_id] = createTensor(                      \
                ts->shape(), ts->dtype(), model->device, model->device_id);    \
            model->weights.name[layer_id]->tensor = ts;                        \
            std::cerr << "[qwen2.cc:setWeights()] Set " #name << " for layer " \
                      << layer_id << std::endl;                                \
        } while (0);                                                           \
        break;

#define MODEL_VALIDITY_CHECK(model)                                            \
    do {                                                                       \
        if (!model) {                                                          \
            std::cerr << "[qwen2.cc:infer()] Model is null, cannot perform "   \
                         "inference."                                          \
                      << std::endl;                                            \
            return -1;                                                         \
        }                                                                      \
    } while (0)

#define LOG_SHAPE(stage, tensr, name)                                          \
    do {                                                                       \
        std::cerr << "[qwen2.cc:" << stage << "] " << name << " shape: ";      \
        for (int i = 0, l = tensr->tensor->ndim(); i < l; ++i) {               \
            std::cerr << tensr->tensor->shape()[i];                            \
            if (i != l - 1)                                                    \
                std::cerr << " x ";                                            \
        }                                                                      \
        std::cerr << std::endl;                                                \
    } while (0)
// Define some helper functions here, init model weights array/kvcache
// array, etc.
static void initializeArrays(LlaisysQwen2Model *model);

__C {

    struct LlaisysQwen2Model {
        LlaisysQwen2Meta meta;
        LlaisysQwen2Weights weights;
        llaisys::kvcache::simple::KVCache **kvcaches;
        llaisysDeviceType_t device;
        int device_id;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device,
        int *device_ids, int ndevice) {
        if (!meta) {
            std::cerr
                << "[qwen2.cc:create()] Meta is null, cannot create model."
                << std::endl;
            return nullptr;
        }

        LlaisysQwen2Model *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids ? device_ids[0] : 0;

        initializeArrays(model);
        std::cerr << "[qwen2.cc:create()] Model created." << std::endl;

        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:destroy()] Model is null, nothing to destroy."
                << std::endl;
            return;
        }

        // Free KV caches
        for (size_t i = 0; i < model->meta.nlayer; ++i) {
            delete model->kvcaches[i];
        }
        delete[] model->kvcaches;
        std::cerr << "[qwen2.cc:destroy()] Destroyed all KV caches."
                  << std::endl;

        // Free weight arrays
        delete[] model->weights.attn_norm_w;
        delete[] model->weights.attn_q_w;
        delete[] model->weights.attn_q_b;
        delete[] model->weights.attn_k_w;
        delete[] model->weights.attn_k_b;
        delete[] model->weights.attn_v_w;
        delete[] model->weights.attn_v_b;
        delete[] model->weights.attn_o_w;
        delete[] model->weights.mlp_norm_w;
        delete[] model->weights.mlp_gate_w;
        delete[] model->weights.mlp_up_w;
        delete[] model->weights.mlp_down_w;
        std::cerr << "[qwen2.cc:destroy()] Destroyed all weight arrays."
                  << std::endl;

        delete model;
        std::cerr << "[qwen2.cc:destroy()] Model destroyed." << std::endl;
    }

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        struct LlaisysQwen2Model * model) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:weights()] Model is null, cannot get weights."
                << std::endl;
            return nullptr;
        }
        /**
         * Display all tensor shape here for debugging
         */
        std::cerr << "[qwen2.cc:weights()] Model weights shapes:" << std::endl;

        LOG_SHAPE("weights()", model->weights.in_embed, "in_embed");
        LOG_SHAPE("weights()", model->weights.out_embed, "out_embed");
        LOG_SHAPE("weights()", model->weights.out_norm_w, "out_norm_w");

        auto nlayer = model->meta.nlayer;
        for (size_t i = 0; i < nlayer; ++i) {
            std::cerr << "[qwen2.cc:weights()] Layer " << i
                      << " weights:" << std::endl;
            LOG_SHAPE("weights()", model->weights.attn_norm_w[i],
                      "attn_norm_w");
            LOG_SHAPE("weights()", model->weights.attn_q_w[i], "attn_q_w");
            LOG_SHAPE("weights()", model->weights.attn_q_b[i], "attn_q_b");
            LOG_SHAPE("weights()", model->weights.attn_k_w[i], "attn_k_w");
            LOG_SHAPE("weights()", model->weights.attn_k_b[i], "attn_k_b");
            LOG_SHAPE("weights()", model->weights.attn_v_w[i], "attn_v_w");
            LOG_SHAPE("weights()", model->weights.attn_v_b[i], "attn_v_b");
            LOG_SHAPE("weights()", model->weights.attn_o_w[i], "attn_o_w");
            LOG_SHAPE("weights()", model->weights.mlp_norm_w[i], "mlp_norm_w");
            LOG_SHAPE("weights()", model->weights.mlp_gate_w[i], "mlp_gate_w");
            LOG_SHAPE("weights()", model->weights.mlp_up_w[i], "mlp_up_w");
            LOG_SHAPE("weights()", model->weights.mlp_down_w[i], "mlp_down_w");
        }

        return &model->weights;
    }

    __export void llaisysQwen2SetWeights(struct LlaisysQwen2Model * model,
                                         int name, int layer_id,
                                         llaisysTensor_t tensor) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:setWeights()] Model is null, cannot set weights."
                << std::endl;
            return;
        }

        size_t nlayer = model->meta.nlayer;
        switch (name) {
            CASE(0, in_embed, tensor)          // in_embed
            CASE(1, out_embed, tensor)         // out_embed
            CASE(2, out_norm_w, tensor)        // out_norm_w
            CASE_ARRAY(3, attn_norm_w, tensor) // attn_norm_w
            CASE_ARRAY(4, attn_q_w, tensor)    // attn_q_w
            CASE_ARRAY(5, attn_q_b, tensor)    // attn_q_b
            CASE_ARRAY(6, attn_k_w, tensor)    // attn_k_w
            CASE_ARRAY(7, attn_k_b, tensor)    // attn_k_b
            CASE_ARRAY(8, attn_v_w, tensor)    // attn_v_w
            CASE_ARRAY(9, attn_v_b, tensor)    // attn_v_b
            CASE_ARRAY(10, attn_o_w, tensor)   // attn_o_w
            CASE_ARRAY(11, mlp_norm_w, tensor) // mlp_norm_w
            CASE_ARRAY(12, mlp_gate_w, tensor) // mlp_gate_w
            CASE_ARRAY(13, mlp_up_w, tensor)   // mlp_up_w
            CASE_ARRAY(14, mlp_down_w, tensor) // mlp_down_w
        default:
            std::cerr << "[qwen2.cc:setWeights()] Unknown weight name: " << name
                      << ", cannot set weight." << std::endl;
            exit(1);
        }

        LOG_SHAPE("setWeights()", tensor, "weight tensor from Python");
    }

    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model * model, int64_t *token_ids, int64_t *pos_ids,
        size_t ntoken, bool prefill) {
        //* -1. Do checking
        MODEL_VALIDITY_CHECK(model);

        auto nlayer = model->meta.nlayer;
        //* 0. If prefill, clean KV Caches
        if (prefill) {
            std::cerr << "[qwen2.cc:infer()] Prefill mode: resetting KV caches."
                      << std::endl;
            for (size_t i = 0; i < nlayer; ++i) model->kvcaches[i]->reset();
        }

        /**
         *! MODEL INFERENCE LOGIC
         */
        using namespace llaisys;
        using tensor = LlaisysTensor *;
        using usize = size_t;
        using i64 = int64_t;

        //* IMPORTANT: store pos_ids as tensor on device for RoPE
        tensor pos_ids_tensor = createTensor({ntoken}, LLAISYS_DTYPE_I64,
                                             model->device, model->device_id);
        pos_ids_tensor->tensor->load(pos_ids);
        std::cerr << "[qwen2.cc:infer()] Loaded position ids." << std::endl;
        LOG_SHAPE("infer()", pos_ids_tensor, "pos_ids");

        //* 1. Copy inputs into tensor
        tensor input_ids = createTensor({ntoken}, LLAISYS_DTYPE_I64,
                                        model->device, model->device_id);
        input_ids->tensor->load(token_ids);
        std::cerr << "[qwen2.cc:infer()] Loaded input token ids." << std::endl;
        LOG_SHAPE("infer()", input_ids, "input_ids");

        //* 2. Token Embedding
        tensor wordvec
            = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                           model->device, model->device_id);
        llaisysEmbedding(wordvec, input_ids, model->weights.in_embed);
        std::cerr << "[qwen2.cc:infer()] Completed token embedding."
                  << std::endl;
        LOG_SHAPE("infer()", wordvec, "wordvec");

        //* 3. Attention Layers
        for (usize layer = 0; layer < nlayer; ++layer) {
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Starting RMS norm before attention." << std::endl;
            //* 3.0 Record a residual
            tensor residual = wordvec;

            //* 3.a RMS Norm before Attention
            tensor attn_normed
                = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                               model->device, model->device_id);
            llaisysRmsNorm(attn_normed, wordvec,
                           model->weights.attn_norm_w[layer],
                           model->meta.epsilon);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed RMS norm before attention." << std::endl;

            //* 3.b QKV Projection
            tensor q_proj = createTensor(
                {ntoken, model->meta.nh * model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            tensor k_proj = createTensor(
                {ntoken, model->meta.nkvh * model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            tensor v_proj = createTensor(
                {ntoken, model->meta.nkvh * model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);

            LOG_SHAPE("infer().qkv", attn_normed, "attn_normed");
            LOG_SHAPE("infer().qkv", q_proj, "q_proj");
            LOG_SHAPE("infer().qkv", model->weights.attn_q_w[layer],
                      "attn_q_w");
            LOG_SHAPE("infer().qkv", model->weights.attn_q_b[layer],
                      "attn_q_b");

            llaisysLinear(q_proj, attn_normed, model->weights.attn_q_w[layer],
                          model->weights.attn_q_b[layer]);
            llaisysLinear(k_proj, attn_normed, model->weights.attn_k_w[layer],
                          model->weights.attn_k_b[layer]);
            llaisysLinear(v_proj, attn_normed, model->weights.attn_v_w[layer],
                          model->weights.attn_v_b[layer]);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed QKV projection." << std::endl;

            //* 3.c RoPE Encoding for Q, K
            std::vector<usize> q_shape{ntoken, model->meta.nh, model->meta.dh};
            std::vector<usize> k_shape{ntoken, model->meta.nkvh,
                                       model->meta.dh};
            std::vector<usize> v_shape{ntoken, model->meta.nkvh,
                                       model->meta.dh};
            q_proj = tensorView(q_proj, q_shape.data(),
                                q_shape.size()); // reshape for RoPE
            k_proj = tensorView(k_proj, k_shape.data(),
                                k_shape.size()); // reshape for RoPE
            v_proj = tensorView(v_proj, v_shape.data(),
                                v_shape.size()); // reshape for attn

            llaisysROPE(q_proj, q_proj, pos_ids_tensor, model->meta.theta);
            llaisysROPE(k_proj, k_proj, pos_ids_tensor, model->meta.theta);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed RoPE encoding for Q and K." << std::endl;

            //* 3.d Update KV Cache
            model->kvcaches[layer]->insert(
                k_proj->tensor, v_proj->tensor, ntoken,
                model->kvcaches[layer]->getCacheSize());
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Updated KV cache." << std::endl;

            //* 3.e Self-Attention
            tensor attn_out = createTensor(
                {ntoken, model->meta.nh, model->meta.dh}, model->meta.dtype,
                model->device, model->device_id);
            float scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));

            auto _kcache = model->kvcaches[layer]->getKeysSlice();
            tensor kcache
                = createTensor(_kcache->shape(), _kcache->dtype(),
                               _kcache->deviceType(), _kcache->deviceId());
            kcache->tensor = _kcache;

            auto _vcache = model->kvcaches[layer]->getValuesSlice();
            tensor vcache
                = createTensor(_vcache->shape(), _vcache->dtype(),
                               _vcache->deviceType(), _vcache->deviceId());
            vcache->tensor = _vcache;

            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Starting self-attention computation." << std::endl;
            attn_out = createTensor({ntoken, model->meta.nh, model->meta.dh},
                                    model->meta.dtype, model->device,
                                    model->device_id);

            LOG_SHAPE("infer().attn", attn_out, "attn_out");
            LOG_SHAPE("infer().attn", q_proj, "q_proj");
            LOG_SHAPE("infer().attn", kcache, "kcache");
            LOG_SHAPE("infer().attn", vcache, "vcache");
            
            llaisysSelfAttention(attn_out, q_proj, kcache, vcache, scale);

            //* 3.f Output Projection
            tensor attn_proj
                = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                               model->device, model->device_id);
            llaisysLinear(attn_proj, attn_out, model->weights.attn_o_w[layer],
                          nullptr); // no bias
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed attention output projection."
                      << std::endl;

            //* 3.g Residual after attention
            llaisysAdd(wordvec, residual, attn_proj);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed residual addition after attention."
                      << std::endl;

            //* 3.h update residual
            residual = wordvec;
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Starting MLP block." << std::endl;

            //* 3.i MLP block
            tensor mlp_normed
                = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                               model->device, model->device_id);
            llaisysRmsNorm(mlp_normed, wordvec,
                           model->weights.mlp_norm_w[layer],
                           model->meta.epsilon);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed RMS norm before MLP." << std::endl;

            //* 3.j MLP projections
            tensor mlp_gate
                = createTensor({ntoken, model->meta.di}, model->meta.dtype,
                               model->device, model->device_id);
            tensor mlp_up
                = createTensor({ntoken, model->meta.dh}, model->meta.dtype,
                               model->device, model->device_id);
            tensor mlp_down
                = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                               model->device, model->device_id);
            llaisysLinear(mlp_gate, mlp_normed,
                          model->weights.mlp_gate_w[layer], nullptr); // no bias
            llaisysLinear(mlp_up, mlp_normed, model->weights.mlp_up_w[layer],
                          nullptr); // no bias
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed MLP gate and up projections."
                      << std::endl;

            llaisysSwiGLU(mlp_down, mlp_gate, mlp_up);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed SwiGLU activation." << std::endl;

            //* 3.k Final MLP output projection
            tensor mlp_out
                = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                               model->device, model->device_id);
            llaisysLinear(mlp_out, mlp_down, model->weights.mlp_down_w[layer],
                          nullptr); // no bias
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed final MLP output projection."
                      << std::endl;

            //* 3.l Final residual addition
            llaisysAdd(wordvec, residual, mlp_out);
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Completed final residual addition." << std::endl;

            //* 3.final clean up all intermediate tensors in this layer
            delete attn_normed;
            delete q_proj;
            delete k_proj;
            delete v_proj;
            delete attn_out;
            delete attn_proj;
            delete mlp_normed;
            delete mlp_gate;
            delete mlp_up;
            delete mlp_down;
            delete mlp_out;
            delete residual;
            std::cerr << "[qwen2.cc:infer()] Layer " << layer
                      << ": Cleaned up intermediate tensors." << std::endl;

            //* Logging
            std::cerr << "[qwen2.cc:infer()] Layer " << layer << ": Completed."
                      << std::endl;
        }

        //* 4. Final transform and output projection
        tensor final_norm
            = createTensor({ntoken, model->meta.hs}, model->meta.dtype,
                           model->device, model->device_id);
        llaisysRmsNorm(final_norm, wordvec, model->weights.out_norm_w,
                       model->meta.epsilon);
        std::cerr << "[qwen2.cc:infer()] Completed final RMS norm."
                  << std::endl;

        //* 5. Get logits
        tensor logits
            = createTensor({ntoken, model->meta.voc}, model->meta.dtype,
                           model->device, model->device_id);
        llaisysLinear(logits, final_norm, model->weights.out_embed, nullptr);
        std::cerr << "[qwen2.cc:infer()] Completed output projection to logits."
                  << std::endl;

        //* 6. Get the last token's logits and argmax
        auto logits_output = logits->tensor->slice(
            ntoken - 1, 0, model->meta.voc); // last token
        tensor last_token_logits
            = createTensor(logits_output->shape(), model->meta.dtype,
                           model->device, model->device_id);
        last_token_logits->tensor = logits_output;
        std::cerr << "[qwen2.cc:infer()] Sliced out last token logits."
                  << std::endl;
        LOG_SHAPE("infer()", logits, "logits");
        LOG_SHAPE("infer()", last_token_logits, "last_token_logits");

        //* 7. Argmax to get next token id
        tensor next_token_id_tensor = createTensor(
            {1}, LLAISYS_DTYPE_I64, model->device, model->device_id);
        tensor next_token_logits_tensor
            = createTensor({1, model->meta.voc}, model->meta.dtype,
                           model->device, model->device_id);
        llaisysArgmax(next_token_id_tensor, next_token_logits_tensor,
                      last_token_logits);
        std::cerr
            << "[qwen2.cc:infer()] Completed argmax to get next token id: "
            << *((i64 *)next_token_id_tensor->tensor->data()) << std::endl;

        //* 8. Clean up all tensors
        delete input_ids;
        delete wordvec;
        delete pos_ids_tensor;
        delete final_norm;
        delete logits;
        delete last_token_logits;
        delete next_token_logits_tensor;

        //* 9. Return next token id
        i64 next_token_id = *((i64 *)next_token_id_tensor->tensor->data());
        delete next_token_id_tensor;
        return next_token_id;
    }
}

static void initializeArrays(LlaisysQwen2Model *model) {
    size_t nlayer = model->meta.nlayer;

    // KV Cache init
    model->kvcaches = new llaisys::kvcache::simple::KVCache *[nlayer];
    for (size_t i = 0; i < nlayer; ++i) {
        model->kvcaches[i] = new llaisys::kvcache::simple::KVCache(
            model->meta.maxseq, model->meta.nkvh, model->meta.dh,
            model->meta.dh, model->meta.dtype, model->device, model->device_id);
    }

    std::cerr << "[qwen2.cc:initializeArrays()] Initialized KV caches for "
              << nlayer << " layers." << std::endl;

    // Weight init
    model->weights.attn_norm_w = new llaisysTensor_t[nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_up_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[nlayer];

    std::cerr << "[qwen2.cc:initializeArrays()] Initialized Qwen2 model with "
              << nlayer << " layers." << std::endl;
}