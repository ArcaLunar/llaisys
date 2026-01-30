#include "llaisys/models/qwen2.h"
#include "../kvcache/simple.hpp"
#include "llaisys.h"
#include <cassert>
#include <iostream>

#define CASE(id, name, val)                                                    \
    case id:                                                                   \
        if (layer_id != -1) {                                                  \
            std::cerr << "[qwen2.cc:setWeights()] " #name                      \
                         " should not have layer_id"                           \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
        model->weights.name = val;                                             \
        std::cerr << "[qwen2.cc:setWeights()] Set " #name << std::endl;        \
        break;

#define CASE_ARRAY(id, name, val)                                              \
    case id:                                                                   \
        if (layer_id < 0 || layer_id >= static_cast<int>(nlayer)) {            \
            std::cerr << "[qwen2.cc:setWeights()] " #name                      \
                         " layer_id out of range"                              \
                      << std::endl;                                            \
            exit(1);                                                           \
        }                                                                      \
        model->weights.name[layer_id] = val;                                   \
        std::cerr << "[qwen2.cc:setWeights()] Set " #name << " for layer "     \
                  << layer_id << std::endl;                                    \
        break;

// Define some helper functions here, init model weights array/kvcache array,
// etc.
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
        return &model->weights;
    }

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model,
                                            int64_t *token_ids, size_t ntoken) {
        if (!model) {
            std::cerr
                << "[qwen2.cc:infer()] Model is null, cannot perform inference."
                << std::endl;
            return -1;
        }
        // Dummy implementation, just return the last token id + 1
        assert(false && "Inference not implemented yet.");
        return token_ids[ntoken - 1] + 1;
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