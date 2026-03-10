# Cosmoe

Cosmoe is an inference engine for large mixture-of-experts (MoE) models on Apple Silicon machines that leverages dynamic offloading to disk in order to enable inference of models whose _total parameter size exceeds available memory_.

## Motivation

Mixture of Experts (MoE) models have become predominant in the field of LLM inference, due to their ability to scale to large parameter capacities while maintaining high throughput. Despite the dramatically reduced compute requirements, inference engines such as llama.cpp and vLLM require the entire model to be resident in VRAM [1], which imposes significant constraints and high barrier to entry for consumer and locally-hosted LLM deployments.

Apple Silicon machines have emerged as an unexpectedly popular platform for local LLM inference due to their low power consumption and unified memory architecture, which allows the CPU and GPU to share the same physical memory. This eliminates a significant memory transfer bottleneck which hampers expert offloading on discrete GPU systems--on Apple Silicon, parameter buffers loaded from disk into a shared Metal buffer are immediately accessible to the GPU. The offloading cost then reduces to the NVMe read bandwidth bottleneck, which we hide by interleaving I/O calls with GPU compute.

---

[1] llama.cpp allows for MoE weights to be [dynamically offloaded to RAM](https://github.com/ggml-org/llama.cpp/pull/15077). However, this still requires that the machine's sum RAM + VRAM be sufficient to hold the total size of the model.