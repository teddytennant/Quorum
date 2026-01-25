# Roadmap

Quorum development is organized into phases, each building on the previous.

## Phase 1: Foundation (Current)

Core infrastructure for loading and inspecting GGUF models.

- [x] GGUF parser for headers, metadata, and tensor tables
- [x] CLI with `info` command
- [x] Tensor view helpers
- [x] Memory-mapped tensor access
- [x] Quantization format support (30+ formats)

## Phase 2: Inference

CPU-based inference implementation for correctness validation.

- [ ] Tokenizer loading and integration
- [ ] Token embedding lookup
- [ ] Attention mechanism (standard multi-head)
- [ ] Feed-forward network layers
- [ ] RMSNorm / LayerNorm
- [ ] RoPE positional encoding
- [ ] KV cache implementation
- [ ] Sampling strategies (greedy, top-k, top-p)
- [ ] Basic text generation loop

## Phase 3: Metal Acceleration

GPU acceleration using Apple's Metal framework.

- [ ] Metal compute pipeline setup
- [ ] Matrix multiplication kernels
- [ ] Optimized attention kernels
- [ ] Unified memory management
- [ ] Batch processing support
- [ ] Performance profiling tools

## Phase 4: Expert Offloading

Memory-efficient MoE inference through SSD caching.

- [ ] Expert identification and routing
- [ ] SSD-backed expert storage
- [ ] LRU cache for active experts
- [ ] Async expert prefetching
- [ ] Memory pressure monitoring
- [ ] Dynamic expert loading/unloading

## Phase 5: Advanced Features

Enhanced capabilities for production use.

- [ ] MLA (Multi-Head Latent Attention) support
- [ ] Speculative decoding
- [ ] Continuous batching
- [ ] Streaming output
- [ ] Model quantization tools
- [ ] Benchmarking suite

## Target Milestone

**Run GLM-4.7-Flash on M3 MacBook Air (8GB)**

This requires completing through Phase 4:
- Full inference pipeline (Phase 2)
- Metal acceleration (Phase 3)
- Expert offloading to handle 32 experts with 4 active (Phase 4)

## Non-Goals

Things Quorum is **not** trying to do:

- Support every model architecture (focused on MoE/GLM)
- Replace llama.cpp for general use
- Provide training capabilities
- Support non-Apple platforms for GPU acceleration

## Contributing

See [Contributing](./contributing.md) for how to help with roadmap items.
