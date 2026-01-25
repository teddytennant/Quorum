# Quorum

**Quorum** is a Zig-based inference engine designed for running Mixture-of-Experts (MoE) models on memory-constrained Apple Silicon devices.

## Goal

The primary goal is to run **GLM-4.7-Flash** on an M3 MacBook Air with 8GB of Unified Memory through:

- **Int4/Int8 quantization** for model compression
- **SSD-backed expert caching** for memory-efficient expert offloading
- **Apple Silicon (Metal) acceleration** for GPU compute

## Features

- **GGUF Format Support**: Complete parser for GGUF v2/v3 files including headers, metadata, and tensor tables
- **Quantization Aware**: Supports 30+ quantization formats (q2_k through q8_k, iq variants, and more)
- **Memory Efficient**: Uses memory-mapped I/O for zero-copy tensor access
- **Apple Silicon Ready**: Build system configured for Metal framework integration

## Current Status

Quorum is in **Phase 1** of development. The foundation work includes:

- GGUF parser for header, metadata, and tensor tables
- CLI with `info` command to inspect model files
- Tensor view helper for basic size calculations

See the [Roadmap](./roadmap.md) for planned features.

## Quick Start

```bash
# Build
zig build

# Inspect a model
./zig-out/bin/quorum info path/to/model.gguf
```

## License

MIT License
