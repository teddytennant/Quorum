# Quorum

Quorum is a Zig-based inference engine designed for running Mixture-of-Experts (MoE) models on memory-constrained Apple Silicon devices. The primary goal is to run GLM-4.7-Flash on an M3 MacBook Air with 8GB of Unified Memory through int4/int8 quantization and SSD-backed expert caching.

## Features

- **GGUF Format Support**: Complete parser for GGUF v2/v3 files including headers, metadata, and tensor tables
- **Quantization Aware**: Supports 30+ quantization formats (q2_k through q8_k, iq variants, and more)
- **Memory Efficient**: Uses memory-mapped I/O for zero-copy tensor access
- **Apple Silicon Ready**: Build system configured for Metal framework integration

## Requirements

- Zig 0.15.x or later
- macOS (for Metal acceleration) or Linux

## Installation

Clone the repository and build:

```sh
git clone https://github.com/teddytennant/quorum.git
cd quorum
zig build
```

The executable will be placed in `zig-out/bin/quorum`.

## Usage

### Inspect Model Information

Display comprehensive information about a GGUF model file:

```sh
./zig-out/bin/quorum info path/to/model.gguf
```

Example output:

```
Loading model: model.gguf

=== GGUF File Info ===
Version: 3
Tensor count: 291
Metadata KV count: 25

=== Model Metadata ===
general.architecture: glm4
general.name: GLM-4.7-Flash
general.quantization_version: 2

=== Architecture: glm4 ===
glm4.context_length: 131072
glm4.embedding_length: 4096
glm4.block_count: 40
glm4.attention.head_count: 32
glm4.expert_count: 32
glm4.expert_used_count: 4

=== Tensors (291 total) ===
First 10 tensors:
  token_embd.weight: q4_k [151552, 4096]
  blk.0.attn_norm.weight: f32 [4096]
  ...

=== Memory Estimate ===
Total tensor data: 4.72 GB
```

### Help

```sh
./zig-out/bin/quorum help
```

## Project Structure

```
quorum/
├── build.zig          # Zig build configuration
├── src/
│   ├── main.zig       # CLI entry point and commands
│   ├── gguf.zig       # GGUF file format parser
│   └── tensor.zig     # Tensor view utilities
├── README.md
└── LICENSE
```

## Architecture

### GGUF Parser (`src/gguf.zig`)

The GGUF parser provides complete support for the GGUF file format:

- **Header Parsing**: Validates magic number (`GGUF`), version (2-3), tensor/metadata counts
- **Metadata Extraction**: Supports all 13 GGUF metadata value types
- **Tensor Information**: Parses tensor names, dimensions, data types, shapes, and byte offsets
- **Memory Mapping**: Uses POSIX mmap for efficient tensor data access

#### Supported Quantization Formats

| Format | Block Size | Elements/Block |
|--------|-----------|----------------|
| f32    | 4         | 1              |
| f16    | 2         | 1              |
| bf16   | 2         | 1              |
| q4_0   | 18        | 32             |
| q4_1   | 20        | 32             |
| q5_0   | 22        | 32             |
| q5_1   | 24        | 32             |
| q8_0   | 34        | 32             |
| q8_1   | 36        | 32             |
| q2_k   | 84        | 256            |
| q3_k   | 110       | 256            |
| q4_k   | 144       | 256            |
| q5_k   | 176       | 256            |
| q6_k   | 210       | 256            |
| q8_k   | 292       | 256            |

Additional formats: iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq4_nl, iq3_s, iq2_s, iq4_xs, i8, i16, i32, i64, f64

### Tensor View (`src/tensor.zig`)

Provides utilities for working with tensor data:

- `elementCount()`: Calculate total number of elements
- `storageBytes()`: Calculate actual bytes used (accounting for quantization)

## Development

### Running Tests

```sh
zig build test
```

### Run with Arguments

```sh
zig build run -- info model.gguf
```

## Roadmap

### Phase 1: Foundation (Current)
- [x] GGUF parser for headers, metadata, and tensor tables
- [x] CLI with `info` command
- [x] Tensor view helpers

### Phase 2: Inference
- [ ] CPU reference forward pass
- [ ] Tokenizer integration
- [ ] Basic text generation

### Phase 3: Acceleration
- [ ] Metal compute backend
- [ ] GPU kernel optimization
- [ ] Unified memory management

### Phase 4: Expert Offloading
- [ ] Expert cache implementation
- [ ] SSD streaming for expert weights
- [ ] Dynamic expert loading/unloading

### Phase 5: Advanced Features
- [ ] MLA (Multi-Head Latent Attention) support
- [ ] Speculative decoding
- [ ] Continuous batching

## License

MIT License - see [LICENSE](LICENSE) for details.