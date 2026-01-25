# Architecture

Quorum follows a modular architecture designed for clarity and extensibility.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (main.zig)                       │
│                    Command parsing & output                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   GGUF Parser       │       │   Tensor Utilities  │
│   (gguf.zig)        │       │   (tensor.zig)      │
│                     │       │                     │
│ • Header parsing    │       │ • Element counting  │
│ • Metadata reading  │       │ • Storage calc      │
│ • Tensor info       │       │                     │
│ • Memory mapping    │       │                     │
└─────────────────────┘       └─────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                     GGUF File (mmap)                         │
│              Zero-copy access to tensor data                 │
└─────────────────────────────────────────────────────────────┘
```

## Modules

### `main.zig` - CLI Entry Point

The command-line interface handles:

- Argument parsing
- Command dispatch (`info`, `help`)
- Formatted output display
- Error handling and user feedback

### `gguf.zig` - GGUF Parser

The core GGUF file format parser. See [GGUF Parser](./gguf-parser.md) for details.

Key responsibilities:
- Header validation (magic number, version)
- Metadata extraction (all 13 value types)
- Tensor table parsing
- Memory-mapped tensor data access

### `tensor.zig` - Tensor Utilities

Helper utilities for tensor operations. See [Tensor Utilities](./tensor-utilities.md) for details.

Provides:
- Element count calculation
- Storage byte calculation (quantization-aware)

## Design Principles

### Zero Dependencies

Quorum uses only the Zig standard library. No external dependencies means:
- Simpler builds
- Smaller binaries
- Fewer supply chain concerns

### Memory Efficiency

- **Memory mapping**: Tensor data is accessed via `mmap` for zero-copy reads
- **Arena allocation**: String data uses arena allocators for batch deallocation
- **Lazy loading**: Only metadata and tensor info are parsed upfront

### Quantization Awareness

All tensor operations account for quantization:
- Block-based storage calculations
- Support for 30+ quantization formats
- Accurate memory estimates

### Platform Abstraction

The build system conditionally links platform-specific frameworks:
- **macOS**: Metal, Foundation, CoreGraphics
- **Linux**: Standard POSIX APIs

## Future Architecture

Planned modules for later phases:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Tokenizer     │  │   Inference     │  │   Expert Cache  │
│                 │  │                 │  │                 │
│ • Encode/Decode │  │ • Forward pass  │  │ • SSD streaming │
│ • Vocab loading │  │ • KV cache      │  │ • LRU eviction  │
└─────────────────┘  │ • Sampling      │  │ • Prefetching   │
                     └─────────────────┘  └─────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Metal Backend  │
                     │                 │
                     │ • GPU kernels   │
                     │ • Unified mem   │
                     └─────────────────┘
```
