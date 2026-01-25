# Quorum

Quorum is a Zig-based inference engine focused on MoE models with expert offloading and Apple Silicon acceleration. The initial goal is to run GLM-4.7-Flash on an M3 MacBook Air with 8GB of Unified Memory by using int4/int8 quantization and SSD-backed expert caching.

## Status

Phase 1 foundation work is in progress:

- GGUF parser for header, metadata, and tensor tables
- CLI with `info` command to inspect model files
- Tensor view helper for basic size calculations

## Build

Requires Zig 0.15.x.

```sh
zig build
```

## Usage

```sh
./zig-out/bin/quorum info path/to/model.gguf
```

## Roadmap

- CPU reference forward pass
- Metal compute backend
- Expert cache and SSD streaming
- MLA attention support
