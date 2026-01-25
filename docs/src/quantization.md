# Quantization Formats

Quorum supports all major quantization formats used in the GGML ecosystem. This page documents each format's characteristics.

## Overview

Quantization reduces model size by storing weights in lower precision formats. GGML uses **block-based quantization** where weights are grouped into blocks, each with shared scale factors.

## Format Categories

### Floating Point

Full precision formats with no quantization.

| Format | Bits | Block Size | Elements/Block | Description |
|--------|------|------------|----------------|-------------|
| `f32`  | 32   | 4 bytes    | 1              | Single precision float |
| `f16`  | 16   | 2 bytes    | 1              | Half precision float |
| `bf16` | 16   | 2 bytes    | 1              | Brain floating point |
| `f64`  | 64   | 8 bytes    | 1              | Double precision float |

### Integer

Fixed-point integer formats.

| Format | Bits | Block Size | Elements/Block | Description |
|--------|------|------------|----------------|-------------|
| `i8`   | 8    | 1 byte     | 1              | Signed 8-bit integer |
| `i16`  | 16   | 2 bytes    | 1              | Signed 16-bit integer |
| `i32`  | 32   | 4 bytes    | 1              | Signed 32-bit integer |
| `i64`  | 64   | 8 bytes    | 1              | Signed 64-bit integer |

### Legacy Quantization (Q4/Q5/Q8)

Original GGML quantization formats with 32 elements per block.

| Format | Block Size | Elements/Block | Bits/Weight | Description |
|--------|------------|----------------|-------------|-------------|
| `q4_0` | 18 bytes   | 32             | 4.5         | 4-bit, single scale |
| `q4_1` | 20 bytes   | 32             | 5.0         | 4-bit, scale + min |
| `q5_0` | 22 bytes   | 32             | 5.5         | 5-bit, single scale |
| `q5_1` | 24 bytes   | 32             | 6.0         | 5-bit, scale + min |
| `q8_0` | 34 bytes   | 32             | 8.5         | 8-bit, single scale |
| `q8_1` | 36 bytes   | 32             | 9.0         | 8-bit, scale + min |

### K-Quants

Improved quantization with 256 elements per block and better quality.

| Format | Block Size | Elements/Block | Bits/Weight | Description |
|--------|------------|----------------|-------------|-------------|
| `q2_k` | 84 bytes   | 256            | 2.6         | 2-bit with k-quant |
| `q3_k` | 110 bytes  | 256            | 3.4         | 3-bit with k-quant |
| `q4_k` | 144 bytes  | 256            | 4.5         | 4-bit with k-quant |
| `q5_k` | 176 bytes  | 256            | 5.5         | 5-bit with k-quant |
| `q6_k` | 210 bytes  | 256            | 6.6         | 6-bit with k-quant |
| `q8_k` | 292 bytes  | 256            | 9.1         | 8-bit with k-quant |

### I-Quants (Importance Matrix)

Advanced formats using importance-weighted quantization for better quality at extreme compression.

| Format    | Description |
|-----------|-------------|
| `iq1_s`   | 1-bit importance quantization |
| `iq2_xxs` | 2-bit extra extra small |
| `iq2_xs`  | 2-bit extra small |
| `iq2_s`   | 2-bit small |
| `iq3_xxs` | 3-bit extra extra small |
| `iq3_s`   | 3-bit small |
| `iq4_nl`  | 4-bit non-linear |
| `iq4_xs`  | 4-bit extra small |

## Choosing a Format

### Quality vs Size Tradeoff

```
Quality ◄──────────────────────────────────────► Size
   │                                               │
   │  f32  f16  q8_k  q6_k  q5_k  q4_k  q3_k  q2_k │
   │   │    │    │     │     │     │     │     │   │
   │  100% 50%  ~28%  ~21%  ~17%  ~14%  ~11%  ~8% │
   │                                               │
   └───────────────────────────────────────────────┘
                    Size (% of f32)
```

### Recommendations

| Use Case | Recommended Format |
|----------|-------------------|
| Maximum quality | `f16` or `q8_k` |
| Good quality, moderate size | `q5_k` or `q6_k` |
| Balanced | `q4_k` (most popular) |
| Memory constrained | `q3_k` or `q2_k` |
| Extreme compression | `iq2_xs` or `iq1_s` |

### Memory Requirements

For a 7B parameter model:

| Format | Approximate Size |
|--------|-----------------|
| f32    | 28 GB |
| f16    | 14 GB |
| q8_k   | ~8 GB |
| q4_k   | ~4 GB |
| q2_k   | ~2.5 GB |

## Code Reference

The `GGMLType` enum in `gguf.zig` defines all formats:

```zig
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // ... see source for complete list

    pub fn blockSize(self: GGMLType) usize { ... }
    pub fn blockElements(self: GGMLType) usize { ... }
};
```
