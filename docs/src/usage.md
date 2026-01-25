# Usage

## Commands

### `info` - Inspect Model Information

Display comprehensive information about a GGUF model file:

```bash
./zig-out/bin/quorum info path/to/model.gguf
```

#### Example Output

```
Loading model: glm-4.7-flash-q4_k.gguf

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
  blk.0.attn_qkv.weight: q4_k [4096, 4608]
  blk.0.attn_output.weight: q4_k [4096, 4096]
  blk.0.ffn_norm.weight: f32 [4096]
  blk.0.ffn_gate.weight: q4_k [4096, 13696]
  blk.0.ffn_up.weight: q4_k [4096, 13696]
  blk.0.ffn_down.weight: q4_k [13696, 4096]
  blk.1.attn_norm.weight: f32 [4096]
  blk.1.attn_qkv.weight: q4_k [4096, 4608]
  ... and 281 more

=== Memory Estimate ===
Total tensor data: 4.72 GB

=== Tensor Lookup ===
token_embd.weight: offset 128 bytes, 620756992 elements, 357564416 storage bytes
```

#### Output Sections

| Section | Description |
|---------|-------------|
| **GGUF File Info** | Format version and counts |
| **Model Metadata** | Architecture, name, quantization info |
| **Architecture** | Model-specific parameters (context length, layers, heads, experts) |
| **Tensors** | List of tensors with data types and shapes |
| **Memory Estimate** | Total tensor data size in GB |
| **Tensor Lookup** | Detailed info for common embedding tensors |

### `help` - Show Help

```bash
./zig-out/bin/quorum help
# or
./zig-out/bin/quorum --help
./zig-out/bin/quorum -h
```

## Running via Zig Build

You can also run Quorum directly through the build system:

```bash
zig build run -- info model.gguf
```

This is useful during development as it automatically rebuilds if sources changed.
