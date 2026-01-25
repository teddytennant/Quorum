# GGUF Parser

The GGUF parser (`src/gguf.zig`) provides complete support for the GGUF file format used by llama.cpp and related projects.

## Overview

GGUF (GGML Universal Format) is a binary format for storing LLM weights and metadata. Quorum's parser handles:

- **GGUF v2 and v3** file formats
- **All 13 metadata value types**
- **30+ tensor quantization formats**
- **Memory-mapped tensor data access**

## Core Types

### `GGUFFile`

The main file handle that provides access to all model data.

```zig
pub const GGUFFile = struct {
    header: Header,
    metadata: std.StringHashMap(MetaValue),
    tensors: []TensorInfo,
    // ...
};
```

#### Opening a File

```zig
var model = try gguf.GGUFFile.open(allocator, "model.gguf");
defer model.close();
```

#### Accessing Metadata

```zig
// Get string metadata
const arch = model.getMetadataString("general.architecture");

// Get integer metadata
const ctx_len = model.getMetadataU32("llama.context_length");

// Direct access to all metadata
if (model.metadata.get("general.name")) |value| {
    switch (value) {
        .string => |s| std.debug.print("Name: {s}\n", .{s}),
        else => {},
    }
}
```

#### Working with Tensors

```zig
// Find a tensor by name
if (model.getTensor("token_embd.weight")) |tensor| {
    std.debug.print("Shape: {any}\n", .{tensor.shape[0..tensor.n_dims]});
    std.debug.print("Type: {s}\n", .{@tagName(tensor.dtype)});

    // Get raw tensor data (memory-mapped)
    if (model.getTensorData(tensor)) |data| {
        // data is a []const u8 slice
    }
}

// Iterate all tensors
for (model.tensors) |tensor| {
    std.debug.print("{s}: {s}\n", .{tensor.name, @tagName(tensor.dtype)});
}
```

### `Header`

```zig
pub const Header = struct {
    magic: u32,           // 0x46554747 ("GGUF")
    version: u32,         // 2 or 3
    tensor_count: u64,
    metadata_kv_count: u64,
};
```

### `TensorInfo`

```zig
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,          // Number of dimensions (1-4)
    shape: [4]u64,        // Dimension sizes
    dtype: GGMLType,      // Quantization format
    offset: u64,          // Offset in tensor data section
    size_bytes: u64,      // Calculated storage size
};
```

### `MetaValue`

A tagged union supporting all GGUF metadata types:

```zig
pub const MetaValue = union(MetaValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    uint64: u64,
    int64: i64,
    float32: f32,
    float64: f64,
    bool_: bool,
    string: []const u8,
    array: []const MetaValue,
};
```

### `GGMLType`

Enum representing all supported quantization formats. See [Quantization Formats](./quantization.md) for the complete list.

Key methods:

```zig
// Get block size in bytes
const block_bytes = GGMLType.q4_k.blockSize();  // 144

// Get elements per block
const block_elems = GGMLType.q4_k.blockElements();  // 256
```

## File Format

### Structure

```
┌─────────────────────────────────────┐
│           Header (24 bytes)          │
│  magic (4) | version (4) | counts   │
├─────────────────────────────────────┤
│         Metadata Key-Values          │
│  Repeated metadata_kv_count times    │
├─────────────────────────────────────┤
│         Tensor Information           │
│  Repeated tensor_count times         │
├─────────────────────────────────────┤
│      Padding (to alignment)          │
├─────────────────────────────────────┤
│           Tensor Data                │
│      (memory-mapped access)          │
└─────────────────────────────────────┘
```

### Alignment

Tensor data is aligned to a configurable boundary (default: 32 bytes). The alignment can be overridden via the `general.alignment` metadata key.

## Error Handling

The parser returns descriptive errors:

| Error | Cause |
|-------|-------|
| `InvalidMagic` | File doesn't start with "GGUF" |
| `UnsupportedVersion` | Version not 2 or 3 |
| Reader errors | I/O or allocation failures |

## Memory Management

- **Arena allocator**: All string data (tensor names, metadata keys/values) uses an arena for efficient cleanup
- **Memory mapping**: Tensor data accessed via `mmap` for zero-copy reads
- **Explicit cleanup**: Call `model.close()` to release all resources
