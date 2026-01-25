# Tensor Utilities

The tensor utilities module (`src/tensor.zig`) provides helper functions for working with tensor data.

## TensorView

A lightweight view into tensor data that provides utility methods without owning the data.

```zig
pub const TensorView = struct {
    dtype: gguf.GGMLType,
    n_dims: u32,
    shape: [4]u64,
    data: []const u8,
};
```

### Creating a TensorView

```zig
const tensor_mod = @import("tensor.zig");

// From a GGUFFile tensor
if (model.getTensor("token_embd.weight")) |info| {
    if (model.getTensorData(info)) |data| {
        const view = tensor_mod.TensorView{
            .dtype = info.dtype,
            .n_dims = info.n_dims,
            .shape = info.shape,
            .data = data,
        };

        std.debug.print("Elements: {}\n", .{view.elementCount()});
        std.debug.print("Storage: {} bytes\n", .{view.storageBytes()});
    }
}
```

## Methods

### `elementCount()`

Returns the total number of logical elements in the tensor.

```zig
pub fn elementCount(self: TensorView) u64
```

Calculates the product of all dimensions:

```zig
// For a tensor with shape [4096, 4096]
// elementCount() returns 16,777,216
```

### `storageBytes()`

Returns the actual storage size in bytes, accounting for quantization.

```zig
pub fn storageBytes(self: TensorView) u64
```

This correctly handles block-based quantization formats:

```zig
// For a q4_k tensor with 16,777,216 elements:
// - Block size: 144 bytes
// - Elements per block: 256
// - Number of blocks: 65,536
// - Storage: 65,536 * 144 = 9,437,184 bytes

// Compare to f32:
// - Storage: 16,777,216 * 4 = 67,108,864 bytes
// - Compression ratio: ~7x
```

## Quantization-Aware Calculations

The storage calculation accounts for the block structure of quantized formats:

```zig
pub fn storageBytes(self: TensorView) u64 {
    const elements = self.elementCount();
    const block_elements = self.dtype.blockElements();
    const block_size = self.dtype.blockSize();
    const n_blocks = (elements + block_elements - 1) / block_elements;
    return n_blocks * block_size;
}
```

### Block Sizes by Format

| Format | Block Size (bytes) | Elements/Block | Bits/Element |
|--------|-------------------|----------------|--------------|
| f32    | 4                 | 1              | 32           |
| f16    | 2                 | 1              | 16           |
| q8_0   | 34                | 32             | 8.5          |
| q4_k   | 144               | 256            | 4.5          |
| q2_k   | 84                | 256            | 2.6          |

See [Quantization Formats](./quantization.md) for the complete reference.

## Usage Example

```zig
const std = @import("std");
const gguf = @import("gguf.zig");
const tensor_mod = @import("tensor.zig");

pub fn analyzeModel(model: *gguf.GGUFFile) void {
    var total_elements: u64 = 0;
    var total_storage: u64 = 0;

    for (model.tensors) |info| {
        if (model.getTensorData(&info)) |data| {
            const view = tensor_mod.TensorView{
                .dtype = info.dtype,
                .n_dims = info.n_dims,
                .shape = info.shape,
                .data = data,
            };

            total_elements += view.elementCount();
            total_storage += view.storageBytes();
        }
    }

    const theoretical_f32 = total_elements * 4;
    const compression = @as(f64, @floatFromInt(theoretical_f32)) /
                        @as(f64, @floatFromInt(total_storage));

    std.debug.print("Total elements: {}\n", .{total_elements});
    std.debug.print("Storage: {} bytes\n", .{total_storage});
    std.debug.print("Compression vs f32: {d:.1}x\n", .{compression});
}
```
