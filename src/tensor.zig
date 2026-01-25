const std = @import("std");
const gguf = @import("gguf.zig");

pub const TensorView = struct {
    dtype: gguf.GGMLType,
    n_dims: u32,
    shape: [4]u64,
    data: []const u8,

    pub fn elementCount(self: TensorView) u64 {
        var count: u64 = 1;
        for (self.shape[0..self.n_dims]) |dim| {
            count *= dim;
        }
        return count;
    }

    pub fn storageBytes(self: TensorView) u64 {
        const elements = self.elementCount();
        const block_elements = self.dtype.blockElements();
        const block_size = self.dtype.blockSize();
        const n_blocks = (elements + block_elements - 1) / block_elements;
        return n_blocks * block_size;
    }
};
