const std = @import("std");
const gguf = @import("gguf.zig");

pub const TensorView = struct {
    dtype: gguf.GGMLType,
    n_dims: u32,
    shape: [4]u64,
    data: []const u8,

    pub fn elementCount(self: TensorView) error{Overflow}!u64 {
        var count: u64 = 1;
        for (self.shape[0..self.n_dims]) |dim| {
            count = try std.math.mul(u64, count, dim);
        }
        return count;
    }

    pub fn storageBytes(self: TensorView) error{Overflow}!u64 {
        const elements = try self.elementCount();
        const block_elements = self.dtype.blockElements();
        const block_size: u64 = @intCast(self.dtype.blockSize());
        const n_blocks = (elements + block_elements - 1) / block_elements;
        return std.math.mul(u64, n_blocks, block_size);
    }
};

test "element count 1D tensor" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 1,
        .shape = [4]u64{ 128, 1, 1, 1 },
        .data = &[_]u8{},
    };
    try std.testing.expectEqual(@as(u64, 128), try view.elementCount());
}

test "element count 2D tensor" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 2,
        .shape = [4]u64{ 128, 256, 1, 1 },
        .data = &[_]u8{},
    };
    try std.testing.expectEqual(@as(u64, 128 * 256), try view.elementCount());
}

test "element count 3D tensor" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 3,
        .shape = [4]u64{ 64, 32, 16, 1 },
        .data = &[_]u8{},
    };
    try std.testing.expectEqual(@as(u64, 64 * 32 * 16), try view.elementCount());
}

test "element count 4D tensor" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 4,
        .shape = [4]u64{ 2, 3, 4, 5 },
        .data = &[_]u8{},
    };
    try std.testing.expectEqual(@as(u64, 2 * 3 * 4 * 5), try view.elementCount());
}

test "element count overflow is caught" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 2,
        .shape = [4]u64{ std.math.maxInt(u64), 2, 1, 1 },
        .data = &[_]u8{},
    };
    try std.testing.expectError(error.Overflow, view.elementCount());
}

test "element count overflow with large 3D shape" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 3,
        .shape = [4]u64{ 1 << 32, 1 << 32, 2, 1 },
        .data = &[_]u8{},
    };
    try std.testing.expectError(error.Overflow, view.elementCount());
}

test "storage bytes for f32 tensor" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 1,
        .shape = [4]u64{ 100, 1, 1, 1 },
        .data = &[_]u8{},
    };
    // f32: blockSize=4, blockElements=1, so 100 elements => 100 blocks * 4 bytes = 400
    try std.testing.expectEqual(@as(u64, 400), try view.storageBytes());
}

test "storage bytes for f16 tensor" {
    const view = TensorView{
        .dtype = .f16,
        .n_dims = 2,
        .shape = [4]u64{ 64, 64, 1, 1 },
        .data = &[_]u8{},
    };
    // f16: blockSize=2, blockElements=1, so 4096 elements => 4096 * 2 = 8192
    try std.testing.expectEqual(@as(u64, 8192), try view.storageBytes());
}

test "storage bytes for q4_0 tensor" {
    const view = TensorView{
        .dtype = .q4_0,
        .n_dims = 1,
        .shape = [4]u64{ 256, 1, 1, 1 },
        .data = &[_]u8{},
    };
    // q4_0: blockSize=18, blockElements=32, so 256 elements => 8 blocks * 18 = 144
    try std.testing.expectEqual(@as(u64, 144), try view.storageBytes());
}

test "storage bytes for q8_0 tensor" {
    const view = TensorView{
        .dtype = .q8_0,
        .n_dims = 1,
        .shape = [4]u64{ 128, 1, 1, 1 },
        .data = &[_]u8{},
    };
    // q8_0: blockSize=34, blockElements=32, so 128 elements => 4 blocks * 34 = 136
    try std.testing.expectEqual(@as(u64, 136), try view.storageBytes());
}

test "storage bytes overflow is caught" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 2,
        .shape = [4]u64{ std.math.maxInt(u64), 2, 1, 1 },
        .data = &[_]u8{},
    };
    try std.testing.expectError(error.Overflow, view.storageBytes());
}

test "element count 0D tensor (scalar)" {
    const view = TensorView{
        .dtype = .f32,
        .n_dims = 0,
        .shape = [4]u64{ 1, 1, 1, 1 },
        .data = &[_]u8{},
    };
    // With 0 dims, the loop doesn't execute, so count remains 1
    try std.testing.expectEqual(@as(u64, 1), try view.elementCount());
}
