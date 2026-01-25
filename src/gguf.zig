const std = @import("std");
const fs = std.fs;
const mem = std.mem;
const posix = std.posix;
const Allocator = mem.Allocator;
const Io = std.Io;

/// GGUF magic number: "GGUF" in little-endian
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

/// Supported GGUF versions
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;

/// Default alignment for tensor data
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Read buffer size
const READ_BUFFER_SIZE: usize = 8192;

/// GGUF metadata value types
pub const MetaValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// GGUF tensor data types (quantization formats)
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    bf16 = 29,
    _,

    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .f64 => 8,
            else => 1,
        };
    }

    pub fn blockElements(self: GGMLType) usize {
        return switch (self) {
            .f32, .f16, .bf16, .i8, .i16, .i32, .i64, .f64 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            else => 1,
        };
    }
};

/// Metadata value union
pub const MetaValue = union(MetaValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: []const MetaValue,
    uint64: u64,
    int64: i64,
    float64: f64,
};

/// GGUF file header
pub const Header = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

/// Tensor information
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    shape: [4]u64,
    dtype: GGMLType,
    offset: u64,
    size_bytes: u64,
};

/// Main GGUF file handle
pub const GGUFFile = struct {
    allocator: Allocator,
    file: fs.File,
    header: Header,
    metadata: std.StringHashMap(MetaValue),
    tensors: []TensorInfo,
    tensor_data_offset: u64,
    alignment: usize,
    string_arena: std.heap.ArenaAllocator,
    read_buffer: [READ_BUFFER_SIZE]u8,
    file_reader: fs.File.Reader,
    mapped_data: ?[]align(std.heap.page_size_min) const u8,

    const ReaderError = Io.Reader.Error;
    const ParseError = ReaderError || Allocator.Error;

    pub fn open(allocator: Allocator, path: []const u8) !GGUFFile {
        const file = try fs.cwd().openFile(path, .{});
        errdefer file.close();

        var self = GGUFFile{
            .allocator = allocator,
            .file = file,
            .header = undefined,
            .metadata = std.StringHashMap(MetaValue).init(allocator),
            .tensors = &[_]TensorInfo{},
            .tensor_data_offset = 0,
            .alignment = GGUF_DEFAULT_ALIGNMENT,
            .string_arena = std.heap.ArenaAllocator.init(allocator),
            .read_buffer = undefined,
            .file_reader = undefined,
            .mapped_data = null,
        };
        errdefer self.string_arena.deinit();

        self.file_reader = self.file.reader(&self.read_buffer);

        try self.parseHeader();
        try self.parseMetadata();
        try self.parseTensors();
        try self.mapTensorData();

        return self;
    }

    pub fn dataOffset(self: *const GGUFFile, tensor: *const TensorInfo) u64 {
        return self.tensor_data_offset + tensor.offset;
    }

    pub fn close(self: *GGUFFile) void {
        if (self.mapped_data) |mapped| {
            posix.munmap(mapped);
        }
        self.file.close();
        self.metadata.deinit();
        if (self.tensors.len > 0) {
            self.allocator.free(self.tensors);
        }
        self.string_arena.deinit();
    }

    // Helper functions for reading integers using Zig 0.15 API
    fn readU32(reader: *Io.Reader) ReaderError!u32 {
        const bytes = try reader.takeArray(4);
        return mem.readInt(u32, bytes, .little);
    }

    fn readU64(reader: *Io.Reader) ReaderError!u64 {
        const bytes = try reader.takeArray(8);
        return mem.readInt(u64, bytes, .little);
    }

    fn readI8(reader: *Io.Reader) ReaderError!i8 {
        const bytes = try reader.takeArray(1);
        return mem.readInt(i8, bytes, .little);
    }

    fn readU8(reader: *Io.Reader) ReaderError!u8 {
        const bytes = try reader.takeArray(1);
        return bytes[0];
    }

    fn readI16(reader: *Io.Reader) ReaderError!i16 {
        const bytes = try reader.takeArray(2);
        return mem.readInt(i16, bytes, .little);
    }

    fn readU16(reader: *Io.Reader) ReaderError!u16 {
        const bytes = try reader.takeArray(2);
        return mem.readInt(u16, bytes, .little);
    }

    fn readI32(reader: *Io.Reader) ReaderError!i32 {
        const bytes = try reader.takeArray(4);
        return mem.readInt(i32, bytes, .little);
    }

    fn readI64(reader: *Io.Reader) ReaderError!i64 {
        const bytes = try reader.takeArray(8);
        return mem.readInt(i64, bytes, .little);
    }

    fn readF32(reader: *Io.Reader) ReaderError!f32 {
        const bytes = try reader.takeArray(4);
        return @bitCast(mem.readInt(u32, bytes, .little));
    }

    fn readF64(reader: *Io.Reader) ReaderError!f64 {
        const bytes = try reader.takeArray(8);
        return @bitCast(mem.readInt(u64, bytes, .little));
    }

    fn readBytes(reader: *Io.Reader, dest: []u8) ReaderError!void {
        try reader.readSliceAll(dest);
    }

    fn skipBytes(reader: *Io.Reader, count: u64) ReaderError!void {
        var remaining = count;
        while (remaining > 0) {
            const to_skip = @min(remaining, 4096);
            _ = try reader.take(@intCast(to_skip));
            remaining -= to_skip;
        }
    }

    fn parseHeader(self: *GGUFFile) !void {
        const reader = &self.file_reader.interface;

        self.header.magic = try readU32(reader);
        if (self.header.magic != GGUF_MAGIC) {
            return error.InvalidMagic;
        }

        self.header.version = try readU32(reader);
        if (self.header.version < GGUF_VERSION_MIN or self.header.version > GGUF_VERSION_MAX) {
            return error.UnsupportedVersion;
        }

        self.header.tensor_count = try readU64(reader);
        self.header.metadata_kv_count = try readU64(reader);
    }

    fn parseMetadata(self: *GGUFFile) !void {
        const reader = &self.file_reader.interface;
        const arena = self.string_arena.allocator();

        var i: u64 = 0;
        while (i < self.header.metadata_kv_count) : (i += 1) {
            const key_len = try readU64(reader);
            const key = try arena.alloc(u8, key_len);
            try readBytes(reader, key);

            const value_type_int = try readU32(reader);
            const value_type: MetaValueType = @enumFromInt(value_type_int);

            const value = try readMetaValue(reader, value_type, arena);

            if (mem.eql(u8, key, "general.alignment")) {
                if (value == .uint32) {
                    self.alignment = value.uint32;
                }
            }

            try self.metadata.put(key, value);
        }
    }

    fn readMetaValue(reader: *Io.Reader, value_type: MetaValueType, arena: Allocator) ParseError!MetaValue {
        return switch (value_type) {
            .uint8 => MetaValue{ .uint8 = try readU8(reader) },
            .int8 => MetaValue{ .int8 = try readI8(reader) },
            .uint16 => MetaValue{ .uint16 = try readU16(reader) },
            .int16 => MetaValue{ .int16 = try readI16(reader) },
            .uint32 => MetaValue{ .uint32 = try readU32(reader) },
            .int32 => MetaValue{ .int32 = try readI32(reader) },
            .uint64 => MetaValue{ .uint64 = try readU64(reader) },
            .int64 => MetaValue{ .int64 = try readI64(reader) },
            .float32 => MetaValue{ .float32 = try readF32(reader) },
            .float64 => MetaValue{ .float64 = try readF64(reader) },
            .bool_ => MetaValue{ .bool_ = (try readU8(reader)) != 0 },
            .string => blk: {
                const len = try readU64(reader);
                const str = try arena.alloc(u8, len);
                try readBytes(reader, str);
                break :blk MetaValue{ .string = str };
            },
            .array => blk: {
                const elem_type_int = try readU32(reader);
                const elem_type: MetaValueType = @enumFromInt(elem_type_int);
                const len = try readU64(reader);

                var j: u64 = 0;
                while (j < len) : (j += 1) {
                    try skipMetaValue(reader, elem_type);
                }
                break :blk MetaValue{ .array = &[_]MetaValue{} };
            },
        };
    }

    fn skipMetaValue(reader: *Io.Reader, value_type: MetaValueType) ParseError!void {
        switch (value_type) {
            .uint8, .int8, .bool_ => _ = try readU8(reader),
            .uint16, .int16 => _ = try readU16(reader),
            .uint32, .int32, .float32 => _ = try readU32(reader),
            .uint64, .int64, .float64 => _ = try readU64(reader),
            .string => {
                const len = try readU64(reader);
                try skipBytes(reader, len);
            },
            .array => {
                const elem_type_int = try readU32(reader);
                const elem_type: MetaValueType = @enumFromInt(elem_type_int);
                const len = try readU64(reader);
                var j: u64 = 0;
                while (j < len) : (j += 1) {
                    try skipMetaValue(reader, elem_type);
                }
            },
        }
    }

    fn parseTensors(self: *GGUFFile) !void {
        const reader = &self.file_reader.interface;
        const arena = self.string_arena.allocator();

        if (self.header.tensor_count == 0) {
            return;
        }

        self.tensors = try self.allocator.alloc(TensorInfo, self.header.tensor_count);
        errdefer self.allocator.free(self.tensors);

        for (self.tensors) |*tensor| {
            const name_len = try readU64(reader);
            const name = try arena.alloc(u8, name_len);
            try readBytes(reader, name);
            tensor.name = name;

            tensor.n_dims = try readU32(reader);
            tensor.shape = [4]u64{ 1, 1, 1, 1 };
            for (0..tensor.n_dims) |d| {
                tensor.shape[d] = try readU64(reader);
            }

            const dtype_int = try readU32(reader);
            tensor.dtype = @enumFromInt(dtype_int);
            tensor.offset = try readU64(reader);

            var n_elements: u64 = 1;
            for (tensor.shape[0..tensor.n_dims]) |dim| {
                n_elements *= dim;
            }
            const block_size = tensor.dtype.blockSize();
            const block_elements = tensor.dtype.blockElements();
            const n_blocks = (n_elements + block_elements - 1) / block_elements;
            tensor.size_bytes = n_blocks * block_size;
        }

        const current_pos = try self.file.getPos();
        self.tensor_data_offset = alignUp(current_pos, self.alignment);
    }

    fn mapTensorData(self: *GGUFFile) !void {
        const file_len = try self.file.getEndPos();
        const mapped = try posix.mmap(
            null,
            @intCast(file_len),
            posix.PROT.READ,
            .{ .TYPE = .SHARED },
            self.file.handle,
            0,
        );
        self.mapped_data = mapped;
    }

    pub fn getMetadataString(self: *GGUFFile, key: []const u8) ?[]const u8 {
        if (self.metadata.get(key)) |value| {
            if (value == .string) {
                return value.string;
            }
        }
        return null;
    }

    pub fn getMetadataU32(self: *GGUFFile, key: []const u8) ?u32 {
        if (self.metadata.get(key)) |value| {
            return switch (value) {
                .uint32 => value.uint32,
                .uint64 => @intCast(value.uint64),
                .int32 => @intCast(value.int32),
                else => null,
            };
        }
        return null;
    }

    pub fn getTensor(self: *GGUFFile, name: []const u8) ?*const TensorInfo {
        for (self.tensors) |*tensor| {
            if (mem.eql(u8, tensor.name, name)) {
                return tensor;
            }
        }
        return null;
    }

    pub fn getTensorData(self: *const GGUFFile, tensor: *const TensorInfo) ?[]const u8 {
        const mapped = self.mapped_data orelse return null;
        const start = self.dataOffset(tensor);
        const end = start + tensor.size_bytes;
        if (end > mapped.len) {
            return null;
        }
        return mapped[@intCast(start)..@intCast(end)];
    }
};

fn alignUp(value: u64, alignment: usize) u64 {
    const align_u64: u64 = @intCast(alignment);
    return (value + align_u64 - 1) & ~(align_u64 - 1);
}

test "GGUF magic constant" {
    try std.testing.expectEqual(@as(u32, 0x46554747), GGUF_MAGIC);
}

test "GGMLType block sizes" {
    try std.testing.expectEqual(@as(usize, 4), GGMLType.f32.blockSize());
    try std.testing.expectEqual(@as(usize, 2), GGMLType.f16.blockSize());
    try std.testing.expectEqual(@as(usize, 34), GGMLType.q8_0.blockSize());
    try std.testing.expectEqual(@as(usize, 144), GGMLType.q4_k.blockSize());
}

test "align up" {
    try std.testing.expectEqual(@as(u64, 32), alignUp(1, 32));
    try std.testing.expectEqual(@as(u64, 32), alignUp(32, 32));
    try std.testing.expectEqual(@as(u64, 64), alignUp(33, 32));
}
