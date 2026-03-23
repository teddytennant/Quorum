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

/// Maximum number of metadata key-value pairs (prevents DoS from malformed files)
const MAX_METADATA_KV_COUNT: u64 = 100_000;

/// Maximum metadata key/string length in bytes (1 MB)
const MAX_KEY_LEN: u64 = 1_048_576;

/// Maximum number of tensor dimensions
const MAX_TENSOR_DIMS: u32 = 4;

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
            .iq2_xxs => 66,
            .iq2_xs => 74,
            .iq3_xxs => 98,
            .iq1_s => 50,
            .iq4_nl => 34,
            .iq3_s => 110,
            .iq2_s => 82,
            .iq4_xs => 36,
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
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl, .iq4_xs => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq3_s, .iq2_s => 256,
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
    tensor_map: std.StringHashMap(usize),
    tensor_data_offset: u64,
    alignment: usize,
    string_arena: std.heap.ArenaAllocator,
    read_buffer: [READ_BUFFER_SIZE]u8,
    file_reader: fs.File.Reader,
    mapped_data: ?[]align(std.heap.page_size_min) const u8,

    const ReaderError = Io.Reader.Error;
    const ParseError = ReaderError || Allocator.Error || error{InvalidFormat};

    pub fn open(allocator: Allocator, path: []const u8) !GGUFFile {
        const file = try fs.cwd().openFile(path, .{});
        errdefer file.close();

        var self = GGUFFile{
            .allocator = allocator,
            .file = file,
            .header = undefined,
            .metadata = std.StringHashMap(MetaValue).init(allocator),
            .tensors = &[_]TensorInfo{},
            .tensor_map = std.StringHashMap(usize).init(allocator),
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
        self.tensor_map.deinit();
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

        if (self.header.metadata_kv_count > MAX_METADATA_KV_COUNT) {
            return error.InvalidFormat;
        }

        var i: u64 = 0;
        while (i < self.header.metadata_kv_count) : (i += 1) {
            const key_len = try readU64(reader);
            if (key_len > MAX_KEY_LEN) {
                return error.InvalidFormat;
            }
            const key = try arena.alloc(u8, key_len);
            try readBytes(reader, key);

            const value_type_int = try readU32(reader);
            const value_type: MetaValueType = @enumFromInt(value_type_int);

            const value = try readMetaValue(reader, value_type, arena);

            if (mem.eql(u8, key, "general.alignment")) {
                const raw_align: usize = switch (value) {
                    .uint8 => value.uint8,
                    .uint16 => value.uint16,
                    .uint32 => value.uint32,
                    .uint64 => std.math.cast(usize, value.uint64) orelse return error.InvalidFormat,
                    .int8 => std.math.cast(usize, value.int8) orelse return error.InvalidFormat,
                    .int16 => std.math.cast(usize, value.int16) orelse return error.InvalidFormat,
                    .int32 => std.math.cast(usize, value.int32) orelse return error.InvalidFormat,
                    .int64 => std.math.cast(usize, value.int64) orelse return error.InvalidFormat,
                    else => self.alignment,
                };
                // Alignment must be a power of two and non-zero (alignUp uses bitmask arithmetic)
                if (raw_align == 0 or (raw_align & (raw_align - 1)) != 0) {
                    return error.InvalidFormat;
                }
                self.alignment = raw_align;
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
                if (len > MAX_KEY_LEN) return error.InvalidFormat;
                const str = try arena.alloc(u8, len);
                try readBytes(reader, str);
                break :blk MetaValue{ .string = str };
            },
            .array => blk: {
                const elem_type_int = try readU32(reader);
                const elem_type: MetaValueType = @enumFromInt(elem_type_int);
                const len = try readU64(reader);
                const len_usize = std.math.cast(usize, len) orelse return error.InvalidFormat;

                const elems = try arena.alloc(MetaValue, len_usize);
                for (elems) |*elem| {
                    elem.* = try readMetaValue(reader, elem_type, arena);
                }
                break :blk MetaValue{ .array = elems };
            },
        };
    }

    fn parseTensors(self: *GGUFFile) !void {
        const reader = &self.file_reader.interface;
        const arena = self.string_arena.allocator();

        if (self.header.tensor_count == 0) {
            return;
        }

        const tensor_count = std.math.cast(usize, self.header.tensor_count) orelse return error.InvalidFormat;
        self.tensors = try self.allocator.alloc(TensorInfo, tensor_count);
        errdefer self.allocator.free(self.tensors);

        for (self.tensors) |*tensor| {
            const name_len = try readU64(reader);
            const name = try arena.alloc(u8, name_len);
            try readBytes(reader, name);
            tensor.name = name;

            tensor.n_dims = try readU32(reader);
            if (tensor.n_dims > MAX_TENSOR_DIMS) {
                return error.InvalidFormat;
            }
            tensor.shape = [4]u64{ 1, 1, 1, 1 };
            for (0..tensor.n_dims) |d| {
                tensor.shape[d] = try readU64(reader);
            }

            const dtype_int = try readU32(reader);
            tensor.dtype = @enumFromInt(dtype_int);
            tensor.offset = try readU64(reader);

            var n_elements: u64 = 1;
            for (tensor.shape[0..tensor.n_dims]) |dim| {
                n_elements = std.math.mul(u64, n_elements, dim) catch return error.InvalidFormat;
            }
            const block_size = tensor.dtype.blockSize();
            const block_elements = tensor.dtype.blockElements();
            const n_blocks = (n_elements + block_elements - 1) / block_elements;
            tensor.size_bytes = n_blocks * block_size;
        }

        // Build name->index map for O(1) tensor lookup
        for (self.tensors, 0..) |*tensor, idx| {
            try self.tensor_map.put(tensor.name, idx);
        }

        const current_pos = try self.file.getPos();
        self.tensor_data_offset = alignUp(current_pos, self.alignment);
    }

    fn mapTensorData(self: *GGUFFile) !void {
        const file_len = try self.file.getEndPos();
        const mmap_size = std.math.cast(usize, file_len) orelse return error.InvalidFormat;
        const mapped = try posix.mmap(
            null,
            mmap_size,
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
                .uint64 => std.math.cast(u32, value.uint64),
                .int32 => std.math.cast(u32, value.int32),
                else => null,
            };
        }
        return null;
    }

    pub fn getTensor(self: *GGUFFile, name: []const u8) ?*const TensorInfo {
        const idx = self.tensor_map.get(name) orelse return null;
        return &self.tensors[idx];
    }

    pub fn getTensorData(self: *const GGUFFile, tensor: *const TensorInfo) ?[]const u8 {
        const mapped = self.mapped_data orelse return null;
        const start = self.dataOffset(tensor);
        const end = start + tensor.size_bytes;
        if (end > mapped.len) {
            return null;
        }
        const start_usize = std.math.cast(usize, start) orelse return null;
        const end_usize = std.math.cast(usize, end) orelse return null;
        return mapped[start_usize..end_usize];
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

test "align up edge cases" {
    // Zero value aligns to zero
    try std.testing.expectEqual(@as(u64, 0), alignUp(0, 32));
    // Value of 1 aligns up to alignment
    try std.testing.expectEqual(@as(u64, 32), alignUp(1, 32));
    // Value just below alignment
    try std.testing.expectEqual(@as(u64, 32), alignUp(31, 32));
    // Value exactly at alignment
    try std.testing.expectEqual(@as(u64, 32), alignUp(32, 32));
    // Value just above alignment
    try std.testing.expectEqual(@as(u64, 64), alignUp(33, 32));
    // Alignment of 1 should return the same value
    try std.testing.expectEqual(@as(u64, 0), alignUp(0, 1));
    try std.testing.expectEqual(@as(u64, 1), alignUp(1, 1));
    try std.testing.expectEqual(@as(u64, 100), alignUp(100, 1));
    // Different alignment values
    try std.testing.expectEqual(@as(u64, 64), alignUp(33, 64));
    try std.testing.expectEqual(@as(u64, 64), alignUp(64, 64));
    try std.testing.expectEqual(@as(u64, 128), alignUp(65, 64));
}

test "GGMLType block sizes for common types" {
    // Unquantized types
    try std.testing.expectEqual(@as(usize, 4), GGMLType.f32.blockSize());
    try std.testing.expectEqual(@as(usize, 2), GGMLType.f16.blockSize());
    try std.testing.expectEqual(@as(usize, 8), GGMLType.f64.blockSize());
    // Quantized types
    try std.testing.expectEqual(@as(usize, 18), GGMLType.q4_0.blockSize());
    try std.testing.expectEqual(@as(usize, 34), GGMLType.q8_0.blockSize());
    // Integer types
    try std.testing.expectEqual(@as(usize, 1), GGMLType.i8.blockSize());
    try std.testing.expectEqual(@as(usize, 2), GGMLType.i16.blockSize());
    try std.testing.expectEqual(@as(usize, 4), GGMLType.i32.blockSize());
    try std.testing.expectEqual(@as(usize, 8), GGMLType.i64.blockSize());
}

test "GGMLType block elements for common types" {
    // Unquantized types: 1 element per block
    try std.testing.expectEqual(@as(usize, 1), GGMLType.f32.blockElements());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.f16.blockElements());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.f64.blockElements());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.i8.blockElements());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.i16.blockElements());
    try std.testing.expectEqual(@as(usize, 1), GGMLType.i32.blockElements());
    // Quantized 4/5/8-bit: 32 elements per block
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q4_0.blockElements());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q4_1.blockElements());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q5_0.blockElements());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q5_1.blockElements());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q8_0.blockElements());
    try std.testing.expectEqual(@as(usize, 32), GGMLType.q8_1.blockElements());
    // K-quant types: 256 elements per block
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q2_k.blockElements());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q3_k.blockElements());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q4_k.blockElements());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q5_k.blockElements());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q6_k.blockElements());
    try std.testing.expectEqual(@as(usize, 256), GGMLType.q8_k.blockElements());
}

test "MAX_TENSOR_DIMS constant matches shape array size" {
    // Ensure our constant matches the actual shape array size in TensorInfo
    const info = TensorInfo{
        .name = "",
        .n_dims = 0,
        .shape = [4]u64{ 1, 1, 1, 1 },
        .dtype = .f32,
        .offset = 0,
        .size_bytes = 0,
    };
    try std.testing.expectEqual(@as(usize, MAX_TENSOR_DIMS), info.shape.len);
}

test "metadata limits are reasonable" {
    try std.testing.expect(MAX_METADATA_KV_COUNT > 0);
    try std.testing.expect(MAX_METADATA_KV_COUNT <= 100_000);
    try std.testing.expect(MAX_KEY_LEN > 0);
    try std.testing.expect(MAX_KEY_LEN <= 1_048_576);
}
