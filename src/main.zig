const std = @import("std");
const gguf = @import("gguf.zig");
const tensor_mod = @import("tensor.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "info")) {
        if (args.len < 3) {
            std.debug.print("Error: 'info' requires a model path\n", .{});
            std.debug.print("Usage: quorum info <model.gguf>\n", .{});
            return;
        }
        try infoCommand(allocator, args[2]);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage();
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        printUsage();
    }
}

fn printUsage() void {
    const usage =
        \\Quorum - MoE Inference Engine for GLM-4.7-Flash
        \\
        \\Usage: quorum <command> [options]
        \\
        \\Commands:
        \\  info <model.gguf>    Display model information
        \\  help                 Show this help message
        \\
        \\Examples:
        \\  quorum info model.gguf
        \\
    ;
    std.debug.print("{s}", .{usage});
}

fn infoCommand(allocator: std.mem.Allocator, path: []const u8) !void {
    std.debug.print("Loading model: {s}\n\n", .{path});

    var model = gguf.GGUFFile.open(allocator, path) catch |err| {
        std.debug.print("Error loading model: {}\n", .{err});
        return;
    };
    defer model.close();

    // Print header info
    std.debug.print("=== GGUF File Info ===\n", .{});
    std.debug.print("Version: {}\n", .{model.header.version});
    std.debug.print("Tensor count: {}\n", .{model.header.tensor_count});
    std.debug.print("Metadata KV count: {}\n", .{model.header.metadata_kv_count});
    std.debug.print("\n", .{});

    // Print key metadata
    std.debug.print("=== Model Metadata ===\n", .{});
    printMetadataIfExists(&model, "general.architecture");
    printMetadataIfExists(&model, "general.name");
    printMetadataIfExists(&model, "general.quantization_version");
    printMetadataIfExists(&model, "general.file_type");

    // Architecture-specific metadata
    const arch = model.getMetadataString("general.architecture") orelse "unknown";
    std.debug.print("\n=== Architecture: {s} ===\n", .{arch});

    // Try to print common architecture params
    printArchMetadata(&model, arch, "context_length");
    printArchMetadata(&model, arch, "embedding_length");
    printArchMetadata(&model, arch, "block_count");
    printArchMetadata(&model, arch, "attention.head_count");
    printArchMetadata(&model, arch, "attention.head_count_kv");
    printArchMetadata(&model, arch, "expert_count");
    printArchMetadata(&model, arch, "expert_used_count");

    // Print tensor summary
    std.debug.print("\n=== Tensors ({} total) ===\n", .{model.tensors.len});
    if (model.tensors.len > 0) {
        std.debug.print("First 10 tensors:\n", .{});
        const count = @min(model.tensors.len, 10);
        for (model.tensors[0..count]) |tensor| {
            std.debug.print("  {s}: ", .{tensor.name});
            std.debug.print("{s} ", .{@tagName(tensor.dtype)});
            std.debug.print("[", .{});
            var first = true;
            for (tensor.shape[0..tensor.n_dims]) |dim| {
                if (!first) std.debug.print(", ", .{});
                std.debug.print("{}", .{dim});
                first = false;
            }
            std.debug.print("]\n", .{});
        }
        if (model.tensors.len > 10) {
            std.debug.print("  ... and {} more\n", .{model.tensors.len - 10});
        }
    }

    // Estimate memory
    std.debug.print("\n=== Memory Estimate ===\n", .{});
    var total_bytes: u64 = 0;
    for (model.tensors) |tensor| {
        total_bytes += tensor.size_bytes;
    }
    const gb = @as(f64, @floatFromInt(total_bytes)) / (1024 * 1024 * 1024);
    std.debug.print("Total tensor data: {d:.2} GB\n", .{gb});

    printTensorExamples(&model);
}

fn printMetadataIfExists(model: *gguf.GGUFFile, key: []const u8) void {
    if (model.metadata.get(key)) |value| {
        std.debug.print("{s}: ", .{key});
        switch (value) {
            .uint8 => |v| std.debug.print("{}\n", .{v}),
            .int8 => |v| std.debug.print("{}\n", .{v}),
            .uint16 => |v| std.debug.print("{}\n", .{v}),
            .int16 => |v| std.debug.print("{}\n", .{v}),
            .uint32 => |v| std.debug.print("{}\n", .{v}),
            .int32 => |v| std.debug.print("{}\n", .{v}),
            .uint64 => |v| std.debug.print("{}\n", .{v}),
            .int64 => |v| std.debug.print("{}\n", .{v}),
            .float32 => |v| std.debug.print("{d}\n", .{v}),
            .float64 => |v| std.debug.print("{d}\n", .{v}),
            .bool_ => |v| std.debug.print("{}\n", .{v}),
            .string => |v| std.debug.print("{s}\n", .{v}),
            .array => |v| std.debug.print("[array of {} elements]\n", .{v.len}),
        }
    }
}

fn printTensorExamples(model: *gguf.GGUFFile) void {
    const examples = [_][]const u8{
        "tok_embeddings.weight",
        "output.weight",
        "token_embd.weight",
        "model.embed_tokens.weight",
    };

    std.debug.print("\n=== Tensor Lookup ===\n", .{});
    var found_any = false;
    for (examples) |name| {
        if (model.getTensor(name)) |info| {
            const data = model.getTensorData(info) orelse &[_]u8{};
            const view = tensor_mod.TensorView{
                .dtype = info.dtype,
                .n_dims = info.n_dims,
                .shape = info.shape,
                .data = data,
            };
            const offset = model.dataOffset(info);
            std.debug.print("{s}: offset {} bytes, {} elements, {} storage bytes\n", .{
                name,
                offset,
                view.elementCount(),
                view.storageBytes(),
            });
            found_any = true;
        }
    }

    if (!found_any) {
        std.debug.print("No example tensors found.\n", .{});
    }
}

fn printArchMetadata(model: *gguf.GGUFFile, arch: []const u8, suffix: []const u8) void {
    var buffer: [256]u8 = undefined;
    const key = std.fmt.bufPrint(&buffer, "{s}.{s}", .{ arch, suffix }) catch return;
    printMetadataIfExists(model, key);
}

test "basic functionality" {
    // Placeholder test
    try std.testing.expect(true);
}
