# Getting Started

## Requirements

- **Zig 0.15.x** or later
- **macOS** (for Metal acceleration) or **Linux**

## Installation

### Clone the Repository

```bash
git clone https://github.com/teddytennant/quorum.git
cd quorum
```

### Build

```bash
zig build
```

The executable will be placed in `zig-out/bin/quorum`.

### Verify Installation

```bash
./zig-out/bin/quorum help
```

You should see:

```
Quorum - MoE Inference Engine for GLM-4.7-Flash

Usage: quorum <command> [options]

Commands:
  info <model.gguf>    Display model information
  help                 Show this help message

Examples:
  quorum info model.gguf
```

## Running Tests

```bash
zig build test
```

## Project Structure

```
quorum/
├── build.zig          # Zig build configuration
├── src/
│   ├── main.zig       # CLI entry point and commands
│   ├── gguf.zig       # GGUF file format parser
│   └── tensor.zig     # Tensor view utilities
├── docs/              # Documentation (this site)
├── README.md
└── LICENSE
```

## Next Steps

- Learn about the [CLI usage](./usage.md)
- Understand the [architecture](./architecture.md)
- Check supported [quantization formats](./quantization.md)
