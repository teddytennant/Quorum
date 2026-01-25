# Contributing

Contributions to Quorum are welcome! This guide will help you get started.

## Development Setup

### Prerequisites

- Zig 0.15.x or later
- Git
- A GGUF model file for testing (optional)

### Clone and Build

```bash
git clone https://github.com/teddytennant/quorum.git
cd quorum
zig build
```

### Run Tests

```bash
zig build test
```

## Code Style

Quorum follows standard Zig conventions:

- Use `snake_case` for functions and variables
- Use `PascalCase` for types
- Use `SCREAMING_SNAKE_CASE` for constants
- Keep lines under 100 characters when practical
- Use explicit error handling with `try` and `catch`

## Project Structure

```
src/
├── main.zig      # CLI entry point
├── gguf.zig      # GGUF parser (core functionality)
└── tensor.zig    # Tensor utilities
```

### Adding New Features

1. **GGUF-related**: Add to `gguf.zig`
2. **Tensor operations**: Add to `tensor.zig`
3. **CLI commands**: Add to `main.zig`
4. **New module**: Create new file, import in relevant places

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run tests
zig build test

# Test with a real model (if available)
zig build run -- info path/to/model.gguf
```

### 4. Commit

Write clear commit messages:

```bash
git commit -m "Add support for XYZ quantization format"
```

### 5. Submit a Pull Request

- Describe what your changes do
- Reference any related issues
- Include test results if relevant

## Areas for Contribution

### Phase 2: Inference
- CPU reference forward pass implementation
- Tokenizer integration
- Text generation sampling

### Phase 3: Metal Backend
- Metal compute shader development
- GPU memory management
- Performance optimization

### Phase 4: Expert Offloading
- SSD-backed expert cache
- Prefetching strategies
- Memory pressure handling

### Documentation
- Usage examples
- Tutorial content
- API documentation

### Testing
- Unit tests for edge cases
- Integration tests with various models
- Performance benchmarks

## Reporting Issues

When reporting bugs, include:

- Zig version (`zig version`)
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Model file details (if relevant)

## Questions?

Open an issue for questions about:
- Implementation approach
- Architecture decisions
- Feature requests
