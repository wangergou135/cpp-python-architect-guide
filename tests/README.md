# Tests

## Overview
This directory contains test cases and testing infrastructure for the C++/Python architect guide repository, including:

- Unit tests for code examples
- Integration tests for complete workflows
- Performance benchmarks
- Documentation validation tests
- CI/CD pipeline tests

## Structure
- `cpp/` - C++ related tests using frameworks like Google Test, Catch2
- `python/` - Python tests using pytest, unittest
- `integration/` - Cross-language integration tests
- `performance/` - Performance benchmarks and profiling tests
- `docs/` - Documentation validation and link checking

## Running Tests

### C++ Tests
```bash
cd tests/cpp
mkdir build && cd build
cmake ..
make
ctest
```

### Python Tests
```bash
cd tests/python
python -m pytest
```

### All Tests
```bash
# Run from repository root
make test  # or equivalent build system command
```

## Prerequisites
- All dependencies for respective language examples
- Testing frameworks: Google Test (C++), pytest (Python)
- CMake and appropriate build tools