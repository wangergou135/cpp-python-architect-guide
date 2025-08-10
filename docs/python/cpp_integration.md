# Python与C++集成指南 / Python-C++ Integration Guide

*最后更新 / Last Updated: 2025-08-10 10:38:31 UTC*

## Python C API使用 / Python C API Usage

### 基础概念 / Basic Concepts
- Python对象引用计数 / Python object reference counting
- GIL（全局解释器锁）/ Global Interpreter Lock
- 类型转换 / Type conversion
- 内存管理 / Memory management

### 示例代码 / Example Code
```cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* 
example_function(PyObject* self, PyObject* args) {
    const char* input;
    if (!PyArg_ParseTuple(args, "s", &input))
        return NULL;
    
    // 处理数据 / Process data
    return Py_BuildValue("s", result);
}
```

## pybind11使用指南 / pybind11 Usage Guide

### 安装配置 / Installation & Setup
```bash
pip install pybind11
```

### CMake配置 / CMake Configuration
```cmake
cmake_minimum_required(VERSION 3.10)
project(example_module)

find_package(pybind11 REQUIRED)
pybind11_add_module(example_module src/main.cpp)
```

## 参考资料 / References
- [Python/C API Reference Manual](https://docs.python.org/3/c-api/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)