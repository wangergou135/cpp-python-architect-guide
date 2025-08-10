# C++ 技术指南 / C++ Technical Guide

本指南涵盖了C++开发中的核心概念、最佳实践和高级特性。
This guide covers core concepts, best practices, and advanced features in C++ development.

## 目录 / Contents

1. [基础概念强化 / Core Concepts](basics.md)
   - 内存模型 / Memory Model
     ```cpp
     // Example of memory layout
     struct MyStruct {
         int a;      // 4 bytes
         double b;   // 8 bytes
         char c;     // 1 byte + padding
     };
     ```
   - RAII (资源获取即初始化) / Resource Acquisition Is Initialization
     ```cpp
     class FileHandler {
         std::ifstream file;
     public:
         FileHandler(const std::string& filename) : file(filename) {}
         ~FileHandler() { if(file.is_open()) file.close(); }
     };
     ```
   - 智能指针 / Smart Pointers
     ```cpp
     auto ptr = std::make_unique<int>(42);
     std::shared_ptr<int> shared = std::make_shared<int>(100);
     ```
   - 左值与右值 / Lvalues and Rvalues
     ```cpp
     int&& rvalue = std::move(42);
     ```

2. [现代C++特性 / Modern C++ Features](modern.md)
   - C++11/14/17/20 新特性 / New Features
     ```cpp
     // C++17 Structured Binding
     std::map<int, std::string> map {{1, "one"}};
     auto const& [key, value] = *map.begin();
     ```
   - Lambda表达式 / Lambda Expressions
     ```cpp
     auto add = [](auto a, auto b) { return a + b; };
     ```
   - 移动语义 / Move Semantics
     ```cpp
     std::vector<int> createVector() {
         return std::vector<int>{1, 2, 3}; // Move constructor called
     }
     ```
   - 并发编程 / Concurrent Programming
     ```cpp
     std::async(std::launch::async, []{ 
         std::cout << "Async task\n"; 
     });
     ```

3. [设计模式 / Design Patterns](patterns.md)
   - 创建型模式 / Creational Patterns
     ```cpp
     // Singleton Example
     class Singleton {
     public:
         static Singleton& getInstance() {
             static Singleton instance;
             return instance;
         }
     private:
         Singleton() = default;
     };
     ```
   - 结构型模式 / Structural Patterns
   - 行为型模式 / Behavioral Patterns
   - C++实现示例 / C++ Implementation Examples

4. [性能优化 / Performance Optimization](performance.md)
   - 编译期优化 / Compile-time Optimization
     ```cpp
     constexpr int fibonacci(int n) {
         return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2);
     }
     ```
   - 运行时优化 / Runtime Optimization
   - 内存优化 / Memory Optimization
     ```cpp
     // Memory Pool Example
     std::pmr::synchronized_pool_resource pool;
     std::pmr::vector<int> vec{&pool};
     ```
   - 并发优化 / Concurrency Optimization

5. [工程实践 / Engineering Practices](engineering.md)
   - 项目结构 / Project Structure
     ```
     project/
     ├── include/
     ├── src/
     ├── tests/
     └── CMakeLists.txt
     ```
   - 构建系统 / Build System
   - 测试框架 / Testing Framework
     ```cpp
     // Google Test Example
     TEST(MyTest, Addition) {
         EXPECT_EQ(1 + 1, 2);
     }
     ```
   - CI/CD Pipeline
     ```yaml
     # Example GitHub Actions workflow
     jobs:
       build:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v2
           - name: Build
             run: cmake . && make
     ```

## Python 互操作 / Python Interoperability

示例：使用 pybind11 封装 C++ 类 / Example: Wrapping C++ class with pybind11
```cpp
#include <pybind11/pybind11.h>

class Calculator {
public:
    int add(int a, int b) { return a + b; }
};

PYBIND11_MODULE(example, m) {
    py::class_<Calculator>(m, "Calculator")
        .def(py::init<>())
        .def("add", &Calculator::add);
}
```

## 更新日志 / Changelog

Last Updated: 2025-08-10 10:15:14 UTC
最后更新：2025-08-10 10:15:14 UTC
