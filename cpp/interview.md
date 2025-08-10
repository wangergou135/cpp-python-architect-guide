# C++ Interview Guide

A comprehensive guide for C++ technical interviews covering fundamental concepts, advanced topics, common questions, and best practices.

## Table of Contents

1. [Fundamental Concepts](#fundamental-concepts)
2. [Advanced Topics](#advanced-topics)
3. [Common Interview Questions](#common-interview-questions)
4. [Best Practices](#best-practices)
5. [Real-World Scenarios](#real-world-scenarios)
6. [Coding Patterns and Anti-Patterns](#coding-patterns-and-anti-patterns)
7. [Performance Considerations](#performance-considerations)
8. [Modern C++ Practices and Trends](#modern-cpp-practices-and-trends)

## Fundamental Concepts

### Core Language Features

#### 1. Data Types and Variables
- **Primitive types**: int, char, float, double, bool
- **Type modifiers**: signed, unsigned, short, long
- **Auto keyword**: Type deduction in C++11+
- **Const correctness**: const variables, const functions, const pointers

**Interview Question**: "Explain the difference between `const int*`, `int* const`, and `const int* const`"

#### 2. Memory Management
- **Stack vs Heap**: Automatic vs dynamic memory allocation
- **Pointers and References**: Differences, when to use each
- **Smart Pointers** (C++11+): unique_ptr, shared_ptr, weak_ptr
- **RAII** (Resource Acquisition Is Initialization)

```cpp
// Example: Smart pointer usage
std::unique_ptr<int> ptr = std::make_unique<int>(42);
std::shared_ptr<std::vector<int>> vec = std::make_shared<std::vector<int>>();
```

#### 3. Object-Oriented Programming
- **Classes and Objects**: Encapsulation, member functions
- **Inheritance**: public, protected, private inheritance
- **Polymorphism**: Virtual functions, pure virtual functions
- **Abstract classes and interfaces**

### Control Flow and Functions

#### 1. Functions
- **Function overloading**: Same name, different parameters
- **Default parameters**: Parameter defaults
- **Function pointers**: Pointing to functions
- **Lambda expressions** (C++11+): Anonymous functions

```cpp
// Lambda example
auto lambda = [](int x, int y) -> int { return x + y; };
```

#### 2. Templates
- **Function templates**: Generic functions
- **Class templates**: Generic classes
- **Template specialization**: Specific implementations
- **SFINAE** (Substitution Failure Is Not An Error)

## Advanced Topics

### 1. Move Semantics and Perfect Forwarding (C++11+)

#### Move Semantics
- **Rvalue references**: `&&` syntax
- **Move constructors and move assignment operators**
- **std::move**: Converting lvalue to rvalue reference

```cpp
class MyClass {
public:
    // Move constructor
    MyClass(MyClass&& other) noexcept 
        : data(std::move(other.data)) {
        other.data = nullptr;
    }
    
    // Move assignment operator
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};
```

#### Perfect Forwarding
- **Universal references**: Template parameter `T&&`
- **std::forward**: Preserving value category

### 2. Template Metaprogramming

#### Template Specialization
```cpp
template<typename T>
struct TypeTraits {
    static constexpr bool is_pointer = false;
};

template<typename T>
struct TypeTraits<T*> {
    static constexpr bool is_pointer = true;
};
```

#### SFINAE and Type Traits
```cpp
template<typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
safe_divide(T a, T b) {
    return b != 0 ? a / b : 0;
}
```

### 3. Concurrency and Multithreading

#### Threading Primitives
- **std::thread**: Creating and managing threads
- **std::mutex**: Mutual exclusion
- **std::condition_variable**: Thread synchronization
- **std::atomic**: Lock-free programming

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>

class ThreadSafeQueue {
private:
    std::queue<int> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    
public:
    void push(int item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        condition_.notify_one();
    }
    
    int pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        int item = queue_.front();
        queue_.pop();
        return item;
    }
};
```

### 4. Standard Template Library (STL)

#### Containers
- **Sequence containers**: vector, deque, list, array
- **Associative containers**: set, map, multiset, multimap
- **Unordered containers**: unordered_set, unordered_map
- **Container adapters**: stack, queue, priority_queue

#### Algorithms
- **Sorting**: sort, stable_sort, partial_sort
- **Searching**: find, binary_search, lower_bound
- **Numeric**: accumulate, inner_product
- **Custom predicates and function objects**

```cpp
// Using algorithms with lambdas
std::vector<int> vec = {1, 2, 3, 4, 5};
auto count = std::count_if(vec.begin(), vec.end(), 
                          [](int x) { return x % 2 == 0; });
```

## Common Interview Questions

### 1. Language Fundamentals

**Q: What is the difference between struct and class in C++?**
A: The only difference is default access level - struct members are public by default, class members are private by default.

**Q: Explain virtual destructors. When and why should you use them?**
A: Virtual destructors ensure proper cleanup when deleting objects through base class pointers. Always use virtual destructors in base classes intended for inheritance.

**Q: What is the Rule of Three/Five/Zero?**
A: 
- **Rule of Three**: If you define destructor, copy constructor, or copy assignment operator, you probably need all three
- **Rule of Five**: Adds move constructor and move assignment operator
- **Rule of Zero**: Prefer using smart pointers and RAII to avoid manual resource management

### 2. Memory Management

**Q: Explain memory leaks and how to prevent them.**
A: Memory leaks occur when dynamically allocated memory is not freed. Prevention:
- Use smart pointers (RAII)
- Proper exception handling
- Tools like Valgrind, AddressSanitizer

**Q: What's the difference between stack and heap allocation?**
A:
- **Stack**: Fast, automatic cleanup, limited size, LIFO order
- **Heap**: Slower, manual management, larger size, random access

### 3. Object-Oriented Design

**Q: Explain polymorphism and provide an example.**
```cpp
class Shape {
public:
    virtual double area() const = 0;  // Pure virtual function
    virtual ~Shape() = default;
};

class Circle : public Shape {
private:
    double radius_;
public:
    Circle(double r) : radius_(r) {}
    double area() const override { return M_PI * radius_ * radius_; }
};

class Rectangle : public Shape {
private:
    double width_, height_;
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    double area() const override { return width_ * height_; }
};
```

### 4. Templates and Generic Programming

**Q: What is template specialization? Provide an example.**
```cpp
// Generic template
template<typename T>
class Container {
public:
    void info() { std::cout << "Generic container\n"; }
};

// Specialization for bool
template<>
class Container<bool> {
public:
    void info() { std::cout << "Specialized bool container\n"; }
};
```

## Best Practices

### 1. Code Organization

#### Header Guards and Include Guidelines
```cpp
// Use include guards or #pragma once
#ifndef MYHEADER_H
#define MYHEADER_H
// header content
#endif

// Or prefer:
#pragma once
```

#### Forward Declarations
- Use forward declarations to reduce compile times
- Only include necessary headers
- Separate interface (.h) from implementation (.cpp)

### 2. Resource Management

#### RAII (Resource Acquisition Is Initialization)
```cpp
class FileWrapper {
private:
    std::FILE* file_;
public:
    explicit FileWrapper(const char* filename) 
        : file_(std::fopen(filename, "r")) {
        if (!file_) throw std::runtime_error("Failed to open file");
    }
    
    ~FileWrapper() {
        if (file_) std::fclose(file_);
    }
    
    // Delete copy constructor and assignment
    FileWrapper(const FileWrapper&) = delete;
    FileWrapper& operator=(const FileWrapper&) = delete;
    
    // Move semantics
    FileWrapper(FileWrapper&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }
};
```

### 3. Modern C++ Guidelines

#### Use auto for Type Deduction
```cpp
// Prefer
auto it = container.begin();
auto lambda = [](int x) { return x * 2; };

// Over
std::vector<std::string>::iterator it = container.begin();
```

#### Range-based for loops
```cpp
// Prefer
for (const auto& item : container) {
    process(item);
}

// Over traditional loops when possible
```

## Real-World Scenarios

### 1. Performance-Critical Applications

#### Cache-Friendly Code
```cpp
// Structure of Arrays (better cache locality)
struct ParticleSystem {
    std::vector<float> positions_x;
    std::vector<float> positions_y;
    std::vector<float> velocities_x;
    std::vector<float> velocities_y;
};

// Instead of Array of Structures
struct Particle {
    float pos_x, pos_y;
    float vel_x, vel_y;
};
std::vector<Particle> particles;  // Worse cache performance
```

### 2. Large-Scale Systems

#### Design Patterns Implementation
```cpp
// Singleton Pattern (thread-safe, C++11)
class DatabaseConnection {
private:
    DatabaseConnection() = default;
    
public:
    static DatabaseConnection& getInstance() {
        static DatabaseConnection instance;
        return instance;
    }
    
    // Delete copy constructor and assignment
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;
};
```

#### Observer Pattern
```cpp
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const std::string& message) = 0;
};

class Subject {
private:
    std::vector<std::weak_ptr<Observer>> observers_;
    
public:
    void addObserver(std::shared_ptr<Observer> observer) {
        observers_.push_back(observer);
    }
    
    void notify(const std::string& message) {
        // Remove expired weak_ptrs and notify valid ones
        observers_.erase(
            std::remove_if(observers_.begin(), observers_.end(),
                          [&](const std::weak_ptr<Observer>& wp) {
                              if (auto sp = wp.lock()) {
                                  sp->update(message);
                                  return false;
                              }
                              return true;  // Remove expired
                          }),
            observers_.end());
    }
};
```

### 3. Error Handling Strategies

#### Exception Safety Guarantees
```cpp
class VectorWrapper {
private:
    std::vector<int> data_;
    
public:
    void push_back_safe(int value) {
        // Strong exception safety guarantee
        std::vector<int> temp = data_;  // Copy
        temp.push_back(value);          // May throw
        data_ = std::move(temp);        // No-throw swap
    }
};
```

## Coding Patterns and Anti-Patterns

### Good Patterns

#### 1. RAII for Resource Management
```cpp
class MutexLock {
    std::mutex& mtx_;
public:
    explicit MutexLock(std::mutex& m) : mtx_(m) { mtx_.lock(); }
    ~MutexLock() { mtx_.unlock(); }
};
```

#### 2. Factory Pattern
```cpp
class ShapeFactory {
public:
    static std::unique_ptr<Shape> createShape(const std::string& type) {
        if (type == "circle") return std::make_unique<Circle>(1.0);
        if (type == "rectangle") return std::make_unique<Rectangle>(1.0, 1.0);
        return nullptr;
    }
};
```

### Anti-Patterns to Avoid

#### 1. Raw Pointer Ownership
```cpp
// BAD: Unclear ownership
Shape* createShape() {
    return new Circle(5.0);  // Who deletes this?
}

// GOOD: Clear ownership
std::unique_ptr<Shape> createShape() {
    return std::make_unique<Circle>(5.0);
}
```

#### 2. Premature Optimization
```cpp
// BAD: Complex optimization without profiling
void processData(std::vector<int>& data) {
    // Complex bit manipulation that's hard to read
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (data[i] << 1) & 0xFFFFFFFE;  // What does this do?
    }
}

// GOOD: Clear intent, optimize later if needed
void processData(std::vector<int>& data) {
    for (auto& value : data) {
        value = makeEven(value);  // Clear intent
    }
}
```

## Performance Considerations

### 1. Memory Layout and Cache Efficiency

#### Data Structure Design
```cpp
// Cache-friendly: data stored contiguously
class EfficientMatrix {
private:
    std::vector<double> data_;
    size_t rows_, cols_;
    
public:
    double& operator()(size_t row, size_t col) {
        return data_[row * cols_ + col];  // Row-major order
    }
};
```

### 2. Algorithm Complexity

#### Container Selection Based on Use Case
```cpp
// O(1) average case for lookups
std::unordered_map<std::string, int> fastLookup;

// O(log n) for sorted data with range queries  
std::map<std::string, int> sortedData;

// O(1) for stack operations
std::stack<int> lifoData;
```

### 3. Compiler Optimizations

#### Enabling Compiler Optimizations
```cpp
// Use appropriate compiler flags
// -O2 or -O3 for release builds
// -march=native for target architecture
// -flto for link-time optimization

// Help compiler with branch prediction
if ([[likely]] condition) {
    // Common path
} else {
    // Rare path
}
```

## Modern C++ Practices and Trends

### C++11/14/17/20 Features

#### 1. C++11 Innovations
- **Auto keyword**: Type deduction
- **Range-based for loops**: Simplified iteration
- **Lambda expressions**: Anonymous functions
- **Smart pointers**: Automatic memory management
- **Move semantics**: Efficient resource transfer

#### 2. C++14 Enhancements
- **Generic lambdas**: Template lambdas
- **std::make_unique**: Safe unique_ptr creation
- **Binary literals**: 0b prefix for binary numbers

#### 3. C++17 Features
```cpp
// Structured bindings
auto [key, value] = map.insert({1, "one"});

// if constexpr for template conditionals
template<typename T>
void process(T value) {
    if constexpr (std::is_integral_v<T>) {
        // Handle integers
    } else {
        // Handle other types
    }
}

// std::optional for nullable values
std::optional<int> divide(int a, int b) {
    if (b != 0) return a / b;
    return std::nullopt;
}
```

#### 4. C++20 Additions
```cpp
// Concepts for template constraints
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// Ranges library
#include <ranges>
auto even_numbers = numbers | std::views::filter([](int n) { return n % 2 == 0; });

// Coroutines for asynchronous programming
generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}
```

### Current Industry Trends

#### 1. Zero-Cost Abstractions
- Template metaprogramming for compile-time computation
- constexpr functions for compile-time evaluation
- CRTP (Curiously Recurring Template Pattern) for static polymorphism

#### 2. Functional Programming Concepts
```cpp
// Immutable data structures
class ImmutableVector {
private:
    std::shared_ptr<const std::vector<int>> data_;
    
public:
    ImmutableVector add(int value) const {
        auto new_data = std::make_shared<std::vector<int>>(*data_);
        new_data->push_back(value);
        return ImmutableVector{new_data};
    }
};
```

#### 3. Safety and Security
- **Memory safety**: Smart pointers, RAII
- **Type safety**: Strong typing, concepts
- **Thread safety**: std::atomic, lock-free programming

### Tools and Ecosystem

#### 1. Build Systems
- **CMake**: Cross-platform build system
- **Conan**: Package manager
- **vcpkg**: Microsoft's package manager

#### 2. Static Analysis Tools
- **Clang Static Analyzer**: Bug detection
- **PVS-Studio**: Commercial static analyzer
- **Cppcheck**: Open-source static analyzer

#### 3. Testing Frameworks
- **Google Test**: Unit testing framework
- **Catch2**: Header-only test framework
- **Boost.Test**: Part of Boost libraries

## Interview Preparation Tips

### 1. Coding Interview Strategy
1. **Understand the problem**: Ask clarifying questions
2. **Think out loud**: Explain your thought process
3. **Start simple**: Basic solution first, then optimize
4. **Test your code**: Walk through examples
5. **Discuss trade-offs**: Time/space complexity, maintainability

### 2. System Design for C++ Applications
- **Performance requirements**: Latency, throughput
- **Memory constraints**: Available RAM, cache considerations
- **Concurrency model**: Threading, async I/O
- **Platform considerations**: Windows, Linux, embedded systems

### 3. Common Pitfalls to Avoid
- **Memory leaks**: Use smart pointers
- **Undefined behavior**: Array bounds, null pointer dereference
- **Race conditions**: Proper synchronization
- **Exception safety**: RAII, strong exception guarantee

Remember: The key to C++ interviews is demonstrating not just language knowledge, but understanding of when and why to use specific features, performance implications, and software engineering best practices.
