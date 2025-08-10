# C++ Interview Guide

## Table of Contents
1. [Memory Management](#memory-management)
2. [STL Containers and Algorithms](#stl-containers-and-algorithms)
3. [Modern C++ Features](#modern-c-features)
4. [Multi-threading and Concurrency](#multi-threading-and-concurrency)
5. [Design Patterns in C++](#design-patterns-in-c)

---

## Memory Management

### RAII (Resource Acquisition Is Initialization)

**Concept**: RAII is a programming idiom where resource acquisition is tied to object lifetime. Resources are acquired in constructors and released in destructors.

**Key Benefits**:
- Automatic resource cleanup
- Exception safety
- Clear ownership semantics

**Code Example**:
```cpp
class FileManager {
private:
    std::FILE* file;
public:
    FileManager(const std::string& filename) : file(std::fopen(filename.c_str(), "r")) {
        if (!file) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    ~FileManager() {
        if (file) {
            std::fclose(file);
        }
    }
    
    // Disable copy to prevent double-deletion
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
    
    // Enable move semantics
    FileManager(FileManager&& other) noexcept : file(other.file) {
        other.file = nullptr;
    }
    
    FileManager& operator=(FileManager&& other) noexcept {
        if (this != &other) {
            if (file) std::fclose(file);
            file = other.file;
            other.file = nullptr;
        }
        return *this;
    }
    
    std::FILE* get() const { return file; }
};
```

### Smart Pointers

**unique_ptr**: Exclusive ownership of a resource
```cpp
#include <memory>

class Resource {
public:
    Resource(int id) : id_(id) { std::cout << "Resource " << id_ << " created\n"; }
    ~Resource() { std::cout << "Resource " << id_ << " destroyed\n"; }
    void use() { std::cout << "Using resource " << id_ << "\n"; }
private:
    int id_;
};

// Factory function
std::unique_ptr<Resource> createResource(int id) {
    return std::make_unique<Resource>(id);
}

// Usage
auto resource = createResource(42);
resource->use();
// Automatic cleanup when resource goes out of scope
```

**shared_ptr**: Shared ownership with reference counting
```cpp
std::shared_ptr<Resource> resource1 = std::make_shared<Resource>(1);
std::shared_ptr<Resource> resource2 = resource1; // Shared ownership
std::cout << "Reference count: " << resource1.use_count() << std::endl; // 2

// Weak references to break cycles
std::weak_ptr<Resource> weak_ref = resource1;
if (auto locked = weak_ref.lock()) {
    locked->use(); // Safe access
}
```

**Common Interview Questions**:
1. **Q**: What is the difference between `unique_ptr` and `shared_ptr`?
   **A**: `unique_ptr` provides exclusive ownership with zero overhead, while `shared_ptr` allows shared ownership using reference counting but has higher overhead.

2. **Q**: How do you handle circular references with smart pointers?
   **A**: Use `weak_ptr` to break cycles. One object holds a `shared_ptr` and the other holds a `weak_ptr`.

3. **Q**: When would you use `make_unique` vs `new`?
   **A**: Always prefer `make_unique` for exception safety and better performance (single allocation).

---

## STL Containers and Algorithms

### Container Selection Guide

**Sequence Containers**:
- `vector`: Dynamic array, best for most use cases
- `deque`: Double-ended queue, efficient insertion/deletion at both ends
- `list`: Doubly-linked list, efficient insertion/deletion anywhere
- `array`: Fixed-size array, stack-allocated

**Associative Containers**:
- `set`/`map`: Ordered, typically red-black trees, O(log n) operations
- `unordered_set`/`unordered_map`: Hash tables, O(1) average operations
- `multiset`/`multimap`: Allow duplicate keys

**Code Example - Container Performance**:
```cpp
#include <vector>
#include <list>
#include <unordered_map>
#include <algorithm>
#include <chrono>

// Performance comparison
void vectorVsList() {
    const size_t SIZE = 100000;
    
    // Vector: better cache locality
    std::vector<int> vec;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < SIZE; ++i) {
        vec.push_back(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Vector insertion: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs\n";
    
    // List: better for insertion in middle
    std::list<int> lst;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < SIZE; ++i) {
        lst.push_back(i);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "List insertion: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " μs\n";
}
```

### STL Algorithms

**Key Categories**:
- Non-modifying: `find`, `count`, `equal`, `search`
- Modifying: `copy`, `fill`, `replace`, `remove`
- Sorting: `sort`, `partial_sort`, `nth_element`
- Set operations: `set_union`, `set_intersection`

**Advanced Algorithm Usage**:
```cpp
#include <algorithm>
#include <numeric>
#include <functional>

void algorithmExamples() {
    std::vector<int> data{5, 2, 8, 1, 9, 3};
    
    // Custom sorting with lambda
    std::sort(data.begin(), data.end(), [](int a, int b) {
        return a > b; // Descending order
    });
    
    // Find first element greater than 5
    auto it = std::find_if(data.begin(), data.end(), [](int x) {
        return x > 5;
    });
    
    // Transform and accumulate
    std::vector<int> squared;
    std::transform(data.begin(), data.end(), std::back_inserter(squared),
                   [](int x) { return x * x; });
    
    int sum = std::accumulate(squared.begin(), squared.end(), 0);
    
    // Parallel algorithms (C++17)
    std::sort(std::execution::par, data.begin(), data.end());
}
```

**Interview Questions**:
1. **Q**: When would you use `std::list` over `std::vector`?
   **A**: When you need frequent insertion/deletion in the middle and don't require random access.

2. **Q**: What's the difference between `std::map` and `std::unordered_map`?
   **A**: `map` maintains sorted order (O(log n) operations), `unordered_map` uses hash table (O(1) average).

---

## Modern C++ Features

### C++11 Key Features

**Move Semantics**:
```cpp
class MovableClass {
private:
    std::unique_ptr<int[]> data;
    size_t size;
    
public:
    // Constructor
    MovableClass(size_t n) : data(std::make_unique<int[]>(n)), size(n) {}
    
    // Move constructor
    MovableClass(MovableClass&& other) noexcept 
        : data(std::move(other.data)), size(other.size) {
        other.size = 0;
    }
    
    // Move assignment
    MovableClass& operator=(MovableClass&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            size = other.size;
            other.size = 0;
        }
        return *this;
    }
    
    // Copy operations deleted for simplicity
    MovableClass(const MovableClass&) = delete;
    MovableClass& operator=(const MovableClass&) = delete;
};
```

**Lambda Expressions**:
```cpp
// Basic lambda
auto add = [](int a, int b) { return a + b; };

// Capture by value and reference
int multiplier = 10;
auto lambda = [multiplier](int x) mutable { 
    return x * multiplier++; 
};

// Generic lambda (C++14)
auto generic = [](auto x, auto y) { return x + y; };

// IIFE (Immediately Invoked Function Expression)
int result = [](int x) { return x * x; }(5);
```

### C++14/17/20 Features

**C++14 - Generic Lambdas and auto return**:
```cpp
// Generic lambda
auto multiply = [](auto a, auto b) { return a * b; };

// Variable templates
template<typename T>
constexpr T pi = T(3.1415926535897932385);

// Make functions
auto vec = std::make_unique<std::vector<int>>();
```

**C++17 - Structured Bindings and std::optional**:
```cpp
#include <optional>
#include <tuple>

// Structured bindings
std::map<std::string, int> map{{"apple", 1}, {"banana", 2}};
for (const auto& [key, value] : map) {
    std::cout << key << ": " << value << "\n";
}

// Optional
std::optional<int> divide(int a, int b) {
    if (b != 0) {
        return a / b;
    }
    return std::nullopt;
}

// Usage
if (auto result = divide(10, 2)) {
    std::cout << "Result: " << *result << "\n";
}
```

**C++20 - Concepts and Ranges**:
```cpp
#include <concepts>
#include <ranges>

// Concepts
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// Ranges
void rangesExample() {
    std::vector<int> numbers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    auto even_squares = numbers 
        | std::views::filter([](int n) { return n % 2 == 0; })
        | std::views::transform([](int n) { return n * n; });
    
    for (int value : even_squares) {
        std::cout << value << " ";
    }
}
```

**Interview Questions**:
1. **Q**: What is perfect forwarding?
   **A**: Technique to pass arguments to another function while preserving their value category (lvalue/rvalue).

2. **Q**: Explain the difference between `std::move` and `std::forward`.
   **A**: `std::move` unconditionally casts to rvalue, `std::forward` preserves the value category in template contexts.

---

## Multi-threading and Concurrency

### Thread Management

**Basic Threading**:
```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

class ThreadSafeCounter {
private:
    mutable std::mutex mtx;
    int count = 0;
    
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx);
        ++count;
    }
    
    int get() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
};

// Producer-Consumer pattern
class ProducerConsumer {
private:
    std::queue<int> buffer;
    mutable std::mutex mtx;
    std::condition_variable cv;
    bool finished = false;
    
public:
    void produce(int item) {
        std::lock_guard<std::mutex> lock(mtx);
        buffer.push(item);
        cv.notify_one();
    }
    
    std::optional<int> consume() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !buffer.empty() || finished; });
        
        if (buffer.empty()) {
            return std::nullopt;
        }
        
        int item = buffer.front();
        buffer.pop();
        return item;
    }
    
    void finish() {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;
        cv.notify_all();
    }
};
```

### Atomic Operations

**Lock-free Programming**:
```cpp
#include <atomic>

class LockFreeStack {
private:
    struct Node {
        std::atomic<int> data;
        Node* next;
        Node(int value) : data(value), next(nullptr) {}
    };
    
    std::atomic<Node*> head{nullptr};
    
public:
    void push(int value) {
        Node* new_node = new Node(value);
        new_node->next = head.load();
        while (!head.compare_exchange_weak(new_node->next, new_node));
    }
    
    bool pop(int& result) {
        Node* old_head = head.load();
        while (old_head && !head.compare_exchange_weak(old_head, old_head->next));
        
        if (old_head) {
            result = old_head->data;
            delete old_head;
            return true;
        }
        return false;
    }
};
```

### Async Programming

**Futures and Promises**:
```cpp
#include <future>
#include <chrono>

// Async task execution
std::future<int> asyncComputation() {
    return std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 42;
    });
}

// Promise-based communication
void promiseExample() {
    std::promise<int> promise;
    std::future<int> future = promise.get_future();
    
    std::thread worker([&promise]() {
        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        promise.set_value(100);
    });
    
    // Wait for result
    int result = future.get();
    worker.join();
}
```

**Interview Questions**:
1. **Q**: What's the difference between `std::mutex` and `std::shared_mutex`?
   **A**: `shared_mutex` allows multiple readers or one writer, `mutex` allows only one accessor.

2. **Q**: Explain the ABA problem in lock-free programming.
   **A**: When a value changes from A to B and back to A, compare-and-swap may succeed incorrectly.

---

## Design Patterns in C++

### Creational Patterns

**Singleton Pattern (Thread-safe)**:
```cpp
class Singleton {
private:
    static std::unique_ptr<Singleton> instance;
    static std::once_flag init_flag;
    
    Singleton() = default;
    
public:
    static Singleton& getInstance() {
        std::call_once(init_flag, []() {
            instance = std::make_unique<Singleton>();
        });
        return *instance;
    }
    
    // Delete copy operations
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};

// Modern C++11 approach (guaranteed thread-safe)
class ModernSingleton {
public:
    static ModernSingleton& getInstance() {
        static ModernSingleton instance;
        return instance;
    }
    
private:
    ModernSingleton() = default;
    ModernSingleton(const ModernSingleton&) = delete;
    ModernSingleton& operator=(const ModernSingleton&) = delete;
};
```

**Factory Pattern**:
```cpp
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual std::unique_ptr<Shape> clone() const = 0;
};

class Circle : public Shape {
public:
    void draw() const override {
        std::cout << "Drawing Circle\n";
    }
    
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Circle>(*this);
    }
};

class ShapeFactory {
public:
    enum class ShapeType { CIRCLE, RECTANGLE, TRIANGLE };
    
    static std::unique_ptr<Shape> createShape(ShapeType type) {
        switch (type) {
            case ShapeType::CIRCLE:
                return std::make_unique<Circle>();
            // Add other shapes...
            default:
                throw std::invalid_argument("Unknown shape type");
        }
    }
};
```

### Behavioral Patterns

**Observer Pattern (Modern C++)**:
```cpp
#include <functional>

template<typename... Args>
class Signal {
private:
    mutable std::mutex mtx;
    std::vector<std::function<void(Args...)>> slots;
    
public:
    template<typename F>
    void connect(F&& f) {
        std::lock_guard<std::mutex> lock(mtx);
        slots.emplace_back(std::forward<F>(f));
    }
    
    void emit(Args... args) {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& slot : slots) {
            slot(args...);
        }
    }
};

// Usage
Signal<int> signal;
signal.connect([](int value) {
    std::cout << "Received: " << value << "\n";
});
signal.emit(42);
```

**Strategy Pattern with Templates**:
```cpp
template<typename SortStrategy>
class Sorter {
private:
    SortStrategy strategy;
    
public:
    template<typename Container>
    void sort(Container& container) {
        strategy(container);
    }
};

struct QuickSort {
    template<typename Container>
    void operator()(Container& container) {
        std::sort(container.begin(), container.end());
    }
};

struct MergeSort {
    template<typename Container>
    void operator()(Container& container) {
        std::stable_sort(container.begin(), container.end());
    }
};

// Usage
Sorter<QuickSort> quickSorter;
Sorter<MergeSort> mergeSorter;
```

### Structural Patterns

**PIMPL (Pointer to Implementation)**:
```cpp
// Header file (widget.h)
class Widget {
public:
    Widget();
    ~Widget();
    Widget(const Widget& other);
    Widget& operator=(const Widget& other);
    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;
    
    void doSomething();
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// Implementation file (widget.cpp)
class Widget::Impl {
public:
    void doSomething() {
        // Implementation details hidden
    }
    
private:
    // Private members not exposed in header
    std::vector<int> data;
    ComplexDependency dependency;
};

Widget::Widget() : pImpl(std::make_unique<Impl>()) {}
Widget::~Widget() = default;

// Copy constructor
Widget::Widget(const Widget& other) 
    : pImpl(std::make_unique<Impl>(*other.pImpl)) {}

// Move constructor
Widget::Widget(Widget&& other) noexcept = default;

void Widget::doSomething() {
    pImpl->doSomething();
}
```

**Interview Questions**:
1. **Q**: What are the benefits of the PIMPL idiom?
   **A**: Reduces compilation dependencies, provides binary stability, and hides implementation details.

2. **Q**: How does RAII relate to design patterns?
   **A**: RAII is fundamental to many patterns like Proxy, Guard, and ensures exception safety.

---

## Best Practices and Real-world Scenarios

### Performance Optimization

**Memory Optimization**:
```cpp
// Prefer stack allocation
void goodFunction() {
    std::array<int, 1000> data; // Stack allocated
    // Use data...
}

// Avoid unnecessary allocations
void avoidThis() {
    for (int i = 0; i < 1000; ++i) {
        std::vector<int> temp(100); // Allocates every iteration!
    }
}

void betterApproach() {
    std::vector<int> reusable;
    for (int i = 0; i < 1000; ++i) {
        reusable.clear();
        reusable.reserve(100); // Reuse capacity
        // Use reusable...
    }
}
```

**Cache-friendly Programming**:
```cpp
struct SoA { // Structure of Arrays - better cache locality
    std::vector<float> x, y, z;
    void addPoint(float px, float py, float pz) {
        x.push_back(px);
        y.push_back(py);
        z.push_back(pz);
    }
};

struct AoS { // Array of Structures - worse cache locality
    struct Point { float x, y, z; };
    std::vector<Point> points;
};
```

### Common Pitfalls

1. **Memory Leaks with Exceptions**:
```cpp
// Bad: exception unsafe
void unsafe() {
    Resource* ptr = new Resource();
    riskyOperation(); // May throw
    delete ptr; // Never reached if exception thrown
}

// Good: RAII
void safe() {
    auto ptr = std::make_unique<Resource>();
    riskyOperation(); // Exception safe
} // Automatic cleanup
```

2. **Iterator Invalidation**:
```cpp
// Dangerous
std::vector<int> vec{1, 2, 3, 4, 5};
for (auto it = vec.begin(); it != vec.end(); ++it) {
    if (*it % 2 == 0) {
        vec.erase(it); // Invalidates iterator!
    }
}

// Safe
vec.erase(std::remove_if(vec.begin(), vec.end(), 
                        [](int x) { return x % 2 == 0; }), 
          vec.end());
```

### Interview Scenario Questions

**Q**: Design a thread-safe cache with LRU eviction.
**A**: Implement using `std::unordered_map` for O(1) lookup and a doubly-linked list for LRU ordering, protected by a mutex.

**Q**: How would you implement a memory pool allocator?
**A**: Pre-allocate a large block, maintain a free list of fixed-size chunks, use placement new for object construction.

**Q**: Optimize a function that processes millions of small objects.
**A**: Consider object pooling, vectorization, cache optimization, and avoiding virtual function calls in hot paths.

---

This guide covers essential C++ concepts for technical interviews. Focus on understanding the underlying principles, practicing code examples, and being able to explain trade-offs between different approaches.