# C++ Interview Guide

A comprehensive guide covering advanced C++ concepts, best practices, and common interview questions.

## Table of Contents
1. [C++ Language Fundamentals](#c-language-fundamentals)
2. [Memory Management](#memory-management)
3. [STL and Templates](#stl-and-templates)
4. [Modern C++ Features](#modern-c-features)
5. [Multithreading and Synchronization](#multithreading-and-synchronization)
6. [Design Patterns and Best Practices](#design-patterns-and-best-practices)
7. [Performance Optimization](#performance-optimization)

## C++ Language Fundamentals

### What is the difference between stack and heap memory?

**Stack Memory:**
- Automatically managed memory
- Faster allocation/deallocation
- Limited size (typically 1-8MB)
- LIFO (Last In, First Out) structure
- Used for local variables, function parameters

**Heap Memory:**
- Manually managed memory (new/delete)
- Slower allocation/deallocation
- Large memory space available
- No specific ordering
- Used for dynamic allocation

```cpp
void stackHeapExample() {
    // Stack allocation
    int stackVar = 42;          // Automatic cleanup
    char stackArray[100];       // Fixed size, fast access
    
    // Heap allocation
    int* heapVar = new int(42);     // Manual cleanup required
    char* heapArray = new char[100]; // Dynamic size
    
    // Must manually free heap memory
    delete heapVar;
    delete[] heapArray;
}
```

**Common Pitfalls:**
- Stack overflow from large local arrays or deep recursion
- Memory leaks from forgotten delete operations
- Using stack variables after function returns (dangling references)

### Explain virtual functions and vtable implementation

Virtual functions enable polymorphism through dynamic dispatch. The compiler creates a virtual table (vtable) for each class with virtual functions.

```cpp
class Base {
private:
    // Hidden vtable pointer added by compiler
public:
    virtual void func1() { std::cout << "Base::func1\n"; }
    virtual void func2() { std::cout << "Base::func2\n"; }
    virtual ~Base() = default; // Always virtual destructor
};

class Derived : public Base {
public:
    void func1() override { std::cout << "Derived::func1\n"; }
    // func2 inherited from Base
};

// Runtime polymorphism
void demonstrate() {
    Base* ptr = new Derived();
    ptr->func1(); // Calls Derived::func1 via vtable lookup
    delete ptr;   // Calls Derived destructor, then Base destructor
}
```

**vtable Structure:**
```
Base vtable:
[0] -> Base::func1
[1] -> Base::func2
[2] -> Base::~Base

Derived vtable:
[0] -> Derived::func1  // Override
[1] -> Base::func2     // Inherited
[2] -> Derived::~Derived
```

**Best Practices:**
- Always declare virtual destructors in base classes
- Use `override` keyword for clarity
- Consider performance cost of virtual function calls

### How does multiple inheritance work in C++?

Multiple inheritance allows a class to inherit from multiple base classes, but introduces complexity with diamond inheritance and virtual inheritance.

```cpp
class Engine {
public:
    virtual void start() { std::cout << "Engine starting\n"; }
    int enginePower = 100;
};

class Wheels {
public:
    virtual void rotate() { std::cout << "Wheels rotating\n"; }
    int wheelCount = 4;
};

// Multiple inheritance
class Car : public Engine, public Wheels {
public:
    void drive() {
        start();    // Engine::start()
        rotate();   // Wheels::rotate()
    }
};

// Diamond inheritance problem
class Vehicle {
public:
    int id = 1;
};

class LandVehicle : public Vehicle { };
class WaterVehicle : public Vehicle { };

// Ambiguous: which Vehicle::id?
class AmphibiousVehicle : public LandVehicle, public WaterVehicle {
    // Compiler error without virtual inheritance
};

// Solution: Virtual inheritance
class LandVehicleV : public virtual Vehicle { };
class WaterVehicleV : public virtual Vehicle { };
class AmphibiousVehicleV : public LandVehicleV, public WaterVehicleV {
    // Only one Vehicle instance
};
```

**Common Issues:**
- Ambiguous function/variable names
- Diamond inheritance problems
- Complex constructor initialization
- Increased memory overhead with virtual inheritance

### What are the differences between references and pointers?

```cpp
void demonstrateReferencesPointers() {
    int x = 10, y = 20;
    
    // References
    int& ref = x;        // Must be initialized
    ref = 30;            // Changes x to 30
    // int& ref2;        // ERROR: References must be initialized
    // ref = y;          // This assigns y's value to x, doesn't rebind
    
    // Pointers
    int* ptr = &x;       // Can be uninitialized
    *ptr = 40;           // Changes x to 40
    ptr = &y;            // Rebinding pointer to y
    ptr = nullptr;       // Can be null
    
    // Const differences
    const int& constRef = x;        // Cannot modify through reference
    int* const constPtr = &x;       // Constant pointer, can modify value
    const int* ptrToConst = &x;     // Pointer to constant, cannot modify value
    const int* const constPtrToConst = &x; // Both constant
}
```

**Key Differences:**
- References must be initialized, pointers can be uninitialized
- References cannot be reassigned, pointers can
- References cannot be null, pointers can
- No pointer arithmetic on references
- References are generally safer and more convenient

### Explain move semantics and perfect forwarding

Move semantics eliminate unnecessary copying by transferring ownership of resources.

```cpp
class Resource {
private:
    int* data;
    size_t size;
public:
    // Constructor
    Resource(size_t s) : size(s), data(new int[s]) {
        std::cout << "Constructor\n";
    }
    
    // Copy constructor (expensive)
    Resource(const Resource& other) : size(other.size), data(new int[size]) {
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy constructor\n";
    }
    
    // Move constructor (cheap)
    Resource(Resource&& other) noexcept : size(other.size), data(other.data) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Move constructor\n";
    }
    
    // Copy assignment
    Resource& operator=(const Resource& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new int[size];
            std::copy(other.data, other.data + size, data);
        }
        std::cout << "Copy assignment\n";
        return *this;
    }
    
    // Move assignment
    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        std::cout << "Move assignment\n";
        return *this;
    }
    
    ~Resource() { delete[] data; }
};

// Perfect forwarding example
template<typename T>
void wrapper(T&& arg) {
    // Perfect forwarding preserves value category
    process(std::forward<T>(arg));
}

void demonstrateMove() {
    Resource r1(1000);              // Constructor
    Resource r2 = r1;               // Copy constructor
    Resource r3 = std::move(r1);    // Move constructor
    r2 = std::move(r3);             // Move assignment
}
```

**Benefits:**
- Eliminates unnecessary copying
- Enables efficient container operations
- Allows transfer of unique resources

### What is RAII and why is it important?

RAII (Resource Acquisition Is Initialization) ensures automatic resource management by tying resource lifetime to object lifetime.

```cpp
// RAII file handler
class FileHandler {
private:
    std::FILE* file;
public:
    explicit FileHandler(const char* filename) 
        : file(std::fopen(filename, "r")) {
        if (!file) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    ~FileHandler() {
        if (file) {
            std::fclose(file);
        }
    }
    
    // Delete copy operations for simplicity
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    std::FILE* get() const { return file; }
};

// RAII mutex lock
class MutexLock {
private:
    std::mutex& mtx;
public:
    explicit MutexLock(std::mutex& m) : mtx(m) {
        mtx.lock();
    }
    
    ~MutexLock() {
        mtx.unlock();
    }
};

void demonstrateRAII() {
    try {
        FileHandler fh("data.txt");     // File opened
        MutexLock lock(globalMutex);    // Mutex locked
        
        // Do work...
        if (someCondition) {
            throw std::exception();     // Early exit
        }
        
        // Resources automatically cleaned up in destructors
    } catch (...) {
        // File and mutex automatically released
        std::cout << "Exception handled, resources cleaned up\n";
    }
}
```

**Advantages:**
- Exception safety
- Automatic cleanup
- Clear ownership semantics
- Prevents resource leaks

**Interview Tips:**
- Always mention exception safety
- Contrast with manual resource management
- Give examples of standard RAII classes (unique_ptr, lock_guard, etc.)

## Memory Management

### How to prevent memory leaks?

```cpp
// 1. Use smart pointers instead of raw pointers
class ModernExample {
private:
    std::unique_ptr<int[]> data;        // Automatic cleanup
    std::shared_ptr<Resource> shared;   // Reference counting
    
public:
    ModernExample(size_t size) 
        : data(std::make_unique<int[]>(size)),
          shared(std::make_shared<Resource>(100)) {
    }
    
    // No manual destructor needed
};

// 2. Follow Rule of Three/Five/Zero
class RuleOfFive {
private:
    int* ptr;
public:
    // Constructor
    RuleOfFive(int val) : ptr(new int(val)) {}
    
    // Destructor
    ~RuleOfFive() { delete ptr; }
    
    // Copy constructor
    RuleOfFive(const RuleOfFive& other) : ptr(new int(*other.ptr)) {}
    
    // Copy assignment
    RuleOfFive& operator=(const RuleOfFive& other) {
        if (this != &other) {
            delete ptr;
            ptr = new int(*other.ptr);
        }
        return *this;
    }
    
    // Move constructor
    RuleOfFive(RuleOfFive&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    
    // Move assignment
    RuleOfFive& operator=(RuleOfFive&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
};

// 3. Use containers instead of manual arrays
void useContainers() {
    std::vector<int> vec(1000);         // Automatic memory management
    std::array<int, 100> arr{};         // Stack-based, no allocation
    std::string str = "Hello";          // Automatic string management
}

// 4. Proper exception handling
void exceptionSafeFunction() {
    auto ptr1 = std::make_unique<int>(42);
    auto ptr2 = std::make_unique<int>(24);
    
    // If exception thrown here, both ptr1 and ptr2 are cleaned up
    riskyOperation();
}
```

**Prevention Strategies:**
- Use smart pointers (unique_ptr, shared_ptr)
- Follow RAII principles
- Use standard containers
- Implement proper copy/move semantics
- Use static analysis tools (Valgrind, AddressSanitizer)

### What are smart pointers and their types?

```cpp
#include <memory>

// 1. unique_ptr - Exclusive ownership
void uniquePtrExample() {
    auto ptr = std::make_unique<int>(42);
    
    // Transfer ownership
    auto ptr2 = std::move(ptr);  // ptr is now nullptr
    
    // Custom deleter
    auto filePtr = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen("file.txt", "r"), &fclose
    );
}

// 2. shared_ptr - Shared ownership with reference counting
class Node {
public:
    int value;
    std::shared_ptr<Node> next;
    
    Node(int v) : value(v) {}
};

void sharedPtrExample() {
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);
    
    node1->next = node2;  // Reference count of node2 = 2
    
    std::cout << "node2 ref count: " << node2.use_count() << std::endl;
}

// 3. weak_ptr - Non-owning reference to break cycles
class Parent;
class Child {
public:
    std::weak_ptr<Parent> parent;  // Prevents circular reference
};

class Parent {
public:
    std::shared_ptr<Child> child;
};

void weakPtrExample() {
    auto parent = std::make_shared<Parent>();
    auto child = std::make_shared<Child>();
    
    parent->child = child;
    child->parent = parent;  // No circular reference due to weak_ptr
    
    // Check if parent still exists
    if (auto p = child->parent.lock()) {
        // Use p safely
    }
}

// 4. Comparison with raw pointers
void comparePointers() {
    // Raw pointer - manual management
    int* raw = new int(42);
    delete raw;  // Must remember to delete
    
    // Smart pointer - automatic management
    auto smart = std::make_unique<int>(42);
    // Automatically deleted when smart goes out of scope
}
```

**When to use each:**
- `unique_ptr`: Single ownership, most common choice
- `shared_ptr`: Multiple owners, reference counting needed
- `weak_ptr`: Break circular references, temporary access

### Explain placement new and delete operators

Placement new allows constructing objects at specific memory locations without allocating new memory.

```cpp
#include <new>
#include <memory>

class MyClass {
private:
    int data[100];
public:
    MyClass(int val) {
        std::fill(data, data + 100, val);
        std::cout << "MyClass constructed at " << this << std::endl;
    }
    
    ~MyClass() {
        std::cout << "MyClass destructed at " << this << std::endl;
    }
};

void placementNewExample() {
    // 1. Basic placement new
    alignas(MyClass) char buffer[sizeof(MyClass)];
    
    // Construct object in pre-allocated buffer
    MyClass* obj = new(buffer) MyClass(42);
    
    // Must manually call destructor (no delete!)
    obj->~MyClass();
    
    // 2. Placement new with dynamic allocation
    void* memory = std::malloc(sizeof(MyClass));
    MyClass* obj2 = new(memory) MyClass(24);
    
    obj2->~MyClass();
    std::free(memory);
    
    // 3. Custom memory pool
    class MemoryPool {
    private:
        alignas(std::max_align_t) char pool[1024];
        size_t offset = 0;
        
    public:
        void* allocate(size_t size) {
            if (offset + size > sizeof(pool)) return nullptr;
            void* ptr = pool + offset;
            offset += size;
            return ptr;
        }
        
        void reset() { offset = 0; }
    };
    
    MemoryPool pool;
    MyClass* obj3 = new(pool.allocate(sizeof(MyClass))) MyClass(100);
    obj3->~MyClass();
}

// Custom allocator using placement new
template<typename T>
class CustomAllocator {
private:
    T* memory;
    size_t capacity;
    size_t size;
    
public:
    explicit CustomAllocator(size_t cap) 
        : capacity(cap), size(0) {
        memory = static_cast<T*>(std::malloc(capacity * sizeof(T)));
    }
    
    ~CustomAllocator() {
        // Destroy all constructed objects
        for (size_t i = 0; i < size; ++i) {
            memory[i].~T();
        }
        std::free(memory);
    }
    
    template<typename... Args>
    T* construct(Args&&... args) {
        if (size >= capacity) return nullptr;
        
        T* ptr = memory + size++;
        return new(ptr) T(std::forward<Args>(args)...);
    }
};
```

**Use Cases:**
- Memory pools and custom allocators
- Embedded systems with limited memory
- Performance-critical applications
- Object reuse without deallocation

**Important Notes:**
- Must manually call destructor
- Memory alignment considerations
- Exception safety challenges

### How does memory alignment work?

Memory alignment ensures data is stored at addresses that are multiples of the data type's size for optimal CPU access.

```cpp
#include <iostream>
#include <cstdint>

// Alignment examples
struct UnalignedStruct {
    char a;      // 1 byte
    int b;       // 4 bytes
    char c;      // 1 byte
    double d;    // 8 bytes
    // Total: 1 + 3(padding) + 4 + 1 + 7(padding) + 8 = 24 bytes
};

struct AlignedStruct {
    double d;    // 8 bytes (largest alignment requirement)
    int b;       // 4 bytes
    char a;      // 1 byte
    char c;      // 1 byte
    // 2 bytes padding at end for alignment
    // Total: 8 + 4 + 1 + 1 + 2(padding) = 16 bytes
};

// Custom alignment
struct alignas(32) AlignedTo32 {
    int data;
    // Aligned to 32-byte boundary
};

void demonstrateAlignment() {
    std::cout << "UnalignedStruct size: " << sizeof(UnalignedStruct) << std::endl;
    std::cout << "AlignedStruct size: " << sizeof(AlignedStruct) << std::endl;
    std::cout << "AlignedTo32 size: " << sizeof(AlignedTo32) << std::endl;
    
    // Check alignment
    std::cout << "double alignment: " << alignof(double) << std::endl;
    std::cout << "AlignedTo32 alignment: " << alignof(AlignedTo32) << std::endl;
    
    // Memory addresses
    UnalignedStruct u;
    std::cout << "Address of u.a: " << reinterpret_cast<uintptr_t>(&u.a) << std::endl;
    std::cout << "Address of u.b: " << reinterpret_cast<uintptr_t>(&u.b) << std::endl;
    std::cout << "Address of u.c: " << reinterpret_cast<uintptr_t>(&u.c) << std::endl;
    std::cout << "Address of u.d: " << reinterpret_cast<uintptr_t>(&u.d) << std::endl;
}

// SIMD alignment example
void simdAlignment() {
    // Proper alignment for SIMD operations
    alignas(16) float vectorData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Can safely use SIMD instructions on vectorData
    // because it's 16-byte aligned
}
```

**Why Alignment Matters:**
- CPU efficiency: Aligned access is faster
- SIMD operations require specific alignment
- Some architectures require alignment (crash on misaligned access)
- Cache line optimization

### What is memory fragmentation?

Memory fragmentation occurs when free memory is broken into small, non-contiguous blocks.

```cpp
// External fragmentation example
class FragmentationDemo {
private:
    struct Block {
        size_t size;
        bool free;
        Block* next;
    };
    
    Block* freeList;
    
public:
    void demonstrateFragmentation() {
        // Allocate several blocks
        void* ptr1 = allocate(100);  // [Used 100][Free...]
        void* ptr2 = allocate(200);  // [Used 100][Used 200][Free...]
        void* ptr3 = allocate(150);  // [Used 100][Used 200][Used 150][Free...]
        
        // Free every other block
        deallocate(ptr2);            // [Used 100][Free 200][Used 150][Free...]
        
        // Now we have fragmentation:
        // Can't allocate 300 bytes even though we have 200+ bytes free
        // because they're not contiguous
        
        void* ptr4 = allocate(300);  // FAILS due to fragmentation
    }
    
    void* allocate(size_t size);     // Implementation details...
    void deallocate(void* ptr);
};

// Memory pool to reduce fragmentation
template<typename T, size_t PoolSize>
class MemoryPool {
private:
    alignas(T) char pool[PoolSize * sizeof(T)];
    std::vector<bool> used;
    
public:
    MemoryPool() : used(PoolSize, false) {}
    
    T* allocate() {
        for (size_t i = 0; i < PoolSize; ++i) {
            if (!used[i]) {
                used[i] = true;
                return reinterpret_cast<T*>(pool + i * sizeof(T));
            }
        }
        return nullptr;  // Pool exhausted
    }
    
    void deallocate(T* ptr) {
        if (ptr) {
            size_t index = (reinterpret_cast<char*>(ptr) - pool) / sizeof(T);
            if (index < PoolSize) {
                used[index] = false;
            }
        }
    }
};

// Stack allocator to minimize fragmentation
class StackAllocator {
private:
    char* memory;
    size_t size;
    size_t top;
    
public:
    StackAllocator(size_t sz) : size(sz), top(0) {
        memory = new char[size];
    }
    
    ~StackAllocator() {
        delete[] memory;
    }
    
    void* allocate(size_t sz) {
        if (top + sz > size) return nullptr;
        
        void* ptr = memory + top;
        top += sz;
        return ptr;
    }
    
    void reset() {
        top = 0;  // Reset entire allocator
    }
    
    // No individual deallocation - stack-like behavior
};
```

**Types of Fragmentation:**
- **External**: Free memory exists but in small, scattered blocks
- **Internal**: Allocated blocks larger than requested due to alignment

**Solutions:**
- Memory pools for fixed-size allocations
- Stack allocators for temporary allocations
- Compacting garbage collection
- Best-fit vs first-fit allocation strategies

## STL and Templates

### Container types and their implementations

```cpp
#include <vector>
#include <list>
#include <deque>
#include <set>
#include <unordered_map>

// 1. Sequence Containers
void sequenceContainers() {
    // vector: Dynamic array, contiguous memory
    std::vector<int> vec = {1, 2, 3, 4, 5};
    vec.push_back(6);        // O(1) amortized
    vec.insert(vec.begin() + 2, 10); // O(n) - elements shift
    
    // deque: Double-ended queue, chunked memory
    std::deque<int> deq = {1, 2, 3};
    deq.push_front(0);       // O(1) - efficient at both ends
    deq.push_back(4);        // O(1)
    
    // list: Doubly-linked list
    std::list<int> lst = {1, 2, 3};
    lst.insert(std::next(lst.begin()), 10); // O(1) if iterator known
    lst.splice(lst.end(), lst, lst.begin()); // O(1) move operation
}

// 2. Associative Containers
void associativeContainers() {
    // set: Balanced binary search tree (usually Red-Black)
    std::set<int> s = {3, 1, 4, 1, 5}; // Automatically sorted, unique
    s.insert(2);             // O(log n)
    auto it = s.find(4);     // O(log n)
    
    // map: Key-value pairs, ordered by key
    std::map<std::string, int> m;
    m["apple"] = 5;          // O(log n)
    m.emplace("banana", 3);  // O(log n), construct in-place
}

// 3. Unordered Containers (Hash Tables)
void unorderedContainers() {
    std::unordered_map<std::string, int> um;
    um["key1"] = 100;        // O(1) average, O(n) worst case
    um.reserve(1000);        // Prevent rehashing
    
    // Custom hash function
    struct Point {
        int x, y;
        bool operator==(const Point& other) const {
            return x == other.x && y == other.y;
        }
    };
    
    struct PointHash {
        size_t operator()(const Point& p) const {
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };
    
    std::unordered_set<Point, PointHash> pointSet;
}

// Container adaptors
void containerAdaptors() {
    // stack: LIFO, typically uses deque
    std::stack<int> stk;
    stk.push(1);
    int top = stk.top();
    stk.pop();
    
    // queue: FIFO, typically uses deque
    std::queue<int> q;
    q.push(1);
    int front = q.front();
    q.pop();
    
    // priority_queue: Max-heap by default
    std::priority_queue<int> pq;
    pq.push(3);
    pq.push(1);
    pq.push(4);
    while (!pq.empty()) {
        std::cout << pq.top() << " "; // 4, 3, 1
        pq.pop();
    }
}
```

### Iterator categories and their uses

```cpp
#include <iterator>
#include <algorithm>

// Iterator categories hierarchy:
// Input Iterator -> Forward Iterator -> Bidirectional Iterator -> Random Access Iterator

template<typename Iterator>
void demonstrateIteratorCategory(Iterator it) {
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    
    if constexpr (std::is_same_v<category, std::input_iterator_tag>) {
        std::cout << "Input Iterator: Read-only, single-pass\n";
        // Can only read and move forward once
    }
    else if constexpr (std::is_same_v<category, std::forward_iterator_tag>) {
        std::cout << "Forward Iterator: Multi-pass forward iteration\n";
        // Can read/write and move forward multiple times
    }
    else if constexpr (std::is_same_v<category, std::bidirectional_iterator_tag>) {
        std::cout << "Bidirectional Iterator: Forward and backward\n";
        // Can move forward and backward
        --it; // This works
    }
    else if constexpr (std::is_same_v<category, std::random_access_iterator_tag>) {
        std::cout << "Random Access Iterator: Jump to any position\n";
        // Can jump to arbitrary positions
        it += 5;  // This works
        auto diff = it - it; // Pointer arithmetic
    }
}

// Custom iterator example
template<typename T>
class RangeIterator {
private:
    T current;
    T step;
    
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T&;
    
    RangeIterator(T start, T step_size) : current(start), step(step_size) {}
    
    T operator*() const { return current; }
    RangeIterator& operator++() {
        current += step;
        return *this;
    }
    RangeIterator operator++(int) {
        RangeIterator temp = *this;
        ++(*this);
        return temp;
    }
    
    bool operator==(const RangeIterator& other) const {
        return current == other.current;
    }
    bool operator!=(const RangeIterator& other) const {
        return !(*this == other);
    }
};

// Iterator adapter example
template<typename Container>
class ReverseContainer {
private:
    Container& container;
    
public:
    explicit ReverseContainer(Container& c) : container(c) {}
    
    auto begin() { return container.rbegin(); }
    auto end() { return container.rend(); }
};

void iteratorExamples() {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::list<int> lst = {1, 2, 3, 4, 5};
    
    // Different iterator capabilities
    demonstrateIteratorCategory(vec.begin());  // Random access
    demonstrateIteratorCategory(lst.begin());  // Bidirectional
    
    // Custom range iterator
    RangeIterator<int> start(0, 2);
    RangeIterator<int> end(10, 2);
    
    // Use with STL algorithms
    std::for_each(start, end, [](int x) {
        std::cout << x << " "; // 0, 2, 4, 6, 8
    });
    
    // Reverse iteration
    ReverseContainer reverse(vec);
    for (auto x : reverse) {
        std::cout << x << " "; // 5, 4, 3, 2, 1
    }
}
```

### Algorithm complexity guarantees

```cpp
#include <algorithm>
#include <vector>
#include <chrono>

void algorithmComplexities() {
    std::vector<int> vec(1000000);
    std::iota(vec.begin(), vec.end(), 1); // Fill with 1, 2, 3, ...
    
    // Sorting: O(n log n) guaranteed (introsort)
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(vec.begin(), vec.end());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Binary search: O(log n) - requires sorted range
    bool found = std::binary_search(vec.begin(), vec.end(), 500000);
    
    // Linear search: O(n)
    auto it = std::find(vec.begin(), vec.end(), 500000);
    
    // Heap operations: O(log n)
    std::make_heap(vec.begin(), vec.end());    // O(n)
    std::push_heap(vec.begin(), vec.end());    // O(log n)
    std::pop_heap(vec.begin(), vec.end());     // O(log n)
    
    // Partitioning: O(n)
    auto partition_point = std::partition(vec.begin(), vec.end(), 
                                        [](int x) { return x % 2 == 0; });
    
    // nth_element: O(n) average - partial sort
    std::nth_element(vec.begin(), vec.begin() + vec.size()/2, vec.end());
    int median = vec[vec.size()/2];
    
    // STL algorithm complexity categories:
    // O(1): front(), back(), size(), empty()
    // O(log n): binary_search, lower_bound, upper_bound
    // O(n): find, count, copy, fill, transform
    // O(n log n): sort, stable_sort, partial_sort
    // O(n²): none in STL (by design)
}

// Custom algorithm with complexity guarantee
template<typename ForwardIt, typename T>
ForwardIt linear_search(ForwardIt first, ForwardIt last, const T& value) {
    // Complexity: O(n) where n = distance(first, last)
    // Space: O(1)
    for (; first != last; ++first) {
        if (*first == value) {
            return first;
        }
    }
    return last;
}

// Parallel algorithm complexity (C++17)
void parallelAlgorithms() {
    std::vector<int> vec(1000000);
    
    // Parallel sort: O(n log n) with better constant factor
    std::sort(std::execution::par_unseq, vec.begin(), vec.end());
    
    // Parallel transform: O(n) with parallelization
    std::transform(std::execution::par, vec.begin(), vec.end(), vec.begin(),
                   [](int x) { return x * x; });
}
```

### Template metaprogramming examples

```cpp
#include <type_traits>
#include <iostream>

// 1. Compile-time factorial
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// 2. SFINAE (Substitution Failure Is Not An Error)
template<typename T>
typename std::enable_if_t<std::is_integral_v<T>, void>
process(T value) {
    std::cout << "Processing integer: " << value << std::endl;
}

template<typename T>
typename std::enable_if_t<std::is_floating_point_v<T>, void>
process(T value) {
    std::cout << "Processing float: " << value << std::endl;
}

// 3. Modern template metaprogramming with constexpr if (C++17)
template<typename T>
void modernProcess(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Integer: " << value << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Float: " << value << std::endl;
    } else {
        std::cout << "Other type\n";
    }
}

// 4. Variadic templates
template<typename T>
T sum(T value) {
    return value;
}

template<typename T, typename... Args>
T sum(T first, Args... args) {
    return first + sum(args...);
}

// Modern fold expressions (C++17)
template<typename... Args>
auto modernSum(Args... args) {
    return (args + ...); // Right fold
}

// 5. Template specialization
template<typename T>
class Container {
public:
    void info() { std::cout << "Generic container\n"; }
};

template<>
class Container<bool> {
public:
    void info() { std::cout << "Specialized bool container\n"; }
};

// 6. Concepts (C++20)
#if __cplusplus >= 202002L
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T multiply(T a, T b) {
    return a * b;
}
#endif

// 7. Type list and manipulation
template<typename... Types>
struct TypeList {};

template<typename List>
struct Length;

template<typename... Types>
struct Length<TypeList<Types...>> {
    static constexpr size_t value = sizeof...(Types);
};

// 8. Policy-based design
template<typename T, template<typename> class OwnershipPolicy>
class SmartPtr : private OwnershipPolicy<T> {
private:
    T* ptr;
    
public:
    explicit SmartPtr(T* p) : ptr(p) {}
    
    ~SmartPtr() {
        OwnershipPolicy<T>::dispose(ptr);
    }
    
    T& operator*() { return *ptr; }
    T* operator->() { return ptr; }
};

template<typename T>
class DeleteOwnership {
public:
    static void dispose(T* ptr) { delete ptr; }
};

template<typename T>
class ArrayDeleteOwnership {
public:
    static void dispose(T* ptr) { delete[] ptr; }
};

void templateExamples() {
    // Compile-time computation
    constexpr int fact5 = Factorial<5>::value; // Computed at compile time
    
    // SFINAE
    process(42);      // Calls integer version
    process(3.14);    // Calls float version
    
    // Variadic templates
    int total = sum(1, 2, 3, 4, 5); // 15
    auto modern_total = modernSum(1, 2, 3, 4, 5); // 15
    
    // Specialization
    Container<int> intContainer;
    Container<bool> boolContainer;
    intContainer.info();  // "Generic container"
    boolContainer.info(); // "Specialized bool container"
    
    // Type list
    using MyTypes = TypeList<int, double, std::string>;
    constexpr size_t typeCount = Length<MyTypes>::value; // 3
    
    // Policy-based design
    SmartPtr<int, DeleteOwnership> ptr1(new int(42));
    SmartPtr<int, ArrayDeleteOwnership> ptr2(new int[10]);
}
```

### STL allocators and custom allocators

```cpp
#include <memory>
#include <vector>
#include <list>

// 1. Understanding default allocator
template<typename T>
void defaultAllocatorExample() {
    std::allocator<T> alloc;
    
    // Allocate memory for 10 objects
    T* ptr = alloc.allocate(10);
    
    // Construct objects
    for (size_t i = 0; i < 10; ++i) {
        std::allocator_traits<std::allocator<T>>::construct(alloc, ptr + i, T{});
    }
    
    // Destroy objects
    for (size_t i = 0; i < 10; ++i) {
        std::allocator_traits<std::allocator<T>>::destroy(alloc, ptr + i);
    }
    
    // Deallocate memory
    alloc.deallocate(ptr, 10);
}

// 2. Custom allocator: Memory pool
template<typename T>
class PoolAllocator {
private:
    struct Pool {
        alignas(T) char memory[sizeof(T) * 1024];
        std::vector<bool> used;
        
        Pool() : used(1024, false) {}
    };
    
    static inline std::vector<std::unique_ptr<Pool>> pools;
    static inline size_t currentPool = 0;
    
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };
    
    PoolAllocator() = default;
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>&) {}
    
    pointer allocate(size_type n) {
        if (n != 1) {
            throw std::bad_alloc(); // Only single object allocation
        }
        
        // Find free slot in current pool
        if (pools.empty() || currentPool >= pools.size()) {
            pools.push_back(std::make_unique<Pool>());
            currentPool = pools.size() - 1;
        }
        
        auto& pool = *pools[currentPool];
        for (size_t i = 0; i < 1024; ++i) {
            if (!pool.used[i]) {
                pool.used[i] = true;
                return reinterpret_cast<pointer>(pool.memory + i * sizeof(T));
            }
        }
        
        // Current pool full, try next
        ++currentPool;
        return allocate(n);
    }
    
    void deallocate(pointer p, size_type n) {
        // Find which pool this pointer belongs to
        for (auto& pool : pools) {
            char* poolStart = pool->memory;
            char* poolEnd = poolStart + sizeof(pool->memory);
            char* ptr = reinterpret_cast<char*>(p);
            
            if (ptr >= poolStart && ptr < poolEnd) {
                size_t index = (ptr - poolStart) / sizeof(T);
                pool->used[index] = false;
                return;
            }
        }
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>&) const { return false; }
};

// 3. Stack allocator
template<typename T, size_t Size>
class StackAllocator {
private:
    alignas(T) char memory[Size * sizeof(T)];
    size_t offset = 0;
    
public:
    using value_type = T;
    
    template<typename U>
    struct rebind {
        using other = StackAllocator<U, Size>;
    };
    
    T* allocate(size_t n) {
        if (offset + n > Size) {
            throw std::bad_alloc();
        }
        
        T* ptr = reinterpret_cast<T*>(memory + offset * sizeof(T));
        offset += n;
        return ptr;
    }
    
    void deallocate(T* ptr, size_t n) {
        // Stack allocator: only reset when all memory deallocated
        if (ptr == reinterpret_cast<T*>(memory + (offset - n) * sizeof(T))) {
            offset -= n;
        }
    }
    
    void reset() { offset = 0; }
    
    template<typename U>
    bool operator==(const StackAllocator<U, Size>&) const { return false; }
};

// 4. Usage examples
void allocatorExamples() {
    // Default allocator
    std::vector<int> defaultVec = {1, 2, 3, 4, 5};
    
    // Custom pool allocator
    std::vector<int, PoolAllocator<int>> poolVec;
    poolVec.reserve(100);
    for (int i = 0; i < 100; ++i) {
        poolVec.push_back(i);
    }
    
    // Stack allocator
    std::vector<int, StackAllocator<int, 1000>> stackVec;
    stackVec.reserve(500);
    for (int i = 0; i < 500; ++i) {
        stackVec.push_back(i);
    }
    
    // List with custom allocator
    std::list<std::string, PoolAllocator<std::string>> poolList;
    poolList.push_back("Hello");
    poolList.push_back("World");
}

// 5. Allocator-aware container example
template<typename T, typename Allocator = std::allocator<T>>
class SimpleVector {
private:
    T* data;
    size_t size_;
    size_t capacity_;
    Allocator alloc;
    
public:
    explicit SimpleVector(const Allocator& a = Allocator()) 
        : data(nullptr), size_(0), capacity_(0), alloc(a) {}
    
    ~SimpleVector() {
        if (data) {
            for (size_t i = 0; i < size_; ++i) {
                std::allocator_traits<Allocator>::destroy(alloc, data + i);
            }
            std::allocator_traits<Allocator>::deallocate(alloc, data, capacity_);
        }
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        std::allocator_traits<Allocator>::construct(alloc, data + size_, value);
        ++size_;
    }
    
    void reserve(size_t newCapacity) {
        if (newCapacity <= capacity_) return;
        
        T* newData = std::allocator_traits<Allocator>::allocate(alloc, newCapacity);
        
        for (size_t i = 0; i < size_; ++i) {
            std::allocator_traits<Allocator>::construct(alloc, newData + i, 
                                                       std::move(data[i]));
            std::allocator_traits<Allocator>::destroy(alloc, data + i);
        }
        
        if (data) {
            std::allocator_traits<Allocator>::deallocate(alloc, data, capacity_);
        }
        
        data = newData;
        capacity_ = newCapacity;
    }
    
    size_t size() const { return size_; }
    T& operator[](size_t index) { return data[index]; }
};
```

## Modern C++ Features

### Lambda expressions and captures

```cpp
#include <algorithm>
#include <functional>
#include <memory>

void lambdaExpressions() {
    // 1. Basic lambda syntax
    auto simple = []() { return 42; };
    auto withParam = [](int x) { return x * 2; };
    auto withReturn = [](int x) -> double { return x * 1.5; };
    
    // 2. Capture modes
    int local = 10;
    std::unique_ptr<int> ptr = std::make_unique<int>(20);
    
    // Capture by value
    auto captureValue = [local](int x) { 
        return x + local; // local is copied
    };
    
    // Capture by reference
    auto captureRef = [&local](int x) { 
        local += x; // Modifies original local
        return local;
    };
    
    // Mixed capture
    auto mixedCapture = [local, &ptr](int x) {
        return x + local + *ptr;
    };
    
    // Capture all by value/reference
    auto captureAllValue = [=](int x) { return x + local; };
    auto captureAllRef = [&](int x) { local += x; return local; };
    
    // 3. C++14: Generalized capture (init capture)
    auto moveCapture = [p = std::move(ptr)](int x) {
        return x + *p;
    };
    
    auto computedCapture = [result = local * 2](int x) {
        return x + result;
    };
    
    // 4. Mutable lambdas
    auto mutableLambda = [local](int x) mutable {
        local += x; // Modifies captured copy
        return local;
    };
    
    // 5. Generic lambdas (C++14)
    auto genericLambda = [](auto x, auto y) {
        return x + y;
    };
    
    int intResult = genericLambda(1, 2);        // int + int
    double doubleResult = genericLambda(1.5, 2.5); // double + double
    std::string strResult = genericLambda(std::string("Hello"), std::string(" World"));
    
    // 6. Templated lambdas (C++20)
    #if __cplusplus >= 202002L
    auto templatedLambda = []<typename T>(T x, T y) {
        return x + y;
    };
    #endif
    
    // 7. Lambda as function objects
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    // Use with STL algorithms
    std::transform(vec.begin(), vec.end(), vec.begin(),
                   [](int x) { return x * x; });
    
    // Custom comparator
    std::sort(vec.begin(), vec.end(), 
              [](int a, int b) { return a > b; }); // Descending order
    
    // Find with predicate
    auto it = std::find_if(vec.begin(), vec.end(),
                          [](int x) { return x > 10; });
}

// Advanced lambda techniques
void advancedLambdas() {
    // 1. Recursive lambdas
    auto factorial = [](int n) {
        auto impl = [](int n, auto& self) -> int {
            return n <= 1 ? 1 : n * self(n - 1, self);
        };
        return impl(n, impl);
    };
    
    // C++14 approach
    std::function<int(int)> factorialFunc = [&](int n) -> int {
        return n <= 1 ? 1 : n * factorialFunc(n - 1);
    };
    
    // 2. Currying with lambdas
    auto add = [](int x) {
        return [x](int y) {
            return x + y;
        };
    };
    
    auto add5 = add(5);
    int result = add5(3); // 8
    
    // 3. Lambda as template parameter
    template<typename Func>
    void applyToVector(std::vector<int>& vec, Func f) {
        std::transform(vec.begin(), vec.end(), vec.begin(), f);
    }
    
    std::vector<int> nums = {1, 2, 3, 4, 5};
    applyToVector(nums, [](int x) { return x * x; });
    
    // 4. Stateful lambdas
    auto counter = [count = 0]() mutable { return ++count; };
    int first = counter();  // 1
    int second = counter(); // 2
    
    // 5. Perfect forwarding in lambdas
    auto forwardingLambda = [](auto&&... args) {
        return someFunction(std::forward<decltype(args)>(args)...);
    };
}

// Lambda use cases
class LambdaUseCases {
public:
    // 1. Event handling
    void setupEventHandlers() {
        auto onButtonClick = [this](int buttonId) {
            handleButtonClick(buttonId);
        };
        
        auto onKeyPress = [](char key) {
            std::cout << "Key pressed: " << key << std::endl;
        };
    }
    
    // 2. Custom deleter for smart pointers
    void customDeleter() {
        auto fileDeleter = [](FILE* f) {
            if (f) {
                std::fclose(f);
                std::cout << "File closed\n";
            }
        };
        
        std::unique_ptr<FILE, decltype(fileDeleter)> file(
            std::fopen("test.txt", "r"), fileDeleter);
    }
    
    // 3. Factory pattern with lambdas
    std::function<std::unique_ptr<Base>()> createFactory(const std::string& type) {
        if (type == "TypeA") {
            return []() { return std::make_unique<TypeA>(); };
        } else if (type == "TypeB") {
            return []() { return std::make_unique<TypeB>(); };
        }
        return nullptr;
    }
    
private:
    void handleButtonClick(int id) { /* implementation */ }
    
    class Base { public: virtual ~Base() = default; };
    class TypeA : public Base {};
    class TypeB : public Base {};
};
```

### constexpr and compile-time computation

```cpp
#include <array>
#include <string_view>

// 1. constexpr functions
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

constexpr bool isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

// 2. constexpr variables
constexpr int fact5 = factorial(5);  // Computed at compile time
constexpr bool is17Prime = isPrime(17); // true, computed at compile time

// 3. constexpr constructors and classes
class Point {
private:
    double x_, y_;
    
public:
    constexpr Point(double x, double y) : x_(x), y_(y) {}
    
    constexpr double x() const { return x_; }
    constexpr double y() const { return y_; }
    
    constexpr double distance() const {
        return std::sqrt(x_ * x_ + y_ * y_); // sqrt is constexpr in C++26
    }
    
    constexpr Point operator+(const Point& other) const {
        return Point(x_ + other.x_, y_ + other.y_);
    }
};

constexpr Point origin(0, 0);
constexpr Point p1(3, 4);
constexpr Point sum = origin + p1; // Computed at compile time

// 4. constexpr algorithms
template<typename T, size_t N>
constexpr std::array<T, N> generateSequence() {
    std::array<T, N> arr{};
    for (size_t i = 0; i < N; ++i) {
        arr[i] = static_cast<T>(i * i);
    }
    return arr;
}

constexpr auto squares = generateSequence<int, 10>(); // Compile-time array

// 5. constexpr string processing
constexpr size_t stringLength(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') ++len;
    return len;
}

constexpr bool stringEqual(const char* a, const char* b) {
    while (*a && *b && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

constexpr size_t helloLen = stringLength("Hello"); // 5, computed at compile time

// 6. C++20: constexpr dynamic allocation
#if __cplusplus >= 202002L
constexpr int computeSum(int n) {
    auto* arr = new int[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = i;
    }
    
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    
    delete[] arr;
    return sum;
}

constexpr int sum100 = computeSum(100); // Compile-time computation with dynamic allocation
#endif

// 7. consteval (C++20) - must be evaluated at compile time
#if __cplusplus >= 202002L
consteval int mustBeCompileTime(int n) {
    return n * n;
}

constexpr int someValue = 10;
constexpr int result = mustBeCompileTime(someValue); // OK
// int runtime = mustBeCompileTime(getValue()); // ERROR if getValue() not constexpr
#endif

// 8. if constexpr for template metaprogramming
template<typename T>
constexpr auto getValue(T&& t) {
    if constexpr (std::is_same_v<std::decay_t<T>, std::string>) {
        return t.size();
    } else if constexpr (std::is_arithmetic_v<T>) {
        return t * 2;
    } else {
        return sizeof(T);
    }
}

// 9. Compile-time map/lookup table
template<size_t N>
struct CompileTimeMap {
    struct Pair {
        int key;
        int value;
    };
    
    std::array<Pair, N> data;
    
    constexpr CompileTimeMap(std::initializer_list<Pair> init) {
        size_t i = 0;
        for (auto& p : init) {
            data[i++] = p;
        }
    }
    
    constexpr int find(int key) const {
        for (const auto& p : data) {
            if (p.key == key) return p.value;
        }
        return -1; // Not found
    }
};

constexpr auto errorCodes = CompileTimeMap<3>{
    {404, 1}, {500, 2}, {403, 3}
};

constexpr int code404 = errorCodes.find(404); // 1, computed at compile time

void constexprExamples() {
    // All these computations happen at compile time
    static_assert(factorial(5) == 120);
    static_assert(isPrime(17));
    static_assert(squares[3] == 9);
    static_assert(helloLen == 5);
    
    // Runtime vs compile-time
    int runtimeValue = 10;
    
    // This is computed at compile time
    constexpr int compileTime = factorial(5);
    
    // This might be computed at runtime (depends on optimization)
    int maybeRuntime = factorial(runtimeValue);
    
    std::cout << "Compile-time factorial: " << compileTime << std::endl;
    std::cout << "Runtime factorial: " << maybeRuntime << std::endl;
}

### Structured bindings

```cpp
#include <tuple>
#include <map>
#include <array>

// 1. Basic structured bindings
void basicStructuredBindings() {
    // Tuple decomposition
    std::tuple<int, std::string, double> data{42, "hello", 3.14};
    auto [id, name, value] = data;
    
    std::cout << "ID: " << id << ", Name: " << name << ", Value: " << value << std::endl;
    
    // Pair decomposition
    std::pair<int, std::string> p{100, "world"};
    auto [number, text] = p;
    
    // Array decomposition
    std::array<int, 3> arr{1, 2, 3};
    auto [a, b, c] = arr;
    
    // C-style array
    int cArray[2] = {10, 20};
    auto [x, y] = cArray;
}

// 2. Structured bindings with containers
void containerStructuredBindings() {
    std::map<std::string, int> scores{
        {"Alice", 95}, {"Bob", 87}, {"Charlie", 92}
    };
    
    // Iterate with structured bindings
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << std::endl;
    }
    
    // Insert with structured bindings
    auto [it, inserted] = scores.insert({"David", 88});
    if (inserted) {
        std::cout << "Inserted David with score " << it->second << std::endl;
    }
    
    // Find with structured bindings
    if (auto it = scores.find("Alice"); it != scores.end()) {
        auto [name, score] = *it;
        std::cout << "Found: " << name << " -> " << score << std::endl;
    }
}

// 3. Custom types with structured bindings
struct Point3D {
    double x, y, z;
};

// Method 1: Tuple-like interface
template<size_t I>
auto get(const Point3D& p) {
    if constexpr (I == 0) return p.x;
    else if constexpr (I == 1) return p.y;
    else if constexpr (I == 2) return p.z;
}

// Required specializations for tuple-like interface
namespace std {
    template<>
    struct tuple_size<Point3D> : std::integral_constant<size_t, 3> {};
    
    template<size_t I>
    struct tuple_element<I, Point3D> {
        using type = double;
    };
}

// Method 2: Public data members (automatic)
struct Rectangle {
    double width, height;
    // Automatically supports structured bindings
};

void customTypeBindings() {
    Point3D point{1.0, 2.0, 3.0};
    auto [x, y, z] = point; // Uses get<I> functions
    
    Rectangle rect{10.0, 5.0};
    auto [w, h] = rect; // Uses public members directly
    
    std::cout << "Point: (" << x << ", " << y << ", " << z << ")" << std::endl;
    std::cout << "Rectangle: " << w << " x " << h << std::endl;
}

// 4. Reference binding
void referenceBindings() {
    std::tuple<int, std::string> data{42, "hello"};
    
    // Bind to references
    auto& [id, name] = data;
    id = 100;  // Modifies original tuple
    name = "world";
    
    // Const references
    const auto& [constId, constName] = data;
    // constId = 200; // Error: cannot modify const reference
    
    std::cout << "Modified: " << std::get<0>(data) << ", " << std::get<1>(data) << std::endl;
}

// 5. Lambda with structured bindings
void lambdaWithBindings() {
    std::vector<std::pair<std::string, int>> students{
        {"Alice", 95}, {"Bob", 87}, {"Charlie", 92}
    };
    
    // Sort by score
    std::sort(students.begin(), students.end(),
              [](const auto& a, const auto& b) {
                  auto [nameA, scoreA] = a;
                  auto [nameB, scoreB] = b;
                  return scoreA > scoreB;
              });
    
    // Print results
    std::for_each(students.begin(), students.end(),
                  [](const auto& student) {
                      auto [name, score] = student;
                      std::cout << name << ": " << score << std::endl;
                  });
}

// 6. Multiple return values
std::tuple<bool, int, std::string> processData(int input) {
    if (input < 0) {
        return {false, 0, "Invalid input"};
    }
    return {true, input * 2, "Success"};
}

void multipleReturnValues() {
    int userInput = 42;
    auto [success, result, message] = processData(userInput);
    
    if (success) {
        std::cout << "Result: " << result << ", Message: " << message << std::endl;
    } else {
        std::cout << "Error: " << message << std::endl;
    }
}
```

### Concepts and constraints (C++20)

```cpp
#if __cplusplus >= 202002L
#include <concepts>
#include <type_traits>

// 1. Basic concepts
template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Numeric = Integral<T> || FloatingPoint<T>;

// Using concepts
template<Numeric T>
T add(T a, T b) {
    return a + b;
}

// 2. More complex concepts
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<size_t>;
};

template<typename T>
concept Sortable = requires(T t) {
    { t.begin() } -> std::random_access_iterator;
    { t.end() } -> std::random_access_iterator;
    requires std::sortable<typename T::iterator>;
};

// 3. Concepts with parameters
template<typename T, typename U>
concept Comparable = requires(T t, U u) {
    { t == u } -> std::convertible_to<bool>;
    { t != u } -> std::convertible_to<bool>;
    { t < u } -> std::convertible_to<bool>;
    { t > u } -> std::convertible_to<bool>;
};

template<typename Iter, typename Value>
concept IteratorFor = requires(Iter it, Value v) {
    *it = v;
    ++it;
    it++;
    { it == it } -> std::convertible_to<bool>;
    { it != it } -> std::convertible_to<bool>;
};

// 4. Function constraints
template<Container C>
void printContainer(const C& container) {
    for (const auto& item : container) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

template<Sortable S>
void sortAndPrint(S& container) {
    std::sort(container.begin(), container.end());
    printContainer(container);
}

// 5. Abbreviated function templates
void printNumeric(Numeric auto value) {
    std::cout << "Numeric value: " << value << std::endl;
}

auto addTwoNumbers(Numeric auto a, Numeric auto b) {
    return a + b;
}

// 6. Concept specialization
template<typename T>
void process(T value) {
    std::cout << "Generic processing\n";
}

template<Integral T>
void process(T value) {
    std::cout << "Integer processing: " << value << std::endl;
}

template<FloatingPoint T>
void process(T value) {
    std::cout << "Float processing: " << value << std::endl;
}

// 7. Custom concepts for classes
template<typename T>
concept Drawable = requires(T t) {
    t.draw();
    { t.getArea() } -> std::convertible_to<double>;
};

template<typename T>
concept Movable = requires(T t) {
    t.move(0.0, 0.0);
    { t.getPosition() } -> std::convertible_to<std::pair<double, double>>;
};

template<typename T>
concept Shape = Drawable<T> && Movable<T>;

class Circle {
public:
    void draw() const { std::cout << "Drawing circle\n"; }
    double getArea() const { return 3.14159 * radius * radius; }
    void move(double x, double y) { centerX = x; centerY = y; }
    std::pair<double, double> getPosition() const { return {centerX, centerY}; }
    
private:
    double radius = 1.0;
    double centerX = 0.0, centerY = 0.0;
};

template<Shape S>
void manipulateShape(S& shape) {
    shape.draw();
    std::cout << "Area: " << shape.getArea() << std::endl;
    shape.move(10.0, 20.0);
}

// 8. Requires clause
template<typename T>
requires Numeric<T> && (!std::same_as<T, bool>)
T multiply(T a, T b) {
    return a * b;
}

template<typename Iter>
requires std::random_access_iterator<Iter>
void advanceIterator(Iter& it, int n) {
    it += n; // Only works with random access iterators
}

// 9. Concept subsumption
template<typename T>
concept SignedIntegral = Integral<T> && std::signed_integral<T>;

template<Integral T>
void handleInteger(T value) {
    std::cout << "Handling integral type\n";
}

template<SignedIntegral T>  // More constrained, preferred
void handleInteger(T value) {
    std::cout << "Handling signed integral type\n";
}

void conceptExamples() {
    // Basic usage
    int a = 5, b = 10;
    double x = 3.14, y = 2.71;
    
    auto intSum = add(a, b);        // Works with Integral
    auto floatSum = add(x, y);      // Works with FloatingPoint
    
    // Container concepts
    std::vector<int> vec{3, 1, 4, 1, 5};
    std::list<std::string> lst{"hello", "world"};
    
    printContainer(vec);    // Works with Container concept
    printContainer(lst);    // Works with Container concept
    sortAndPrint(vec);      // Works with Sortable concept
    // sortAndPrint(lst);   // Error: list is not Sortable
    
    // Abbreviated templates
    printNumeric(42);       // Works with int
    printNumeric(3.14);     // Works with double
    // printNumeric("hello"); // Error: string is not Numeric
    
    // Concept specialization
    process(42);            // Calls Integral version
    process(3.14);          // Calls FloatingPoint version
    process(std::string{"hello"}); // Calls generic version
    
    // Shape concept
    Circle circle;
    manipulateShape(circle); // Works because Circle satisfies Shape concept
}

#endif // C++20
```

### Ranges and views (C++20)

```cpp
#if __cplusplus >= 202002L
#include <ranges>
#include <vector>
#include <algorithm>
#include <iostream>

void rangesBasics() {
    std::vector<int> numbers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. Basic ranges operations
    auto evens = numbers | std::views::filter([](int n) { return n % 2 == 0; });
    auto squares = evens | std::views::transform([](int n) { return n * n; });
    auto first_three = squares | std::views::take(3);
    
    std::cout << "First three squares of even numbers: ";
    for (auto n : first_three) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // 2. Range algorithms
    std::ranges::sort(numbers, std::greater{});
    std::cout << "Sorted (descending): ";
    std::ranges::copy(numbers, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    
    // 3. Projections
    std::vector<std::string> words{"apple", "banana", "cherry", "date"};
    std::ranges::sort(words, {}, &std::string::size); // Sort by length
    
    std::cout << "Sorted by length: ";
    for (const auto& word : words) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
}

void rangesViews() {
    std::vector<int> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // 1. Common views
    auto filtered = data | std::views::filter([](int x) { return x > 5; });
    auto transformed = data | std::views::transform([](int x) { return x * x; });
    auto dropped = data | std::views::drop(3);        // Skip first 3
    auto taken = data | std::views::take(5);          // Take first 5
    auto reversed = data | std::views::reverse;
    
    // 2. View composition
    auto complex_view = data 
        | std::views::filter([](int x) { return x % 2 == 0; })  // Even numbers
        | std::views::transform([](int x) { return x * x; })     // Square them
        | std::views::take(3)                                    // Take first 3
        | std::views::reverse;                                   // Reverse order
    
    std::cout << "Complex view result: ";
    for (auto n : complex_view) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // 3. Generated views
    auto iota_view = std::views::iota(1, 11);  // Numbers 1 to 10
    auto repeat_view = std::views::repeat(42, 5); // 42 repeated 5 times
    
    // 4. Split view
    std::string text = "hello,world,ranges,views";
    auto words = text | std::views::split(',');
    
    std::cout << "Split words: ";
    for (auto word : words) {
        for (char c : word) {
            std::cout << c;
        }
        std::cout << " ";
    }
    std::cout << std::endl;
}

void customRanges() {
    // 1. Custom view
    template<std::ranges::input_range R>
    class enumerate_view : public std::ranges::view_interface<enumerate_view<R>> {
    private:
        R base_;
        
        template<bool Const>
        class iterator {
        private:
            using Base = std::conditional_t<Const, const R, R>;
            std::ranges::iterator_t<Base> current_;
            size_t index_;
            
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type = std::pair<size_t, std::ranges::range_reference_t<Base>>;
            using difference_type = std::ptrdiff_t;
            
            iterator() = default;
            iterator(std::ranges::iterator_t<Base> it, size_t idx) 
                : current_(it), index_(idx) {}
            
            auto operator*() const {
                return std::make_pair(index_, *current_);
            }
            
            iterator& operator++() {
                ++current_;
                ++index_;
                return *this;
            }
            
            iterator operator++(int) {
                auto tmp = *this;
                ++*this;
                return tmp;
            }
            
            bool operator==(const iterator& other) const {
                return current_ == other.current_;
            }
        };
        
    public:
        enumerate_view() = default;
        enumerate_view(R base) : base_(std::move(base)) {}
        
        auto begin() { return iterator<false>(std::ranges::begin(base_), 0); }
        auto end() { return iterator<false>(std::ranges::end(base_), 0); }
        
        auto begin() const { return iterator<true>(std::ranges::begin(base_), 0); }
        auto end() const { return iterator<true>(std::ranges::end(base_), 0); }
    };
    
    // 2. Range adaptor
    struct enumerate_adaptor {
        template<std::ranges::viewable_range R>
        auto operator()(R&& r) const {
            return enumerate_view{std::forward<R>(r)};
        }
    };
    
    inline constexpr enumerate_adaptor enumerate;
    
    // Usage
    std::vector<std::string> fruits{"apple", "banana", "cherry"};
    
    for (auto [index, fruit] : enumerate(fruits)) {
        std::cout << index << ": " << fruit << std::endl;
    }
}

void rangeAlgorithms() {
    std::vector<int> numbers{5, 2, 8, 1, 9, 3};
    std::vector<std::string> words{"apple", "banana", "cherry", "date"};
    
    // 1. Sorting and searching
    std::ranges::sort(numbers);
    bool found = std::ranges::binary_search(numbers, 5);
    
    auto it = std::ranges::lower_bound(numbers, 5);
    auto [min_it, max_it] = std::ranges::minmax_element(numbers);
    
    // 2. Set operations
    std::vector<int> set1{1, 2, 3, 4, 5};
    std::vector<int> set2{3, 4, 5, 6, 7};
    std::vector<int> result;
    
    std::ranges::set_intersection(set1, set2, std::back_inserter(result));
    
    // 3. Transformations
    std::vector<int> doubled;
    std::ranges::transform(numbers, std::back_inserter(doubled),
                          [](int x) { return x * 2; });
    
    // 4. Folding (C++23 preview)
    // auto sum = std::ranges::fold_left(numbers, 0, std::plus{});
    
    // Current alternative
    auto sum = std::accumulate(numbers.begin(), numbers.end(), 0);
    
    std::cout << "Sum: " << sum << std::endl;
}

void performanceConsiderations() {
    std::vector<int> large_data(1000000);
    std::iota(large_data.begin(), large_data.end(), 1);
    
    // Views are lazy - no computation until iteration
    auto expensive_view = large_data
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x * x; }) // Expensive operation
        | std::views::take(10); // Only first 10 elements computed
    
    // Only when we iterate do the computations happen
    std::cout << "First 10 cubes of even numbers: ";
    for (auto value : expensive_view) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Materialization when needed
    std::vector<int> materialized(expensive_view.begin(), expensive_view.end());
}

#endif // C++20
```

### Coroutines (C++20)

```cpp
#if __cplusplus >= 202002L
#include <coroutine>
#include <iostream>
#include <exception>
#include <thread>
#include <chrono>

// 1. Simple generator coroutine
template<typename T>
struct Generator {
    struct promise_type {
        T current_value;
        
        Generator get_return_object() {
            return Generator{Handle::from_promise(*this)};
        }
        
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        
        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }
        
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };
    
    using Handle = std::coroutine_handle<promise_type>;
    
    Handle h_;
    
    explicit Generator(Handle h) : h_(h) {}
    ~Generator() { if (h_) h_.destroy(); }
    
    // Move only
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&& other) noexcept : h_(other.h_) {
        other.h_ = {};
    }
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (h_) h_.destroy();
            h_ = other.h_;
            other.h_ = {};
        }
        return *this;
    }
    
    bool next() {
        h_.resume();
        return !h_.done();
    }
    
    T value() const {
        return h_.promise().current_value;
    }
    
    // Iterator interface
    struct iterator {
        Generator* gen;
        
        iterator(Generator* g) : gen(g) {
            if (gen && !gen->next()) {
                gen = nullptr;
            }
        }
        
        T operator*() const { return gen->value(); }
        
        iterator& operator++() {
            if (gen && !gen->next()) {
                gen = nullptr;
            }
            return *this;
        }
        
        bool operator==(const iterator& other) const {
            return gen == other.gen;
        }
        
        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }
    };
    
    iterator begin() { return iterator(this); }
    iterator end() { return iterator(nullptr); }
};

// Example generator functions
Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        int next = a + b;
        a = b;
        b = next;
    }
}

Generator<int> range(int start, int end) {
    for (int i = start; i < end; ++i) {
        co_yield i;
    }
}

// 2. Task coroutine for async operations
template<typename T>
struct Task {
    struct promise_type {
        T result;
        std::exception_ptr exception;
        
        Task get_return_object() {
            return Task{Handle::from_promise(*this)};
        }
        
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        
        void return_value(T value) {
            result = value;
        }
        
        void unhandled_exception() {
            exception = std::current_exception();
        }
    };
    
    using Handle = std::coroutine_handle<promise_type>;
    
    Handle h_;
    
    explicit Task(Handle h) : h_(h) {}
    ~Task() { if (h_) h_.destroy(); }
    
    // Move only
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    Task(Task&& other) noexcept : h_(other.h_) {
        other.h_ = {};
    }
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (h_) h_.destroy();
            h_ = other.h_;
            other.h_ = {};
        }
        return *this;
    }
    
    T get() {
        if (!h_.done()) {
            h_.resume();
        }
        if (h_.promise().exception) {
            std::rethrow_exception(h_.promise().exception);
        }
        return h_.promise().result;
    }
    
    bool is_ready() const {
        return h_.done();
    }
};

// Example async functions
Task<int> async_computation(int x) {
    // Simulate async work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    co_return x * x;
}

Task<std::string> async_fetch_data(const std::string& url) {
    // Simulate network request
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    co_return "Data from " + url;
}

// 3. Awaitable type
struct Timer {
    std::chrono::milliseconds duration;
    
    bool await_ready() const noexcept {
        return duration.count() <= 0;
    }
    
    void await_suspend(std::coroutine_handle<> h) const {
        std::thread([h, d = duration]() {
            std::this_thread::sleep_for(d);
            h.resume();
        }).detach();
    }
    
    void await_resume() const noexcept {}
};

Timer delay(std::chrono::milliseconds ms) {
    return Timer{ms};
}

// Example coroutine using awaitable
Task<void> example_async_function() {
    std::cout << "Starting async operation\n";
    
    co_await delay(std::chrono::milliseconds(1000));
    std::cout << "After 1 second delay\n";
    
    auto result = co_await async_computation(42);
    std::cout << "Computation result: " << result << "\n";
    
    auto data = co_await async_fetch_data("https://example.com");
    std::cout << "Fetched: " << data << "\n";
}

// 4. Coroutine examples
void coroutineExamples() {
    // Generator example
    std::cout << "Fibonacci sequence (first 10): ";
    auto fib = fibonacci();
    int count = 0;
    for (auto value : fib) {
        if (count++ >= 10) break;
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Range generator
    std::cout << "Range 5-10: ";
    for (auto value : range(5, 10)) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    
    // Task example
    auto task = async_computation(10);
    std::cout << "Task result: " << task.get() << std::endl;
    
    // Async function
    auto async_task = example_async_function();
    // In a real application, you would await or schedule this task
}

#endif // C++20

## Multithreading and Synchronization

### Thread creation and management

```cpp
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <chrono>
#include <future>

void basicThreading() {
    // 1. Basic thread creation
    std::thread t1([]() {
        std::cout << "Hello from thread " << std::this_thread::get_id() << std::endl;
    });
    
    // 2. Thread with parameters
    auto worker = [](int id, const std::string& message) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Worker " << id << ": " << message << std::endl;
    };
    
    std::thread t2(worker, 1, "Processing data");
    std::thread t3(worker, 2, "Computing results");
    
    // 3. Member function thread
    class TaskProcessor {
    public:
        void process(int taskId) {
            std::cout << "Processing task " << taskId << std::endl;
        }
    };
    
    TaskProcessor processor;
    std::thread t4(&TaskProcessor::process, &processor, 42);
    
    // 4. Thread pool simulation
    std::vector<std::thread> threads;
    const int numThreads = std::thread::hardware_concurrency();
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([i]() {
            std::cout << "Worker thread " << i << " on core\n";
        });
    }
    
    // Wait for all threads
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "All threads completed\n";
}

void threadManagement() {
    std::atomic<bool> stopFlag{false};
    std::atomic<int> counter{0};
    
    // 1. Detached thread (daemon)
    std::thread daemon([&]() {
        while (!stopFlag.load()) {
            counter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    daemon.detach(); // Thread runs independently
    
    // 2. RAII thread wrapper
    class ThreadGuard {
    private:
        std::thread& t;
    public:
        explicit ThreadGuard(std::thread& thread) : t(thread) {}
        ~ThreadGuard() {
            if (t.joinable()) {
                t.join();
            }
        }
        ThreadGuard(const ThreadGuard&) = delete;
        ThreadGuard& operator=(const ThreadGuard&) = delete;
    };
    
    std::thread worker([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "Worker completed\n";
    });
    
    ThreadGuard guard(worker); // Automatically joins on destruction
    
    // 3. Thread interruption simulation
    class InterruptibleThread {
    private:
        std::thread thread_;
        std::atomic<bool> interrupted_{false};
        
    public:
        template<typename F>
        InterruptibleThread(F&& f) {
            thread_ = std::thread([this, f = std::forward<F>(f)]() {
                try {
                    f([this]() { 
                        if (interrupted_.load()) {
                            throw std::runtime_error("Thread interrupted");
                        }
                    });
                } catch (const std::runtime_error&) {
                    std::cout << "Thread was interrupted\n";
                }
            });
        }
        
        void interrupt() {
            interrupted_.store(true);
        }
        
        void join() {
            if (thread_.joinable()) {
                thread_.join();
            }
        }
    };
    
    InterruptibleThread interruptible([](auto checkInterrupt) {
        for (int i = 0; i < 1000; ++i) {
            checkInterrupt(); // Check for interruption
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::cout << "Work completed normally\n";
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    interruptible.interrupt();
    interruptible.join();
    
    // Clean up
    stopFlag.store(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "Counter value: " << counter.load() << std::endl;
}

void threadLocalStorage() {
    // 1. Thread-local storage
    thread_local int tlsCounter = 0;
    
    auto increment = [](int iterations) {
        for (int i = 0; i < iterations; ++i) {
            ++tlsCounter; // Each thread has its own copy
        }
        std::cout << "Thread " << std::this_thread::get_id() 
                 << " counter: " << tlsCounter << std::endl;
    };
    
    std::thread t1(increment, 100);
    std::thread t2(increment, 200);
    std::thread t3(increment, 150);
    
    t1.join();
    t2.join();
    t3.join();
    
    // 2. Thread-local random number generator
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_int_distribution<> dis(1, 100);
    
    auto generateNumbers = [](int count) {
        std::vector<int> numbers;
        for (int i = 0; i < count; ++i) {
            numbers.push_back(dis(gen));
        }
        std::cout << "Thread generated " << count << " numbers\n";
    };
    
    std::vector<std::thread> generators;
    for (int i = 0; i < 3; ++i) {
        generators.emplace_back(generateNumbers, 10);
    }
    
    for (auto& t : generators) {
        t.join();
    }
}

void asyncProgramming() {
    // 1. std::async for simple parallel tasks
    auto future1 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return 42;
    });
    
    auto future2 = std::async(std::launch::async, []() {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        return 3.14;
    });
    
    // Do other work while tasks run
    std::cout << "Tasks started, doing other work...\n";
    
    // Get results (blocks if not ready)
    int result1 = future1.get();
    double result2 = future2.get();
    
    std::cout << "Results: " << result1 << ", " << result2 << std::endl;
    
    // 2. Promise/Future pair
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();
    
    std::thread producer([&promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        promise.set_value("Data from producer");
    });
    
    std::thread consumer([&future]() {
        std::string data = future.get(); // Blocks until data available
        std::cout << "Consumed: " << data << std::endl;
    });
    
    producer.join();
    consumer.join();
    
    // 3. Packaged task
    std::packaged_task<int(int)> task([](int x) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return x * x;
    });
    
    std::future<int> taskResult = task.get_future();
    std::thread taskThread(std::move(task), 10);
    
    std::cout << "Packaged task result: " << taskResult.get() << std::endl;
    taskThread.join();
}

### Different types of mutexes

```cpp
#include <mutex>
#include <shared_mutex>
#include <recursive_mutex>
#include <timed_mutex>

void mutexTypes() {
    // 1. Basic mutex
    std::mutex basicMutex;
    int sharedData = 0;
    
    auto basicWorker = [&]() {
        std::lock_guard<std::mutex> lock(basicMutex);
        ++sharedData; // Critical section
    };
    
    // 2. Recursive mutex - allows same thread to lock multiple times
    std::recursive_mutex recursiveMutex;
    
    std::function<int(int)> recursiveFunction = [&](int n) -> int {
        std::lock_guard<std::recursive_mutex> lock(recursiveMutex);
        if (n <= 1) return 1;
        return n * recursiveFunction(n - 1); // Reentrant call
    };
    
    // 3. Timed mutex - allows timeout on lock attempts
    std::timed_mutex timedMutex;
    
    auto timedWorker = [&](int id) {
        using namespace std::chrono_literals;
        
        if (timedMutex.try_lock_for(100ms)) {
            std::cout << "Thread " << id << " acquired lock\n";
            std::this_thread::sleep_for(50ms);
            timedMutex.unlock();
        } else {
            std::cout << "Thread " << id << " timeout\n";
        }
    };
    
    // 4. Shared mutex (reader-writer lock)
    std::shared_mutex sharedMutex;
    std::string sharedString = "initial data";
    
    auto reader = [&](int id) {
        std::shared_lock<std::shared_mutex> lock(sharedMutex);
        std::cout << "Reader " << id << ": " << sharedString << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };
    
    auto writer = [&](int id) {
        std::unique_lock<std::shared_mutex> lock(sharedMutex);
        sharedString += " + writer" + std::to_string(id);
        std::cout << "Writer " << id << " updated data\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    };
    
    // Test scenarios
    std::vector<std::thread> threads;
    
    // Multiple readers can access simultaneously
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(reader, i);
    }
    
    // Writer gets exclusive access
    threads.emplace_back(writer, 1);
    
    // More readers
    for (int i = 3; i < 6; ++i) {
        threads.emplace_back(reader, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

void lockingStrategies() {
    std::mutex mutex1, mutex2;
    
    // 1. RAII lock guards
    auto useGuards = [&]() {
        std::lock_guard<std::mutex> lock1(mutex1); // Automatic unlock
        // ... critical section ...
    }; // lock1 automatically released
    
    // 2. Unique lock - more flexible
    auto useUniqueLock = [&]() {
        std::unique_lock<std::mutex> lock(mutex1);
        
        // Can manually unlock/lock
        lock.unlock();
        // ... non-critical work ...
        lock.lock();
        // ... critical section ...
        
        // Can transfer ownership
        std::unique_lock<std::mutex> otherLock = std::move(lock);
    };
    
    // 3. Scoped lock - multiple mutexes (C++17)
    auto useMultipleLocks = [&]() {
        // Deadlock-safe locking of multiple mutexes
        std::scoped_lock lock(mutex1, mutex2);
        // Both mutexes locked in safe order
    };
    
    // 4. Try lock patterns
    auto tryLockPattern = [&]() {
        if (mutex1.try_lock()) {
            // Got the lock
            std::lock_guard<std::mutex> lock(mutex1, std::adopt_lock);
            // ... critical section ...
        } else {
            // Couldn't get lock, do alternative work
            std::cout << "Lock not available, doing other work\n";
        }
    };
    
    // 5. Defer lock pattern
    auto deferLockPattern = [&]() {
        std::unique_lock<std::mutex> lock1(mutex1, std::defer_lock);
        std::unique_lock<std::mutex> lock2(mutex2, std::defer_lock);
        
        // Lock both simultaneously (deadlock-safe)
        std::lock(lock1, lock2);
        
        // Now both are locked
    };
}

// Custom mutex wrapper with debugging
template<typename Mutex>
class DebugMutex {
private:
    Mutex mutex_;
    std::atomic<int> lockCount_{0};
    std::thread::id owner_;
    
public:
    void lock() {
        std::cout << "Thread " << std::this_thread::get_id() 
                 << " attempting to lock\n";
        mutex_.lock();
        owner_ = std::this_thread::get_id();
        lockCount_.fetch_add(1);
        std::cout << "Thread " << std::this_thread::get_id() 
                 << " acquired lock (count: " << lockCount_.load() << ")\n";
    }
    
    void unlock() {
        lockCount_.fetch_sub(1);
        std::cout << "Thread " << owner_ << " releasing lock\n";
        owner_ = std::thread::id{};
        mutex_.unlock();
    }
    
    bool try_lock() {
        bool success = mutex_.try_lock();
        if (success) {
            owner_ = std::this_thread::get_id();
            lockCount_.fetch_add(1);
            std::cout << "Thread " << std::this_thread::get_id() 
                     << " try_lock successful\n";
        }
        return success;
    }
};
```

### Condition variables usage

```cpp
#include <condition_variable>
#include <queue>

void conditionVariableBasics() {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::string data;
    
    // Producer thread
    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Prepare data
        {
            std::lock_guard<std::mutex> lock(mtx);
            data = "Important data";
            ready = true;
        }
        
        std::cout << "Producer: Data ready, notifying\n";
        cv.notify_one(); // Wake up one waiting thread
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        std::unique_lock<std::mutex> lock(mtx);
        
        // Wait until data is ready
        cv.wait(lock, [&]() { return ready; });
        
        std::cout << "Consumer: Received data: " << data << std::endl;
    });
    
    producer.join();
    consumer.join();
}

// Producer-Consumer queue using condition variables
template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mtx_;
    std::queue<T> queue_;
    std::condition_variable cv_;
    
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        queue_.push(item);
        cv_.notify_one();
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return !queue_.empty(); });
        
        T result = queue_.front();
        queue_.pop();
        return result;
    }
    
    bool tryPop(T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (queue_.empty()) {
            return false;
        }
        
        item = queue_.front();
        queue_.pop();
        return true;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.empty();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }
};

void producerConsumerExample() {
    ThreadSafeQueue<int> queue;
    std::atomic<bool> finished{false};
    
    // Multiple producers
    std::vector<std::thread> producers;
    for (int i = 0; i < 3; ++i) {
        producers.emplace_back([&queue, i]() {
            for (int j = 0; j < 10; ++j) {
                int value = i * 10 + j;
                queue.push(value);
                std::cout << "Producer " << i << " pushed " << value << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    // Multiple consumers
    std::vector<std::thread> consumers;
    for (int i = 0; i < 2; ++i) {
        consumers.emplace_back([&queue, &finished, i]() {
            while (!finished.load()) {
                int value;
                if (queue.tryPop(value)) {
                    std::cout << "Consumer " << i << " popped " << value << std::endl;
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
            
            // Process remaining items
            int value;
            while (queue.tryPop(value)) {
                std::cout << "Consumer " << i << " (cleanup) popped " << value << std::endl;
            }
        });
    }
    
    // Wait for producers to finish
    for (auto& p : producers) {
        p.join();
    }
    
    finished.store(true);
    
    // Wait for consumers to finish
    for (auto& c : consumers) {
        c.join();
    }
}

// Barrier implementation using condition variables
class Barrier {
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t count_;
    size_t waiting_;
    size_t generation_;
    
public:
    explicit Barrier(size_t count) : count_(count), waiting_(0), generation_(0) {}
    
    void wait() {
        std::unique_lock<std::mutex> lock(mtx_);
        size_t gen = generation_;
        
        if (++waiting_ == count_) {
            // Last thread to arrive
            waiting_ = 0;
            ++generation_;
            cv_.notify_all();
        } else {
            // Wait for all threads to arrive
            cv_.wait(lock, [this, gen]() { return gen != generation_; });
        }
    }
};

void barrierExample() {
    const int numThreads = 4;
    Barrier barrier(numThreads);
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&barrier, i]() {
            // Phase 1
            std::cout << "Thread " << i << " starting phase 1\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(50 * (i + 1)));
            std::cout << "Thread " << i << " finished phase 1\n";
            
            barrier.wait(); // Wait for all threads to complete phase 1
            
            // Phase 2
            std::cout << "Thread " << i << " starting phase 2\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            std::cout << "Thread " << i << " finished phase 2\n";
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}
```

### Atomic operations

```cpp
#include <atomic>

void atomicBasics() {
    // 1. Basic atomic types
    std::atomic<int> atomicInt{0};
    std::atomic<bool> atomicBool{false};
    std::atomic<double> atomicDouble{0.0};
    std::atomic<std::string*> atomicPtr{nullptr};
    
    // 2. Atomic operations
    atomicInt.store(42);                    // Atomic write
    int value = atomicInt.load();           // Atomic read
    int oldValue = atomicInt.exchange(100); // Atomic read-modify-write
    
    // 3. Compare and swap
    int expected = 100;
    bool success = atomicInt.compare_exchange_weak(expected, 200);
    if (success) {
        std::cout << "CAS succeeded, old value was 100\n";
    } else {
        std::cout << "CAS failed, actual value was " << expected << std::endl;
    }
    
    // 4. Arithmetic operations
    atomicInt.fetch_add(10);    // Returns old value, adds 10
    atomicInt.fetch_sub(5);     // Returns old value, subtracts 5
    atomicInt++;                // Equivalent to fetch_add(1) + 1
    ++atomicInt;                // Pre-increment
    
    // 5. Bitwise operations
    std::atomic<int> flags{0b1010};
    flags.fetch_or(0b0101);     // Bitwise OR
    flags.fetch_and(0b1100);    // Bitwise AND
    flags.fetch_xor(0b0011);    // Bitwise XOR
}

void memoryOrdering() {
    std::atomic<int> data{0};
    std::atomic<bool> flag{false};
    
    // 1. Memory ordering examples
    
    // Relaxed ordering - no synchronization, only atomicity
    auto relaxedWriter = [&]() {
        data.store(42, std::memory_order_relaxed);
        flag.store(true, std::memory_order_relaxed);
    };
    
    auto relaxedReader = [&]() {
        while (!flag.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
        }
        int value = data.load(std::memory_order_relaxed);
        std::cout << "Relaxed read: " << value << std::endl;
    };
    
    // Acquire-Release ordering - synchronizes with specific operations
    auto releaseWriter = [&]() {
        data.store(42, std::memory_order_relaxed);
        flag.store(true, std::memory_order_release); // Release barrier
    };
    
    auto acquireReader = [&]() {
        while (!flag.load(std::memory_order_acquire)) { // Acquire barrier
            std::this_thread::yield();
        }
        int value = data.load(std::memory_order_relaxed);
        std::cout << "Acquire-Release read: " << value << std::endl;
    };
    
    // Sequential consistency (default) - strongest ordering
    auto seqWriter = [&]() {
        data.store(42); // memory_order_seq_cst by default
        flag.store(true);
    };
    
    auto seqReader = [&]() {
        while (!flag.load()) {
            std::this_thread::yield();
        }
        int value = data.load();
        std::cout << "Sequential read: " << value << std::endl;
    };
    
    // Test different orderings
    std::thread writer(releaseWriter);
    std::thread reader(acquireReader);
    
    writer.join();
    reader.join();
}

// Lock-free data structures
template<typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        Node* next;
        
        Node(T value) : data(std::move(value)), next(nullptr) {}
    };
    
    std::atomic<Node*> head_{nullptr};
    
public:
    void push(T item) {
        Node* newNode = new Node(std::move(item));
        
        do {
            newNode->next = head_.load();
        } while (!head_.compare_exchange_weak(newNode->next, newNode));
    }
    
    bool pop(T& result) {
        Node* oldHead = head_.load();
        
        do {
            if (oldHead == nullptr) {
                return false; // Stack is empty
            }
        } while (!head_.compare_exchange_weak(oldHead, oldHead->next));
        
        result = oldHead->data;
        delete oldHead;
        return true;
    }
    
    bool empty() const {
        return head_.load() == nullptr;
    }
};

// Atomic reference counting
template<typename T>
class AtomicSharedPtr {
private:
    struct ControlBlock {
        std::atomic<int> refCount{1};
        T* ptr;
        
        ControlBlock(T* p) : ptr(p) {}
        
        void addRef() {
            refCount.fetch_add(1);
        }
        
        void release() {
            if (refCount.fetch_sub(1) == 1) {
                delete ptr;
                delete this;
            }
        }
    };
    
    ControlBlock* control_;
    
public:
    explicit AtomicSharedPtr(T* ptr = nullptr) 
        : control_(ptr ? new ControlBlock(ptr) : nullptr) {}
    
    AtomicSharedPtr(const AtomicSharedPtr& other) : control_(other.control_) {
        if (control_) {
            control_->addRef();
        }
    }
    
    AtomicSharedPtr& operator=(const AtomicSharedPtr& other) {
        if (this != &other) {
            if (control_) {
                control_->release();
            }
            control_ = other.control_;
            if (control_) {
                control_->addRef();
            }
        }
        return *this;
    }
    
    ~AtomicSharedPtr() {
        if (control_) {
            control_->release();
        }
    }
    
    T* get() const {
        return control_ ? control_->ptr : nullptr;
    }
    
    T& operator*() const { return *get(); }
    T* operator->() const { return get(); }
    
    int useCount() const {
        return control_ ? control_->refCount.load() : 0;
    }
};

void lockFreeExamples() {
    // Lock-free stack example
    LockFreeStack<int> stack;
    
    // Producer threads
    std::vector<std::thread> producers;
    for (int i = 0; i < 3; ++i) {
        producers.emplace_back([&stack, i]() {
            for (int j = 0; j < 10; ++j) {
                stack.push(i * 10 + j);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    
    // Consumer threads
    std::vector<std::thread> consumers;
    std::atomic<int> totalPopped{0};
    
    for (int i = 0; i < 2; ++i) {
        consumers.emplace_back([&stack, &totalPopped, i]() {
            int value;
            int count = 0;
            while (totalPopped.load() < 30) {
                if (stack.pop(value)) {
                    ++count;
                    totalPopped.fetch_add(1);
                    std::cout << "Consumer " << i << " popped " << value << std::endl;
                } else {
                    std::this_thread::yield();
                }
            }
            std::cout << "Consumer " << i << " popped " << count << " items total\n";
        });
    }
    
    for (auto& p : producers) {
        p.join();
    }
    
    for (auto& c : consumers) {
        c.join();
    }
}
```

### Memory ordering and memory model

```cpp
void memoryModelExamples() {
    // 1. Store-Load reordering example
    std::atomic<int> x{0}, y{0};
    std::atomic<int> r1{0}, r2{0};
    
    auto thread1 = [&]() {
        x.store(1, std::memory_order_relaxed);
        r1 = y.load(std::memory_order_relaxed);
    };
    
    auto thread2 = [&]() {
        y.store(1, std::memory_order_relaxed);
        r2 = x.load(std::memory_order_relaxed);
    };
    
    // With relaxed ordering, it's possible for both r1 and r2 to be 0
    // due to instruction reordering
    
    // 2. Acquire-Release chains
    std::atomic<int> data1{0}, data2{0};
    std::atomic<bool> flag1{false}, flag2{false};
    
    auto producer = [&]() {
        data1.store(42, std::memory_order_relaxed);
        flag1.store(true, std::memory_order_release); // Release operation
    };
    
    auto intermediate = [&]() {
        while (!flag1.load(std::memory_order_acquire)) { // Acquire operation
            std::this_thread::yield();
        }
        // Synchronized with producer, can safely read data1
        data2.store(data1.load(std::memory_order_relaxed) * 2, std::memory_order_relaxed);
        flag2.store(true, std::memory_order_release);
    };
    
    auto consumer = [&]() {
        while (!flag2.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        // Synchronized with intermediate, can safely read data2
        std::cout << "Final data: " << data2.load(std::memory_order_relaxed) << std::endl;
    };
    
    std::thread t1(producer);
    std::thread t2(intermediate);
    std::thread t3(consumer);
    
    t1.join();
    t2.join();
    t3.join();
    
    // 3. Memory fences
    std::atomic<int> a{0}, b{0};
    
    auto fenceExample1 = [&]() {
        a.store(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release); // Release fence
        b.store(1, std::memory_order_relaxed);
    };
    
    auto fenceExample2 = [&]() {
        while (b.load(std::memory_order_relaxed) == 0) {
            std::this_thread::yield();
        }
        std::atomic_thread_fence(std::memory_order_acquire); // Acquire fence
        assert(a.load(std::memory_order_relaxed) == 1); // Should always pass
    };
}

// Memory ordering performance comparison
void performanceComparison() {
    const int iterations = 1000000;
    std::atomic<int> counter{0};
    
    auto measureTime = [](auto func, const std::string& description) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << description << ": " << duration.count() << " microseconds\n";
    };
    
    // Sequential consistency (strongest, slowest)
    measureTime([&]() {
        for (int i = 0; i < iterations; ++i) {
            counter.store(i, std::memory_order_seq_cst);
            counter.load(std::memory_order_seq_cst);
        }
    }, "Sequential consistency");
    
    // Acquire-Release (balanced)
    measureTime([&]() {
        for (int i = 0; i < iterations; ++i) {
            counter.store(i, std::memory_order_release);
            counter.load(std::memory_order_acquire);
        }
    }, "Acquire-Release");
    
    // Relaxed (weakest, fastest)
    measureTime([&]() {
        for (int i = 0; i < iterations; ++i) {
            counter.store(i, std::memory_order_relaxed);
            counter.load(std::memory_order_relaxed);
        }
    }, "Relaxed");
}
```

## Design Patterns and Best Practices

### Singleton implementation

```cpp
#include <memory>
#include <mutex>
#include <call_once>

// 1. Thread-safe Singleton (C++11 magic statics)
class Singleton {
private:
    Singleton() = default;
    
public:
    static Singleton& getInstance() {
        static Singleton instance; // Thread-safe in C++11+
        return instance;
    }
    
    // Delete copy operations
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    
    void doSomething() {
        std::cout << "Singleton doing work\n";
    }
};

// 2. Thread-safe Singleton with std::call_once
class SingletonCallOnce {
private:
    static std::unique_ptr<SingletonCallOnce> instance_;
    static std::once_flag created_;
    
    SingletonCallOnce() = default;
    
public:
    static SingletonCallOnce& getInstance() {
        std::call_once(created_, []() {
            instance_ = std::unique_ptr<SingletonCallOnce>(new SingletonCallOnce);
        });
        return *instance_;
    }
    
    SingletonCallOnce(const SingletonCallOnce&) = delete;
    SingletonCallOnce& operator=(const SingletonCallOnce&) = delete;
};

std::unique_ptr<SingletonCallOnce> SingletonCallOnce::instance_ = nullptr;
std::once_flag SingletonCallOnce::created_;

// 3. Template Singleton base class
template<typename T>
class SingletonBase {
protected:
    SingletonBase() = default;
    virtual ~SingletonBase() = default;
    
public:
    static T& getInstance() {
        static T instance;
        return instance;
    }
    
    SingletonBase(const SingletonBase&) = delete;
    SingletonBase& operator=(const SingletonBase&) = delete;
};

class ConfigManager : public SingletonBase<ConfigManager> {
    friend class SingletonBase<ConfigManager>; // Allow base class access
    
private:
    std::map<std::string, std::string> config_;
    
    ConfigManager() {
        // Load configuration
        config_["database_url"] = "localhost:5432";
        config_["max_connections"] = "100";
    }
    
public:
    std::string getConfig(const std::string& key) const {
        auto it = config_.find(key);
        return it != config_.end() ? it->second : "";
    }
    
    void setConfig(const std::string& key, const std::string& value) {
        config_[key] = value;
    }
};

// 4. Lazy initialization with double-checked locking (legacy approach)
class SingletonDoubleChecked {
private:
    static SingletonDoubleChecked* instance_;
    static std::mutex mutex_;
    
    SingletonDoubleChecked() = default;
    
public:
    static SingletonDoubleChecked* getInstance() {
        if (instance_ == nullptr) { // First check
            std::lock_guard<std::mutex> lock(mutex_);
            if (instance_ == nullptr) { // Second check
                instance_ = new SingletonDoubleChecked();
            }
        }
        return instance_;
    }
    
    // Note: This approach has issues with memory ordering
    // and is not recommended in modern C++
};

SingletonDoubleChecked* SingletonDoubleChecked::instance_ = nullptr;
std::mutex SingletonDoubleChecked::mutex_;

// 5. Monostate pattern (alternative to Singleton)
class Monostate {
private:
    static int sharedState_;
    static std::string sharedData_;
    
public:
    void setState(int state) { sharedState_ = state; }
    int getState() const { return sharedState_; }
    
    void setData(const std::string& data) { sharedData_ = data; }
    std::string getData() const { return sharedData_; }
};

int Monostate::sharedState_ = 0;
std::string Monostate::sharedData_ = "";

void singletonExamples() {
    // Magic statics singleton
    auto& s1 = Singleton::getInstance();
    auto& s2 = Singleton::getInstance();
    // s1 and s2 refer to the same instance
    
    s1.doSomething();
    
    // Configuration manager
    auto& config = ConfigManager::getInstance();
    std::string dbUrl = config.getConfig("database_url");
    std::cout << "Database URL: " << dbUrl << std::endl;
    
    // Monostate - multiple objects, shared state
    Monostate m1, m2;
    m1.setState(42);
    std::cout << "m2 state: " << m2.getState() << std::endl; // 42
}
```

### Factory pattern variations

```cpp
#include <functional>
#include <unordered_map>

// Base product interface
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual double area() const = 0;
    virtual std::unique_ptr<Shape> clone() const = 0;
};

// Concrete products
class Circle : public Shape {
private:
    double radius_;
    
public:
    explicit Circle(double radius) : radius_(radius) {}
    
    void draw() const override {
        std::cout << "Drawing circle with radius " << radius_ << std::endl;
    }
    
    double area() const override {
        return 3.14159 * radius_ * radius_;
    }
    
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Circle>(radius_);
    }
};

class Rectangle : public Shape {
private:
    double width_, height_;
    
public:
    Rectangle(double width, double height) : width_(width), height_(height) {}
    
    void draw() const override {
        std::cout << "Drawing rectangle " << width_ << "x" << height_ << std::endl;
    }
    
    double area() const override {
        return width_ * height_;
    }
    
    std::unique_ptr<Shape> clone() const override {
        return std::make_unique<Rectangle>(width_, height_);
    }
};

// 1. Simple Factory
class ShapeFactory {
public:
    enum class ShapeType { Circle, Rectangle };
    
    static std::unique_ptr<Shape> createShape(ShapeType type, 
                                            const std::vector<double>& params) {
        switch (type) {
            case ShapeType::Circle:
                if (params.size() >= 1) {
                    return std::make_unique<Circle>(params[0]);
                }
                break;
            case ShapeType::Rectangle:
                if (params.size() >= 2) {
                    return std::make_unique<Rectangle>(params[0], params[1]);
                }
                break;
        }
        return nullptr;
    }
};

// 2. Factory Method Pattern
class ShapeCreator {
public:
    virtual ~ShapeCreator() = default;
    virtual std::unique_ptr<Shape> createShape() const = 0;
    
    // Template method using factory method
    void processShape() const {
        auto shape = createShape();
        shape->draw();
        std::cout << "Area: " << shape->area() << std::endl;
    }
};

class CircleCreator : public ShapeCreator {
private:
    double radius_;
    
public:
    explicit CircleCreator(double radius) : radius_(radius) {}
    
    std::unique_ptr<Shape> createShape() const override {
        return std::make_unique<Circle>(radius_);
    }
};

class RectangleCreator : public ShapeCreator {
private:
    double width_, height_;
    
public:
    RectangleCreator(double width, double height) : width_(width), height_(height) {}
    
    std::unique_ptr<Shape> createShape() const override {
        return std::make_unique<Rectangle>(width_, height_);
    }
};

// 3. Abstract Factory Pattern
class UIFactory {
public:
    virtual ~UIFactory() = default;
    virtual std::unique_ptr<class Button> createButton() const = 0;
    virtual std::unique_ptr<class Window> createWindow() const = 0;
};

class Button {
public:
    virtual ~Button() = default;
    virtual void click() const = 0;
};

class Window {
public:
    virtual ~Window() = default;
    virtual void show() const = 0;
};

// Windows implementations
class WindowsButton : public Button {
public:
    void click() const override {
        std::cout << "Windows button clicked\n";
    }
};

class WindowsWindow : public Window {
public:
    void show() const override {
        std::cout << "Showing Windows window\n";
    }
};

class WindowsUIFactory : public UIFactory {
public:
    std::unique_ptr<Button> createButton() const override {
        return std::make_unique<WindowsButton>();
    }
    
    std::unique_ptr<Window> createWindow() const override {
        return std::make_unique<WindowsWindow>();
    }
};

// Linux implementations
class LinuxButton : public Button {
public:
    void click() const override {
        std::cout << "Linux button clicked\n";
    }
};

class LinuxWindow : public Window {
public:
    void show() const override {
        std::cout << "Showing Linux window\n";
    }
};

class LinuxUIFactory : public UIFactory {
public:
    std::unique_ptr<Button> createButton() const override {
        return std::make_unique<LinuxButton>();
    }
    
    std::unique_ptr<Window> createWindow() const override {
        return std::make_unique<LinuxWindow>();
    }
};

// 4. Registry-based Factory
class ShapeRegistry {
private:
    using Creator = std::function<std::unique_ptr<Shape>(const std::vector<double>&)>;
    std::unordered_map<std::string, Creator> creators_;
    
public:
    void registerCreator(const std::string& name, Creator creator) {
        creators_[name] = creator;
    }
    
    std::unique_ptr<Shape> create(const std::string& name, 
                                const std::vector<double>& params) const {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second(params);
        }
        return nullptr;
    }
    
    std::vector<std::string> getAvailableShapes() const {
        std::vector<std::string> names;
        for (const auto& pair : creators_) {
            names.push_back(pair.first);
        }
        return names;
    }
};

// 5. Self-registering factory
template<typename Base, typename Derived>
class FactoryRegistrar {
public:
    static bool Register(const std::string& name) {
        ShapeRegistry::getInstance().registerCreator(name, 
            [](const std::vector<double>& params) -> std::unique_ptr<Shape> {
                // This would need proper parameter handling for each type
                if constexpr (std::is_same_v<Derived, Circle>) {
                    return params.size() >= 1 ? std::make_unique<Derived>(params[0]) : nullptr;
                } else if constexpr (std::is_same_v<Derived, Rectangle>) {
                    return params.size() >= 2 ? std::make_unique<Derived>(params[0], params[1]) : nullptr;
                }
                return nullptr;
            });
        return true;
    }
};

void factoryExamples() {
    // Simple Factory
    auto circle = ShapeFactory::createShape(ShapeFactory::ShapeType::Circle, {5.0});
    auto rectangle = ShapeFactory::createShape(ShapeFactory::ShapeType::Rectangle, {10.0, 5.0});
    
    if (circle) circle->draw();
    if (rectangle) rectangle->draw();
    
    // Factory Method
    CircleCreator circleCreator(3.0);
    RectangleCreator rectCreator(4.0, 6.0);
    
    circleCreator.processShape();
    rectCreator.processShape();
    
    // Abstract Factory
    std::unique_ptr<UIFactory> factory;
    
    #ifdef WINDOWS
    factory = std::make_unique<WindowsUIFactory>();
    #else
    factory = std::make_unique<LinuxUIFactory>();
    #endif
    
    auto button = factory->createButton();
    auto window = factory->createWindow();
    
    button->click();
    window->show();
    
    // Registry Factory
    ShapeRegistry registry;
    registry.registerCreator("circle", [](const std::vector<double>& params) {
        return params.size() >= 1 ? std::make_unique<Circle>(params[0]) : nullptr;
    });
    
    registry.registerCreator("rectangle", [](const std::vector<double>& params) {
        return params.size() >= 2 ? std::make_unique<Rectangle>(params[0], params[1]) : nullptr;
    });
    
    auto dynamicCircle = registry.create("circle", {7.0});
    if (dynamicCircle) dynamicCircle->draw();
}

### Observer pattern implementation

```cpp
#include <vector>
#include <algorithm>
#include <memory>
#include <functional>

// 1. Traditional Observer Pattern
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const std::string& message) = 0;
};

class Subject {
private:
    std::vector<Observer*> observers_;
    
public:
    virtual ~Subject() = default;
    
    void attach(Observer* observer) {
        observers_.push_back(observer);
    }
    
    void detach(Observer* observer) {
        observers_.erase(
            std::remove(observers_.begin(), observers_.end(), observer),
            observers_.end()
        );
    }
    
    void notify(const std::string& message) {
        for (auto* observer : observers_) {
            observer->update(message);
        }
    }
};

class NewsAgency : public Subject {
private:
    std::string news_;
    
public:
    void setNews(const std::string& news) {
        news_ = news;
        notify("News update: " + news);
    }
    
    std::string getNews() const { return news_; }
};

class NewsChannel : public Observer {
private:
    std::string name_;
    
public:
    explicit NewsChannel(const std::string& name) : name_(name) {}
    
    void update(const std::string& message) override {
        std::cout << name_ << " received: " << message << std::endl;
    }
};

// 2. Modern Observer with std::function and weak_ptr
template<typename... Args>
class ModernSubject {
private:
    using Callback = std::function<void(Args...)>;
    using WeakCallback = std::weak_ptr<Callback>;
    
    std::vector<std::shared_ptr<Callback>> callbacks_;
    
public:
    std::shared_ptr<Callback> subscribe(Callback cb) {
        auto callback = std::make_shared<Callback>(std::move(cb));
        callbacks_.push_back(callback);
        return callback;
    }
    
    void unsubscribe(const std::shared_ptr<Callback>& callback) {
        callbacks_.erase(
            std::remove(callbacks_.begin(), callbacks_.end(), callback),
            callbacks_.end()
        );
    }
    
    void notify(Args... args) {
        // Clean up expired callbacks
        callbacks_.erase(
            std::remove_if(callbacks_.begin(), callbacks_.end(),
                          [](const std::weak_ptr<Callback>& weak) {
                              return weak.expired();
                          }),
            callbacks_.end()
        );
        
        // Notify all active callbacks
        for (const auto& callback : callbacks_) {
            if (callback) {
                (*callback)(args...);
            }
        }
    }
};

// 3. Type-safe Observer with templates
template<typename EventType>
class TypedObserver {
public:
    virtual ~TypedObserver() = default;
    virtual void onEvent(const EventType& event) = 0;
};

template<typename EventType>
class EventDispatcher {
private:
    std::vector<TypedObserver<EventType>*> observers_;
    
public:
    void subscribe(TypedObserver<EventType>* observer) {
        observers_.push_back(observer);
    }
    
    void unsubscribe(TypedObserver<EventType>* observer) {
        observers_.erase(
            std::remove(observers_.begin(), observers_.end(), observer),
            observers_.end()
        );
    }
    
    void dispatch(const EventType& event) {
        for (auto* observer : observers_) {
            observer->onEvent(event);
        }
    }
};

// Event types
struct MouseClickEvent {
    int x, y;
    int button;
};

struct KeyPressEvent {
    char key;
    bool shift, ctrl, alt;
};

class InputHandler : public TypedObserver<MouseClickEvent>,
                     public TypedObserver<KeyPressEvent> {
private:
    std::string name_;
    
public:
    explicit InputHandler(const std::string& name) : name_(name) {}
    
    void onEvent(const MouseClickEvent& event) override {
        std::cout << name_ << " mouse click at (" << event.x << ", " << event.y 
                 << ") button " << event.button << std::endl;
    }
    
    void onEvent(const KeyPressEvent& event) override {
        std::cout << name_ << " key press: " << event.key << std::endl;
    }
};

// 4. Signal/Slot mechanism (similar to Qt)
template<typename... Args>
class Signal {
public:
    using SlotType = std::function<void(Args...)>;
    
private:
    std::vector<SlotType> slots_;
    
public:
    void connect(SlotType slot) {
        slots_.push_back(std::move(slot));
    }
    
    template<typename Object, typename Method>
    void connect(Object* obj, Method method) {
        slots_.push_back([obj, method](Args... args) {
            (obj->*method)(args...);
        });
    }
    
    void emit(Args... args) {
        for (const auto& slot : slots_) {
            slot(args...);
        }
    }
    
    void clear() {
        slots_.clear();
    }
};

class Button {
private:
    std::string text_;
    
public:
    Signal<> clicked;
    Signal<const std::string&> textChanged;
    
    explicit Button(const std::string& text) : text_(text) {}
    
    void click() {
        std::cout << "Button '" << text_ << "' clicked\n";
        clicked.emit();
    }
    
    void setText(const std::string& text) {
        if (text_ != text) {
            text_ = text;
            textChanged.emit(text_);
        }
    }
    
    std::string getText() const { return text_; }
};

class ButtonHandler {
public:
    void onButtonClicked() {
        std::cout << "Button click handled!\n";
    }
    
    void onTextChanged(const std::string& newText) {
        std::cout << "Button text changed to: " << newText << std::endl;
    }
};

void observerExamples() {
    // Traditional Observer
    NewsAgency agency;
    NewsChannel cnn("CNN");
    NewsChannel bbc("BBC");
    
    agency.attach(&cnn);
    agency.attach(&bbc);
    
    agency.setNews("Breaking: New C++ standard released!");
    
    agency.detach(&cnn);
    agency.setNews("Update: Features include concepts and coroutines");
    
    // Modern Observer with std::function
    ModernSubject<int, std::string> modernSubject;
    
    auto subscription1 = modernSubject.subscribe([](int id, const std::string& msg) {
        std::cout << "Subscriber 1 - ID: " << id << ", Message: " << msg << std::endl;
    });
    
    auto subscription2 = modernSubject.subscribe([](int id, const std::string& msg) {
        std::cout << "Subscriber 2 - ID: " << id << ", Message: " << msg << std::endl;
    });
    
    modernSubject.notify(1, "Hello World");
    
    modernSubject.unsubscribe(subscription1);
    modernSubject.notify(2, "Only subscriber 2 should see this");
    
    // Type-safe Observer
    EventDispatcher<MouseClickEvent> mouseDispatcher;
    EventDispatcher<KeyPressEvent> keyDispatcher;
    
    InputHandler handler("GameHandler");
    mouseDispatcher.subscribe(&handler);
    keyDispatcher.subscribe(&handler);
    
    mouseDispatcher.dispatch({100, 200, 1});
    keyDispatcher.dispatch({'A', false, true, false});
    
    // Signal/Slot mechanism
    Button button("OK");
    ButtonHandler buttonHandler;
    
    button.clicked.connect([](){ std::cout << "Lambda handler\n"; });
    button.clicked.connect(&buttonHandler, &ButtonHandler::onButtonClicked);
    button.textChanged.connect(&buttonHandler, &ButtonHandler::onTextChanged);
    
    button.click();
    button.setText("Cancel");
}

### SOLID principles in C++

```cpp
// 1. Single Responsibility Principle (SRP)
// BAD: Class has multiple responsibilities
class BadEmployee {
public:
    void calculatePay() { /* pay calculation logic */ }
    void saveToDatabase() { /* database logic */ }
    void generateReport() { /* reporting logic */ }
    void sendEmail() { /* email logic */ }
};

// GOOD: Each class has a single responsibility
class Employee {
private:
    std::string name_;
    double salary_;
    
public:
    Employee(const std::string& name, double salary) : name_(name), salary_(salary) {}
    
    std::string getName() const { return name_; }
    double getSalary() const { return salary_; }
    void setSalary(double salary) { salary_ = salary; }
};

class PayrollCalculator {
public:
    double calculatePay(const Employee& employee) {
        // Pay calculation logic
        return employee.getSalary() * 1.2; // Example with bonus
    }
};

class EmployeeRepository {
public:
    void save(const Employee& employee) {
        // Database saving logic
        std::cout << "Saving employee " << employee.getName() << " to database\n";
    }
    
    Employee load(const std::string& name) {
        // Database loading logic
        return Employee(name, 50000); // Example
    }
};

class ReportGenerator {
public:
    std::string generateReport(const std::vector<Employee>& employees) {
        std::string report = "Employee Report:\n";
        for (const auto& emp : employees) {
            report += emp.getName() + ": $" + std::to_string(emp.getSalary()) + "\n";
        }
        return report;
    }
};

// 2. Open/Closed Principle (OCP)
// BAD: Must modify existing code to add new shapes
class BadAreaCalculator {
public:
    double calculateArea(const std::vector<std::variant<Circle, Rectangle>>& shapes) {
        double totalArea = 0;
        for (const auto& shape : shapes) {
            std::visit([&](const auto& s) {
                // Must modify this code for new shapes
                if constexpr (std::is_same_v<std::decay_t<decltype(s)>, Circle>) {
                    totalArea += 3.14159 * s.getRadius() * s.getRadius();
                } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>, Rectangle>) {
                    totalArea += s.getWidth() * s.getHeight();
                }
            }, shape);
        }
        return totalArea;
    }
};

// GOOD: Open for extension, closed for modification
class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
};

class Circle : public Shape {
private:
    double radius_;
    
public:
    explicit Circle(double radius) : radius_(radius) {}
    double area() const override { return 3.14159 * radius_ * radius_; }
    double getRadius() const { return radius_; }
};

class Rectangle : public Shape {
private:
    double width_, height_;
    
public:
    Rectangle(double width, double height) : width_(width), height_(height) {}
    double area() const override { return width_ * height_; }
    double getWidth() const { return width_; }
    double getHeight() const { return height_; }
};

class Triangle : public Shape { // New shape - no existing code modification needed
private:
    double base_, height_;
    
public:
    Triangle(double base, double height) : base_(base), height_(height) {}
    double area() const override { return 0.5 * base_ * height_; }
};

class AreaCalculator {
public:
    double calculateTotalArea(const std::vector<std::unique_ptr<Shape>>& shapes) {
        double total = 0;
        for (const auto& shape : shapes) {
            total += shape->area(); // Works with any Shape subclass
        }
        return total;
    }
};

// 3. Liskov Substitution Principle (LSP)
// BAD: Subclass changes expected behavior
class BadBird {
public:
    virtual void fly() { std::cout << "Flying\n"; }
};

class BadPenguin : public BadBird {
public:
    void fly() override {
        throw std::runtime_error("Penguins can't fly!"); // Violates LSP
    }
};

// GOOD: Proper hierarchy respecting LSP
class Bird {
public:
    virtual ~Bird() = default;
    virtual void eat() { std::cout << "Eating\n"; }
    virtual void sleep() { std::cout << "Sleeping\n"; }
};

class FlyingBird : public Bird {
public:
    virtual void fly() { std::cout << "Flying\n"; }
};

class SwimmingBird : public Bird {
public:
    virtual void swim() { std::cout << "Swimming\n"; }
};

class Eagle : public FlyingBird {
public:
    void fly() override { std::cout << "Eagle soaring high\n"; }
};

class Penguin : public SwimmingBird {
public:
    void swim() override { std::cout << "Penguin swimming gracefully\n"; }
};

// 4. Interface Segregation Principle (ISP)
// BAD: Fat interface forces classes to implement unnecessary methods
class BadWorker {
public:
    virtual void work() = 0;
    virtual void eat() = 0;
    virtual void sleep() = 0;
    virtual void takeBreak() = 0;
};

class BadRobot : public BadWorker {
public:
    void work() override { std::cout << "Robot working\n"; }
    void eat() override { /* Robots don't eat - forced to implement */ }
    void sleep() override { /* Robots don't sleep - forced to implement */ }
    void takeBreak() override { /* Robots don't take breaks - forced to implement */ }
};

// GOOD: Segregated interfaces
class Workable {
public:
    virtual ~Workable() = default;
    virtual void work() = 0;
};

class Eatable {
public:
    virtual ~Eatable() = default;
    virtual void eat() = 0;
};

class Sleepable {
public:
    virtual ~Sleepable() = default;
    virtual void sleep() = 0;
};

class Human : public Workable, public Eatable, public Sleepable {
public:
    void work() override { std::cout << "Human working\n"; }
    void eat() override { std::cout << "Human eating\n"; }
    void sleep() override { std::cout << "Human sleeping\n"; }
};

class Robot : public Workable { // Only implements what it needs
public:
    void work() override { std::cout << "Robot working efficiently\n"; }
};

// 5. Dependency Inversion Principle (DIP)
// BAD: High-level module depends on low-level module
class BadEmailService {
public:
    void sendEmail(const std::string& message) {
        // Direct dependency on specific email implementation
        std::cout << "Sending email: " << message << std::endl;
    }
};

class BadUserService {
private:
    BadEmailService emailService_; // Concrete dependency
    
public:
    void registerUser(const std::string& username) {
        // User registration logic
        emailService_.sendEmail("Welcome " + username + "!");
    }
};

// GOOD: Depend on abstractions, not concretions
class MessageService {
public:
    virtual ~MessageService() = default;
    virtual void sendMessage(const std::string& message) = 0;
};

class EmailService : public MessageService {
public:
    void sendMessage(const std::string& message) override {
        std::cout << "Sending email: " << message << std::endl;
    }
};

class SMSService : public MessageService {
public:
    void sendMessage(const std::string& message) override {
        std::cout << "Sending SMS: " << message << std::endl;
    }
};

class SlackService : public MessageService {
public:
    void sendMessage(const std::string& message) override {
        std::cout << "Sending Slack message: " << message << std::endl;
    }
};

class UserService {
private:
    std::unique_ptr<MessageService> messageService_;
    
public:
    explicit UserService(std::unique_ptr<MessageService> service) 
        : messageService_(std::move(service)) {}
    
    void registerUser(const std::string& username) {
        // User registration logic
        messageService_->sendMessage("Welcome " + username + "!");
    }
    
    void setMessageService(std::unique_ptr<MessageService> service) {
        messageService_ = std::move(service);
    }
};

void solidExamples() {
    // SRP Example
    Employee emp("John Doe", 50000);
    PayrollCalculator payroll;
    EmployeeRepository repo;
    ReportGenerator reporter;
    
    double pay = payroll.calculatePay(emp);
    repo.save(emp);
    
    std::vector<Employee> employees = {emp};
    std::string report = reporter.generateReport(employees);
    
    // OCP Example
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 6.0));
    shapes.push_back(std::make_unique<Triangle>(3.0, 4.0)); // New shape added easily
    
    AreaCalculator calculator;
    double totalArea = calculator.calculateTotalArea(shapes);
    std::cout << "Total area: " << totalArea << std::endl;
    
    // LSP Example
    std::vector<std::unique_ptr<Bird>> birds;
    birds.push_back(std::make_unique<Eagle>());
    birds.push_back(std::make_unique<Penguin>());
    
    for (const auto& bird : birds) {
        bird->eat(); // Works for all birds
    }
    
    // ISP Example
    Robot robot;
    Human human;
    
    robot.work(); // Robot only implements what it needs
    
    human.work();
    human.eat();
    human.sleep();
    
    // DIP Example
    auto userService = std::make_unique<UserService>(std::make_unique<EmailService>());
    userService->registerUser("Alice");
    
    // Easy to switch messaging services
    userService->setMessageService(std::make_unique<SMSService>());
    userService->registerUser("Bob");
    
    userService->setMessageService(std::make_unique<SlackService>());
    userService->registerUser("Charlie");
}

### Exception safety guarantees

```cpp
#include <stdexcept>
#include <memory>
#include <vector>

// Exception safety levels:
// 1. No guarantee - may leak resources or corrupt state
// 2. Basic guarantee - no resource leaks, object in valid state
// 3. Strong guarantee - operation succeeds or has no effect (commit/rollback)
// 4. No-throw guarantee - operation never throws exceptions

// 1. Basic guarantee example
class BasicGuaranteeVector {
private:
    std::unique_ptr<int[]> data_;
    size_t size_;
    size_t capacity_;
    
public:
    BasicGuaranteeVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    void push_back_basic(int value) {
        if (size_ >= capacity_) {
            size_t newCapacity = capacity_ == 0 ? 1 : capacity_ * 2;
            auto newData = std::make_unique<int[]>(newCapacity);
            
            // Copy existing data - may throw
            for (size_t i = 0; i < size_; ++i) {
                newData[i] = data_[i]; // If this throws, newData is automatically cleaned up
            }
            
            // Only modify object state after all throwing operations succeed
            data_ = std::move(newData);
            capacity_ = newCapacity;
        }
        
        data_[size_++] = value; // No-throw operation
    }
    
    size_t size() const noexcept { return size_; }
    int& at(size_t index) { 
        if (index >= size_) throw std::out_of_range("Index out of range");
        return data_[index]; 
    }
};

// 2. Strong guarantee example
class StrongGuaranteeVector {
private:
    std::vector<int> data_;
    
public:
    void push_back_strong(int value) {
        // Copy current state
        std::vector<int> temp = data_;
        
        try {
            temp.push_back(value); // May throw
            data_ = std::move(temp); // No-throw swap
        } catch (...) {
            // If push_back throws, data_ remains unchanged
            throw;
        }
    }
    
    // Better implementation using copy-and-swap
    void push_back_copy_swap(int value) {
        StrongGuaranteeVector temp = *this;
        temp.data_.push_back(value); // May throw
        
        // No-throw swap
        data_.swap(temp.data_);
    }
    
    size_t size() const noexcept { return data_.size(); }
    int& at(size_t index) { return data_.at(index); }
};

// 3. No-throw guarantee examples
class NoThrowOperations {
private:
    std::unique_ptr<int> ptr_;
    int value_;
    
public:
    // No-throw constructor
    NoThrowOperations() noexcept : ptr_(nullptr), value_(0) {}
    
    // No-throw destructor (implicitly noexcept)
    ~NoThrowOperations() = default;
    
    // No-throw move constructor
    NoThrowOperations(NoThrowOperations&& other) noexcept 
        : ptr_(std::move(other.ptr_)), value_(other.value_) {
        other.value_ = 0;
    }
    
    // No-throw move assignment
    NoThrowOperations& operator=(NoThrowOperations&& other) noexcept {
        if (this != &other) {
            ptr_ = std::move(other.ptr_);
            value_ = other.value_;
            other.value_ = 0;
        }
        return *this;
    }
    
    // No-throw swap
    void swap(NoThrowOperations& other) noexcept {
        ptr_.swap(other.ptr_);
        std::swap(value_, other.value_);
    }
    
    // No-throw getters
    int getValue() const noexcept { return value_; }
    bool hasPtr() const noexcept { return ptr_ != nullptr; }
};

// 4. RAII for exception safety
class FileHandler {
private:
    FILE* file_;
    
public:
    explicit FileHandler(const std::string& filename) {
        file_ = std::fopen(filename.c_str(), "r");
        if (!file_) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
    }
    
    ~FileHandler() noexcept {
        if (file_) {
            std::fclose(file_);
        }
    }
    
    // Delete copy operations to prevent resource duplication
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    // Move operations
    FileHandler(FileHandler&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }
    
    FileHandler& operator=(FileHandler&& other) noexcept {
        if (this != &other) {
            if (file_) {
                std::fclose(file_);
            }
            file_ = other.file_;
            other.file_ = nullptr;
        }
        return *this;
    }
    
    FILE* get() const noexcept { return file_; }
};

// 5. Exception-safe assignment operator
class ExceptionSafeClass {
private:
    std::string name_;
    std::vector<int> data_;
    std::unique_ptr<int> ptr_;
    
public:
    ExceptionSafeClass(const std::string& name, const std::vector<int>& data) 
        : name_(name), data_(data), ptr_(std::make_unique<int>(42)) {}
    
    // Exception-safe copy assignment using copy-and-swap
    ExceptionSafeClass& operator=(const ExceptionSafeClass& other) {
        if (this != &other) {
            ExceptionSafeClass temp(other); // Copy constructor may throw
            swap(temp); // No-throw swap
        }
        return *this;
    }
    
    void swap(ExceptionSafeClass& other) noexcept {
        name_.swap(other.name_);
        data_.swap(other.data_);
        ptr_.swap(other.ptr_);
    }
    
    // Exception-safe method with rollback
    void updateData(const std::vector<int>& newData, const std::string& newName) {
        // Save current state for rollback
        std::vector<int> oldData = data_;
        std::string oldName = name_;
        
        try {
            data_ = newData; // May throw
            name_ = newName; // May throw
        } catch (...) {
            // Rollback on exception
            data_ = oldData;
            name_ = oldName;
            throw;
        }
    }
};

// 6. Exception safety in template functions
template<typename Container, typename Function>
void safe_transform(Container& container, Function func) {
    using ValueType = typename Container::value_type;
    
    // Create temporary container with transformed values
    Container temp;
    temp.reserve(container.size()); // May throw
    
    for (const auto& item : container) {
        temp.push_back(func(item)); // func or push_back may throw
    }
    
    // If we reach here, all operations succeeded
    container.swap(temp); // No-throw swap
}

void exceptionSafetyExamples() {
    // Basic guarantee
    BasicGuaranteeVector basicVec;
    try {
        basicVec.push_back_basic(1);
        basicVec.push_back_basic(2);
        std::cout << "Basic vector size: " << basicVec.size() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        // Vector is in valid state, no resource leaks
    }
    
    // Strong guarantee
    StrongGuaranteeVector strongVec;
    try {
        strongVec.push_back_strong(1);
        strongVec.push_back_strong(2);
        std::cout << "Strong vector size: " << strongVec.size() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        // Vector state is exactly as it was before the failed operation
    }
    
    // RAII example
    try {
        FileHandler handler("nonexistent.txt");
        // File operations...
    } catch (const std::exception& e) {
        std::cout << "File error: " << e.what() << std::endl;
        // No file handle leaked
    }
    
    // Exception-safe transform
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    try {
        safe_transform(numbers, [](int x) { 
            if (x == 3) throw std::runtime_error("Error at 3");
            return x * x; 
        });
    } catch (const std::exception& e) {
        std::cout << "Transform error: " << e.what() << std::endl;
        // numbers vector is unchanged due to strong guarantee
        std::cout << "Numbers still original: ";
        for (int n : numbers) std::cout << n << " ";
        std::cout << std::endl;
    }
}

## Performance Optimization

### Cache-friendly code

```cpp
#include <chrono>
#include <random>
#include <algorithm>

void cacheUnfriendlyExample() {
    const size_t size = 1000000;
    std::vector<std::vector<int>> matrix(size, std::vector<int>(64, 1));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Cache-unfriendly: accessing columns (non-contiguous memory)
    long long sum = 0;
    for (size_t col = 0; col < 64; ++col) {
        for (size_t row = 0; row < size; ++row) {
            sum += matrix[row][col]; // Poor cache locality
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Cache-unfriendly time: " << duration.count() << " ms\n";
    std::cout << "Sum: " << sum << std::endl;
}

void cacheFriendlyExample() {
    const size_t size = 1000000;
    std::vector<std::vector<int>> matrix(size, std::vector<int>(64, 1));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Cache-friendly: accessing rows (contiguous memory)
    long long sum = 0;
    for (size_t row = 0; row < size; ++row) {
        for (size_t col = 0; col < 64; ++col) {
            sum += matrix[row][col]; // Good cache locality
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Cache-friendly time: " << duration.count() << " ms\n";
    std::cout << "Sum: " << sum << std::endl;
}

// Structure of Arrays (SoA) vs Array of Structures (AoS)
struct Particle {
    float x, y, z;     // Position
    float vx, vy, vz;  // Velocity
    float mass;
    int id;
};

// Array of Structures (AoS) - can have poor cache performance
void updateParticlesAoS(std::vector<Particle>& particles) {
    for (auto& p : particles) {
        p.x += p.vx * 0.016f;  // Only using position and velocity
        p.y += p.vy * 0.016f;  // but loading entire struct
        p.z += p.vz * 0.016f;
    }
}

// Structure of Arrays (SoA) - better cache performance
struct ParticleSystem {
    std::vector<float> x, y, z;       // Positions
    std::vector<float> vx, vy, vz;    // Velocities
    std::vector<float> mass;
    std::vector<int> id;
    
    size_t size() const { return x.size(); }
    
    void reserve(size_t capacity) {
        x.reserve(capacity);
        y.reserve(capacity);
        z.reserve(capacity);
        vx.reserve(capacity);
        vy.reserve(capacity);
        vz.reserve(capacity);
        mass.reserve(capacity);
        id.reserve(capacity);
    }
    
    void addParticle(float px, float py, float pz, 
                    float pvx, float pvy, float pvz, 
                    float pmass, int pid) {
        x.push_back(px);
        y.push_back(py);
        z.push_back(pz);
        vx.push_back(pvx);
        vy.push_back(pvy);
        vz.push_back(pvz);
        mass.push_back(pmass);
        id.push_back(pid);
    }
};

void updateParticlesSoA(ParticleSystem& system) {
    const float dt = 0.016f;
    const size_t count = system.size();
    
    // Better cache utilization - only touch needed data
    for (size_t i = 0; i < count; ++i) {
        system.x[i] += system.vx[i] * dt;
        system.y[i] += system.vy[i] * dt;
        system.z[i] += system.vz[i] * dt;
    }
}

// Cache-friendly data structures
template<typename T, size_t CacheLineSize = 64>
class CacheAlignedVector {
private:
    static constexpr size_t elementsPerCacheLine = CacheLineSize / sizeof(T);
    std::vector<T> data_;
    
public:
    void push_back(const T& value) {
        data_.push_back(value);
    }
    
    // Cache-friendly iteration
    template<typename Function>
    void forEachCacheAligned(Function func) {
        for (size_t i = 0; i < data_.size(); i += elementsPerCacheLine) {
            size_t end = std::min(i + elementsPerCacheLine, data_.size());
            for (size_t j = i; j < end; ++j) {
                func(data_[j]);
            }
            // Process cache line worth of data at once
        }
    }
    
    size_t size() const { return data_.size(); }
    T& operator[](size_t index) { return data_[index]; }
};

void performanceComparison() {
    const size_t numParticles = 1000000;
    
    // AoS version
    std::vector<Particle> particlesAoS(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        particlesAoS[i] = {
            static_cast<float>(i), static_cast<float>(i), static_cast<float>(i),
            1.0f, 1.0f, 1.0f, 1.0f, static_cast<int>(i)
        };
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    updateParticlesAoS(particlesAoS);
    auto end = std::chrono::high_resolution_clock::now();
    auto aosTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // SoA version
    ParticleSystem particlesSoA;
    particlesSoA.reserve(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        particlesSoA.addParticle(
            static_cast<float>(i), static_cast<float>(i), static_cast<float>(i),
            1.0f, 1.0f, 1.0f, 1.0f, static_cast<int>(i)
        );
    }
    
    start = std::chrono::high_resolution_clock::now();
    updateParticlesSoA(particlesSoA);
    end = std::chrono::high_resolution_clock::now();
    auto soaTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "AoS time: " << aosTime.count() << " microseconds\n";
    std::cout << "SoA time: " << soaTime.count() << " microseconds\n";
    std::cout << "SoA speedup: " << static_cast<double>(aosTime.count()) / soaTime.count() << "x\n";
}

### Copy elision and RVO

```cpp
#include <iostream>
#include <string>

class ExpensiveObject {
private:
    std::string data_;
    size_t size_;
    
public:
    // Constructor
    ExpensiveObject(const std::string& data) : data_(data), size_(data.size()) {
        std::cout << "Constructor: " << data_ << std::endl;
    }
    
    // Copy constructor
    ExpensiveObject(const ExpensiveObject& other) : data_(other.data_), size_(other.size_) {
        std::cout << "Copy constructor: " << data_ << std::endl;
    }
    
    // Move constructor
    ExpensiveObject(ExpensiveObject&& other) noexcept 
        : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
        std::cout << "Move constructor: " << data_ << std::endl;
    }
    
    // Copy assignment
    ExpensiveObject& operator=(const ExpensiveObject& other) {
        if (this != &other) {
            data_ = other.data_;
            size_ = other.size_;
            std::cout << "Copy assignment: " << data_ << std::endl;
        }
        return *this;
    }
    
    // Move assignment
    ExpensiveObject& operator=(ExpensiveObject&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
            std::cout << "Move assignment: " << data_ << std::endl;
        }
        return *this;
    }
    
    // Destructor
    ~ExpensiveObject() {
        std::cout << "Destructor: " << data_ << std::endl;
    }
    
    const std::string& getData() const { return data_; }
};

// 1. Return Value Optimization (RVO)
ExpensiveObject createObjectRVO(const std::string& name) {
    return ExpensiveObject(name); // RVO: no copy/move, direct construction at call site
}

// 2. Named Return Value Optimization (NRVO)
ExpensiveObject createObjectNRVO(const std::string& name) {
    ExpensiveObject obj(name); // Local object
    // ... do some work with obj ...
    return obj; // NRVO: direct construction at call site (compiler dependent)
}

// 3. When RVO/NRVO cannot be applied
ExpensiveObject createObjectNoOptimization(bool condition) {
    ExpensiveObject obj1("path1");
    ExpensiveObject obj2("path2");
    
    if (condition) {
        return obj1; // Cannot optimize: multiple return paths
    } else {
        return obj2;
    }
}

// 4. Copy elision in function parameters
void processObject(ExpensiveObject obj) { // Pass by value
    std::cout << "Processing: " << obj.getData() << std::endl;
}

// 5. Forcing moves when RVO isn't possible
ExpensiveObject createAndMove(bool condition) {
    ExpensiveObject obj1("option1");
    ExpensiveObject obj2("option2");
    
    // Explicitly use move to avoid copy
    return condition ? std::move(obj1) : std::move(obj2);
}

// 6. Factory function with guaranteed RVO
template<typename... Args>
ExpensiveObject makeExpensiveObject(Args&&... args) {
    return ExpensiveObject(std::forward<Args>(args)...); // Perfect forwarding + RVO
}

void demonstrateRVO() {
    std::cout << "=== RVO Example ===\n";
    auto obj1 = createObjectRVO("RVO_test");
    // Only constructor and destructor should be called
    
    std::cout << "\n=== NRVO Example ===\n";
    auto obj2 = createObjectNRVO("NRVO_test");
    // May optimize away copy/move (compiler dependent)
    
    std::cout << "\n=== No Optimization Example ===\n";
    auto obj3 = createObjectNoOptimization(true);
    // Will use move constructor
    
    std::cout << "\n=== Pass by Value Example ===\n";
    processObject(createObjectRVO("temp")); // RVO + move for parameter
    
    std::cout << "\n=== Factory Function Example ===\n";
    auto obj4 = makeExpensiveObject("factory_created");
    
    std::cout << "\n=== End of Examples ===\n";
}

// Performance comparison: copy vs move vs RVO
void performanceComparison() {
    const int iterations = 100000;
    
    // Test RVO performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto obj = createObjectRVO("test"); // RVO - most efficient
        (void)obj; // Suppress unused variable warning
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto rvoTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test move performance
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ExpensiveObject temp("test");
        auto obj = std::move(temp); // Move constructor
        (void)obj;
    }
    end = std::chrono::high_resolution_clock::now();
    auto moveTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test copy performance
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ExpensiveObject temp("test");
        auto obj = temp; // Copy constructor
        (void)obj;
    }
    end = std::chrono::high_resolution_clock::now();
    auto copyTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "RVO time: " << rvoTime.count() << " microseconds\n";
    std::cout << "Move time: " << moveTime.count() << " microseconds\n";
    std::cout << "Copy time: " << copyTime.count() << " microseconds\n";
}

### Memory pooling strategies

```cpp
#include <memory>
#include <vector>
#include <list>

// 1. Simple fixed-size memory pool
template<typename T, size_t PoolSize>
class FixedSizePool {
private:
    alignas(T) char memory_[PoolSize * sizeof(T)];
    std::vector<void*> freeList_;
    
public:
    FixedSizePool() {
        // Initialize free list with all blocks
        freeList_.reserve(PoolSize);
        for (size_t i = 0; i < PoolSize; ++i) {
            freeList_.push_back(memory_ + i * sizeof(T));
        }
    }
    
    T* allocate() {
        if (freeList_.empty()) {
            return nullptr; // Pool exhausted
        }
        
        void* ptr = freeList_.back();
        freeList_.pop_back();
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr) {
        if (ptr >= memory_ && ptr < memory_ + sizeof(memory_)) {
            freeList_.push_back(ptr);
        }
    }
    
    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        if (ptr) {
            new(ptr) T(std::forward<Args>(args)...);
        }
        return ptr;
    }
    
    void destroy(T* ptr) {
        if (ptr) {
            ptr->~T();
            deallocate(ptr);
        }
    }
    
    size_t available() const { return freeList_.size(); }
    size_t capacity() const { return PoolSize; }
};

// 2. Growing memory pool
template<typename T>
class GrowingPool {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
        Block* next;
    };
    
    std::vector<std::unique_ptr<Block[]>> chunks_;
    Block* freeHead_;
    size_t chunkSize_;
    size_t currentChunk_;
    
    void addChunk() {
        auto newChunk = std::make_unique<Block[]>(chunkSize_);
        
        // Link all blocks in the new chunk
        for (size_t i = 0; i < chunkSize_ - 1; ++i) {
            newChunk[i].next = &newChunk[i + 1];
        }
        newChunk[chunkSize_ - 1].next = freeHead_;
        freeHead_ = newChunk.get();
        
        chunks_.push_back(std::move(newChunk));
        chunkSize_ *= 2; // Exponential growth
    }
    
public:
    explicit GrowingPool(size_t initialChunkSize = 64) 
        : freeHead_(nullptr), chunkSize_(initialChunkSize), currentChunk_(0) {
        addChunk();
    }
    
    T* allocate() {
        if (!freeHead_) {
            addChunk();
        }
        
        Block* block = freeHead_;
        freeHead_ = freeHead_->next;
        return reinterpret_cast<T*>(block);
    }
    
    void deallocate(T* ptr) {
        Block* block = reinterpret_cast<Block*>(ptr);
        block->next = freeHead_;
        freeHead_ = block;
    }
    
    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        new(ptr) T(std::forward<Args>(args)...);
        return ptr;
    }
    
    void destroy(T* ptr) {
        ptr->~T();
        deallocate(ptr);
    }
};

// 3. Stack allocator for temporary allocations
class StackAllocator {
private:
    char* memory_;
    size_t size_;
    size_t top_;
    
    struct Marker {
        size_t position;
    };
    
public:
    explicit StackAllocator(size_t size) : size_(size), top_(0) {
        memory_ = new char[size];
    }
    
    ~StackAllocator() {
        delete[] memory_;
    }
    
    void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t)) {
        // Align the allocation
        size_t aligned_top = (top_ + alignment - 1) & ~(alignment - 1);
        
        if (aligned_top + bytes > size_) {
            return nullptr; // Not enough space
        }
        
        void* ptr = memory_ + aligned_top;
        top_ = aligned_top + bytes;
        return ptr;
    }
    
    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        void* ptr = allocate(sizeof(T), alignof(T));
        if (ptr) {
            return new(ptr) T(std::forward<Args>(args)...);
        }
        return nullptr;
    }
    
    Marker getMarker() const {
        return {top_};
    }
    
    void freeToMarker(const Marker& marker) {
        top_ = marker.position;
    }
    
    void reset() {
        top_ = 0;
    }
    
    size_t getBytesUsed() const { return top_; }
    size_t getBytesRemaining() const { return size_ - top_; }
};

// 4. Pool allocator for STL containers
template<typename T>
class PoolAllocator {
private:
    static GrowingPool<T> pool_;
    
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };
    
    PoolAllocator() = default;
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>&) {}
    
    pointer allocate(size_type n) {
        if (n == 1) {
            return pool_.allocate();
        }
        return static_cast<pointer>(std::malloc(n * sizeof(T)));
    }
    
    void deallocate(pointer p, size_type n) {
        if (n == 1) {
            pool_.deallocate(p);
        } else {
            std::free(p);
        }
    }
    
    template<typename U>
    bool operator==(const PoolAllocator<U>&) const { return true; }
    
    template<typename U>
    bool operator!=(const PoolAllocator<U>&) const { return false; }
};

template<typename T>
GrowingPool<T> PoolAllocator<T>::pool_;

// Usage examples
void memoryPoolExamples() {
    // Fixed-size pool example
    FixedSizePool<int, 1000> intPool;
    
    std::vector<int*> allocated;
    
    // Allocate objects
    for (int i = 0; i < 100; ++i) {
        int* ptr = intPool.construct(i);
        if (ptr) {
            allocated.push_back(ptr);
        }
    }
    
    std::cout << "Allocated " << allocated.size() << " objects\n";
    std::cout << "Pool has " << intPool.available() << " slots remaining\n";
    
    // Deallocate objects
    for (int* ptr : allocated) {
        intPool.destroy(ptr);
    }
    
    std::cout << "After deallocation: " << intPool.available() << " slots available\n";
    
    // Growing pool example
    GrowingPool<std::string> stringPool;
    
    std::vector<std::string*> strings;
    for (int i = 0; i < 200; ++i) {
        std::string* str = stringPool.construct("String " + std::to_string(i));
        strings.push_back(str);
    }
    
    std::cout << "Created " << strings.size() << " strings\n";
    
    for (std::string* str : strings) {
        stringPool.destroy(str);
    }
    
    // Stack allocator example
    StackAllocator stackAlloc(1024 * 1024); // 1MB stack
    
    auto marker1 = stackAlloc.getMarker();
    
    // Allocate temporary objects
    int* tempInts = static_cast<int*>(stackAlloc.allocate(100 * sizeof(int)));
    auto marker2 = stackAlloc.getMarker();
    
    double* tempDoubles = static_cast<double*>(stackAlloc.allocate(50 * sizeof(double)));
    
    std::cout << "Stack allocator used: " << stackAlloc.getBytesUsed() << " bytes\n";
    
    // Free to markers (no individual deallocations needed)
    stackAlloc.freeToMarker(marker2); // Free doubles
    stackAlloc.freeToMarker(marker1); // Free ints
    
    // STL container with pool allocator
    std::vector<int, PoolAllocator<int>> poolVector;
    std::list<std::string, PoolAllocator<std::string>> poolList;
    
    for (int i = 0; i < 1000; ++i) {
        poolVector.push_back(i);
        poolList.push_back("Item " + std::to_string(i));
    }
    
    std::cout << "Pool-allocated containers created\n";
}

**Interview Tips for C++ Interview Guide:**

1. **Preparation Strategy:**
   - Review fundamental concepts first
   - Practice coding examples
   - Understand trade-offs between different approaches
   - Be ready to explain performance implications

2. **Common Mistakes to Avoid:**
   - Forgetting to handle edge cases
   - Not considering exception safety
   - Ignoring const-correctness
   - Memory management errors

3. **Best Practices to Mention:**
   - Use RAII for resource management
   - Prefer smart pointers over raw pointers
   - Follow the Rule of Three/Five/Zero
   - Use const wherever possible
   - Leverage STL algorithms and containers

4. **Real-world Scenarios:**
   - Performance optimization in production systems
   - Thread safety in multi-threaded applications
   - Memory optimization for embedded systems
   - API design considerations

This comprehensive C++ interview guide covers the essential topics that senior developers should be familiar with, providing both theoretical knowledge and practical examples that can be used in interviews and real-world development scenarios.
```

**Interview Tips for C++ Interview Guide:**

1. **Preparation Strategy:**
   - Review fundamental concepts first
   - Practice coding examples
   - Understand trade-offs between different approaches
   - Be ready to explain performance implications

2. **Common Mistakes to Avoid:**
   - Forgetting to handle edge cases
   - Not considering exception safety
   - Ignoring const-correctness
   - Memory management errors

3. **Best Practices to Mention:**
   - Use RAII for resource management
   - Prefer smart pointers over raw pointers
   - Follow the Rule of Three/Five/Zero
   - Use const wherever possible
   - Leverage STL algorithms and containers

4. **Real-world Scenarios:**
   - Performance optimization in production systems
   - Thread safety in multi-threaded applications
   - Memory optimization for embedded systems
   - API design considerations

This comprehensive C++ interview guide covers the essential topics that senior developers should be familiar with, providing both theoretical knowledge and practical examples that can be used in interviews and real-world development scenarios.
```
```
