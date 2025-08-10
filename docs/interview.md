# C++/Python Architect Interview Preparation Guide

## Table of Contents
1. [Core Language Concepts](#core-language-concepts)
2. [Design Patterns](#design-patterns)
3. [System Design](#system-design)
4. [Performance Optimization](#performance-optimization)
5. [Common Interview Questions and Solutions](#common-interview-questions-and-solutions)
6. [Code Examples](#code-examples)
7. [Architecture Best Practices](#architecture-best-practices)
8. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Core Language Concepts

### C++ Core Concepts

#### Memory Management
- **Stack vs Heap**: Stack allocation is faster but limited; heap allocation is flexible but requires management
- **RAII (Resource Acquisition Is Initialization)**: Resource management through object lifetime
- **Smart Pointers**: `unique_ptr`, `shared_ptr`, `weak_ptr` for automatic memory management

#### Modern C++ Features (C++11/14/17/20)
- **Move Semantics**: Efficient transfer of resources using `std::move`
- **Lambda Expressions**: Anonymous functions for functional programming
- **Auto Keyword**: Type deduction for cleaner code
- **Range-based Loops**: Simplified iteration syntax
- **Constexpr**: Compile-time computation
- **Templates**: Generic programming and template metaprogramming

#### Concurrency
- **Threading**: `std::thread`, `std::mutex`, `std::condition_variable`
- **Atomic Operations**: Lock-free programming with `std::atomic`
- **Memory Model**: Understanding memory ordering and synchronization

### Python Core Concepts

#### Memory Management
- **Garbage Collection**: Reference counting with cycle detection
- **Memory Optimization**: `__slots__`, generators, memory pools
- **CPython Internals**: Understanding bytecode and the Python VM

#### Advanced Features
- **Metaclasses**: Classes that create classes
- **Decorators**: Function and class modification
- **Context Managers**: Resource management with `with` statements
- **Generators and Iterators**: Memory-efficient data processing
- **Async/Await**: Asynchronous programming with `asyncio`

#### Type System
- **Type Hints**: Static type checking with `typing` module
- **Protocols**: Structural subtyping
- **Generics**: Type variables and constraints

---

## Design Patterns

### Creational Patterns

#### Singleton Pattern
**Use Case**: Ensure only one instance of a class exists

**C++ Implementation:**
```cpp
class Singleton {
private:
    static std::unique_ptr<Singleton> instance;
    static std::once_flag initialized;
    
    Singleton() = default;
    
public:
    static Singleton& getInstance() {
        std::call_once(initialized, []() {
            instance = std::make_unique<Singleton>();
        });
        return *instance;
    }
    
    // Delete copy constructor and assignment
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};
```

**Python Implementation:**
```python
class Singleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

#### Factory Pattern
**Use Case**: Create objects without specifying exact classes

**C++ Implementation:**
```cpp
class Shape {
public:
    virtual void draw() = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
public:
    void draw() override { std::cout << "Drawing Circle\n"; }
};

class ShapeFactory {
public:
    static std::unique_ptr<Shape> createShape(const std::string& type) {
        if (type == "circle") {
            return std::make_unique<Circle>();
        }
        throw std::invalid_argument("Unknown shape type");
    }
};
```

### Structural Patterns

#### Adapter Pattern
**Use Case**: Make incompatible interfaces work together

#### Decorator Pattern
**Use Case**: Add behavior to objects dynamically

### Behavioral Patterns

#### Observer Pattern
**Use Case**: Notify multiple objects about state changes

**Python Implementation:**
```python
from abc import ABC, abstractmethod
from typing import List

class Observer(ABC):
    @abstractmethod
    def update(self, message: str) -> None:
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)
    
    def notify(self, message: str) -> None:
        for observer in self._observers:
            observer.update(message)
```

#### Strategy Pattern
**Use Case**: Define family of algorithms and make them interchangeable

---

## System Design

### Scalability Principles

#### Horizontal vs Vertical Scaling
- **Vertical Scaling**: Adding more power (CPU, RAM) to existing machines
- **Horizontal Scaling**: Adding more machines to the pool of resources

#### Load Balancing
- **Round Robin**: Distribute requests evenly
- **Least Connections**: Route to server with fewest active connections
- **Weighted Round Robin**: Assign different weights to servers

#### Caching Strategies
- **Cache-Aside**: Application manages cache
- **Write-Through**: Write to cache and database simultaneously
- **Write-Behind**: Write to cache immediately, database later

### Database Design

#### SQL vs NoSQL
- **SQL**: ACID properties, complex queries, structured data
- **NoSQL**: Horizontal scaling, flexible schema, eventual consistency

#### Database Sharding
- **Horizontal Partitioning**: Split data across multiple databases
- **Vertical Partitioning**: Split table columns across databases

### Microservices Architecture

#### Benefits
- **Independent Deployment**: Services can be deployed separately
- **Technology Diversity**: Different services can use different technologies
- **Fault Isolation**: Failure in one service doesn't bring down the system

#### Challenges
- **Distributed Complexity**: Network calls, latency, failures
- **Data Consistency**: Managing transactions across services
- **Service Discovery**: Finding and communicating with services

---

## Performance Optimization

### C++ Performance

#### Memory Optimization
```cpp
// Use contiguous memory for better cache performance
std::vector<int> data(1000000);  // Good
std::list<int> data_list;        // Poor cache locality

// Avoid unnecessary allocations
void processData(const std::vector<int>& data) {  // Pass by reference
    // Process data
}

// Use move semantics
std::vector<int> createLargeVector() {
    std::vector<int> result(1000000);
    // ... populate result
    return result;  // Move semantics applied automatically
}
```

#### Algorithm Optimization
```cpp
// Use appropriate containers
std::unordered_map<int, std::string> lookup;  // O(1) average lookup
std::map<int, std::string> ordered_lookup;    // O(log n) lookup

// Compiler optimizations
inline int fastMultiply(int x) {  // Hint for inlining
    return x << 1;  // Bit shift instead of multiplication
}
```

### Python Performance

#### Profiling and Optimization
```python
import cProfile
import timeit
from functools import lru_cache

# Profile code
def profile_function():
    cProfile.run('expensive_function()')

# Memoization for expensive calculations
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Use generators for memory efficiency
def process_large_dataset(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield process_line(line)

# NumPy for numerical computations
import numpy as np
# Use vectorized operations instead of loops
data = np.array([1, 2, 3, 4, 5])
result = data * 2  # Faster than loop
```

#### C Extensions
```python
# Using Cython for performance-critical code
%%cython
def fast_sum(int[:] arr):
    cdef int total = 0
    cdef int i
    for i in range(arr.shape[0]):
        total += arr[i]
    return total
```

---

## Common Interview Questions and Solutions

### C++ Interview Questions

#### Q1: What is the difference between `new`/`delete` and `malloc`/`free`?

**Answer:**
- `new`/`delete` are C++ operators that call constructors/destructors
- `malloc`/`free` are C functions that only allocate/deallocate memory
- `new` is type-safe and throws exceptions; `malloc` returns NULL on failure

```cpp
class MyClass {
public:
    MyClass() { std::cout << "Constructor called\n"; }
    ~MyClass() { std::cout << "Destructor called\n"; }
};

// Using new/delete
MyClass* obj1 = new MyClass();  // Constructor called
delete obj1;                    // Destructor called

// Using malloc/free
MyClass* obj2 = (MyClass*)malloc(sizeof(MyClass));  // No constructor
free(obj2);                                          // No destructor
```

#### Q2: Explain RAII and provide an example

**Answer:**
RAII (Resource Acquisition Is Initialization) ensures that resources are properly released when objects go out of scope.

```cpp
class FileManager {
private:
    std::FILE* file;
    
public:
    FileManager(const std::string& filename) {
        file = std::fopen(filename.c_str(), "r");
        if (!file) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    ~FileManager() {
        if (file) {
            std::fclose(file);
        }
    }
    
    // Delete copy operations to prevent double-close
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
};

// Usage
void processFile(const std::string& filename) {
    FileManager fm(filename);  // File opened
    // ... use file
}  // File automatically closed when fm goes out of scope
```

#### Q3: What are the differences between `const`, `constexpr`, and `consteval`?

**Answer:**
- `const`: Value cannot be modified after initialization
- `constexpr`: Can be evaluated at compile time
- `consteval` (C++20): Must be evaluated at compile time

```cpp
const int runtime_const = getValue();        // Runtime constant
constexpr int compile_time = 42;            // Compile-time constant
consteval int must_be_compile_time(int x) { // Must evaluate at compile time
    return x * 2;
}
```

### Python Interview Questions

#### Q1: Explain the GIL and its implications

**Answer:**
The Global Interpreter Lock (GIL) prevents multiple Python threads from executing Python bytecodes simultaneously. This means:
- CPU-bound tasks don't benefit from threading
- I/O-bound tasks can still benefit from threading
- Use multiprocessing for CPU-bound parallelism

```python
import threading
import multiprocessing
import time

# Threading for I/O-bound tasks
def io_bound_task():
    time.sleep(1)  # Simulating I/O
    return "Done"

# Multiprocessing for CPU-bound tasks
def cpu_bound_task(n):
    result = 0
    for i in range(n):
        result += i * i
    return result

# Threading example
threads = []
for _ in range(4):
    t = threading.Thread(target=io_bound_task)
    threads.append(t)
    t.start()

# Multiprocessing example
with multiprocessing.Pool() as pool:
    results = pool.map(cpu_bound_task, [1000000] * 4)
```

#### Q2: What are decorators and how do they work?

**Answer:**
Decorators are a way to modify or enhance functions without permanently modifying their code.

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
            return None
        return wrapper
    return decorator

@timer
@retry(max_attempts=3)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise Exception("Random failure")
    return "Success"
```

#### Q3: Explain metaclasses and provide a practical example

**Answer:**
Metaclasses are classes whose instances are classes. They control class creation.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Database connection"

# Usage
db1 = Database()
db2 = Database()
print(db1 is db2)  # True

# Validation metaclass
class ValidatedMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Validate that all methods have docstrings
        for key, value in attrs.items():
            if callable(value) and not key.startswith('_'):
                if not hasattr(value, '__doc__') or not value.__doc__:
                    raise ValueError(f"Method {key} must have a docstring")
        return super().__new__(mcs, name, bases, attrs)

class API(metaclass=ValidatedMeta):
    def get_user(self, user_id):
        """Retrieve user by ID."""
        pass
```

---

## Code Examples

### Advanced C++ Examples

#### Template Metaprogramming
```cpp
#include <type_traits>
#include <iostream>

// SFINAE (Substitution Failure Is Not An Error)
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
process_value(T value) {
    std::cout << "Processing integral value: " << value << std::endl;
    return value * 2;
}

template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
process_value(T value) {
    std::cout << "Processing floating point value: " << value << std::endl;
    return value * 1.5;
}

// Variadic templates
template<typename T>
void print_values(T&& value) {
    std::cout << value << std::endl;
}

template<typename T, typename... Args>
void print_values(T&& first, Args&&... args) {
    std::cout << first << " ";
    print_values(std::forward<Args>(args)...);
}

// Compile-time computation
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Usage
int main() {
    process_value(42);      // Calls integral version
    process_value(3.14);    // Calls floating point version
    
    print_values(1, 2.5, "hello", 'c');
    
    constexpr int fact5 = Factorial<5>::value;  // Computed at compile time
    std::cout << "5! = " << fact5 << std::endl;
    
    return 0;
}
```

#### Thread-Safe Queue Implementation
```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        condition_.notify_one();
    }
    
    std::optional<T> tryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T result = std::move(queue_.front());
        queue_.pop();
        return result;
    }
    
    T waitAndPop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        T result = std::move(queue_.front());
        queue_.pop();
        return result;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};
```

### Advanced Python Examples

#### Asynchronous Programming
```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any

class AsyncWebScraper:
    def __init__(self, max_concurrent_requests: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str) -> Dict[str, Any]:
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    content = await response.text()
                    return {
                        'url': url,
                        'status': response.status,
                        'content_length': len(content),
                        'success': True
                    }
            except Exception as e:
                return {
                    'url': url,
                    'error': str(e),
                    'success': False
                }
    
    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        tasks = [self.fetch_url(url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
async def main():
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/status/200',
        'https://httpbin.org/status/404'
    ]
    
    start_time = time.time()
    async with AsyncWebScraper(max_concurrent_requests=5) as scraper:
        results = await scraper.scrape_urls(urls)
    
    end_time = time.time()
    
    for result in results:
        print(f"URL: {result['url']}, Success: {result['success']}")
    
    print(f"Total time: {end_time - start_time:.2f} seconds")

# Run the async function
# asyncio.run(main())
```

#### Advanced Data Processing Pipeline
```python
from typing import Iterator, Callable, TypeVar, Generic, Optional
from functools import reduce
import operator
from collections import defaultdict

T = TypeVar('T')
U = TypeVar('U')

class Pipeline(Generic[T]):
    def __init__(self, data: Iterator[T]):
        self.data = data
    
    def map(self, func: Callable[[T], U]) -> 'Pipeline[U]':
        return Pipeline(map(func, self.data))
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Pipeline[T]':
        return Pipeline(filter(predicate, self.data))
    
    def reduce(self, func: Callable[[T, T], T], initial: Optional[T] = None) -> T:
        if initial is not None:
            return reduce(func, self.data, initial)
        return reduce(func, self.data)
    
    def group_by(self, key_func: Callable[[T], U]) -> Dict[U, List[T]]:
        groups = defaultdict(list)
        for item in self.data:
            groups[key_func(item)].append(item)
        return dict(groups)
    
    def take(self, n: int) -> 'Pipeline[T]':
        def take_generator():
            count = 0
            for item in self.data:
                if count >= n:
                    break
                yield item
                count += 1
        return Pipeline(take_generator())
    
    def collect(self) -> List[T]:
        return list(self.data)

# Usage example
data = [
    {'name': 'Alice', 'age': 30, 'department': 'Engineering'},
    {'name': 'Bob', 'age': 25, 'department': 'Marketing'},
    {'name': 'Charlie', 'age': 35, 'department': 'Engineering'},
    {'name': 'Diana', 'age': 28, 'department': 'Marketing'}
]

# Complex data processing pipeline
result = (Pipeline(data)
          .filter(lambda person: person['age'] > 26)
          .map(lambda person: {**person, 'senior': person['age'] > 30})
          .group_by(lambda person: person['department']))

print(result)
```

---

## Architecture Best Practices

### Code Organization

#### C++ Project Structure
```
project/
├── include/           # Public headers
│   └── mylib/
│       ├── core.hpp
│       └── utils.hpp
├── src/               # Implementation files
│   ├── core.cpp
│   └── utils.cpp
├── tests/             # Unit tests
│   ├── test_core.cpp
│   └── test_utils.cpp
├── examples/          # Usage examples
├── docs/              # Documentation
├── CMakeLists.txt     # Build configuration
└── README.md
```

#### Python Package Structure
```
mypackage/
├── mypackage/         # Main package
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│   └── exceptions.py
├── tests/             # Test files
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/              # Documentation
├── setup.py           # Package configuration
├── requirements.txt   # Dependencies
├── README.md
└── .gitignore
```

### Design Principles

#### SOLID Principles

1. **Single Responsibility Principle (SRP)**
   - Each class should have only one reason to change
   
2. **Open/Closed Principle (OCP)**
   - Software entities should be open for extension, closed for modification
   
3. **Liskov Substitution Principle (LSP)**
   - Derived classes must be substitutable for their base classes
   
4. **Interface Segregation Principle (ISP)**
   - Clients shouldn't depend on interfaces they don't use
   
5. **Dependency Inversion Principle (DIP)**
   - Depend on abstractions, not concretions

#### Example: Applying SOLID Principles

```cpp
// Bad: Violates SRP and OCP
class EmailService {
public:
    void sendEmail(const std::string& message) {
        // Format email
        std::string formatted = "<html>" + message + "</html>";
        
        // Send via SMTP
        // ... SMTP implementation
        
        // Log the email
        std::cout << "Email sent: " << formatted << std::endl;
    }
};

// Good: Follows SOLID principles
class MessageFormatter {
public:
    virtual ~MessageFormatter() = default;
    virtual std::string format(const std::string& message) = 0;
};

class EmailSender {
public:
    virtual ~EmailSender() = default;
    virtual void send(const std::string& message) = 0;
};

class Logger {
public:
    virtual ~Logger() = default;
    virtual void log(const std::string& message) = 0;
};

class EmailService {
private:
    std::unique_ptr<MessageFormatter> formatter_;
    std::unique_ptr<EmailSender> sender_;
    std::unique_ptr<Logger> logger_;
    
public:
    EmailService(std::unique_ptr<MessageFormatter> formatter,
                 std::unique_ptr<EmailSender> sender,
                 std::unique_ptr<Logger> logger)
        : formatter_(std::move(formatter))
        , sender_(std::move(sender))
        , logger_(std::move(logger)) {}
    
    void sendEmail(const std::string& message) {
        std::string formatted = formatter_->format(message);
        sender_->send(formatted);
        logger_->log("Email sent: " + formatted);
    }
};
```

### Error Handling Strategies

#### C++ Error Handling
```cpp
#include <expected>  // C++23
#include <system_error>

// Using std::expected for error handling
enum class ParseError {
    InvalidFormat,
    NumberOutOfRange,
    UnexpectedCharacter
};

std::expected<int, ParseError> parseInteger(const std::string& str) {
    if (str.empty()) {
        return std::unexpected(ParseError::InvalidFormat);
    }
    
    try {
        return std::stoi(str);
    } catch (const std::invalid_argument&) {
        return std::unexpected(ParseError::InvalidFormat);
    } catch (const std::out_of_range&) {
        return std::unexpected(ParseError::NumberOutOfRange);
    }
}

// Exception safety guarantees
class SafeContainer {
private:
    std::vector<int> data_;
    
public:
    // Strong exception safety
    void addElement(int value) {
        std::vector<int> temp = data_;  // Copy
        temp.push_back(value);          // May throw
        data_ = std::move(temp);        // No-throw swap
    }
    
    // Basic exception safety
    void processElements() {
        for (auto& element : data_) {
            try {
                element = processValue(element);  // May throw
            } catch (...) {
                // Handle error, maintain invariants
                element = -1;  // Error marker
            }
        }
    }
};
```

#### Python Error Handling
```python
from typing import Union, Optional, TypeVar, Generic
from enum import Enum
import logging

T = TypeVar('T')
E = TypeVar('E')

class Result(Generic[T, E]):
    def __init__(self, value: Optional[T] = None, error: Optional[E] = None):
        if (value is None) == (error is None):
            raise ValueError("Exactly one of value or error must be provided")
        self._value = value
        self._error = error
    
    @property
    def is_ok(self) -> bool:
        return self._value is not None
    
    @property
    def is_err(self) -> bool:
        return self._error is not None
    
    def unwrap(self) -> T:
        if self._value is None:
            raise ValueError(f"Called unwrap on error: {self._error}")
        return self._value
    
    def unwrap_or(self, default: T) -> T:
        return self._value if self._value is not None else default
    
    def map(self, func) -> 'Result':
        if self.is_ok:
            try:
                return Result(value=func(self._value))
            except Exception as e:
                return Result(error=e)
        return Result(error=self._error)

class ValidationError(Exception):
    pass

def validate_email(email: str) -> Result[str, ValidationError]:
    if '@' not in email:
        return Result(error=ValidationError("Email must contain @"))
    if len(email) < 5:
        return Result(error=ValidationError("Email too short"))
    return Result(value=email)

# Context manager for database transactions
class DatabaseTransaction:
    def __init__(self, connection):
        self.connection = connection
        self.transaction = None
    
    def __enter__(self):
        self.transaction = self.connection.begin()
        return self.transaction
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.transaction.commit()
        else:
            self.transaction.rollback()
            logging.error(f"Transaction failed: {exc_val}")
        return False  # Don't suppress exceptions
```

---

## Common Pitfalls and Solutions

### C++ Pitfalls

#### 1. Memory Leaks and Double Deletion
**Problem:**
```cpp
// Bad: Manual memory management
class BadClass {
    int* data;
public:
    BadClass() : data(new int[100]) {}
    ~BadClass() { delete[] data; }  // What if copy constructor is called?
};

BadClass obj1;
BadClass obj2 = obj1;  // Shallow copy! Both objects will delete same memory
```

**Solution:**
```cpp
// Good: Use RAII and smart pointers
class GoodClass {
    std::unique_ptr<int[]> data;
public:
    GoodClass() : data(std::make_unique<int[]>(100)) {}
    
    // Rule of Five
    GoodClass(const GoodClass& other) : data(std::make_unique<int[]>(100)) {
        std::copy(other.data.get(), other.data.get() + 100, data.get());
    }
    
    GoodClass& operator=(const GoodClass& other) {
        if (this != &other) {
            data = std::make_unique<int[]>(100);
            std::copy(other.data.get(), other.data.get() + 100, data.get());
        }
        return *this;
    }
    
    GoodClass(GoodClass&&) = default;
    GoodClass& operator=(GoodClass&&) = default;
    ~GoodClass() = default;  // Automatic cleanup
};
```

#### 2. Iterator Invalidation
**Problem:**
```cpp
// Bad: Modifying container while iterating
std::vector<int> vec = {1, 2, 3, 4, 5};
for (auto it = vec.begin(); it != vec.end(); ++it) {
    if (*it % 2 == 0) {
        vec.erase(it);  // Iterator becomes invalid!
    }
}
```

**Solution:**
```cpp
// Good: Use erase-remove idiom or update iterator
std::vector<int> vec = {1, 2, 3, 4, 5};

// Option 1: erase-remove idiom
vec.erase(std::remove_if(vec.begin(), vec.end(), 
          [](int x) { return x % 2 == 0; }), vec.end());

// Option 2: Update iterator after erase
for (auto it = vec.begin(); it != vec.end();) {
    if (*it % 2 == 0) {
        it = vec.erase(it);  // erase returns next valid iterator
    } else {
        ++it;
    }
}
```

#### 3. Undefined Behavior with Dangling References
**Problem:**
```cpp
// Bad: Returning reference to local variable
const int& dangerous() {
    int local = 42;
    return local;  // Undefined behavior!
}
```

**Solution:**
```cpp
// Good: Return by value or use static/heap allocation
int safe() {
    int local = 42;
    return local;  // Copy returned
}

const int& alseSafe() {
    static int persistent = 42;
    return persistent;  // Static storage duration
}
```

### Python Pitfalls

#### 1. Mutable Default Arguments
**Problem:**
```python
# Bad: Mutable default argument
def add_item(item, target_list=[]):
    target_list.append(item)
    return target_list

list1 = add_item("a")
list2 = add_item("b")  # Same list object as list1!
print(list1)  # ['a', 'b'] - unexpected!
```

**Solution:**
```python
# Good: Use None as default, create new list inside function
def add_item(item, target_list=None):
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list

# Or use copy for existing list
def add_item_safe(item, target_list=None):
    if target_list is None:
        target_list = []
    else:
        target_list = target_list.copy()  # Don't modify original
    target_list.append(item)
    return target_list
```

#### 2. Late Binding Closures
**Problem:**
```python
# Bad: Late binding in loops
functions = []
for i in range(5):
    functions.append(lambda: i)  # All lambdas capture same 'i'

results = [f() for f in functions]
print(results)  # [4, 4, 4, 4, 4] - all return 4!
```

**Solution:**
```python
# Good: Early binding with default parameter
functions = []
for i in range(5):
    functions.append(lambda x=i: x)  # Capture current value of i

results = [f() for f in functions]
print(results)  # [0, 1, 2, 3, 4] - expected result

# Or use functools.partial
import functools

def make_multiplier(x):
    return x * 2

functions = [functools.partial(make_multiplier, i) for i in range(5)]
```

#### 3. Import Statement Pitfalls
**Problem:**
```python
# Bad: Circular imports and import *
# file1.py
from file2 import *
def func1():
    return func2()

# file2.py  
from file1 import *
def func2():
    return func1()
```

**Solution:**
```python
# Good: Specific imports and avoiding circular dependencies
# file1.py
import file2
def func1():
    return file2.func2()

# file2.py
def func2():
    # Import inside function if needed to break circular dependency
    from file1 import func1
    return "Called from func2"

# Or restructure to eliminate circular dependency
# common.py
def shared_function():
    return "shared"

# file1.py
from common import shared_function
def func1():
    return shared_function()

# file2.py
from common import shared_function
def func2():
    return shared_function()
```

#### 4. Class Variable vs Instance Variable Confusion
**Problem:**
```python
# Bad: Mutable class variable
class Counter:
    count = []  # Class variable, shared among all instances
    
    def add(self, value):
        self.count.append(value)  # Modifies class variable!

c1 = Counter()
c2 = Counter()
c1.add(1)
c2.add(2)
print(c1.count)  # [1, 2] - unexpected!
```

**Solution:**
```python
# Good: Proper instance variables
class Counter:
    def __init__(self):
        self.count = []  # Instance variable
    
    def add(self, value):
        self.count.append(value)

# Or if you want a class variable for counting instances
class Counter:
    total_instances = 0  # Class variable for counting
    
    def __init__(self):
        self.count = []  # Instance variable
        Counter.total_instances += 1
    
    def add(self, value):
        self.count.append(value)
```

---

## Summary

This comprehensive interview preparation guide covers the essential concepts, patterns, and practices for C++ and Python architects. Key takeaways:

1. **Master the fundamentals**: Understanding memory management, language features, and core concepts is crucial
2. **Know design patterns**: Be able to implement and explain common patterns in both languages
3. **Understand system design**: Scalability, performance, and architecture principles are essential
4. **Practice code examples**: Be able to write clean, efficient code in both languages
5. **Avoid common pitfalls**: Know the gotchas and how to avoid them
6. **Stay current**: Keep up with modern language features and best practices

Remember to:
- Practice coding problems regularly
- Study real-world system designs
- Understand the trade-offs between different approaches
- Be able to explain your reasoning clearly
- Focus on writing maintainable, efficient code

Good luck with your interviews!
