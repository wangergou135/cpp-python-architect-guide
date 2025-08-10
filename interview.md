# Comprehensive Interview Guide

## Table of Contents
1. [C++ Core Topics](#c-core-topics)
2. [Python Core Topics](#python-core-topics)
3. [Architecture Topics](#architecture-topics)

---

## C++ Core Topics

### Memory Management

#### Key Concepts
- **Stack vs Heap Memory**: Stack memory is automatically managed, heap memory requires manual management
- **RAII (Resource Acquisition Is Initialization)**: Resources are tied to object lifetime
- **Smart Pointers**: Modern C++ automatic memory management

#### Code Examples

```cpp
// Manual memory management (avoid in modern C++)
int* ptr = new int(42);
delete ptr;  // Don't forget to delete!

// RAII with smart pointers (preferred)
#include <memory>

std::unique_ptr<int> ptr = std::make_unique<int>(42);
// Automatically cleaned up when ptr goes out of scope

std::shared_ptr<int> shared = std::make_shared<int>(42);
// Reference counted, cleaned up when last reference is destroyed

// Custom deleter
auto custom_deleter = [](FILE* f) { 
    if (f) fclose(f); 
};
std::unique_ptr<FILE, decltype(custom_deleter)> file_ptr(
    fopen("data.txt", "r"), custom_deleter
);
```

#### Common Interview Questions

**Q: What's the difference between unique_ptr and shared_ptr?**

A: 
- `unique_ptr`: Exclusive ownership, cannot be copied, only moved. Zero overhead.
- `shared_ptr`: Shared ownership with reference counting. Small overhead due to control block.

```cpp
// unique_ptr - exclusive ownership
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
// std::unique_ptr<int> ptr2 = ptr1;  // ERROR: Cannot copy
std::unique_ptr<int> ptr2 = std::move(ptr1);  // OK: Transfer ownership

// shared_ptr - shared ownership
std::shared_ptr<int> shared1 = std::make_shared<int>(42);
std::shared_ptr<int> shared2 = shared1;  // OK: Both point to same object
std::cout << shared1.use_count() << std::endl;  // Prints: 2
```

**Q: What is a memory leak and how do you prevent it?**

A: Memory leak occurs when allocated memory is not freed. Prevention strategies:
1. Use RAII and smart pointers
2. Match every `new` with `delete`, every `new[]` with `delete[]`
3. Use containers instead of raw arrays
4. Use tools like Valgrind, AddressSanitizer

### STL (Standard Template Library)

#### Key Containers

```cpp
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>

// Vector - dynamic array
std::vector<int> vec = {1, 2, 3, 4, 5};
vec.push_back(6);

// Map - sorted key-value pairs
std::map<std::string, int> ages;
ages["Alice"] = 25;
ages["Bob"] = 30;

// Unordered map - hash table
std::unordered_map<std::string, int> fast_lookup;
fast_lookup["key"] = 42;

// Set - sorted unique elements
std::set<int> unique_nums = {3, 1, 4, 1, 5};  // {1, 3, 4, 5}
```

#### Algorithms

```cpp
#include <algorithm>
#include <numeric>

std::vector<int> numbers = {5, 2, 8, 1, 9};

// Sorting
std::sort(numbers.begin(), numbers.end());

// Finding
auto it = std::find(numbers.begin(), numbers.end(), 8);
if (it != numbers.end()) {
    std::cout << "Found 8 at position: " << std::distance(numbers.begin(), it);
}

// Transforming
std::vector<int> doubled(numbers.size());
std::transform(numbers.begin(), numbers.end(), doubled.begin(),
               [](int x) { return x * 2; });

// Accumulating
int sum = std::accumulate(numbers.begin(), numbers.end(), 0);
```

#### Interview Questions

**Q: What's the time complexity of operations on different STL containers?**

A:
- `vector`: Random access O(1), insertion/deletion at end O(1), middle O(n)
- `map`: Search/insert/delete O(log n)
- `unordered_map`: Average O(1), worst case O(n)
- `set`: Search/insert/delete O(log n)

### Modern C++ Features

#### C++11/14/17/20 Features

```cpp
// Auto keyword
auto x = 42;  // int
auto lambda = [](int a, int b) { return a + b; };

// Range-based for loops
std::vector<int> vec = {1, 2, 3, 4, 5};
for (const auto& element : vec) {
    std::cout << element << " ";
}

// Lambda expressions
auto add = [](int a, int b) -> int { 
    return a + b; 
};

// Move semantics
class MyClass {
private:
    std::vector<int> data;
public:
    // Move constructor
    MyClass(MyClass&& other) noexcept : data(std::move(other.data)) {}
    
    // Move assignment
    MyClass& operator=(MyClass&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }
};

// Variadic templates (C++11)
template<typename... Args>
void print(Args... args) {
    ((std::cout << args << " "), ...);  // C++17 fold expression
}

// Structured bindings (C++17)
std::pair<int, std::string> getPair() {
    return {42, "Hello"};
}

auto [num, str] = getPair();

// Concepts (C++20)
template<typename T>
concept Addable = requires(T a, T b) {
    a + b;
};

template<Addable T>
T add(T a, T b) {
    return a + b;
}
```

#### Interview Questions

**Q: Explain move semantics and when you would use it.**

A: Move semantics allow transferring resources from temporary objects instead of copying them, improving performance.

```cpp
class Resource {
private:
    int* data;
    size_t size;

public:
    // Copy constructor (expensive)
    Resource(const Resource& other) : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
    }
    
    // Move constructor (cheap)
    Resource(Resource&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
};

Resource createResource() {
    return Resource(1000);  // Move constructor called automatically
}
```

### Threading

#### Basic Threading

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Creating threads
void worker(int id) {
    std::cout << "Worker " << id << " starting\n";
}

std::thread t1(worker, 1);
std::thread t2(worker, 2);

t1.join();
t2.join();

// Mutex for synchronization
std::mutex mtx;
int shared_counter = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    ++shared_counter;
}

// Atomic operations
std::atomic<int> atomic_counter(0);

void atomic_increment() {
    atomic_counter.fetch_add(1);
}

// Condition variables
std::mutex cv_mtx;
std::condition_variable cv;
bool ready = false;

void producer() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lock(cv_mtx);
        ready = true;
    }
    cv.notify_all();
}

void consumer() {
    std::unique_lock<std::mutex> lock(cv_mtx);
    cv.wait(lock, [] { return ready; });
    std::cout << "Consumer proceeding\n";
}
```

#### Advanced Threading

```cpp
#include <future>
#include <async>

// Future and promises
std::promise<int> prom;
std::future<int> fut = prom.get_future();

std::thread([&prom] {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    prom.set_value(42);
}).detach();

int result = fut.get();  // Blocks until value is set

// std::async
auto future_result = std::async(std::launch::async, []() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 42;
});

int value = future_result.get();

// Thread pool simulation
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
};
```

#### Interview Questions

**Q: What's the difference between std::mutex and std::atomic?**

A:
- `std::mutex`: Heavyweight synchronization, blocks threads, suitable for protecting larger critical sections
- `std::atomic`: Lightweight, lock-free operations, suitable for simple operations like counters

**Q: Explain the producer-consumer problem and how to solve it.**

A: Multiple producers generate data, multiple consumers process it. Solution uses mutex + condition variable or lock-free queues.

### Design Patterns

#### Singleton Pattern

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
    
    // Delete copy constructor and assignment
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};

// Thread-safe Meyer's singleton (C++11)
class MeyersSingleton {
public:
    static MeyersSingleton& getInstance() {
        static MeyersSingleton instance;  // Thread-safe in C++11+
        return instance;
    }
};
```

#### Factory Pattern

```cpp
class Animal {
public:
    virtual ~Animal() = default;
    virtual void makeSound() = 0;
};

class Dog : public Animal {
public:
    void makeSound() override {
        std::cout << "Woof!\n";
    }
};

class Cat : public Animal {
public:
    void makeSound() override {
        std::cout << "Meow!\n";
    }
};

class AnimalFactory {
public:
    static std::unique_ptr<Animal> createAnimal(const std::string& type) {
        if (type == "dog") {
            return std::make_unique<Dog>();
        } else if (type == "cat") {
            return std::make_unique<Cat>();
        }
        return nullptr;
    }
};
```

#### Observer Pattern

```cpp
#include <vector>
#include <algorithm>

class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(int value) = 0;
};

class Subject {
private:
    std::vector<Observer*> observers;
    int state;

public:
    void attach(Observer* observer) {
        observers.push_back(observer);
    }
    
    void detach(Observer* observer) {
        observers.erase(
            std::remove(observers.begin(), observers.end(), observer),
            observers.end()
        );
    }
    
    void setState(int new_state) {
        state = new_state;
        notify();
    }
    
    void notify() {
        for (Observer* obs : observers) {
            obs->update(state);
        }
    }
};
```

---

## Python Core Topics

### GIL (Global Interpreter Lock)

#### Understanding the GIL

The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

```python
import threading
import time

# GIL impact demonstration
def cpu_bound_task(n):
    """CPU-intensive task affected by GIL"""
    total = 0
    for i in range(n):
        total += i * i
    return total

def io_bound_task():
    """I/O task releases GIL"""
    time.sleep(1)
    return "I/O complete"

# Single-threaded
start = time.time()
result1 = cpu_bound_task(1000000)
result2 = cpu_bound_task(1000000)
single_threaded_time = time.time() - start

# Multi-threaded (won't improve CPU-bound performance)
start = time.time()
threads = []
for _ in range(2):
    t = threading.Thread(target=cpu_bound_task, args=(1000000,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
multi_threaded_time = time.time() - start

print(f"Single-threaded: {single_threaded_time:.2f}s")
print(f"Multi-threaded: {multi_threaded_time:.2f}s")
```

#### Working Around the GIL

```python
import multiprocessing
import concurrent.futures
import asyncio

# 1. Multiprocessing for CPU-bound tasks
def cpu_task(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_task, [1000000, 1000000, 1000000])

# 2. AsyncIO for I/O-bound tasks
async def fetch_data(url):
    # Simulated async I/O
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    tasks = [fetch_data(f"url{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results

# 3. Threading for I/O-bound tasks
def io_task():
    time.sleep(1)
    return "I/O complete"

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(io_task) for _ in range(5)]
    results = [f.result() for f in futures]
```

#### Interview Questions

**Q: When should you use threading vs multiprocessing vs asyncio in Python?**

A:
- **Threading**: I/O-bound tasks (file operations, network requests)
- **Multiprocessing**: CPU-bound tasks (mathematical calculations, data processing)
- **AsyncIO**: I/O-bound tasks with many concurrent operations (web servers, crawlers)

### Decorators

#### Basic Decorators

```python
import functools
import time
from typing import Callable, Any

# Simple decorator
def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

# Timing decorator
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
    return "Done"
```

#### Parameterized Decorators

```python
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise ValueError("Random failure")
    return "Success!"

# Caching decorator
def lru_cache(maxsize: int = 128):
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                # Remove oldest entry (simplified LRU)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            cache[key] = result
            return result
        return wrapper
    return decorator
```

#### Class-based Decorators

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    return "Hello!"

# Property decorators
class Temperature:
    def __init__(self):
        self._celsius = 0
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9
```

#### Interview Questions

**Q: What's the difference between @property and @staticmethod/@classmethod?**

A:
- `@property`: Creates computed attributes that can be accessed like regular attributes
- `@staticmethod`: Methods that don't need access to self or cls
- `@classmethod`: Methods that receive the class as the first argument

### Async Programming

#### Basic Async/Await

```python
import asyncio
import aiohttp
import time

# Basic async function
async def async_function():
    await asyncio.sleep(1)
    return "Async result"

# Running async code
async def main():
    result = await async_function()
    print(result)

# Multiple concurrent operations
async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple_urls():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Event loop management
def run_async():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(main())
        return result
    finally:
        loop.close()
```

#### Advanced Async Patterns

```python
import asyncio
from asyncio import Queue
import random

# Producer-Consumer with async
async def producer(queue: Queue, producer_id: int):
    for i in range(5):
        item = f"item-{producer_id}-{i}"
        await queue.put(item)
        print(f"Producer {producer_id} produced {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))

async def consumer(queue: Queue, consumer_id: int):
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            print(f"Consumer {consumer_id} consumed {item}")
            queue.task_done()
            await asyncio.sleep(random.uniform(0.1, 0.3))
        except asyncio.TimeoutError:
            break

async def producer_consumer_example():
    queue = Queue(maxsize=10)
    
    # Create producers and consumers
    producers = [producer(queue, i) for i in range(2)]
    consumers = [consumer(queue, i) for i in range(3)]
    
    # Run producers
    await asyncio.gather(*producers)
    
    # Wait for all items to be processed
    await queue.join()
    
    # Cancel consumers
    consumer_tasks = [asyncio.create_task(c) for c in consumers]
    await asyncio.sleep(1)  # Let consumers timeout
    for task in consumer_tasks:
        task.cancel()

# Async context managers
class AsyncDatabase:
    async def __aenter__(self):
        print("Connecting to database...")
        await asyncio.sleep(0.1)
        self.connection = "fake_connection"
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection...")
        await asyncio.sleep(0.1)
        self.connection = None
    
    async def query(self, sql):
        await asyncio.sleep(0.1)
        return f"Result for: {sql}"

async def database_example():
    async with AsyncDatabase() as db:
        result = await db.query("SELECT * FROM users")
        print(result)

# Async generators
async def async_range(start, stop):
    current = start
    while current < stop:
        yield current
        current += 1
        await asyncio.sleep(0.1)

async def async_generator_example():
    async for num in async_range(0, 5):
        print(f"Got: {num}")
```

#### Interview Questions

**Q: What's the difference between asyncio.gather() and asyncio.wait()?**

A:
- `asyncio.gather()`: Waits for all tasks to complete, returns results in order, fails fast on first exception
- `asyncio.wait()`: More control over completion conditions, returns done and pending sets

### Memory Management

#### Reference Counting and Garbage Collection

```python
import gc
import sys
import weakref

# Reference counting
class MyClass:
    def __init__(self, name):
        self.name = name
    
    def __del__(self):
        print(f"MyClass {self.name} is being destroyed")

obj = MyClass("test")
print(f"Reference count: {sys.getrefcount(obj)}")  # Includes the argument reference

# Creating circular references
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)

# This creates a circular reference
parent = Node("parent")
child = Node("child")
parent.add_child(child)

# Weak references to avoid circular references
class WeakNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self._parent = None
    
    @property
    def parent(self):
        return self._parent() if self._parent else None
    
    @parent.setter
    def parent(self, value):
        self._parent = weakref.ref(value) if value else None
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)

# Memory profiling
import tracemalloc

def memory_intensive_function():
    # Simulate memory usage
    data = []
    for i in range(100000):
        data.append([i] * 100)
    return data

tracemalloc.start()
result = memory_intensive_function()
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

#### Memory Optimization Techniques

```python
import array
from collections import deque
import sys

# Use appropriate data structures
# Regular list
regular_list = [1, 2, 3, 4, 5]
print(f"List size: {sys.getsizeof(regular_list)} bytes")

# Array for numeric data
numeric_array = array.array('i', [1, 2, 3, 4, 5])
print(f"Array size: {sys.getsizeof(numeric_array)} bytes")

# Generators for large datasets
def large_sequence():
    for i in range(1000000):
        yield i * 2

# Memory-efficient vs memory-hungry
# Memory hungry
def process_file_memory_hungry(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()  # Loads entire file into memory
    return [line.strip().upper() for line in lines]

# Memory efficient
def process_file_memory_efficient(filename):
    with open(filename, 'r') as f:
        for line in f:  # Process one line at a time
            yield line.strip().upper()

# Using __slots__ to reduce memory overhead
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Memory usage comparison
regular_obj = RegularClass(1, 2)
slotted_obj = SlottedClass(1, 2)

print(f"Regular object size: {sys.getsizeof(regular_obj)} bytes")
print(f"Slotted object size: {sys.getsizeof(slotted_obj)} bytes")
```

### Design Patterns

#### Singleton Pattern

```python
class Singleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.value = 0
            self._initialized = True

# Thread-safe singleton
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# Decorator-based singleton
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        self.connection = "database_connection"
```

#### Factory Pattern

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type.lower() == "dog":
            return Dog()
        elif animal_type.lower() == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Abstract factory
class AbstractFactory(ABC):
    @abstractmethod
    def create_animal(self):
        pass
    
    @abstractmethod
    def create_food(self):
        pass

class DogFactory(AbstractFactory):
    def create_animal(self):
        return Dog()
    
    def create_food(self):
        return "Dog food"

class CatFactory(AbstractFactory):
    def create_animal(self):
        return Cat()
    
    def create_food(self):
        return "Cat food"
```

#### Observer Pattern

```python
from abc import ABC, abstractmethod
from typing import List

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None
    
    def attach(self, observer: Observer):
        self._observers.append(observer)
    
    def detach(self, observer: Observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, subject):
        print(f"{self.name} received update: {subject.state}")

# Usage
subject = Subject()
observer1 = ConcreteObserver("Observer 1")
observer2 = ConcreteObserver("Observer 2")

subject.attach(observer1)
subject.attach(observer2)

subject.state = "New State"  # Both observers will be notified
```

---

## Architecture Topics

### System Design

#### Scalability Fundamentals

```python
# Load Balancer Example
import random
from typing import List

class Server:
    def __init__(self, server_id: str, capacity: int = 100):
        self.server_id = server_id
        self.capacity = capacity
        self.current_load = 0
    
    def handle_request(self, request):
        if self.current_load < self.capacity:
            self.current_load += 1
            return f"Server {self.server_id} handled request: {request}"
        else:
            return f"Server {self.server_id} is overloaded"
    
    def finish_request(self):
        if self.current_load > 0:
            self.current_load -= 1

class LoadBalancer:
    def __init__(self, servers: List[Server]):
        self.servers = servers
    
    def round_robin(self, request):
        # Simple round-robin implementation
        server = self.servers[hash(request) % len(self.servers)]
        return server.handle_request(request)
    
    def least_connections(self, request):
        # Route to server with least connections
        server = min(self.servers, key=lambda s: s.current_load)
        return server.handle_request(request)
    
    def weighted_round_robin(self, request):
        # Route based on server capacity
        available_servers = [s for s in self.servers if s.current_load < s.capacity]
        if available_servers:
            server = random.choice(available_servers)
            return server.handle_request(request)
        return "All servers overloaded"

# Caching Strategy
import time
from functools import wraps

class Cache:
    def __init__(self, ttl: int = 300):  # 5 minutes TTL
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())
    
    def invalidate(self, key):
        if key in self.cache:
            del self.cache[key]

def cached(cache_instance):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            result = cache_instance.get(cache_key)
            if result is None:
                result = func(*args, **kwargs)
                cache_instance.set(cache_key, result)
            return result
        return wrapper
    return decorator

# Database Sharding Example
class DatabaseShard:
    def __init__(self, shard_id: str):
        self.shard_id = shard_id
        self.data = {}
    
    def insert(self, key, value):
        self.data[key] = value
    
    def get(self, key):
        return self.data.get(key)

class ShardedDatabase:
    def __init__(self, num_shards: int = 4):
        self.shards = [DatabaseShard(f"shard_{i}") for i in range(num_shards)]
    
    def _get_shard(self, key):
        shard_index = hash(key) % len(self.shards)
        return self.shards[shard_index]
    
    def insert(self, key, value):
        shard = self._get_shard(key)
        shard.insert(key, value)
    
    def get(self, key):
        shard = self._get_shard(key)
        return shard.get(key)
```

#### Interview Questions

**Q: How would you design a URL shortener like bit.ly?**

A:
1. **Requirements**: Shorten URLs, redirect to original, analytics, custom aliases
2. **Scale**: 100M URLs per day, 100:1 read/write ratio
3. **Design**:
   - Base62 encoding for short URLs
   - Database sharding by URL hash
   - Cache for popular URLs
   - Analytics service for tracking

```python
import hashlib
import base64
from typing import Optional

class URLShortener:
    def __init__(self):
        self.url_database = {}  # In practice, this would be a distributed database
        self.cache = Cache()
        self.counter = 1000000  # Starting counter for unique IDs
    
    def encode_base62(self, num: int) -> str:
        """Encode number to base62 string"""
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if num == 0:
            return chars[0]
        
        result = ""
        while num > 0:
            result = chars[num % 62] + result
            num //= 62
        return result
    
    def shorten_url(self, long_url: str, custom_alias: Optional[str] = None) -> str:
        if custom_alias:
            if custom_alias in self.url_database:
                raise ValueError("Custom alias already exists")
            short_code = custom_alias
        else:
            self.counter += 1
            short_code = self.encode_base62(self.counter)
        
        self.url_database[short_code] = {
            'long_url': long_url,
            'created_at': time.time(),
            'click_count': 0
        }
        
        return f"https://short.ly/{short_code}"
    
    def expand_url(self, short_code: str) -> Optional[str]:
        # Check cache first
        cached_url = self.cache.get(short_code)
        if cached_url:
            return cached_url
        
        # Get from database
        url_data = self.url_database.get(short_code)
        if url_data:
            long_url = url_data['long_url']
            # Update click count
            url_data['click_count'] += 1
            # Cache the result
            self.cache.set(short_code, long_url)
            return long_url
        
        return None
```

### Microservices

#### Service Communication Patterns

```python
import asyncio
import json
from typing import Dict, Any
import httpx

# Service Discovery
class ServiceRegistry:
    def __init__(self):
        self.services = {}
    
    def register(self, service_name: str, host: str, port: int):
        if service_name not in self.services:
            self.services[service_name] = []
        self.services[service_name].append(f"http://{host}:{port}")
    
    def discover(self, service_name: str):
        return self.services.get(service_name, [])

# API Gateway
class APIGateway:
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.rate_limits = {}
    
    async def route_request(self, path: str, method: str, data: Dict[Any, Any] = None):
        # Simple routing logic
        service_name = path.split('/')[1] if path.startswith('/') else path
        services = self.service_registry.discover(service_name)
        
        if not services:
            return {"error": "Service not found"}, 404
        
        # Load balancing - simple round robin
        service_url = services[0]  # In practice, implement proper load balancing
        
        # Rate limiting check
        if not self._check_rate_limit(service_name):
            return {"error": "Rate limit exceeded"}, 429
        
        # Forward request
        async with httpx.AsyncClient() as client:
            try:
                if method.upper() == "GET":
                    response = await client.get(f"{service_url}{path}")
                elif method.upper() == "POST":
                    response = await client.post(f"{service_url}{path}", json=data)
                
                return response.json(), response.status_code
            except httpx.RequestError:
                return {"error": "Service unavailable"}, 503
    
    def _check_rate_limit(self, service_name: str) -> bool:
        # Simplified rate limiting
        current_time = time.time()
        if service_name not in self.rate_limits:
            self.rate_limits[service_name] = []
        
        # Remove old requests (older than 1 minute)
        self.rate_limits[service_name] = [
            req_time for req_time in self.rate_limits[service_name]
            if current_time - req_time < 60
        ]
        
        # Check if under limit (100 requests per minute)
        if len(self.rate_limits[service_name]) < 100:
            self.rate_limits[service_name].append(current_time)
            return True
        return False

# Circuit Breaker Pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Event Sourcing Example
class Event:
    def __init__(self, event_type: str, data: Dict[Any, Any], timestamp: float = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or time.time()

class EventStore:
    def __init__(self):
        self.events = []
    
    def append(self, event: Event):
        self.events.append(event)
    
    def get_events(self, aggregate_id: str = None):
        if aggregate_id:
            return [e for e in self.events if e.data.get('aggregate_id') == aggregate_id]
        return self.events

class AccountAggregate:
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.balance = 0
        self.version = 0
    
    def apply_event(self, event: Event):
        if event.event_type == "AccountCreated":
            self.balance = event.data['initial_balance']
        elif event.event_type == "MoneyDeposited":
            self.balance += event.data['amount']
        elif event.event_type == "MoneyWithdrawn":
            self.balance -= event.data['amount']
        self.version += 1
    
    def deposit(self, amount: float, event_store: EventStore):
        event = Event("MoneyDeposited", {
            'aggregate_id': self.account_id,
            'amount': amount
        })
        event_store.append(event)
        self.apply_event(event)
    
    def withdraw(self, amount: float, event_store: EventStore):
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        
        event = Event("MoneyWithdrawn", {
            'aggregate_id': self.account_id,
            'amount': amount
        })
        event_store.append(event)
        self.apply_event(event)
```

#### Interview Questions

**Q: How do you handle data consistency in microservices?**

A: Several patterns:
1. **Saga Pattern**: Orchestrated or choreographed transactions
2. **Event Sourcing**: Store events, not state
3. **CQRS**: Separate read and write models
4. **Eventual Consistency**: Accept temporary inconsistency for availability

### Performance

#### Performance Optimization Techniques

```cpp
// C++ Performance Examples
#include <chrono>
#include <vector>
#include <algorithm>
#include <memory>

// 1. Memory locality - cache-friendly data structures
struct Point3D_AoS {  // Array of Structures (cache-unfriendly for specific operations)
    float x, y, z;
};

struct Point3D_SoA {  // Structure of Arrays (cache-friendly)
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

// 2. Move semantics for expensive objects
class ExpensiveResource {
private:
    std::vector<int> data;
public:
    ExpensiveResource(size_t size) : data(size, 42) {}
    
    // Copy constructor (expensive)
    ExpensiveResource(const ExpensiveResource& other) : data(other.data) {
        std::cout << "Expensive copy!\n";
    }
    
    // Move constructor (cheap)
    ExpensiveResource(ExpensiveResource&& other) noexcept : data(std::move(other.data)) {
        std::cout << "Cheap move!\n";
    }
};

// 3. Template specialization for performance
template<typename T>
void process_data(const std::vector<T>& data) {
    // Generic implementation
    for (const auto& item : data) {
        // Process item
    }
}

// Specialized version for better performance with specific types
template<>
void process_data<int>(const std::vector<int>& data) {
    // Optimized implementation for integers
    // Maybe use SIMD instructions
}

// 4. Memory pool for frequent allocations
template<typename T>
class MemoryPool {
private:
    std::vector<T> pool;
    std::vector<T*> free_list;
    size_t pool_size;

public:
    MemoryPool(size_t size) : pool_size(size) {
        pool.resize(pool_size);
        for (size_t i = 0; i < pool_size; ++i) {
            free_list.push_back(&pool[i]);
        }
    }
    
    T* allocate() {
        if (free_list.empty()) {
            throw std::runtime_error("Pool exhausted");
        }
        T* ptr = free_list.back();
        free_list.pop_back();
        return ptr;
    }
    
    void deallocate(T* ptr) {
        free_list.push_back(ptr);
    }
};
```

```python
# Python Performance Examples
import time
import numpy as np
from numba import jit
import cProfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 1. Profiling and timing
def profile_function():
    def slow_function():
        total = 0
        for i in range(1000000):
            total += i * i
        return total
    
    # Time measurement
    start = time.time()
    result = slow_function()
    end = time.time()
    print(f"Function took: {end - start:.4f} seconds")
    
    # Profiling
    cProfile.run('slow_function()')

# 2. NumPy for numerical computations
def numpy_performance():
    # Pure Python (slow)
    python_list = list(range(1000000))
    start = time.time()
    python_result = [x * 2 for x in python_list]
    python_time = time.time() - start
    
    # NumPy (fast)
    numpy_array = np.arange(1000000)
    start = time.time()
    numpy_result = numpy_array * 2
    numpy_time = time.time() - start
    
    print(f"Python time: {python_time:.4f}s")
    print(f"NumPy time: {numpy_time:.4f}s")
    print(f"Speedup: {python_time / numpy_time:.2f}x")

# 3. JIT compilation with Numba
@jit(nopython=True)
def numba_function(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

def compare_numba():
    def python_function(n):
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    n = 1000000
    
    # Python version
    start = time.time()
    python_result = python_function(n)
    python_time = time.time() - start
    
    # Numba version (first call includes compilation)
    start = time.time()
    numba_result = numba_function(n)
    numba_time = time.time() - start
    
    print(f"Python: {python_time:.4f}s")
    print(f"Numba: {numba_time:.4f}s")

# 4. Async I/O for concurrent operations
async def fetch_data(url, session):
    # Simulate API call
    await asyncio.sleep(1)
    return f"Data from {url}"

async def concurrent_requests():
    urls = [f"https://api{i}.example.com" for i in range(10)]
    
    # Sequential (slow)
    start = time.time()
    results = []
    for url in urls:
        result = await fetch_data(url, None)
        results.append(result)
    sequential_time = time.time() - start
    
    # Concurrent (fast)
    start = time.time()
    tasks = [fetch_data(url, None) for url in urls]
    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")

# 5. Caching for expensive computations
import functools

@functools.lru_cache(maxsize=None)
def expensive_fibonacci(n):
    if n < 2:
        return n
    return expensive_fibonacci(n-1) + expensive_fibonacci(n-2)

# 6. Generator expressions for memory efficiency
def memory_efficient_processing():
    # Memory inefficient
    numbers = [i for i in range(1000000)]
    squares = [x * x for x in numbers]
    
    # Memory efficient
    numbers_gen = (i for i in range(1000000))
    squares_gen = (x * x for x in numbers_gen)
    
    # Process only what you need
    first_10_squares = [next(squares_gen) for _ in range(10)]
    return first_10_squares
```

#### Interview Questions

**Q: How would you optimize a slow database query?**

A: 
1. **Indexing**: Add appropriate indexes
2. **Query optimization**: Rewrite inefficient queries
3. **Denormalization**: Trade storage for speed
4. **Caching**: Cache frequently accessed data
5. **Partitioning**: Split large tables
6. **Read replicas**: Distribute read load

### Security

#### Common Security Patterns

```python
import hashlib
import hmac
import secrets
import jwt
from cryptography.fernet import Fernet
import bcrypt
from functools import wraps

# 1. Password hashing
class PasswordManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# 2. JWT Token management
class JWTManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, expiration_hours: int = 24) -> str:
        """Generate JWT token"""
        import datetime
        payload = {
            'user_id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=expiration_hours),
            'iat': datetime.datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

# 3. Data encryption
class DataEncryption:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data).decode()

# 4. Rate limiting
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(current_time)
            return True
        return False

# 5. Input validation and sanitization
import re
from html import escape

class InputValidator:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def sanitize_html(input_string: str) -> str:
        """Sanitize HTML input to prevent XSS"""
        return escape(input_string)
    
    @staticmethod
    def validate_sql_injection(input_string: str) -> bool:
        """Basic SQL injection detection"""
        dangerous_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)',
            r'(\b(UNION|OR|AND)\b.*\b(SELECT)\b)',
            r'(\'|\"|;|--|\|\|)'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                return False
        return True

# 6. Secure session management
class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id: str) -> str:
        """Create secure session"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        return session_id
    
    def validate_session(self, session_id: str, max_age: int = 3600) -> bool:
        """Validate session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check if session has expired
        if current_time - session['last_activity'] > max_age:
            del self.sessions[session_id]
            return False
        
        # Update last activity
        session['last_activity'] = current_time
        return True
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# 7. API Security decorator
def require_auth(jwt_manager: JWTManager):
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            # Extract token from Authorization header
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return {'error': 'Missing or invalid authorization header'}, 401
            
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            
            try:
                payload = jwt_manager.verify_token(token)
                request.user_id = payload['user_id']
                return func(request, *args, **kwargs)
            except ValueError as e:
                return {'error': str(e)}, 401
        
        return wrapper
    return decorator

# Usage example
jwt_manager = JWTManager("your-secret-key")

@require_auth(jwt_manager)
def protected_endpoint(request):
    return {'message': f'Hello user {request.user_id}'}
```

#### Interview Questions

**Q: How do you prevent SQL injection attacks?**

A:
1. **Parameterized queries**: Use prepared statements
2. **Input validation**: Validate all user inputs
3. **Least privilege**: Database users should have minimal permissions
4. **Escape special characters**: Though parameterization is preferred
5. **Use ORM**: Object-Relational Mapping tools often prevent SQL injection

**Q: Explain OAuth 2.0 flow.**

A: OAuth 2.0 authorization code flow:
1. Client redirects user to authorization server
2. User authenticates and grants permission
3. Authorization server redirects back with authorization code
4. Client exchanges code for access token
5. Client uses access token to access protected resources

---

## Additional Resources

### Recommended Reading
- **C++**: "Effective Modern C++" by Scott Meyers
- **Python**: "Fluent Python" by Luciano Ramalho
- **Architecture**: "Designing Data-Intensive Applications" by Martin Kleppmann

### Practice Platforms
- LeetCode for algorithmic problems
- System Design Interview questions
- GitHub projects for hands-on experience

### Key Takeaways for Interviews
1. **Start simple**: Begin with basic solution, then optimize
2. **Ask clarifying questions**: Understand requirements fully
3. **Consider trade-offs**: Discuss pros and cons of different approaches
4. **Code quality**: Write clean, readable code
5. **Testing**: Discuss how you would test your solution

*Last updated: 2025-01-11*
