# Interview Preparation Guide

## 1. C++ Core Concepts

### Memory Management (RAII, Smart Pointers)
- **RAII (Resource Acquisition Is Initialization)**: Resources are tied to object lifetime
- **Smart Pointers**: Automatic memory management with reference counting
- **Memory Leaks**: Understanding stack vs heap allocation
- **Custom Allocators**: When and how to implement them

### Move Semantics
- **Rvalue References**: Understanding `&&` and perfect forwarding
- **Move Constructors**: Efficient resource transfer
- **std::move and std::forward**: When to use each
- **Copy Elision**: Return Value Optimization (RVO)

### Template Meta-programming
- **SFINAE**: Substitution Failure Is Not An Error
- **Variadic Templates**: Parameter packs and template recursion
- **Concepts (C++20)**: Type constraints and requirements
- **Template Specialization**: Partial and explicit specialization

### STL Containers and Algorithms
- **Container Performance**: Time complexities and memory layout
- **Iterator Categories**: Input, output, forward, bidirectional, random access
- **Algorithm Complexity**: Understanding Big O notation
- **Custom Comparators**: Function objects and lambdas

### Multi-threading and Synchronization
- **Thread Safety**: Race conditions and data races
- **Synchronization Primitives**: mutex, condition_variable, atomic
- **Memory Ordering**: Happens-before relationships
- **Lock-free Programming**: Compare-and-swap operations

Example code snippets:
```cpp
// Smart pointer example with custom deleter
std::unique_ptr<Resource> res = std::make_unique<Resource>();

// RAII example
class FileHandler {
    FILE* file_;
public:
    FileHandler(const char* filename) : file_(fopen(filename, "r")) {
        if (!file_) throw std::runtime_error("Failed to open file");
    }
    ~FileHandler() { if (file_) fclose(file_); }
    FILE* get() const { return file_; }
};

// Move semantics example
class MoveableResource {
    std::unique_ptr<int[]> data_;
    size_t size_;
    
public:
    // Move constructor
    MoveableResource(MoveableResource&& other) noexcept 
        : data_(std::move(other.data_)), size_(other.size_) {
        other.size_ = 0;
    }
    
    // Move assignment
    MoveableResource& operator=(MoveableResource&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }
};

// Thread-safe singleton with call_once
class Singleton {
    static std::unique_ptr<Singleton> instance_;
    static std::once_flag once_flag_;
    
    Singleton() = default;

public:
    static Singleton* getInstance() {
        std::call_once(once_flag_, []() {
            instance_.reset(new Singleton());
        });
        return instance_.get();
    }
};

// Template metaprogramming example
template<typename T>
struct is_pointer : std::false_type {};

template<typename T>
struct is_pointer<T*> : std::true_type {};

template<typename T>
constexpr bool is_pointer_v = is_pointer<T>::value;

// Thread-safe queue implementation
template<typename T>
class ThreadSafeQueue {
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        condition_.notify_one();
    }
    
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = queue_.front();
        queue_.pop();
        return true;
    }
    
    void wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        item = queue_.front();
        queue_.pop();
    }
};
```

## 2. Python Advanced Features

### GIL Understanding
- **Global Interpreter Lock**: Why Python has true parallelism limitations
- **CPU-bound vs I/O-bound**: When GIL matters and when it doesn't
- **Multiprocessing vs Threading**: Choosing the right concurrency model
- **GIL Release**: Understanding when Python releases the GIL

### Decorators and Context Managers
- **Function Decorators**: Modifying behavior without changing code
- **Class Decorators**: Decorating entire classes
- **Decorator Patterns**: Caching, validation, timing, authentication
- **Context Managers**: Resource management with `__enter__` and `__exit__`

### Asyncio and Coroutines
- **Event Loop**: Understanding the core of asyncio
- **Coroutines vs Generators**: Differences and use cases
- **async/await**: Modern asynchronous programming
- **Concurrency Patterns**: Gathering, semaphores, queues

### Metaclasses
- **Class Creation Process**: How Python creates classes
- **`__new__` vs `__init__`**: Class instantiation process
- **Metaclass Use Cases**: ORMs, singletons, validation
- **Type System**: Understanding Python's type hierarchy

### Memory Management
- **Reference Counting**: How Python manages memory
- **Garbage Collection**: Cyclic reference detection
- **Memory Profiling**: Tools and techniques
- **Memory Optimization**: Slots, interning, weak references

Example code snippets:
```python
import time
import asyncio
import weakref
from functools import wraps
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Advanced decorator with parameters
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

# Context manager example
@contextmanager
def database_transaction():
    conn = get_database_connection()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()

# Advanced async example with semaphore
class AsyncRateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(max_requests)
        self.requests = []
    
    async def acquire(self):
        await self.semaphore.acquire()
        now = time.time()
        # Clean old requests
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        self.requests.append(now)
    
    def release(self):
        self.semaphore.release()

async def fetch_data_with_rate_limit(url: str, rate_limiter: AsyncRateLimiter):
    await rate_limiter.acquire()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    finally:
        rate_limiter.release()

# Metaclass example
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = self._create_connection()
    
    def _create_connection(self):
        # Simulate database connection
        return "database_connection"

# Memory-efficient class with slots
class Point:
    __slots__ = ['x', 'y']
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

# Weak reference example for avoiding memory leaks
class Observer:
    def __init__(self):
        self._observers = weakref.WeakSet()
    
    def subscribe(self, observer):
        self._observers.add(observer)
    
    def notify(self, event):
        for observer in self._observers:
            observer.handle_event(event)

# Generator with advanced features
def fibonacci_generator(limit: Optional[int] = None):
    a, b = 0, 1
    count = 0
    while limit is None or count < limit:
        yield a
        a, b = b, a + b
        count += 1

# Coroutine example
async def process_data_pipeline(data_source):
    async for item in data_source:
        # Transform data
        processed = await transform_item(item)
        # Validate
        if await validate_item(processed):
            yield processed
```

## 3. System Design & Architecture

### Microservices Architecture
- **Service Decomposition**: Domain-driven design principles
- **Communication Patterns**: Synchronous vs asynchronous messaging
- **Service Discovery**: Dynamic service registration and lookup
- **Data Management**: Database per service pattern
- **Distributed Transactions**: Saga pattern and eventual consistency

### Load Balancing
- **Load Balancing Algorithms**: Round-robin, weighted, least connections
- **Health Checks**: Service monitoring and failover
- **Session Affinity**: Sticky sessions and stateless design
- **Geographic Distribution**: CDN and edge computing

### Caching Strategies
- **Cache Patterns**: Cache-aside, write-through, write-behind
- **Cache Levels**: Browser, CDN, application, database
- **Cache Invalidation**: TTL, manual invalidation, event-driven
- **Distributed Caching**: Redis, Memcached, consistency models

### Database Scaling
- **Vertical vs Horizontal Scaling**: Scale-up vs scale-out
- **Sharding Strategies**: Range-based, hash-based, directory-based
- **Replication**: Master-slave, master-master, read replicas
- **CAP Theorem**: Consistency, availability, partition tolerance trade-offs

### Message Queues
- **Queue Types**: FIFO, priority, delay queues
- **Message Patterns**: Point-to-point, publish-subscribe
- **Reliability**: At-least-once, at-most-once, exactly-once delivery
- **Backpressure**: Handling overloaded consumers

Example implementations:
```python
import random
import hashlib
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Advanced Load Balancer with multiple algorithms
class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"

@dataclass
class Server:
    id: str
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    response_time_avg: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0

class LoadBalancer:
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.servers: List[Server] = []
        self.strategy = strategy
        self.current_index = 0
        
    def add_server(self, server: Server):
        self.servers.append(server)
    
    def remove_server(self, server_id: str):
        self.servers = [s for s in self.servers if s.id != server_id]
    
    def get_healthy_servers(self) -> List[Server]:
        return [s for s in self.servers if s.is_healthy]
    
    def get_next_server(self) -> Optional[Server]:
        healthy_servers = self.get_healthy_servers()
        if not healthy_servers:
            return None
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(healthy_servers)
    
    def _round_robin(self, servers: List[Server]) -> Server:
        server = servers[self.current_index % len(servers)]
        self.current_index = (self.current_index + 1) % len(servers)
        return server
    
    def _weighted_round_robin(self, servers: List[Server]) -> Server:
        total_weight = sum(s.weight for s in servers)
        random_weight = random.randint(1, total_weight)
        
        current_weight = 0
        for server in servers:
            current_weight += server.weight
            if random_weight <= current_weight:
                return server
        return servers[0]
    
    def _least_connections(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.active_connections)
    
    def _least_response_time(self, servers: List[Server]) -> Server:
        return min(servers, key=lambda s: s.response_time_avg)

# Distributed Cache Implementation
class CacheNode:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.data: Dict[str, Any] = {}
        self.last_access: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.data:
            self.last_access[key] = time.time()
            return self.data[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        self.data[key] = value
        self.last_access[key] = time.time()
        if ttl:
            # In real implementation, would use a timer or background process
            pass
    
    def delete(self, key: str):
        self.data.pop(key, None)
        self.last_access.pop(key, None)

class ConsistentHashRing:
    def __init__(self, replicas: int = 3):
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.nodes: Dict[str, CacheNode] = {}
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: CacheNode):
        self.nodes[node.node_id] = node
        for i in range(self.replicas):
            virtual_key = f"{node.node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node.node_id
    
    def remove_node(self, node_id: str):
        if node_id in self.nodes:
            del self.nodes[node_id]
            keys_to_remove = [k for k, v in self.ring.items() if v == node_id]
            for key in keys_to_remove:
                del self.ring[key]
    
    def get_node(self, key: str) -> Optional[CacheNode]:
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        # Find the first node clockwise
        for ring_key in sorted(self.ring.keys()):
            if hash_value <= ring_key:
                return self.nodes[self.ring[ring_key]]
        # Wrap around to the first node
        first_key = min(self.ring.keys())
        return self.nodes[self.ring[first_key]]

class DistributedCache:
    def __init__(self):
        self.hash_ring = ConsistentHashRing()
    
    def add_node(self, node_id: str):
        node = CacheNode(node_id)
        self.hash_ring.add_node(node)
    
    def get(self, key: str) -> Optional[Any]:
        node = self.hash_ring.get_node(key)
        return node.get(key) if node else None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        node = self.hash_ring.get_node(key)
        if node:
            node.set(key, value, ttl)

# Rate Limiter Implementation
class RateLimiter:
    def __init__(self, max_requests: int, window_size: int):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        requests = self.requests[identifier]
        self.requests[identifier] = [req for req in requests if now - req < self.window_size]
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        return False

# Message Queue Implementation
class Message:
    def __init__(self, data: Any, priority: int = 0, delay: float = 0):
        self.data = data
        self.priority = priority
        self.timestamp = time.time() + delay
        self.retry_count = 0
        self.max_retries = 3

class MessageQueue:
    def __init__(self, max_size: Optional[int] = None):
        self.messages: List[Message] = []
        self.max_size = max_size
        self.dead_letter_queue: List[Message] = []
    
    def enqueue(self, message: Message) -> bool:
        if self.max_size and len(self.messages) >= self.max_size:
            return False
        
        # Insert based on priority and timestamp
        inserted = False
        for i, existing in enumerate(self.messages):
            if (message.priority > existing.priority or 
                (message.priority == existing.priority and message.timestamp < existing.timestamp)):
                self.messages.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            self.messages.append(message)
        return True
    
    def dequeue(self) -> Optional[Message]:
        now = time.time()
        for i, message in enumerate(self.messages):
            if message.timestamp <= now:
                return self.messages.pop(i)
        return None
    
    def requeue_failed(self, message: Message):
        message.retry_count += 1
        if message.retry_count > message.max_retries:
            self.dead_letter_queue.append(message)
        else:
            message.timestamp = time.time() + (2 ** message.retry_count)  # Exponential backoff
            self.enqueue(message)

# Circuit Breaker Pattern
class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
```

## 4. Design Patterns

### Creational Patterns

#### Factory Pattern
```python
from abc import ABC, abstractmethod

class DatabaseConnection(ABC):
    @abstractmethod
    def connect(self):
        pass

class MySQLConnection(DatabaseConnection):
    def connect(self):
        return "Connected to MySQL"

class PostgreSQLConnection(DatabaseConnection):
    def connect(self):
        return "Connected to PostgreSQL"

class DatabaseFactory:
    @staticmethod
    def create_connection(db_type: str) -> DatabaseConnection:
        if db_type == "mysql":
            return MySQLConnection()
        elif db_type == "postgresql":
            return PostgreSQLConnection()
        else:
            raise ValueError(f"Unknown database type: {db_type}")
```

#### Builder Pattern
```cpp
class Computer {
private:
    std::string cpu_;
    std::string ram_;
    std::string storage_;
    std::string gpu_;

public:
    void setCPU(const std::string& cpu) { cpu_ = cpu; }
    void setRAM(const std::string& ram) { ram_ = ram; }
    void setStorage(const std::string& storage) { storage_ = storage; }
    void setGPU(const std::string& gpu) { gpu_ = gpu; }
    
    std::string getSpecs() const {
        return "CPU: " + cpu_ + ", RAM: " + ram_ + 
               ", Storage: " + storage_ + ", GPU: " + gpu_;
    }
};

class ComputerBuilder {
private:
    Computer computer_;

public:
    ComputerBuilder& withCPU(const std::string& cpu) {
        computer_.setCPU(cpu);
        return *this;
    }
    
    ComputerBuilder& withRAM(const std::string& ram) {
        computer_.setRAM(ram);
        return *this;
    }
    
    ComputerBuilder& withStorage(const std::string& storage) {
        computer_.setStorage(storage);
        return *this;
    }
    
    ComputerBuilder& withGPU(const std::string& gpu) {
        computer_.setGPU(gpu);
        return *this;
    }
    
    Computer build() {
        return computer_;
    }
};

// Usage
Computer computer = ComputerBuilder()
    .withCPU("Intel i7")
    .withRAM("16GB")
    .withStorage("1TB SSD")
    .withGPU("RTX 3080")
    .build();
```

### Structural Patterns

#### Adapter Pattern
```python
class LegacyPrinter:
    def old_print(self, text: str):
        print(f"Legacy: {text}")

class ModernPrinter:
    def print(self, text: str):
        print(f"Modern: {text}")

class PrinterAdapter:
    def __init__(self, legacy_printer: LegacyPrinter):
        self.legacy_printer = legacy_printer
    
    def print(self, text: str):
        self.legacy_printer.old_print(text)

# Usage
legacy = LegacyPrinter()
adapter = PrinterAdapter(legacy)
adapter.print("Hello World")  # Uses legacy printer through adapter
```

#### Composite Pattern
```cpp
#include <vector>
#include <memory>

class Component {
public:
    virtual ~Component() = default;
    virtual void operation() = 0;
    virtual void add(std::shared_ptr<Component> component) {}
    virtual void remove(std::shared_ptr<Component> component) {}
};

class Leaf : public Component {
public:
    void operation() override {
        std::cout << "Leaf operation\n";
    }
};

class Composite : public Component {
private:
    std::vector<std::shared_ptr<Component>> children_;

public:
    void operation() override {
        std::cout << "Composite operation\n";
        for (auto& child : children_) {
            child->operation();
        }
    }
    
    void add(std::shared_ptr<Component> component) override {
        children_.push_back(component);
    }
    
    void remove(std::shared_ptr<Component> component) override {
        children_.erase(
            std::remove(children_.begin(), children_.end(), component),
            children_.end()
        );
    }
};
```

### Behavioral Patterns

#### Observer Pattern
```python
from typing import List, Protocol

class Observer(Protocol):
    def update(self, data: Any) -> None:
        ...

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
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()

class ConcreteObserver:
    def __init__(self, name: str):
        self.name = name
    
    def update(self, data: Any):
        print(f"{self.name} received update: {data}")
```

#### Strategy Pattern
```cpp
class SortStrategy {
public:
    virtual ~SortStrategy() = default;
    virtual void sort(std::vector<int>& data) = 0;
};

class BubbleSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        // Bubble sort implementation
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data.size() - i - 1; ++j) {
                if (data[j] > data[j + 1]) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }
};

class QuickSort : public SortStrategy {
public:
    void sort(std::vector<int>& data) override {
        quickSort(data, 0, data.size() - 1);
    }
    
private:
    void quickSort(std::vector<int>& arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    int partition(std::vector<int>& arr, int low, int high) {
        int pivot = arr[high];
        int i = (low - 1);
        
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        return (i + 1);
    }
};

class Sorter {
private:
    std::unique_ptr<SortStrategy> strategy_;

public:
    void setStrategy(std::unique_ptr<SortStrategy> strategy) {
        strategy_ = std::move(strategy);
    }
    
    void sort(std::vector<int>& data) {
        if (strategy_) {
            strategy_->sort(data);
        }
    }
};
```

#### Command Pattern
```python
from abc import ABC, abstractmethod
from typing import List

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class Light:
    def __init__(self):
        self.is_on = False
    
    def turn_on(self):
        self.is_on = True
        print("Light is ON")
    
    def turn_off(self):
        self.is_on = False
        print("Light is OFF")

class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_on()
    
    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.turn_off()
    
    def undo(self):
        self.light.turn_on()

class RemoteControl:
    def __init__(self):
        self.commands: List[Command] = []
        self.last_command: Optional[Command] = None
    
    def set_command(self, slot: int, command: Command):
        if len(self.commands) <= slot:
            self.commands.extend([None] * (slot - len(self.commands) + 1))
        self.commands[slot] = command
    
    def press_button(self, slot: int):
        if slot < len(self.commands) and self.commands[slot]:
            self.commands[slot].execute()
            self.last_command = self.commands[slot]
    
    def press_undo(self):
        if self.last_command:
            self.last_command.undo()
```

## 5. Common Interview Questions & Solutions

### C++ Questions:

#### 1. Implement a Thread-safe Queue
```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;

public:
    ThreadSafeQueue() = default;
    
    ThreadSafeQueue(const ThreadSafeQueue& other) {
        std::lock_guard<std::mutex> lock(other.mutex_);
        queue_ = other.queue_;
    }
    
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        condition_.notify_one();
    }
    
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> res = std::make_shared<T>(std::move(queue_.front()));
        queue_.pop();
        return res;
    }
    
    void wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        item = std::move(queue_.front());
        queue_.pop();
    }
    
    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        std::shared_ptr<T> res = std::make_shared<T>(std::move(queue_.front()));
        queue_.pop();
        return res;
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

#### 2. Create a Memory Pool
```cpp
#include <memory>
#include <vector>
#include <stack>

template<typename T, size_t PoolSize = 1024>
class MemoryPool {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
    };
    
    std::vector<Block> pool_;
    std::stack<T*> free_blocks_;
    
public:
    MemoryPool() : pool_(PoolSize) {
        for (auto& block : pool_) {
            free_blocks_.push(reinterpret_cast<T*>(&block));
        }
    }
    
    ~MemoryPool() {
        // All objects should be deallocated before pool destruction
    }
    
    template<typename... Args>
    T* allocate(Args&&... args) {
        if (free_blocks_.empty()) {
            throw std::bad_alloc();
        }
        
        T* ptr = free_blocks_.top();
        free_blocks_.pop();
        
        new(ptr) T(std::forward<Args>(args)...);
        return ptr;
    }
    
    void deallocate(T* ptr) {
        if (ptr) {
            ptr->~T();
            free_blocks_.push(ptr);
        }
    }
    
    size_t available() const {
        return free_blocks_.size();
    }
    
    size_t capacity() const {
        return PoolSize;
    }
};
```

#### 3. Design a Smart Pointer
```cpp
template<typename T>
class unique_ptr {
private:
    T* ptr_;
    
public:
    explicit unique_ptr(T* ptr = nullptr) : ptr_(ptr) {}
    
    ~unique_ptr() {
        delete ptr_;
    }
    
    // Move constructor
    unique_ptr(unique_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    // Move assignment
    unique_ptr& operator=(unique_ptr&& other) noexcept {
        if (this != &other) {
            delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    // Delete copy constructor and copy assignment
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;
    
    T& operator*() const {
        return *ptr_;
    }
    
    T* operator->() const {
        return ptr_;
    }
    
    T* get() const {
        return ptr_;
    }
    
    T* release() {
        T* temp = ptr_;
        ptr_ = nullptr;
        return temp;
    }
    
    void reset(T* ptr = nullptr) {
        delete ptr_;
        ptr_ = ptr;
    }
    
    explicit operator bool() const {
        return ptr_ != nullptr;
    }
};
```

#### 4. Implement Move Semantics
```cpp
class MoveableString {
private:
    char* data_;
    size_t size_;
    size_t capacity_;
    
public:
    // Constructor
    MoveableString(const char* str = "") {
        size_ = strlen(str);
        capacity_ = size_ + 1;
        data_ = new char[capacity_];
        strcpy(data_, str);
    }
    
    // Copy constructor
    MoveableString(const MoveableString& other) 
        : size_(other.size_), capacity_(other.capacity_) {
        data_ = new char[capacity_];
        strcpy(data_, other.data_);
    }
    
    // Move constructor
    MoveableString(MoveableString&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Copy assignment
    MoveableString& operator=(const MoveableString& other) {
        if (this != &other) {
            delete[] data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_ = new char[capacity_];
            strcpy(data_, other.data_);
        }
        return *this;
    }
    
    // Move assignment
    MoveableString& operator=(MoveableString&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    ~MoveableString() {
        delete[] data_;
    }
    
    const char* c_str() const { return data_; }
    size_t size() const { return size_; }
};
```

### Python Questions:

#### 1. Create a Context Manager
```python
import threading
import time
from contextlib import contextmanager

class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        self.transaction = None
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connection = f"Connection to {self.connection_string}"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Error occurred: {exc_val}")
            if self.transaction:
                print("Rolling back transaction")
        else:
            if self.transaction:
                print("Committing transaction")
        
        print("Closing connection")
        self.connection = None
        return False  # Don't suppress exceptions
    
    def begin_transaction(self):
        self.transaction = "Active transaction"
        return self
    
    def execute(self, query: str):
        if not self.connection:
            raise RuntimeError("No active connection")
        print(f"Executing: {query}")
        return f"Result of {query}"

# Alternative context manager using generator
@contextmanager
def file_manager(filename: str, mode: str = 'r'):
    print(f"Opening {filename}")
    file = open(filename, mode)
    try:
        yield file
    except Exception as e:
        print(f"Error handling file: {e}")
        raise
    finally:
        print(f"Closing {filename}")
        file.close()

# Usage
with DatabaseConnection("postgresql://localhost:5432/db") as db:
    with db.begin_transaction():
        db.execute("SELECT * FROM users")
```

#### 2. Implement a Decorator with Parameters
```python
import functools
import time
import logging
from typing import Callable, Any, Optional

def rate_limit(max_calls: int, time_window: float):
    def decorator(func: Callable) -> Callable:
        calls = []
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                # Remove calls outside the time window
                calls[:] = [call_time for call_time in calls if now - call_time < time_window]
                
                if len(calls) >= max_calls:
                    raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window} seconds")
                
                calls.append(now)
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0, 
                      exceptions: tuple = (Exception,)):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    wait_time = backoff_factor ** attempt
                    logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

def cache_with_ttl(ttl: float = 300):
    def decorator(func: Callable) -> Callable:
        cache = {}
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            with lock:
                now = time.time()
                
                # Check if cached result exists and is valid
                if key in cache:
                    result, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return result
                    else:
                        del cache[key]
                
                # Call function and cache result
                result = func(*args, **kwargs)
                cache[key] = (result, now)
                return result
        
        return wrapper
    return decorator

# Usage examples
@rate_limit(max_calls=5, time_window=60)
@retry_with_backoff(max_retries=3, backoff_factor=2.0)
@cache_with_ttl(ttl=300)
def api_call(endpoint: str) -> dict:
    # Simulate API call
    if random.random() < 0.3:  # 30% chance of failure
        raise requests.RequestException("API call failed")
    return {"data": f"Response from {endpoint}"}
```

#### 3. Build an Async Framework
```python
import asyncio
import inspect
from typing import Callable, Any, Awaitable, Dict, List
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    exception: Exception = None
    dependencies: List[str] = None
    priority: int = 0

class AsyncFramework:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
        self.semaphore = asyncio.Semaphore(max_workers)
        self.running = False
    
    async def add_task(self, task_id: str, func: Callable, *args, 
                      dependencies: List[str] = None, priority: int = 0, **kwargs):
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies or [],
            priority=priority
        )
        self.tasks[task_id] = task
        
        # Check if dependencies are met
        if await self._dependencies_met(task):
            await self.task_queue.put(task)
    
    async def _dependencies_met(self, task: Task) -> bool:
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _execute_task(self, task: Task):
        async with self.semaphore:
            try:
                task.status = TaskStatus.RUNNING
                
                if inspect.iscoroutinefunction(task.func):
                    result = await task.func(*task.args, **task.kwargs)
                else:
                    # Run synchronous function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, task.func, *task.args)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                self.results[task.id] = result
                
                # Check for dependent tasks
                await self._check_dependent_tasks(task.id)
                
            except Exception as e:
                task.exception = e
                task.status = TaskStatus.FAILED
                print(f"Task {task.id} failed: {e}")
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        for task in self.tasks.values():
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.dependencies and 
                await self._dependencies_met(task)):
                await self.task_queue.put(task)
    
    async def _worker(self):
        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self._execute_task(task)
                self.task_queue.task_done()
            except asyncio.TimeoutError:
                continue
    
    async def start(self):
        self.running = True
        workers = [asyncio.create_task(self._worker()) for _ in range(self.max_workers)]
        
        # Add initial tasks with no dependencies
        for task in self.tasks.values():
            if not task.dependencies:
                await self.task_queue.put(task)
        
        return workers
    
    async def stop(self):
        self.running = False
        await self.task_queue.join()
    
    async def wait_for_completion(self):
        while any(task.status in [TaskStatus.PENDING, TaskStatus.RUNNING] 
                 for task in self.tasks.values()):
            await asyncio.sleep(0.1)
    
    def get_result(self, task_id: str) -> Any:
        return self.results.get(task_id)

# Usage example
async def example_usage():
    framework = AsyncFramework(max_workers=5)
    
    # Add tasks with dependencies
    await framework.add_task("task1", lambda: "Result 1")
    await framework.add_task("task2", lambda: "Result 2")
    await framework.add_task("task3", lambda x, y: f"Combined: {x}, {y}", 
                           dependencies=["task1", "task2"])
    
    workers = await framework.start()
    await framework.wait_for_completion()
    await framework.stop()
    
    for worker in workers:
        worker.cancel()
    
    print(framework.get_result("task3"))
```

#### 4. Design a Metaclass
```python
import inspect
from typing import Dict, Any, List, Callable

class ValidationMeta(type):
    """Metaclass that adds automatic validation to class attributes"""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Collect validation rules
        validators = {}
        
        for key, value in namespace.items():
            if hasattr(value, '_validators'):
                validators[key] = value._validators
        
        # Add validation method
        def validate(self):
            errors = []
            for attr_name, attr_validators in validators.items():
                attr_value = getattr(self, attr_name, None)
                for validator in attr_validators:
                    try:
                        validator(attr_value)
                    except ValueError as e:
                        errors.append(f"{attr_name}: {e}")
            
            if errors:
                raise ValueError(f"Validation failed: {', '.join(errors)}")
            return True
        
        namespace['validate'] = validate
        namespace['_validators'] = validators
        
        return super().__new__(mcs, name, bases, namespace)

def validator(*validation_funcs):
    """Decorator to add validators to a method or property"""
    def decorator(func):
        func._validators = validation_funcs
        return func
    return decorator

def min_length(min_len: int):
    def validate(value):
        if value is None or len(str(value)) < min_len:
            raise ValueError(f"Must be at least {min_len} characters long")
    return validate

def max_value(max_val: int):
    def validate(value):
        if value is not None and value > max_val:
            raise ValueError(f"Must be less than or equal to {max_val}")
    return validate

def required():
    def validate(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("This field is required")
    return validate

class SingletonMeta(type):
    """Metaclass that creates singleton instances"""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class ORM_Meta(type):
    """Simplified ORM metaclass"""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Skip for base Model class
        if name == 'Model':
            return super().__new__(mcs, name, bases, namespace)
        
        # Collect field definitions
        fields = {}
        for key, value in list(namespace.items()):
            if isinstance(value, Field):
                fields[key] = value
                value.name = key
                # Remove from namespace to avoid conflicts
                del namespace[key]
        
        namespace['_fields'] = fields
        namespace['_table_name'] = kwargs.get('table_name', name.lower())
        
        # Add ORM methods
        def save(self):
            # Simplified save logic
            field_names = list(self._fields.keys())
            values = [getattr(self, name, None) for name in field_names]
            print(f"INSERT INTO {self._table_name} ({', '.join(field_names)}) "
                  f"VALUES ({', '.join(repr(v) for v in values)})")
        
        @classmethod
        def find_by_id(cls, id_value):
            print(f"SELECT * FROM {cls._table_name} WHERE id = {id_value}")
            return cls()
        
        namespace['save'] = save
        namespace['find_by_id'] = find_by_id
        
        return super().__new__(mcs, name, bases, namespace)

class Field:
    def __init__(self, field_type: type, required: bool = False, default: Any = None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.name = None

# Usage examples
class User(metaclass=ValidationMeta):
    def __init__(self, name: str, email: str, age: int):
        self.name = name
        self.email = email
        self.age = age
    
    @validator(required(), min_length(2))
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
    
    @validator(required(), min_length(5))
    def email(self):
        return self._email
    
    @email.setter
    def email(self, value):
        self._email = value
    
    @validator(required(), max_value(150))
    def age(self):
        return self._age
    
    @age.setter
    def age(self, value):
        self._age = value

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "database_connection"

class Model(metaclass=ORM_Meta):
    pass

class Product(Model, table_name='products'):
    id = Field(int, required=True)
    name = Field(str, required=True)
    price = Field(float, required=True)
    description = Field(str, default="")
```

### System Design Questions:

#### 1. Design a Rate Limiter
```python
import time
import threading
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from typing import Dict, Optional

class RateLimiter(ABC):
    @abstractmethod
    def is_allowed(self, key: str) -> bool:
        pass

class TokenBucketRateLimiter(RateLimiter):
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, dict] = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        with self.lock:
            now = time.time()
            
            if key not in self.buckets:
                self.buckets[key] = {
                    'tokens': self.capacity,
                    'last_refill': now
                }
            
            bucket = self.buckets[key]
            
            # Refill tokens
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * self.refill_rate
            bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            # Check if request can be served
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            
            return False

class SlidingWindowRateLimiter(RateLimiter):
    def __init__(self, max_requests: int, window_size: int):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        with self.lock:
            now = time.time()
            window_start = now - self.window_size
            
            # Remove old requests
            while self.requests[key] and self.requests[key][0] < window_start:
                self.requests[key].popleft()
            
            # Check if under limit
            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(now)
                return True
            
            return False

class DistributedRateLimiter(RateLimiter):
    def __init__(self, redis_client, max_requests: int, window_size: int):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_size = window_size
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        pipe = self.redis.pipeline()
        
        # Sliding window log implementation
        pipe.zremrangebyscore(key, 0, now - self.window_size)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self.window_size + 1)
        
        results = pipe.execute()
        request_count = results[1]
        
        return request_count < self.max_requests
```

#### 2. Create a Distributed Cache
Already implemented in the System Design section above.

#### 3. Implement Service Discovery
```python
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceInstance:
    id: str
    name: str
    host: str
    port: int
    metadata: Dict[str, str] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_heartbeat: float = field(default_factory=time.time)
    health_check_url: Optional[str] = None

class ServiceRegistry:
    def __init__(self, heartbeat_interval: float = 30.0, health_check_timeout: float = 60.0):
        self.services: Dict[str, ServiceInstance] = {}
        self.service_by_name: Dict[str, List[str]] = {}
        self.heartbeat_interval = heartbeat_interval
        self.health_check_timeout = health_check_timeout
        self.lock = threading.RLock()
        self.subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        self._monitor_thread = None
    
    def register_service(self, service: ServiceInstance) -> bool:
        with self.lock:
            self.services[service.id] = service
            
            if service.name not in self.service_by_name:
                self.service_by_name[service.name] = []
            
            if service.id not in self.service_by_name[service.name]:
                self.service_by_name[service.name].append(service.id)
            
            service.last_heartbeat = time.time()
            service.status = ServiceStatus.HEALTHY
            
            self._notify_subscribers(service.name, 'registered', service)
            return True
    
    def deregister_service(self, service_id: str) -> bool:
        with self.lock:
            if service_id in self.services:
                service = self.services[service_id]
                del self.services[service_id]
                
                if service.name in self.service_by_name:
                    if service_id in self.service_by_name[service.name]:
                        self.service_by_name[service.name].remove(service_id)
                    
                    if not self.service_by_name[service.name]:
                        del self.service_by_name[service.name]
                
                self._notify_subscribers(service.name, 'deregistered', service)
                return True
            return False
    
    def heartbeat(self, service_id: str) -> bool:
        with self.lock:
            if service_id in self.services:
                service = self.services[service_id]
                service.last_heartbeat = time.time()
                
                if service.status != ServiceStatus.HEALTHY:
                    service.status = ServiceStatus.HEALTHY
                    self._notify_subscribers(service.name, 'health_changed', service)
                
                return True
            return False
    
    def discover_services(self, service_name: str) -> List[ServiceInstance]:
        with self.lock:
            if service_name in self.service_by_name:
                service_ids = self.service_by_name[service_name]
                return [self.services[sid] for sid in service_ids 
                       if self.services[sid].status == ServiceStatus.HEALTHY]
            return []
    
    def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        with self.lock:
            return self.services.get(service_id)
    
    def subscribe(self, service_name: str, callback: Callable):
        with self.lock:
            if service_name not in self.subscribers:
                self.subscribers[service_name] = []
            self.subscribers[service_name].append(callback)
    
    def _notify_subscribers(self, service_name: str, event: str, service: ServiceInstance):
        if service_name in self.subscribers:
            for callback in self.subscribers[service_name]:
                try:
                    callback(event, service)
                except Exception as e:
                    print(f"Error notifying subscriber: {e}")
    
    def start_monitoring(self):
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_services)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_services(self):
        while self._running:
            current_time = time.time()
            services_to_check = []
            
            with self.lock:
                for service in self.services.values():
                    if (current_time - service.last_heartbeat > self.health_check_timeout and
                        service.status == ServiceStatus.HEALTHY):
                        services_to_check.append(service)
            
            # Mark unhealthy services
            for service in services_to_check:
                with self.lock:
                    if service.id in self.services:  # Double-check service still exists
                        service.status = ServiceStatus.UNHEALTHY
                        self._notify_subscribers(service.name, 'health_changed', service)
            
            time.sleep(self.heartbeat_interval)

class LoadBalancedServiceClient:
    def __init__(self, registry: ServiceRegistry, service_name: str):
        self.registry = registry
        self.service_name = service_name
        self.current_index = 0
    
    def call_service(self, method: str, *args, **kwargs):
        services = self.registry.discover_services(self.service_name)
        if not services:
            raise Exception(f"No healthy instances of {self.service_name} available")
        
        # Simple round-robin load balancing
        service = services[self.current_index % len(services)]
        self.current_index = (self.current_index + 1) % len(services)
        
        # Simulate service call
        return self._make_request(service, method, *args, **kwargs)
    
    def _make_request(self, service: ServiceInstance, method: str, *args, **kwargs):
        # This would contain actual HTTP/gRPC/etc. call logic
        print(f"Calling {method} on {service.host}:{service.port}")
        return {"status": "success", "service": service.id}
```

#### 4. Design a Message Queue
Already implemented in the System Design section above.

## 6. Best Practices & Tips

### Code Quality

#### C++ Best Practices
- **RAII**: Always use Resource Acquisition Is Initialization
- **Const Correctness**: Make everything const that can be const
- **Smart Pointers**: Prefer smart pointers over raw pointers
- **Move Semantics**: Implement move constructors and assignment operators
- **Exception Safety**: Write exception-safe code with strong guarantee
- **Template Best Practices**: Use SFINAE and concepts for better error messages

```cpp
// Good: RAII and const correctness
class FileProcessor {
private:
    std::unique_ptr<FILE, decltype(&fclose)> file_;
    
public:
    explicit FileProcessor(const std::string& filename) 
        : file_(fopen(filename.c_str(), "r"), &fclose) {
        if (!file_) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    std::string readLine() const {
        // Implementation...
        return "";
    }
};

// Good: Exception safety
template<typename T>
void safe_push_back(std::vector<T>& vec, const T& item) {
    vec.reserve(vec.size() + 1);  // Strong exception guarantee
    vec.push_back(item);
}
```

#### Python Best Practices
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations for better code documentation
- **Context Managers**: Use `with` statements for resource management
- **List Comprehensions**: Prefer comprehensions over loops when appropriate
- **Generator Functions**: Use generators for memory-efficient iteration
- **Error Handling**: Use specific exception types

```python
# Good: Type hints and context managers
from typing import List, Optional, Iterator
from pathlib import Path

def process_files(file_paths: List[Path]) -> Iterator[str]:
    """Process files and yield results."""
    for path in file_paths:
        try:
            with path.open('r', encoding='utf-8') as file:
                yield file.read().strip()
        except IOError as e:
            logging.error(f"Failed to process {path}: {e}")
            continue

# Good: Use of dataclasses and enums
from dataclasses import dataclass
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    status: Status
    created_at: datetime
    data: Optional[dict] = None
```

### Testing Strategies

#### Unit Testing Best Practices
- **Test Isolation**: Each test should be independent
- **AAA Pattern**: Arrange, Act, Assert
- **Test Naming**: Use descriptive test names
- **Mock External Dependencies**: Isolate units under test
- **Edge Cases**: Test boundary conditions and error scenarios

```cpp
// C++ unit test example with Google Test
TEST(ThreadSafeQueueTest, PushAndPopSingleItem) {
    // Arrange
    ThreadSafeQueue<int> queue;
    int test_value = 42;
    
    // Act
    queue.push(test_value);
    int result;
    bool success = queue.try_pop(result);
    
    // Assert
    EXPECT_TRUE(success);
    EXPECT_EQ(result, test_value);
    EXPECT_TRUE(queue.empty());
}

TEST(ThreadSafeQueueTest, ConcurrentPushPop) {
    ThreadSafeQueue<int> queue;
    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};
    
    auto pusher = [&]() {
        for (int i = 0; i < 1000; ++i) {
            queue.push(i);
            ++push_count;
        }
    };
    
    auto popper = [&]() {
        int value;
        while (pop_count < 1000) {
            if (queue.try_pop(value)) {
                ++pop_count;
            }
            std::this_thread::yield();
        }
    };
    
    std::thread t1(pusher);
    std::thread t2(popper);
    
    t1.join();
    t2.join();
    
    EXPECT_EQ(push_count, 1000);
    EXPECT_EQ(pop_count, 1000);
}
```

```python
# Python unit test example with pytest
import pytest
from unittest.mock import Mock, patch
import asyncio

class TestAsyncFramework:
    @pytest.fixture
    def framework(self):
        return AsyncFramework(max_workers=2)
    
    @pytest.mark.asyncio
    async def test_simple_task_execution(self, framework):
        # Arrange
        result_value = "test_result"
        test_func = Mock(return_value=result_value)
        
        # Act
        await framework.add_task("test_task", test_func)
        workers = await framework.start()
        await framework.wait_for_completion()
        await framework.stop()
        
        # Assert
        assert framework.get_result("test_task") == result_value
        test_func.assert_called_once()
        
        # Cleanup
        for worker in workers:
            worker.cancel()
    
    @pytest.mark.asyncio
    async def test_task_dependencies(self, framework):
        # Arrange
        results = []
        
        def task1():
            results.append("task1")
            return "result1"
        
        def task2():
            results.append("task2")
            return "result2"
        
        def task3(r1, r2):
            results.append("task3")
            return f"combined: {r1}, {r2}"
        
        # Act
        await framework.add_task("task1", task1)
        await framework.add_task("task2", task2)
        await framework.add_task("task3", task3, dependencies=["task1", "task2"])
        
        workers = await framework.start()
        await framework.wait_for_completion()
        await framework.stop()
        
        # Assert
        assert "task1" in results
        assert "task2" in results
        assert "task3" in results
        assert results.index("task3") > max(results.index("task1"), results.index("task2"))
        
        # Cleanup
        for worker in workers:
            worker.cancel()

# Integration testing
@pytest.mark.integration
class TestServiceIntegration:
    @pytest.fixture
    def service_registry(self):
        registry = ServiceRegistry()
        registry.start_monitoring()
        yield registry
        registry.stop_monitoring()
    
    def test_service_lifecycle(self, service_registry):
        # Test service registration, discovery, and deregistration
        service = ServiceInstance(
            id="test-service-1",
            name="user-service",
            host="localhost",
            port=8080
        )
        
        # Register service
        assert service_registry.register_service(service)
        
        # Discover service
        services = service_registry.discover_services("user-service")
        assert len(services) == 1
        assert services[0].id == "test-service-1"
        
        # Heartbeat
        assert service_registry.heartbeat("test-service-1")
        
        # Deregister service
        assert service_registry.deregister_service("test-service-1")
        
        # Verify service is gone
        services = service_registry.discover_services("user-service")
        assert len(services) == 0
```

### Performance Optimization

#### C++ Performance Tips
- **Memory Locality**: Optimize data structures for cache efficiency
- **Algorithm Complexity**: Choose appropriate algorithms and data structures
- **Compiler Optimizations**: Use appropriate compiler flags
- **Profile-Guided Optimization**: Measure before optimizing
- **Avoid Premature Optimization**: Profile first, optimize later

```cpp
// Good: Cache-friendly data structure
struct Point3D {
    float x, y, z;  // Better cache locality than separate arrays
};

std::vector<Point3D> points;  // Better than std::vector<float> x, y, z;

// Good: Algorithm selection
void processLargeDataset(std::vector<int>& data) {
    // For large datasets, prefer algorithms with better complexity
    std::sort(data.begin(), data.end());  // O(n log n)
    // Instead of selection sort O(n²)
}

// Good: Move semantics to avoid copies
std::vector<std::string> createLargeVector() {
    std::vector<std::string> result;
    result.reserve(1000000);
    
    for (int i = 0; i < 1000000; ++i) {
        result.emplace_back("String " + std::to_string(i));  // Construct in place
    }
    
    return result;  // Move semantics will avoid copy
}
```

#### Python Performance Tips
- **Use Built-in Functions**: They're usually implemented in C
- **List Comprehensions**: Faster than equivalent loops
- **Generator Expressions**: Memory-efficient for large datasets
- **NumPy**: Use for numerical computations
- **Cython/PyPy**: For CPU-intensive code
- **Async/Await**: For I/O-bound operations

```python
# Good: Using built-ins and comprehensions
import numpy as np
from collections import defaultdict, Counter

# Efficient data processing
def process_numbers(numbers: List[int]) -> dict:
    # Use built-in functions
    return {
        'sum': sum(numbers),
        'max': max(numbers),
        'min': min(numbers),
        'count': len(numbers),
        'squared': [x*x for x in numbers],  # List comprehension
        'even_count': sum(1 for x in numbers if x % 2 == 0)  # Generator
    }

# Use NumPy for numerical operations
def calculate_statistics(data: np.ndarray) -> dict:
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'percentiles': np.percentile(data, [25, 50, 75]),
        'correlation': np.corrcoef(data)
    }

# Efficient string operations
def process_text(text: str) -> dict:
    words = text.lower().split()
    return {
        'word_count': Counter(words),
        'unique_words': len(set(words)),
        'avg_word_length': sum(len(word) for word in words) / len(words)
    }
```

### Security Considerations

#### Common Security Issues
- **Buffer Overflows**: Always validate input sizes
- **SQL Injection**: Use parameterized queries
- **Cross-Site Scripting (XSS)**: Sanitize user input
- **Authentication/Authorization**: Implement proper access controls
- **Data Encryption**: Encrypt sensitive data at rest and in transit
- **Input Validation**: Never trust user input

```cpp
// Good: Safe string handling
#include <string>
#include <algorithm>

class SafeString {
private:
    std::string data_;
    
public:
    SafeString(const char* input, size_t max_length = 1024) {
        if (input && strlen(input) <= max_length) {
            data_ = input;
        } else {
            throw std::invalid_argument("Invalid input string");
        }
    }
    
    const std::string& get() const { return data_; }
};

// Good: Secure random number generation
#include <random>
#include <cryptopp/cryptlib.h>

class SecureRandom {
private:
    std::random_device rd_;
    std::mt19937 gen_;
    
public:
    SecureRandom() : gen_(rd_()) {}
    
    int generateInt(int min, int max) {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen_);
    }
};
```

```python
# Good: Secure Python practices
import hashlib
import secrets
import bcrypt
from cryptography.fernet import Fernet
import sqlalchemy

class SecureUserManager:
    def __init__(self, db_connection):
        self.db = db_connection
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
    
    def hash_password(self, password: str) -> str:
        """Securely hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, username: str, password: str, email: str):
        """Create user with proper validation and SQL injection protection."""
        # Input validation
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        
        if not self._is_valid_email(email):
            raise ValueError("Invalid email format")
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Use parameterized query to prevent SQL injection
        query = """
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        """
        self.db.execute(query, (username, password_hash, email))
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    def _is_valid_email(self, email: str) -> bool:
        """Simple email validation."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
```

## 7. Interview Process Tips

### Technical Discussion Strategies

#### Approach to Problem Solving
1. **Clarify Requirements**: Ask questions to understand the problem fully
2. **Start Simple**: Begin with a basic solution, then optimize
3. **Think Out Loud**: Explain your thought process
4. **Consider Edge Cases**: Discuss boundary conditions and error scenarios
5. **Analyze Complexity**: Discuss time and space complexity
6. **Discuss Trade-offs**: Explain different approaches and their pros/cons

#### Communication Best Practices
- **Be Structured**: Follow a logical progression in explanations
- **Use Examples**: Provide concrete examples to illustrate concepts
- **Ask for Feedback**: Check if the interviewer wants more detail
- **Admit Uncertainties**: It's okay to say "I'm not sure, but I think..."
- **Stay Calm**: Don't panic if you don't know something immediately

### Code Review Best Practices

#### What to Look For
- **Correctness**: Does the code do what it's supposed to do?
- **Readability**: Is the code easy to understand?
- **Performance**: Are there obvious performance issues?
- **Security**: Are there potential security vulnerabilities?
- **Maintainability**: Is the code easy to modify and extend?
- **Testing**: Is the code testable and well-tested?

#### How to Give Feedback
```python
# Example code review comments

# Good feedback: Specific and constructive
"""
Consider using a list comprehension here for better performance:
result = [process_item(item) for item in items if item.is_valid()]

Instead of:
result = []
for item in items:
    if item.is_valid():
        result.append(process_item(item))
"""

# Good feedback: Suggests improvement with reasoning
"""
This function is doing too many things. Consider splitting it into:
1. validate_input() - handles input validation
2. process_data() - handles the core logic
3. format_output() - handles result formatting

This would improve testability and maintainability.
"""

# Good feedback: Points out potential issues
"""
This code might have a race condition if accessed by multiple threads.
Consider adding synchronization or using thread-safe data structures.
"""
```

### System Design Approach

#### Step-by-Step Process
1. **Requirements Gathering** (5-10 minutes)
   - Functional requirements: What should the system do?
   - Non-functional requirements: Scale, performance, availability
   - Constraints: Budget, timeline, technology limitations

2. **Capacity Estimation** (5-10 minutes)
   - Users: How many active users?
   - Data: How much data will be stored?
   - Traffic: Requests per second, read/write ratio
   - Storage: Database size, cache requirements

3. **System Interface Definition** (5-10 minutes)
   - API design: REST endpoints, parameters, responses
   - Data models: Key entities and relationships

4. **High-Level Design** (10-15 minutes)
   - Draw major components and their interactions
   - Show data flow through the system
   - Identify key services and databases

5. **Detailed Design** (10-15 minutes)
   - Dive deeper into critical components
   - Discuss algorithms and data structures
   - Address scalability and reliability concerns

6. **Scale the Design** (10-15 minutes)
   - Identify bottlenecks
   - Discuss scaling strategies
   - Add caching, load balancing, etc.

#### Common System Design Patterns
```python
# Microservices Architecture Example
class SystemDesign:
    """
    Design a URL Shortener like bit.ly
    
    Requirements:
    - Shorten long URLs
    - Redirect short URLs to original
    - Custom aliases (optional)
    - Analytics
    - 100M URLs per month
    - 10:1 read/write ratio
    """
    
    def design_components(self):
        return {
            'api_gateway': 'Route requests, rate limiting, authentication',
            'url_service': 'Shorten URLs, generate short codes',
            'redirect_service': 'Handle redirects, update analytics',
            'analytics_service': 'Track clicks, generate reports',
            'database': 'Store URL mappings (SQL for consistency)',
            'cache': 'Cache popular URLs (Redis)',
            'load_balancer': 'Distribute traffic across services'
        }
    
    def capacity_estimation(self):
        return {
            'new_urls_per_month': 100_000_000,
            'new_urls_per_second': 100_000_000 / (30 * 24 * 3600),  # ~40 URLs/sec
            'read_requests_per_second': 40 * 10,  # 400 reads/sec
            'storage_per_year': '100M URLs * 500 bytes = 50GB',
            'cache_memory': '20% of daily requests = 8GB'
        }
    
    def api_design(self):
        return {
            'create_short_url': {
                'method': 'POST',
                'endpoint': '/api/v1/shorten',
                'request': {'long_url': 'string', 'custom_alias': 'string?'},
                'response': {'short_url': 'string', 'expiration': 'datetime'}
            },
            'redirect': {
                'method': 'GET',
                'endpoint': '/{short_code}',
                'response': '302 redirect to original URL'
            },
            'analytics': {
                'method': 'GET',
                'endpoint': '/api/v1/analytics/{short_code}',
                'response': {'clicks': 'number', 'locations': 'array'}
            }
        }
```

### Problem-Solving Framework

#### The STAR Method
When discussing past experiences:
- **Situation**: Describe the context
- **Task**: Explain what needed to be done
- **Action**: Detail what you did
- **Result**: Share the outcome

#### Debugging Approach
1. **Reproduce the Issue**: Can you make it happen consistently?
2. **Isolate the Problem**: Narrow down to the smallest failing case
3. **Gather Information**: Logs, stack traces, system metrics
4. **Form Hypotheses**: What could be causing this?
5. **Test Hypotheses**: Systematically test each possibility
6. **Fix and Verify**: Implement fix and confirm it works
7. **Prevent Recurrence**: Add tests, monitoring, documentation

#### Algorithm Problem-Solving Steps
1. **Understand the Problem**: Read carefully, ask clarifying questions
2. **Work Through Examples**: Manually solve small cases
3. **Identify Patterns**: Look for known algorithm patterns
4. **Choose Data Structures**: Pick appropriate data structures
5. **Write Pseudocode**: Outline the algorithm before coding
6. **Implement**: Write clean, readable code
7. **Test**: Walk through with examples, consider edge cases
8. **Optimize**: Improve time/space complexity if needed

```python
# Example: Problem-solving demonstration
def find_two_sum(nums: List[int], target: int) -> List[int]:
    """
    Given an array of integers and a target, return indices of two numbers
    that add up to target.
    
    Approach:
    1. Brute force: O(n²) - check all pairs
    2. Hash map: O(n) - store complements
    
    I'll use the hash map approach for better performance.
    """
    seen = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []  # No solution found

# Test with examples
assert find_two_sum([2, 7, 11, 15], 9) == [0, 1]  # 2 + 7 = 9
assert find_two_sum([3, 2, 4], 6) == [1, 2]       # 2 + 4 = 6
```

#### General Interview Tips
- **Prepare Stories**: Have concrete examples of your work ready
- **Practice Coding**: Use platforms like LeetCode, HackerRank
- **Mock Interviews**: Practice with peers or online platforms
- **Know Your Resume**: Be able to discuss everything you've listed
- **Research the Company**: Understand their tech stack and challenges
- **Prepare Questions**: Ask thoughtful questions about the role and company
- **Stay Updated**: Keep up with industry trends and new technologies
- **Be Honest**: Don't pretend to know things you don't
- **Show Enthusiasm**: Demonstrate genuine interest in the role
- **Follow Up**: Send a thank-you note after the interview
