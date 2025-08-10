# Python Interview Guide

## Table of Contents
1. [GIL and Concurrency](#gil-and-concurrency)
2. [Decorators and Context Managers](#decorators-and-context-managers)
3. [Async Programming](#async-programming)
4. [Memory Management and Garbage Collection](#memory-management-and-garbage-collection)
5. [Python Design Patterns](#python-design-patterns)

---

## GIL and Concurrency

### Understanding the Global Interpreter Lock (GIL)

**What is the GIL?**
The Global Interpreter Lock is a mutex that protects access to Python objects, preventing multiple native threads from executing Python bytecodes simultaneously.

**Key Implications**:
- CPU-bound tasks don't benefit from threading
- I/O-bound tasks can still benefit from threading
- Only one thread can execute Python code at a time

**Code Example - GIL Impact**:
```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound_task(n):
    """CPU-intensive task that won't benefit from threading due to GIL"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

def io_bound_task(duration):
    """I/O-bound task that benefits from threading"""
    time.sleep(duration)
    return f"Task completed after {duration}s"

# Demonstrating GIL impact
def compare_threading_vs_processing():
    import time
    
    # CPU-bound task comparison
    start = time.time()
    # Sequential execution
    results = [cpu_bound_task(1000000) for _ in range(4)]
    sequential_time = time.time() - start
    
    # Threading (won't help due to GIL)
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_bound_task, [1000000] * 4))
    threading_time = time.time() - start
    
    # Multiprocessing (bypasses GIL)
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_bound_task, [1000000] * 4))
    multiprocessing_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Threading: {threading_time:.2f}s")
    print(f"Multiprocessing: {multiprocessing_time:.2f}s")
```

### Threading Strategies

**Thread-Safe Data Structures**:
```python
import threading
import queue
from collections import deque
import time

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def get_value(self):
        with self._lock:
            return self._value

class ProducerConsumer:
    def __init__(self, maxsize=10):
        self.queue = queue.Queue(maxsize=maxsize)
        self.finished = threading.Event()
    
    def producer(self, items):
        for item in items:
            self.queue.put(item)
            time.sleep(0.1)  # Simulate work
        self.finished.set()
    
    def consumer(self):
        while not self.finished.is_set() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1)
                print(f"Consumed: {item}")
                self.queue.task_done()
            except queue.Empty:
                continue

# Advanced threading with condition variables
class BoundedBuffer:
    def __init__(self, capacity):
        self.buffer = deque()
        self.capacity = capacity
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
    
    def put(self, item):
        with self.not_full:
            while len(self.buffer) >= self.capacity:
                self.not_full.wait()
            self.buffer.append(item)
            self.not_empty.notify()
    
    def get(self):
        with self.not_empty:
            while len(self.buffer) == 0:
                self.not_empty.wait()
            item = self.buffer.popleft()
            self.not_full.notify()
            return item
```

### Multiprocessing and IPC

**Process Communication**:
```python
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

class SharedCounter:
    def __init__(self):
        self.manager = mp.Manager()
        self.counter = self.manager.Value('i', 0)
        self.lock = self.manager.Lock()
    
    def increment(self):
        with self.lock:
            self.counter.value += 1
    
    def get_value(self):
        return self.counter.value

# Shared memory example
def create_shared_array(size):
    # Create shared memory for numpy array
    shm = shared_memory.SharedMemory(create=True, size=size * 8)  # 8 bytes per float64
    array = np.ndarray((size,), dtype=np.float64, buffer=shm.buf)
    return shm, array

def worker_function(shm_name, shape, dtype):
    # Access existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    # Modify the shared array
    array[:] = np.random.random(shape)
    shm.close()

# Pool-based parallel processing
def parallel_map_example():
    def square(x):
        return x ** 2
    
    with mp.Pool(processes=4) as pool:
        numbers = range(1000000)
        result = pool.map(square, numbers)
    return result
```

**Interview Questions**:
1. **Q**: Why doesn't threading improve CPU-bound tasks in Python?
   **A**: Due to the GIL, only one thread can execute Python bytecode at a time, making threading ineffective for CPU-intensive tasks.

2. **Q**: When would you use multiprocessing vs threading?
   **A**: Multiprocessing for CPU-bound tasks (bypasses GIL), threading for I/O-bound tasks (simpler, less overhead).

3. **Q**: How do you share data between processes safely?
   **A**: Use multiprocessing.Manager(), shared memory, queues, or pipes with proper synchronization.

---

## Decorators and Context Managers

### Decorator Patterns

**Basic Decorators**:
```python
import functools
import time
from typing import Callable, Any

# Simple function decorator
def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Parameterized decorator
def retry(max_attempts: int = 3, delay: float = 1.0):
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
                    time.sleep(delay)
        return wrapper
    return decorator

# Class-based decorator
class CallCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} called {self.count} times")
        return self.func(*args, **kwargs)

# Method decorator
def property_validator(validation_func):
    def decorator(setter_func):
        @functools.wraps(setter_func)
        def wrapper(self, value):
            if not validation_func(value):
                raise ValueError(f"Invalid value: {value}")
            return setter_func(self, value)
        return wrapper
    return decorator

class Person:
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    @property_validator(lambda x: isinstance(x, str) and len(x) > 0)
    def name(self, value):
        self._name = value
```

**Advanced Decorator Patterns**:
```python
import asyncio
from typing import TypeVar, Callable, Union
import weakref

T = TypeVar('T')

# Caching decorator with TTL
class TTLCache:
    def __init__(self, ttl: float = 300):  # 5 minutes default
        self.ttl = ttl
        self.cache = {}
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            if key in self.cache:
                value, timestamp = self.cache[key]
                if now - timestamp < self.ttl:
                    return value
            
            result = func(*args, **kwargs)
            self.cache[key] = (result, now)
            return result
        return wrapper

# Async decorator
def async_retry(max_attempts: int = 3):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return wrapper
    return decorator

# Singleton decorator using weak references
def singleton(cls):
    instances = weakref.WeakValueDictionary()
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        key = (cls, args, tuple(sorted(kwargs.items())))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]
    return wrapper
```

### Context Managers

**Basic Context Managers**:
```python
from contextlib import contextmanager, ExitStack
import tempfile
import os

class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        # Simulate connection
        self.connection = f"Connected to {self.connection_string}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
            # Return False to propagate exception
        self.connection = None
        return False

# Function-based context manager
@contextmanager
def file_manager(filename, mode='r'):
    f = None
    try:
        f = open(filename, mode)
        yield f
    except Exception as e:
        print(f"Error handling file: {e}")
        raise
    finally:
        if f:
            f.close()

# Advanced context manager with resource cleanup
class ResourcePool:
    def __init__(self, size: int):
        self.size = size
        self.pool = [f"Resource_{i}" for i in range(size)]
        self.used = set()
        self.lock = threading.Lock()
    
    @contextmanager
    def acquire_resource(self):
        resource = None
        try:
            with self.lock:
                if not self.pool:
                    raise RuntimeError("No resources available")
                resource = self.pool.pop()
                self.used.add(resource)
            yield resource
        finally:
            if resource:
                with self.lock:
                    self.used.discard(resource)
                    self.pool.append(resource)

# Multiple context managers
@contextmanager
def multi_file_manager(*filenames):
    files = []
    try:
        for filename in filenames:
            files.append(open(filename, 'r'))
        yield files
    finally:
        for f in files:
            f.close()

# Using ExitStack for dynamic context management
def process_multiple_files(filenames):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fname, 'r')) for fname in filenames]
        # Process all files
        for f in files:
            print(f"Processing {f.name}")
```

**Async Context Managers**:
```python
import aiofiles
import aiohttp

class AsyncDatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    async def __aenter__(self):
        print(f"Async connecting to {self.connection_string}")
        # Simulate async connection
        await asyncio.sleep(0.1)
        self.connection = f"Async connected to {self.connection_string}"
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Async closing database connection")
        await asyncio.sleep(0.1)  # Simulate cleanup
        self.connection = None

# Async file processing
async def process_large_file(filename):
    async with aiofiles.open(filename, 'r') as f:
        async for line in f:
            # Process line asynchronously
            await asyncio.sleep(0.001)  # Simulate processing
            yield line.strip()

# HTTP session management
async def fetch_data(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = fetch_url(session, url)
            tasks.append(task)
        return await asyncio.gather(*tasks)

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()
```

**Interview Questions**:
1. **Q**: What's the difference between `__enter__`/`__exit__` and `@contextmanager`?
   **A**: `__enter__`/`__exit__` are class-based context managers, while `@contextmanager` creates function-based context managers using generators.

2. **Q**: How do decorators preserve function metadata?
   **A**: Using `functools.wraps()` which copies `__name__`, `__doc__`, `__module__`, etc. from the original function.

3. **Q**: When would you use a class-based decorator vs function-based?
   **A**: Class-based decorators when you need to maintain state between calls, function-based for simpler transformations.

---

## Async Programming

### Asyncio Fundamentals

**Core Concepts**:
```python
import asyncio
import aiohttp
import time
from typing import List, Optional, Any

# Basic async function
async def fetch_data(url: str, delay: float = 1.0) -> str:
    await asyncio.sleep(delay)  # Simulate I/O
    return f"Data from {url}"

# Running async functions
async def main():
    # Sequential execution
    start = time.time()
    result1 = await fetch_data("http://api1.com")
    result2 = await fetch_data("http://api2.com")
    sequential_time = time.time() - start
    
    # Concurrent execution
    start = time.time()
    results = await asyncio.gather(
        fetch_data("http://api1.com"),
        fetch_data("http://api2.com"),
        fetch_data("http://api3.com")
    )
    concurrent_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")

# Error handling in async code
async def safe_fetch(url: str) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                return await response.text()
    except asyncio.TimeoutError:
        print(f"Timeout fetching {url}")
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None
```

### Advanced Async Patterns

**Producer-Consumer with Asyncio**:
```python
import asyncio
from asyncio import Queue
import random

class AsyncProducerConsumer:
    def __init__(self, queue_size: int = 10):
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = asyncio.Event()
    
    async def producer(self, name: str, items: List[Any]):
        for item in items:
            await self.queue.put(item)
            print(f"Producer {name} produced: {item}")
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
    async def consumer(self, name: str):
        while not self.stop_event.is_set():
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                print(f"Consumer {name} consumed: {item}")
                await asyncio.sleep(random.uniform(0.2, 0.8))
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
    
    async def run(self):
        # Start consumers
        consumers = [
            asyncio.create_task(self.consumer(f"Consumer-{i}"))
            for i in range(2)
        ]
        
        # Start producers
        producers = [
            asyncio.create_task(self.producer("Producer-1", list(range(10)))),
            asyncio.create_task(self.producer("Producer-2", list(range(10, 20))))
        ]
        
        # Wait for all items to be produced
        await asyncio.gather(*producers)
        
        # Wait for queue to be empty
        await self.queue.join()
        
        # Stop consumers
        self.stop_event.set()
        await asyncio.gather(*consumers, return_exceptions=True)

# Async rate limiting
class RateLimiter:
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                await asyncio.sleep(sleep_time)
                return await self.acquire()
            
            self.calls.append(now)

# Async context manager for rate limiting
class RateLimitedSession:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.session = None
    
    async def __aenter__(self):
        await self.rate_limiter.acquire()
        self.session = aiohttp.ClientSession()
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
```

### Async Iterators and Generators

**Async Iterators**:
```python
class AsyncFileReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.file is None:
            self.file = await aiofiles.open(self.filename, 'r')
        
        line = await self.file.readline()
        if line:
            return line.strip()
        else:
            await self.file.close()
            raise StopAsyncIteration

# Async generator
async def async_range(start: int, stop: int, delay: float = 0.1):
    for i in range(start, stop):
        await asyncio.sleep(delay)
        yield i

async def fibonacci_async(n: int):
    a, b = 0, 1
    for _ in range(n):
        yield a
        await asyncio.sleep(0.01)  # Yield control
        a, b = b, a + b

# Usage
async def use_async_iterators():
    # Async file reading
    async for line in AsyncFileReader("large_file.txt"):
        print(f"Processing line: {line}")
    
    # Async range
    async for num in async_range(0, 10):
        print(f"Number: {num}")
    
    # Async fibonacci
    fib_sequence = []
    async for fib in fibonacci_async(10):
        fib_sequence.append(fib)
    print(f"Fibonacci: {fib_sequence}")
```

### Event Loop Management

**Custom Event Loop Handling**:
```python
import asyncio
import concurrent.futures
import threading

class AsyncManager:
    def __init__(self):
        self.loop = None
        self.thread = None
    
    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.start()
    
    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join()
    
    def run_coroutine(self, coro):
        if not self.loop:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

# Mixing sync and async code
def sync_function_with_async():
    async def async_work():
        await asyncio.sleep(1)
        return "Async result"
    
    # Option 1: Run in existing loop
    if asyncio._get_running_loop():
        task = asyncio.create_task(async_work())
        return asyncio.run(async_work())
    
    # Option 2: Create new loop
    return asyncio.run(async_work())

# Async task management
class TaskManager:
    def __init__(self):
        self.tasks = set()
    
    def create_task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task
    
    async def wait_all(self):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
    
    def cancel_all(self):
        for task in self.tasks:
            task.cancel()
```

**Interview Questions**:
1. **Q**: What's the difference between `asyncio.gather()` and `asyncio.wait()`?
   **A**: `gather()` returns results in order and raises on first exception, `wait()` provides more control with return conditions and exception handling.

2. **Q**: How do you handle CPU-bound tasks in async code?
   **A**: Use `asyncio.get_event_loop().run_in_executor()` with `ThreadPoolExecutor` or `ProcessPoolExecutor`.

3. **Q**: What's the difference between `await` and `yield from`?
   **A**: `await` is for async/await syntax, `yield from` is for generator delegation (legacy async pattern).

---

## Memory Management and Garbage Collection

### Understanding Python Memory Model

**Object Memory Layout**:
```python
import sys
import gc
import weakref
from pympler import tracker, muppy, summary

# Memory usage analysis
def analyze_memory_usage():
    # Object size inspection
    objects = [
        42,                    # int
        3.14,                  # float
        "hello",               # string
        [1, 2, 3],            # list
        {"a": 1},             # dict
        {1, 2, 3},            # set
    ]
    
    for obj in objects:
        print(f"{type(obj).__name__}: {sys.getsizeof(obj)} bytes")

# Memory tracking
class MemoryTracker:
    def __init__(self):
        self.tracker = tracker.SummaryTracker()
    
    def start_tracking(self):
        self.tracker.print_diff()
    
    def get_memory_diff(self):
        return self.tracker.diff()
    
    def print_memory_summary(self):
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)

# Memory-efficient data structures
class MemoryEfficientClass:
    __slots__ = ['x', 'y', 'z']  # Reduces memory overhead
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Memory pooling example
class ObjectPool:
    def __init__(self, create_func, max_size=100):
        self.create_func = create_func
        self.pool = []
        self.max_size = max_size
    
    def acquire(self):
        if self.pool:
            return self.pool.pop()
        return self.create_func()
    
    def release(self, obj):
        if len(self.pool) < self.max_size:
            # Reset object state if needed
            self.pool.append(obj)
```

### Garbage Collection Optimization

**GC Control and Tuning**:
```python
import gc
import time
from typing import List, Optional

class GCOptimizer:
    def __init__(self):
        self.original_thresholds = gc.get_threshold()
    
    def disable_gc_temporarily(self):
        """Context manager to temporarily disable GC"""
        class GCDisabled:
            def __enter__(self):
                gc.disable()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                gc.enable()
        
        return GCDisabled()
    
    def tune_gc_thresholds(self, gen0=1000, gen1=15, gen2=15):
        """Tune GC thresholds for better performance"""
        gc.set_threshold(gen0, gen1, gen2)
    
    def restore_gc_thresholds(self):
        gc.set_threshold(*self.original_thresholds)
    
    def force_cleanup(self):
        """Force garbage collection and return collected objects"""
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        return collected

# Memory leak detection
class MemoryLeakDetector:
    def __init__(self):
        self.baseline = None
    
    def set_baseline(self):
        gc.collect()  # Clean up first
        self.baseline = len(gc.get_objects())
    
    def check_leaks(self) -> int:
        if self.baseline is None:
            raise RuntimeError("Baseline not set")
        
        gc.collect()
        current_objects = len(gc.get_objects())
        return current_objects - self.baseline
    
    def find_leaking_objects(self, limit=10):
        if self.baseline is None:
            return []
        
        current_objects = gc.get_objects()
        new_objects = current_objects[self.baseline:]
        
        # Group by type
        type_counts = {}
        for obj in new_objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        # Return top leaking types
        return sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

# Weak references for cache implementation
class WeakCache:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key, factory_func):
        """Get cached value or create new one"""
        if key in self._cache:
            return self._cache[key]
        
        value = factory_func()
        self._cache[key] = value
        return value
    
    def clear(self):
        self._cache.clear()
    
    def size(self):
        return len(self._cache)

# Memory-efficient data processing
def process_large_dataset_efficiently(filename: str):
    """Process large files without loading everything into memory"""
    def read_chunks(file_handle, chunk_size=8192):
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            yield chunk
    
    total_size = 0
    with open(filename, 'rb') as f:
        for chunk in read_chunks(f):
            total_size += len(chunk)
            # Process chunk
            del chunk  # Explicit cleanup for large objects
    
    return total_size
```

### Memory Profiling and Optimization

**Advanced Memory Profiling**:
```python
import tracemalloc
import linecache
import os
from contextlib import contextmanager

class MemoryProfiler:
    def __init__(self):
        self.snapshots = []
    
    def start_tracing(self):
        tracemalloc.start()
    
    def take_snapshot(self, description=""):
        if not tracemalloc.is_tracing():
            raise RuntimeError("Memory tracing not started")
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((description, snapshot))
        return snapshot
    
    def compare_snapshots(self, index1=0, index2=-1):
        if len(self.snapshots) < 2:
            raise ValueError("Need at least 2 snapshots")
        
        _, snap1 = self.snapshots[index1]
        _, snap2 = self.snapshots[index2]
        
        top_stats = snap2.compare_to(snap1, 'lineno')
        return top_stats
    
    def print_top_allocations(self, snapshot_index=-1, limit=10):
        if not self.snapshots:
            raise ValueError("No snapshots available")
        
        _, snapshot = self.snapshots[snapshot_index]
        top_stats = snapshot.statistics('lineno')
        
        print(f"Top {limit} allocations:")
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback.format()[-1]
            print(f"#{index}: {frame}")
            print(f"    {stat.size / 1024 / 1024:.1f} MB, {stat.count} blocks")

# Context manager for memory profiling
@contextmanager
def memory_profiler():
    profiler = MemoryProfiler()
    profiler.start_tracing()
    profiler.take_snapshot("start")
    
    try:
        yield profiler
    finally:
        profiler.take_snapshot("end")
        stats = profiler.compare_snapshots(0, 1)
        
        print("Memory usage changes:")
        for stat in stats[:10]:
            if stat.size_diff > 0:
                print(f"+{stat.size_diff / 1024:.1f} KB: {stat.traceback.format()[-1]}")

# Memory optimization techniques
class OptimizedDataProcessor:
    def __init__(self):
        self.gc_optimizer = GCOptimizer()
    
    def process_with_gc_optimization(self, data_list):
        """Process data with GC optimizations"""
        with self.gc_optimizer.disable_gc_temporarily():
            # Batch processing without GC interruption
            results = []
            for i, item in enumerate(data_list):
                result = self._process_item(item)
                results.append(result)
                
                # Periodic manual GC
                if i % 1000 == 0:
                    collected = self.gc_optimizer.force_cleanup()
                    if collected:
                        print(f"Collected {collected} objects at item {i}")
            
            return results
    
    def _process_item(self, item):
        # Simulate processing
        return item * 2
    
    def process_with_slots(self):
        """Demonstrate __slots__ memory savings"""
        
        class RegularClass:
            def __init__(self, a, b, c):
                self.a, self.b, self.c = a, b, c
        
        class SlottedClass:
            __slots__ = ['a', 'b', 'c']
            def __init__(self, a, b, c):
                self.a, self.b, self.c = a, b, c
        
        # Memory comparison
        regular_objects = [RegularClass(i, i+1, i+2) for i in range(10000)]
        slotted_objects = [SlottedClass(i, i+1, i+2) for i in range(10000)]
        
        regular_size = sum(sys.getsizeof(obj) + sys.getsizeof(obj.__dict__) 
                          for obj in regular_objects)
        slotted_size = sum(sys.getsizeof(obj) for obj in slotted_objects)
        
        print(f"Regular class total: {regular_size / 1024:.1f} KB")
        print(f"Slotted class total: {slotted_size / 1024:.1f} KB")
        print(f"Memory savings: {(regular_size - slotted_size) / regular_size * 100:.1f}%")
```

**Interview Questions**:
1. **Q**: How does Python's garbage collection work?
   **A**: Python uses reference counting as primary mechanism, with cyclic garbage collection for breaking reference cycles using generational collection.

2. **Q**: What are `__slots__` and when should you use them?
   **A**: `__slots__` restricts attribute creation and saves memory by avoiding `__dict__` creation. Use for classes with many instances and fixed attributes.

3. **Q**: How do you detect memory leaks in Python?
   **A**: Use `tracemalloc`, `gc.get_objects()`, weak references, and memory profiling tools like `pympler` or `memory_profiler`.

---

## Python Design Patterns

### Creational Patterns

**Singleton Pattern (Multiple Implementations)**:
```python
import threading
from abc import ABC, abstractmethod

# Thread-safe singleton with metaclass
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Connected to database"

# Decorator-based singleton
def singleton_decorator(cls):
    instances = {}
    lock = threading.Lock()
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton_decorator
class CacheManager:
    def __init__(self):
        self.cache = {}

# Borg pattern (shared state)
class BorgPattern:
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state
        if not hasattr(self, 'initialized'):
            self.data = {}
            self.initialized = True

# Factory pattern with registration
class ShapeFactory:
    _shapes = {}
    
    @classmethod
    def register(cls, shape_type, shape_class):
        cls._shapes[shape_type] = shape_class
    
    @classmethod
    def create(cls, shape_type, **kwargs):
        if shape_type not in cls._shapes:
            raise ValueError(f"Unknown shape type: {shape_type}")
        return cls._shapes[shape_type](**kwargs)

class Shape(ABC):
    @abstractmethod
    def draw(self):
        pass

class Circle(Shape):
    def __init__(self, radius=1):
        self.radius = radius
    
    def draw(self):
        return f"Circle with radius {self.radius}"

class Rectangle(Shape):
    def __init__(self, width=1, height=1):
        self.width = width
        self.height = height
    
    def draw(self):
        return f"Rectangle {self.width}x{self.height}"

# Register shapes
ShapeFactory.register("circle", Circle)
ShapeFactory.register("rectangle", Rectangle)
```

### Structural Patterns

**Adapter Pattern**:
```python
from abc import ABC, abstractmethod

# Target interface
class MediaPlayer(ABC):
    @abstractmethod
    def play(self, audio_type, filename):
        pass

# Adaptee (existing interface)
class Mp3Player:
    def play_mp3(self, filename):
        return f"Playing MP3: {filename}"

class Mp4Player:
    def play_mp4(self, filename):
        return f"Playing MP4: {filename}"

# Adapter
class MediaAdapter(MediaPlayer):
    def __init__(self):
        self.mp3_player = Mp3Player()
        self.mp4_player = Mp4Player()
    
    def play(self, audio_type, filename):
        if audio_type.lower() == "mp3":
            return self.mp3_player.play_mp3(filename)
        elif audio_type.lower() == "mp4":
            return self.mp4_player.play_mp4(filename)
        else:
            raise ValueError(f"Unsupported audio type: {audio_type}")

# Decorator pattern for caching
from functools import wraps
import time

class CacheDecorator:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self.cache = {}
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            if key in self.cache:
                value, timestamp = self.cache[key]
                if now - timestamp < self.ttl:
                    return value
            
            result = func(*args, **kwargs)
            self.cache[key] = (result, now)
            return result
        return wrapper

# Proxy pattern for lazy loading
class ImageProxy:
    def __init__(self, filename):
        self.filename = filename
        self._image = None
    
    def display(self):
        if self._image is None:
            self._image = self._load_image()
        return self._image.display()
    
    def _load_image(self):
        print(f"Loading image: {self.filename}")
        # Simulate expensive image loading
        time.sleep(0.1)
        return Image(self.filename)

class Image:
    def __init__(self, filename):
        self.filename = filename
    
    def display(self):
        return f"Displaying: {self.filename}"
```

### Behavioral Patterns

**Observer Pattern**:
```python
from abc import ABC, abstractmethod
from typing import List, Any
import weakref

class Observer(ABC):
    @abstractmethod
    def update(self, subject, event_type, data):
        pass

class Subject:
    def __init__(self):
        self._observers = weakref.WeakSet()
    
    def attach(self, observer: Observer):
        self._observers.add(observer)
    
    def detach(self, observer: Observer):
        self._observers.discard(observer)
    
    def notify(self, event_type, data=None):
        # Create list to avoid modification during iteration
        observers = list(self._observers)
        for observer in observers:
            try:
                observer.update(self, event_type, data)
            except Exception as e:
                print(f"Error notifying observer: {e}")

class StockPrice(Subject):
    def __init__(self, symbol, price):
        super().__init__()
        self.symbol = symbol
        self._price = price
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        old_price = self._price
        self._price = value
        self.notify("price_change", {
            'old_price': old_price,
            'new_price': value,
            'symbol': self.symbol
        })

class StockTrader(Observer):
    def __init__(self, name):
        self.name = name
    
    def update(self, subject, event_type, data):
        if event_type == "price_change":
            symbol = data['symbol']
            new_price = data['new_price']
            print(f"{self.name}: {symbol} price changed to ${new_price}")

# Strategy pattern with duck typing
class SortingStrategy(ABC):
    @abstractmethod
    def sort(self, data):
        pass

class BubbleSort(SortingStrategy):
    def sort(self, data):
        data = data.copy()
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data

class QuickSort(SortingStrategy):
    def sort(self, data):
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class SortContext:
    def __init__(self, strategy: SortingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: SortingStrategy):
        self.strategy = strategy
    
    def sort(self, data):
        return self.strategy.sort(data)

# Command pattern
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def undo(self):
        pass

class TextEditor:
    def __init__(self):
        self.content = ""
    
    def write(self, text):
        self.content += text
    
    def delete(self, length):
        self.content = self.content[:-length]
    
    def get_content(self):
        return self.content

class WriteCommand(Command):
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text
    
    def execute(self):
        self.editor.write(self.text)
    
    def undo(self):
        self.editor.delete(len(self.text))

class CommandInvoker:
    def __init__(self):
        self.history = []
        self.current_position = -1
    
    def execute_command(self, command: Command):
        # Remove any commands after current position
        self.history = self.history[:self.current_position + 1]
        
        command.execute()
        self.history.append(command)
        self.current_position += 1
    
    def undo(self):
        if self.current_position >= 0:
            command = self.history[self.current_position]
            command.undo()
            self.current_position -= 1
    
    def redo(self):
        if self.current_position < len(self.history) - 1:
            self.current_position += 1
            command = self.history[self.current_position]
            command.execute()
```

### Python-Specific Patterns

**Context Manager Pattern**:
```python
import sqlite3
from contextlib import contextmanager
import tempfile
import os

# Database transaction context manager
@contextmanager
def database_transaction(connection):
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

# File operation context manager
@contextmanager
def temporary_file_manager(suffix='.tmp'):
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'w') as tmp_file:
            yield tmp_file, path
    finally:
        if os.path.exists(path):
            os.unlink(path)

# Resource pool context manager
class ConnectionPool:
    def __init__(self, create_connection, max_connections=10):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.pool = []
        self.used = set()
        self.lock = threading.Lock()
    
    @contextmanager
    def get_connection(self):
        connection = self._acquire_connection()
        try:
            yield connection
        finally:
            self._release_connection(connection)
    
    def _acquire_connection(self):
        with self.lock:
            if self.pool:
                connection = self.pool.pop()
            elif len(self.used) < self.max_connections:
                connection = self.create_connection()
            else:
                raise RuntimeError("No connections available")
            
            self.used.add(connection)
            return connection
    
    def _release_connection(self, connection):
        with self.lock:
            self.used.discard(connection)
            self.pool.append(connection)

# Descriptor pattern for validation
class ValidatedAttribute:
    def __init__(self, validator_func, error_message="Invalid value"):
        self.validator_func = validator_func
        self.error_message = error_message
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name, None)
    
    def __set__(self, obj, value):
        if not self.validator_func(value):
            raise ValueError(self.error_message)
        setattr(obj, self.name, value)

class Person:
    name = ValidatedAttribute(
        lambda x: isinstance(x, str) and len(x) > 0,
        "Name must be a non-empty string"
    )
    age = ValidatedAttribute(
        lambda x: isinstance(x, int) and 0 <= x <= 150,
        "Age must be an integer between 0 and 150"
    )
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

**Interview Questions**:
1. **Q**: What's the difference between `__new__` and `__init__` in implementing Singleton?
   **A**: `__new__` controls object creation (better for Singleton), `__init__` initializes existing objects.

2. **Q**: How do you implement the Observer pattern with weak references?
   **A**: Use `weakref.WeakSet()` to avoid circular references and allow automatic cleanup of observers.

3. **Q**: When would you use the Borg pattern instead of Singleton?
   **A**: When you want multiple instances that share state, rather than restricting to a single instance.

---

## Best Practices and Real-world Scenarios

### Performance Optimization

**Profiling and Optimization**:
```python
import cProfile
import pstats
from functools import wraps
import timeit

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        return result
    return wrapper

# Memory-efficient iteration
def process_large_file_efficiently(filename):
    """Process large files line by line"""
    with open(filename, 'r') as f:
        for line_number, line in enumerate(f, 1):
            # Process line without loading entire file
            yield line_number, line.strip()

# Efficient data structures
from collections import deque, defaultdict, Counter
import bisect

class EfficientDataProcessor:
    def __init__(self):
        self.data = deque()  # O(1) append/pop from both ends
        self.lookup = {}     # O(1) average lookup
        self.sorted_data = []  # For binary search
    
    def add_data(self, item):
        self.data.append(item)
        bisect.insort(self.sorted_data, item)  # Keep sorted
    
    def find_nearest(self, target):
        """Find nearest value using binary search"""
        pos = bisect.bisect_left(self.sorted_data, target)
        if pos == 0:
            return self.sorted_data[0]
        if pos == len(self.sorted_data):
            return self.sorted_data[-1]
        
        before = self.sorted_data[pos - 1]
        after = self.sorted_data[pos]
        return after if (after - target) < (target - before) else before
```

### Error Handling and Debugging

**Robust Error Handling**:
```python
import logging
import traceback
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class APIError(Exception):
    def __init__(self, message, status_code=None, error_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code

@contextmanager
def error_handler(logger=None, reraise=True):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        yield
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if reraise:
            raise

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=3, backoff_base=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    wait_time = backoff_base ** attempt
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        return wrapper
    return decorator
```

### Common Pitfalls

**Avoiding Common Mistakes**:
```python
# Mutable default arguments
def bad_function(items=[]):  # Don't do this
    items.append("new")
    return items

def good_function(items=None):  # Do this instead
    if items is None:
        items = []
    items.append("new")
    return items

# Late binding closures
def create_multipliers_bad():
    return [lambda x: x * i for i in range(5)]  # All use i=4

def create_multipliers_good():
    return [lambda x, i=i: x * i for i in range(5)]  # Capture i

# Circular imports
# Use local imports or restructure modules
def expensive_operation():
    from some_module import expensive_function  # Local import
    return expensive_function()
```

### Interview Scenario Questions

**Q**: Design a Python application to handle millions of concurrent users.
**A**: Use async/await with frameworks like FastAPI/aiohttp, implement connection pooling, use Redis for session storage, and horizontal scaling with load balancers.

**Q**: How would you optimize a Python application that's using too much memory?
**A**: Profile with tracemalloc, use generators instead of lists, implement object pooling, use `__slots__`, and consider using numpy for numeric data.

**Q**: Implement a distributed cache in Python.
**A**: Use Redis with consistent hashing, implement cache warming strategies, handle cache invalidation, and use async operations for non-blocking cache access.

---

This comprehensive guide covers essential Python concepts for technical interviews. Focus on understanding the trade-offs between different approaches and practicing real-world applications of these patterns and techniques.