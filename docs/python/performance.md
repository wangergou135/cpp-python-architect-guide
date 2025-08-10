# Performance Optimization in Python

## Core Concepts

### Memory Management
```python
import sys
import gc

# Memory usage monitoring
def get_size(obj):
    """Get size of object and its contents in bytes."""
    marked = {id(obj)}
    obj_q = [obj]
    size = 0

    while obj_q:
        size += sum(sys.getsizeof(v) for v in obj_q)
        obj_q = [
            v for o in obj_q 
            for v in gc.get_referents(o) 
            if id(v) not in marked and not isinstance(v, type)
        ]
        marked.update(id(v) for v in obj_q)
    return size
```

### CPU Profiling
```python
import cProfile
import pstats

def profile_func(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats()
        return result
    return wrapper
```

## Optimization Techniques

### Data Structures
```python
from collections import defaultdict, deque
from typing import Dict, List

# Using appropriate data structures
def optimize_lookups():
    # Dictionary for O(1) lookups
    lookup_dict: Dict[str, int] = {}
    
    # defaultdict to avoid key checks
    counts = defaultdict(int)
    
    # deque for efficient queue operations
    queue = deque(maxlen=1000)
```

### List Comprehensions
```python
# Efficient list operations
def list_operations():
    # List comprehension (faster than loops)
    squares = [x**2 for x in range(1000)]
    
    # Generator expression (memory efficient)
    sum_squares = sum(x**2 for x in range(1000))
```

### Numpy Operations
```python
import numpy as np

def numpy_optimizations():
    # Vectorized operations
    arr = np.array([1, 2, 3, 4, 5])
    squared = arr ** 2  # Faster than Python loops
    
    # Efficient matrix operations
    matrix = np.random.rand(1000, 1000)
    result = matrix.dot(matrix.T)
```

## Memory Optimization

### Using __slots__
```python
class OptimizedClass:
    __slots__ = ['x', 'y']  # Restricts attributes
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### Memory Efficient Collections
```python
from array import array
from collections import namedtuple

# Using array for homogeneous data
numbers = array('i', range(10000))

# Using namedtuple for immutable records
Point = namedtuple('Point', ['x', 'y'])
points = [Point(x, x*2) for x in range(1000)]
```

## Multiprocessing and Threading

### Process Pool
```python
from multiprocessing import Pool

def cpu_bound_task(n):
    return sum(i * i for i in range(n))

def parallel_processing():
    with Pool() as pool:
        results = pool.map(cpu_bound_task, [10**6] * 4)
```

### Thread Pool
```python
from concurrent.futures import ThreadPoolExecutor
import requests

def io_bound_task(url):
    return requests.get(url).text

def parallel_io():
    urls = ['http://example.com'] * 10
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(io_bound_task, urls))
```

## Code Optimization

### Function Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### Generators for Large Data
```python
def process_large_file(filename):
    with open(filename) as f:
        # Memory efficient iteration
        for line in f:
            yield line.strip()
```

## Database Optimization

### Connection Pooling
```python
import psycopg2
from psycopg2 import pool

class DatabasePool:
    def __init__(self):
        self.pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,
            database="mydb",
            user="user",
            password="password",
            host="localhost"
        )

    def execute_query(self, query, params=None):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
        finally:
            self.pool.putconn(conn)
```

## Profiling and Monitoring

### Custom Timer
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    start = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start
    print(f"{description}: {elapsed_time:0.4f} seconds")
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function to profile
    large_list = [i**2 for i in range(10**6)]
    return sum(large_list)
```

## Best Practices

1. Use built-in functions and libraries
2. Implement caching where appropriate
3. Choose appropriate data structures
4. Use generators for large datasets
5. Profile before optimizing
6. Consider using PyPy for CPU-intensive tasks
7. Implement connection pooling for databases
8. Use multiprocessing for CPU-bound tasks
9. Use threading for I/O-bound tasks
10. Keep memory usage in check

## Advanced Techniques

### Custom Memory Management
```python
class MemoryPool:
    def __init__(self, size):
        self.size = size
        self.memory = [None] * size
        self.free = list(range(size))
    
    def allocate(self):
        if not self.free:
            raise MemoryError("Pool exhausted")
        return self.free.pop()
    
    def deallocate(self, index):
        self.memory[index] = None
        self.free.append(index)
```

### Lazy Evaluation
```python
class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = self.function(obj)
        setattr(obj, self.name, value)
        return value
```

*Last updated: 2025-08-10 11:08:24 UTC*