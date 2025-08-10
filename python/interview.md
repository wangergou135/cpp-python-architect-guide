# Python Interview Guide

A comprehensive guide covering advanced Python concepts, best practices, and common interview questions for Python developers and architects.

## Table of Contents
1. [Python Fundamentals](#python-fundamentals)
2. [Python Memory Management](#python-memory-management)
3. [Advanced Python Concepts](#advanced-python-concepts)
4. [Web Frameworks](#web-frameworks)
5. [Python Performance](#python-performance)
6. [Testing and Development](#testing-and-development)

## Python Fundamentals

### GIL and its impact

The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecodes simultaneously.

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# CPU-bound task to demonstrate GIL impact
def cpu_bound_task(n):
    """CPU-intensive task that will be affected by GIL"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# I/O-bound task where GIL is less problematic
def io_bound_task(duration):
    """I/O task that releases GIL during sleep"""
    time.sleep(duration)
    return f"Task completed after {duration} seconds"

def demonstrate_gil_impact():
    # Single-threaded execution
    start_time = time.time()
    results = [cpu_bound_task(1000000) for _ in range(4)]
    single_thread_time = time.time() - start_time
    
    # Multi-threaded execution (affected by GIL)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_bound_task, [1000000] * 4))
    multi_thread_time = time.time() - start_time
    
    # Multi-process execution (bypasses GIL)
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_bound_task, [1000000] * 4))
    multi_process_time = time.time() - start_time
    
    print(f"Single thread time: {single_thread_time:.2f}s")
    print(f"Multi-thread time: {multi_thread_time:.2f}s")
    print(f"Multi-process time: {multi_process_time:.2f}s")
    print(f"Thread speedup: {single_thread_time / multi_thread_time:.2f}x")
    print(f"Process speedup: {single_thread_time / multi_process_time:.2f}x")

# Releasing GIL with C extensions
import ctypes
import time

def gil_releasing_sleep(duration):
    """Example of operation that releases GIL"""
    # time.sleep() releases GIL, allowing other threads to run
    time.sleep(duration)
    return "Sleep completed"

def demonstrate_gil_release():
    """Show how I/O operations release GIL"""
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(gil_releasing_sleep, 1) for _ in range(4)]
        results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    print(f"4 parallel 1-second sleeps took: {total_time:.2f}s")
    # Should be ~1 second, not 4, proving GIL was released

# Working around GIL limitations
class GILWorkArounds:
    @staticmethod
    def use_multiprocessing():
        """Use multiprocessing for CPU-bound tasks"""
        from multiprocessing import Pool
        
        def square(x):
            return x ** 2
        
        with Pool(processes=4) as pool:
            result = pool.map(square, range(100))
        return result
    
    @staticmethod
    def use_asyncio():
        """Use asyncio for I/O-bound concurrency"""
        import asyncio
        
        async def async_task(n):
            await asyncio.sleep(0.1)  # Simulated I/O
            return n ** 2
        
        async def main():
            tasks = [async_task(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            return results
        
        return asyncio.run(main())
    
    @staticmethod
    def use_numpy():
        """NumPy operations release GIL"""
        import numpy as np
        
        # NumPy operations are implemented in C and release GIL
        arr = np.random.random((1000, 1000))
        result = np.dot(arr, arr.T)  # Matrix multiplication releases GIL
        return result.sum()

# GIL monitoring
def monitor_gil_contention():
    """Simple GIL contention monitoring"""
    import sys
    import threading
    
    gil_count = {'switches': 0}
    
    def count_gil_switches():
        old_switch_count = sys.getswitchinterval()
        while True:
            new_count = threading.active_count()
            gill_count['switches'] += 1
            time.sleep(0.001)
            if gill_count['switches'] > 1000:
                break
    
    # This is a simplified example
    # Real GIL monitoring requires more sophisticated tools
    print("GIL monitoring would require specialized tools like py-spy or gil_load")
```

### Metaclasses and class decorators

Metaclasses control class creation, while class decorators modify classes after creation.

```python
# 1. Basic metaclass example
class SingletonMeta(type):
    """Metaclass that creates singleton instances"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    def __init__(self, host="localhost"):
        self.host = host
        self.connected = False
    
    def connect(self):
        self.connected = True
        print(f"Connected to {self.host}")

# 2. Metaclass for automatic registration
class RegisteredMeta(type):
    """Metaclass that automatically registers classes"""
    registry = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        if hasattr(cls, 'registry_name'):
            mcs.registry[cls.registry_name] = cls
        return cls

class BaseHandler(metaclass=RegisteredMeta):
    pass

class HttpHandler(BaseHandler):
    registry_name = 'http'
    
    def handle(self, request):
        return "HTTP handled"

class FtpHandler(BaseHandler):
    registry_name = 'ftp'
    
    def handle(self, request):
        return "FTP handled"

# 3. Metaclass for ORM-like field validation
class FieldMeta(type):
    """Metaclass for creating ORM-like models with field validation"""
    
    def __new__(mcs, name, bases, namespace):
        fields = {}
        validators = {}
        
        for key, value in list(namespace.items()):
            if isinstance(value, Field):
                fields[key] = value
                validators[key] = value.validator
                # Replace field descriptor with property
                namespace[key] = value.create_property(key)
        
        namespace['_fields'] = fields
        namespace['_validators'] = validators
        return super().__new__(mcs, name, bases, namespace)

class Field:
    def __init__(self, field_type=str, required=True, validator=None):
        self.field_type = field_type
        self.required = required
        self.validator = validator or (lambda x: True)
    
    def create_property(self, name):
        def getter(obj):
            return getattr(obj, f'_{name}', None)
        
        def setter(obj, value):
            if self.required and value is None:
                raise ValueError(f"{name} is required")
            if not isinstance(value, self.field_type):
                raise TypeError(f"{name} must be {self.field_type.__name__}")
            if not self.validator(value):
                raise ValueError(f"Invalid value for {name}")
            setattr(obj, f'_{name}', value)
        
        return property(getter, setter)

class Model(metaclass=FieldMeta):
    def __init__(self, **kwargs):
        for name, field in self._fields.items():
            value = kwargs.get(name)
            if field.required and value is None:
                raise ValueError(f"{name} is required")
            setattr(self, name, value)

class User(Model):
    name = Field(str, required=True, validator=lambda x: len(x) > 0)
    age = Field(int, required=True, validator=lambda x: 0 <= x <= 150)
    email = Field(str, required=True, validator=lambda x: '@' in x)

# 4. Class decorators
def add_repr(cls):
    """Class decorator that adds a __repr__ method"""
    def __repr__(self):
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{cls.__name__}({attrs})'
    
    cls.__repr__ = __repr__
    return cls

def add_equality(cls):
    """Class decorator that adds equality comparison"""
    def __eq__(self, other):
        if not isinstance(other, cls):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
    
    cls.__eq__ = __eq__
    cls.__hash__ = __hash__
    return cls

def immutable(cls):
    """Class decorator that makes a class immutable"""
    original_setattr = cls.__setattr__
    
    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            raise AttributeError(f"Cannot modify immutable object")
        original_setattr(self, name, value)
    
    def __init_wrapper__(original_init):
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            object.__setattr__(self, '_initialized', True)
        return __init__
    
    cls.__setattr__ = __setattr__
    cls.__init__ = __init_wrapper__(cls.__init__)
    return cls

@add_repr
@add_equality
@immutable
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 5. Advanced metaclass: Attribute access logging
class LoggingMeta(type):
    """Metaclass that logs attribute access"""
    
    def __new__(mcs, name, bases, namespace):
        # Wrap all methods to add logging
        for key, value in namespace.items():
            if callable(value) and not key.startswith('_'):
                namespace[key] = mcs.log_calls(value, key)
        
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Override __getattribute__ to log attribute access
        original_getattribute = cls.__getattribute__
        
        def logged_getattribute(self, name):
            result = original_getattribute(self, name)
            if not name.startswith('_'):
                print(f"Accessed {cls.__name__}.{name}")
            return result
        
        cls.__getattribute__ = logged_getattribute
        return cls
    
    @staticmethod
    def log_calls(func, name):
        def wrapper(self, *args, **kwargs):
            print(f"Calling {self.__class__.__name__}.{name}")
            return func(self, *args, **kwargs)
        return wrapper

class Calculator(metaclass=LoggingMeta):
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

# Usage examples
def metaclass_examples():
    # Singleton metaclass
    db1 = DatabaseConnection("server1")
    db2 = DatabaseConnection("server2")
    print(f"Same instance: {db1 is db2}")  # True
    
    # Registration metaclass
    print("Registered handlers:", RegisteredMeta.registry)
    handler = RegisteredMeta.registry['http']()
    print(handler.handle("request"))
    
    # Field validation metaclass
    try:
        user = User(name="John", age=30, email="john@example.com")
        print(f"Created user: {user.name}, {user.age}, {user.email}")
        
        # This will raise an error
        user.age = -5
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Class decorators
    p1 = Point(1, 2)
    p2 = Point(1, 2)
    print(f"Points equal: {p1 == p2}")
    print(f"Point repr: {p1}")
    
    try:
        p1.x = 5  # Will raise error due to immutable decorator
    except AttributeError as e:
        print(f"Immutability error: {e}")
    
    # Logging metaclass
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")
```

### Context managers and with statement

Context managers ensure proper resource cleanup and can manage complex setup/teardown scenarios.

```python
import contextlib
import threading
import time
import tempfile
import os
from typing import Generator, Any

# 1. Basic context manager protocol
class FileManager:
    """Traditional context manager using __enter__ and __exit__"""
    
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening file {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing file {self.filename}")
        if self.file:
            self.file.close()
        
        # Handle exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
            return False  # Propagate exception
        return True

# 2. Context manager using contextlib.contextmanager decorator
@contextlib.contextmanager
def database_transaction():
    """Context manager for database transactions"""
    print("Beginning transaction")
    try:
        yield "transaction_object"
        print("Committing transaction")
    except Exception as e:
        print(f"Rolling back transaction due to: {e}")
        raise
    finally:
        print("Cleaning up database connection")

@contextlib.contextmanager
def timing_context(name: str):
    """Context manager for timing code execution"""
    start_time = time.time()
    print(f"Starting {name}")
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{name} took {end_time - start_time:.2f} seconds")

@contextlib.contextmanager
def temporary_attribute(obj, attr_name, temp_value):
    """Temporarily change an object's attribute"""
    old_value = getattr(obj, attr_name, None)
    setattr(obj, attr_name, temp_value)
    try:
        yield obj
    finally:
        if old_value is not None:
            setattr(obj, attr_name, old_value)
        else:
            delattr(obj, attr_name)

# 3. Complex context managers
class ManagedResource:
    """Example of a resource that needs careful management"""
    def __init__(self, name):
        self.name = name
        self.is_locked = False
        self.lock = threading.Lock()
    
    def acquire(self):
        self.lock.acquire()
        self.is_locked = True
        print(f"Acquired resource: {self.name}")
    
    def release(self):
        if self.is_locked:
            self.is_locked = False
            self.lock.release()
            print(f"Released resource: {self.name}")

@contextlib.contextmanager
def managed_resource_context(resource: ManagedResource):
    """Context manager for ManagedResource"""
    resource.acquire()
    try:
        yield resource
    finally:
        resource.release()

# 4. Nested context managers and ExitStack
def demonstrate_exit_stack():
    """Show how to manage multiple context managers dynamically"""
    filenames = ['temp1.txt', 'temp2.txt', 'temp3.txt']
    
    with contextlib.ExitStack() as stack:
        files = []
        for filename in filenames:
            file = stack.enter_context(open(filename, 'w'))
            files.append(file)
            file.write(f"Content for {filename}\n")
        
        # All files are automatically closed when exiting the with block
        print(f"Opened {len(files)} files")
    
    # Clean up temporary files
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)

# 5. Async context managers
import asyncio

class AsyncResource:
    def __init__(self, name):
        self.name = name
    
    async def __aenter__(self):
        print(f"Async acquiring {self.name}")
        await asyncio.sleep(0.1)  # Simulate async setup
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        print(f"Async releasing {self.name}")
        await asyncio.sleep(0.1)  # Simulate async cleanup
        return False

@contextlib.asynccontextmanager
async def async_timing_context(name: str):
    """Async context manager for timing"""
    start_time = time.time()
    print(f"Starting async {name}")
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Async {name} took {end_time - start_time:.2f} seconds")

# 6. Context manager for thread safety
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    @contextlib.contextmanager
    def get_lock(self):
        """Context manager that provides thread-safe access"""
        self._lock.acquire()
        try:
            yield self
        finally:
            self._lock.release()
    
    def increment(self):
        self._value += 1
    
    def get_value(self):
        return self._value

# 7. Context manager for configuration changes
class ConfigurationManager:
    def __init__(self):
        self.config = {'debug': False, 'timeout': 30}
    
    @contextlib.contextmanager
    def temporary_config(self, **kwargs):
        """Temporarily modify configuration"""
        old_config = self.config.copy()
        self.config.update(kwargs)
        try:
            yield self.config
        finally:
            self.config = old_config

# 8. Error handling context manager
@contextlib.contextmanager
def ignore_errors(*exceptions):
    """Context manager that ignores specific exceptions"""
    try:
        yield
    except exceptions as e:
        print(f"Ignored exception: {e}")

@contextlib.contextmanager
def retry_context(max_attempts=3, delay=1):
    """Context manager that retries operations"""
    for attempt in range(max_attempts):
        try:
            yield attempt + 1
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

# Usage examples
def context_manager_examples():
    # Basic file manager
    with FileManager('test.txt', 'w') as f:
        f.write("Hello, World!")
    
    # Database transaction
    with database_transaction() as tx:
        print(f"Working with {tx}")
    
    # Timing context
    with timing_context("slow operation"):
        time.sleep(0.1)
    
    # Temporary attribute change
    class TestObject:
        value = "original"
    
    obj = TestObject()
    print(f"Before: {obj.value}")
    
    with temporary_attribute(obj, 'value', 'temporary'):
        print(f"During: {obj.value}")
    
    print(f"After: {obj.value}")
    
    # Managed resource
    resource = ManagedResource("important_resource")
    with managed_resource_context(resource):
        print(f"Using {resource.name}")
    
    # ExitStack for multiple resources
    demonstrate_exit_stack()
    
    # Thread-safe counter
    counter = ThreadSafeCounter()
    
    def worker():
        for _ in range(100):
            with counter.get_lock():
                counter.increment()
    
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    print(f"Final counter value: {counter.get_value()}")
    
    # Configuration manager
    config_mgr = ConfigurationManager()
    print(f"Original config: {config_mgr.config}")
    
    with config_mgr.temporary_config(debug=True, timeout=60):
        print(f"Temporary config: {config_mgr.config}")
    
    print(f"Restored config: {config_mgr.config}")
    
    # Error handling
    with ignore_errors(ValueError, TypeError):
        raise ValueError("This will be ignored")
    
    print("Continuing after ignored error")
    
    # Retry context
    attempt_count = 0
    with retry_context(max_attempts=3):
        attempt_count += 1
        if attempt_count < 3:
            raise RuntimeError(f"Simulated failure on attempt {attempt_count}")
        print("Success on final attempt!")

# Async context manager example
async def async_context_examples():
    async with AsyncResource("async_file"):
        print("Working with async resource")
    
    async with async_timing_context("async operation"):
        await asyncio.sleep(0.2)

# Run async examples
def run_async_examples():
    asyncio.run(async_context_examples())
```

### Generator expressions and iterators

Generators provide memory-efficient iteration and enable powerful functional programming patterns.

```python
import itertools
import sys
from typing import Iterator, Generator, Iterable, Any
from collections.abc import Iterator as ABCIterator

# 1. Basic generators and yield
def simple_generator(n: int) -> Generator[int, None, None]:
    """Basic generator that yields numbers from 0 to n-1"""
    for i in range(n):
        print(f"Generating {i}")
        yield i

def fibonacci_generator() -> Generator[int, None, None]:
    """Infinite Fibonacci sequence generator"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def range_with_step(start: int, stop: int, step: int = 1) -> Generator[int, None, None]:
    """Custom range generator with step"""
    current = start
    while current < stop:
        yield current
        current += step

# 2. Generator expressions
def generator_expressions_demo():
    """Demonstrate various generator expressions"""
    # Basic generator expression
    squares = (x ** 2 for x in range(10))
    print("Squares:", list(squares))
    
    # Conditional generator expression
    even_squares = (x ** 2 for x in range(10) if x % 2 == 0)
    print("Even squares:", list(even_squares))
    
    # Nested generator expression
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flattened = (item for row in matrix for item in row)
    print("Flattened matrix:", list(flattened))
    
    # Generator with transformation
    words = ["hello", "world", "python", "generator"]
    uppercased = (word.upper() for word in words if len(word) > 5)
    print("Long words uppercased:", list(uppercased))

# 3. Advanced generators with send(), throw(), and close()
def advanced_generator() -> Generator[int, str, str]:
    """Generator that can receive values and handle exceptions"""
    result = []
    try:
        while True:
            # Receive value sent to generator
            received = yield len(result)
            if received:
                result.append(received)
                print(f"Received: {received}, List now: {result}")
    except GeneratorExit:
        print("Generator is being closed")
        return "Generator closed gracefully"
    except Exception as e:
        print(f"Exception in generator: {e}")
        return f"Generator closed due to exception: {e}"

def demonstrate_generator_communication():
    """Show how to communicate with generators"""
    gen = advanced_generator()
    
    # Start the generator
    first_value = next(gen)
    print(f"First value: {first_value}")
    
    # Send values to generator
    response1 = gen.send("hello")
    print(f"Response 1: {response1}")
    
    response2 = gen.send("world")
    print(f"Response 2: {response2}")
    
    # Throw exception to generator
    try:
        gen.throw(ValueError, "Test exception")
    except StopIteration as e:
        print(f"Generator returned: {e.value}")

# 4. Custom iterator classes
class CountDown:
    """Custom iterator for counting down"""
    
    def __init__(self, start: int):
        self.start = start
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

class InfiniteSequence:
    """Iterator that generates infinite sequence with a function"""
    
    def __init__(self, func, start=0):
        self.func = func
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        result = self.func(self.current)
        self.current += 1
        return result

# 5. Generator-based data processing pipeline
def read_large_file(filename: str) -> Generator[str, None, None]:
    """Generator for reading large files line by line"""
    try:
        with open(filename, 'r') as file:
            for line in file:
                yield line.strip()
    except FileNotFoundError:
        # For demo purposes, generate fake data
        for i in range(1000):
            yield f"Line {i}: This is sample data for processing"

def filter_lines(lines: Iterable[str], keyword: str) -> Generator[str, None, None]:
    """Filter lines containing a keyword"""
    for line in lines:
        if keyword.lower() in line.lower():
            yield line

def transform_lines(lines: Iterable[str]) -> Generator[dict, None, None]:
    """Transform lines into structured data"""
    for i, line in enumerate(lines):
        yield {
            'line_number': i + 1,
            'content': line,
            'word_count': len(line.split()),
            'length': len(line)
        }

def process_in_batches(items: Iterable[Any], batch_size: int) -> Generator[list, None, None]:
    """Process items in batches"""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:  # Yield remaining items
        yield batch

# 6. Memory-efficient data processing
def memory_efficient_processing():
    """Demonstrate memory-efficient data processing pipeline"""
    # Create processing pipeline
    raw_lines = read_large_file("nonexistent.txt")  # Will generate fake data
    filtered_lines = filter_lines(raw_lines, "data")
    structured_data = transform_lines(filtered_lines)
    batched_data = process_in_batches(structured_data, 5)
    
    # Process batches
    total_processed = 0
    for batch_num, batch in enumerate(batched_data):
        if batch_num >= 3:  # Process only first 3 batches for demo
            break
        
        print(f"Processing batch {batch_num + 1} with {len(batch)} items")
        for item in batch:
            total_processed += 1
            # Simulate processing
            if total_processed <= 5:  # Show first 5 items
                print(f"  Item {item['line_number']}: {item['content'][:50]}...")
    
    print(f"Total items processed: {total_processed}")

# 7. Generator-based coroutines (pre-async/await)
def grep_coroutine(pattern: str) -> Generator[None, str, None]:
    """Coroutine that filters lines matching a pattern"""
    print(f"Starting grep for pattern: {pattern}")
    try:
        while True:
            line = yield
            if pattern in line:
                print(f"Match found: {line}")
    except GeneratorExit:
        print("Grep coroutine closing")

def broadcast_coroutine(targets: list) -> Generator[None, str, None]:
    """Coroutine that broadcasts messages to multiple targets"""
    while True:
        message = yield
        for target in targets:
            target.send(message)

def demonstrate_coroutines():
    """Show generator-based coroutines in action"""
    # Create grep coroutines
    grep1 = grep_coroutine("python")
    grep2 = grep_coroutine("generator")
    
    # Start coroutines
    next(grep1)
    next(grep2)
    
    # Create broadcaster
    broadcaster = broadcast_coroutine([grep1, grep2])
    next(broadcaster)
    
    # Send messages
    messages = [
        "This is about python programming",
        "Generators are powerful in python",
        "This message has no keywords",
        "Generator expressions are concise",
        "Python generators are memory efficient"
    ]
    
    for message in messages:
        broadcaster.send(message)
    
    # Close coroutines
    grep1.close()
    grep2.close()

# 8. itertools usage examples
def itertools_examples():
    """Demonstrate powerful itertools functions"""
    # Infinite iterators
    count_iter = itertools.count(start=10, step=2)
    print("Count iterator (first 5):", list(itertools.islice(count_iter, 5)))
    
    cycle_iter = itertools.cycle(['A', 'B', 'C'])
    print("Cycle iterator (first 8):", list(itertools.islice(cycle_iter, 8)))
    
    repeat_iter = itertools.repeat('hello', 3)
    print("Repeat iterator:", list(repeat_iter))
    
    # Combinatorial iterators
    data = [1, 2, 3, 4]
    print("Permutations:", list(itertools.permutations(data, 2)))
    print("Combinations:", list(itertools.combinations(data, 2)))
    print("Product:", list(itertools.product([1, 2], ['A', 'B'])))
    
    # Filtering iterators
    numbers = range(10)
    evens = itertools.filterfalse(lambda x: x % 2, numbers)
    print("Evens using filterfalse:", list(evens))
    
    # Grouping
    data = [1, 1, 2, 2, 2, 3, 3, 1, 1]
    grouped = itertools.groupby(data)
    print("Grouped data:")
    for key, group in grouped:
        print(f"  {key}: {list(group)}")
    
    # Chain multiple iterables
    list1 = [1, 2, 3]
    list2 = [4, 5, 6]
    list3 = [7, 8, 9]
    chained = itertools.chain(list1, list2, list3)
    print("Chained lists:", list(chained))

# 9. Performance comparison: list vs generator
def performance_comparison():
    """Compare memory usage of lists vs generators"""
    
    def create_list(n):
        return [x ** 2 for x in range(n)]
    
    def create_generator(n):
        return (x ** 2 for x in range(n))
    
    n = 10000
    
    # List approach
    print("Creating list...")
    squares_list = create_list(n)
    list_size = sys.getsizeof(squares_list)
    
    # Generator approach
    print("Creating generator...")
    squares_gen = create_generator(n)
    gen_size = sys.getsizeof(squares_gen)
    
    print(f"List size: {list_size} bytes")
    print(f"Generator size: {gen_size} bytes")
    print(f"Memory saved: {list_size - gen_size} bytes")
    print(f"Memory efficiency: {gen_size / list_size:.1%} of list size")

# Usage examples
def generator_examples():
    print("=== Basic Generators ===")
    gen = simple_generator(3)
    for value in gen:
        print(f"Got: {value}")
    
    print("\n=== Fibonacci Generator ===")
    fib = fibonacci_generator()
    first_10_fibs = [next(fib) for _ in range(10)]
    print(f"First 10 Fibonacci numbers: {first_10_fibs}")
    
    print("\n=== Generator Expressions ===")
    generator_expressions_demo()
    
    print("\n=== Advanced Generator Communication ===")
    demonstrate_generator_communication()
    
    print("\n=== Custom Iterators ===")
    countdown = CountDown(5)
    print("Countdown:", list(countdown))
    
    powers_of_2 = InfiniteSequence(lambda x: 2 ** x)
    first_8_powers = [next(powers_of_2) for _ in range(8)]
    print(f"First 8 powers of 2: {first_8_powers}")
    
    print("\n=== Memory-Efficient Processing ===")
    memory_efficient_processing()
    
    print("\n=== Generator Coroutines ===")
    demonstrate_coroutines()
    
    print("\n=== Itertools Examples ===")
    itertools_examples()
    
    print("\n=== Performance Comparison ===")
    performance_comparison()

### Descriptors and properties

Descriptors are Python's way of customizing attribute access and are the mechanism behind properties, methods, and static methods.

```python
import weakref
from typing import Any, Type, Optional

# 1. Basic descriptor protocol
class LoggedAttribute:
    """Descriptor that logs attribute access"""
    
    def __init__(self, name: str = None):
        self.name = name
        self.private_name = None
    
    def __set_name__(self, owner: Type, name: str):
        """Called when descriptor is assigned to a class attribute"""
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        """Called when attribute is accessed"""
        if instance is None:
            return self
        
        value = getattr(instance, self.private_name, None)
        print(f"Getting {self.name}: {value}")
        return value
    
    def __set__(self, instance: Any, value: Any) -> None:
        """Called when attribute is set"""
        print(f"Setting {self.name}: {value}")
        setattr(instance, self.private_name, value)
    
    def __delete__(self, instance: Any) -> None:
        """Called when attribute is deleted"""
        print(f"Deleting {self.name}")
        delattr(instance, self.private_name)

# 2. Validation descriptor
class ValidatedAttribute:
    """Descriptor with validation"""
    
    def __init__(self, validator_func, error_msg="Invalid value"):
        self.validator = validator_func
        self.error_msg = error_msg
        self.name = None
        self.private_name = None
    
    def __set_name__(self, owner: Type, name: str):
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)
    
    def __set__(self, instance: Any, value: Any) -> None:
        if not self.validator(value):
            raise ValueError(f"{self.error_msg}: {value}")
        setattr(instance, self.private_name, value)

# 3. Type-enforcing descriptor
class TypedAttribute:
    """Descriptor that enforces type checking"""
    
    def __init__(self, expected_type: Type, allow_none: bool = False):
        self.expected_type = expected_type
        self.allow_none = allow_none
        self.name = None
        self.private_name = None
    
    def __set_name__(self, owner: Type, name: str):
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)
    
    def __set__(self, instance: Any, value: Any) -> None:
        if value is None and not self.allow_none:
            raise ValueError(f"{self.name} cannot be None")
        
        if value is not None and not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        
        setattr(instance, self.private_name, value)

# 4. Computed property descriptor
class ComputedProperty:
    """Descriptor for computed properties with caching"""
    
    def __init__(self, func, cache=True):
        self.func = func
        self.cache = cache
        self.name = None
        self.cache_name = None
    
    def __set_name__(self, owner: Type, name: str):
        self.name = name
        self.cache_name = f'_cached_{name}'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        if instance is None:
            return self
        
        if self.cache:
            # Check cache first
            if hasattr(instance, self.cache_name):
                return getattr(instance, self.cache_name)
            
            # Compute and cache
            value = self.func(instance)
            setattr(instance, self.cache_name, value)
            return value
        else:
            return self.func(instance)
    
    def __set__(self, instance: Any, value: Any) -> None:
        raise AttributeError(f"'{self.name}' is a computed property and cannot be set")
    
    def invalidate_cache(self, instance: Any) -> None:
        """Manually invalidate cache"""
        if hasattr(instance, self.cache_name):
            delattr(instance, self.cache_name)

# 5. WeakRef descriptor for avoiding circular references
class WeakRefAttribute:
    """Descriptor that stores weak references"""
    
    def __init__(self):
        self.name = None
        self.private_name = None
    
    def __set_name__(self, owner: Type, name: str):
        self.name = name
        self.private_name = f'_{name}_ref'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        if instance is None:
            return self
        
        weak_ref = getattr(instance, self.private_name, None)
        if weak_ref is None:
            return None
        
        # Try to get the object from weak reference
        obj = weak_ref()
        if obj is None:
            # Object was garbage collected
            delattr(instance, self.private_name)
            return None
        
        return obj
    
    def __set__(self, instance: Any, value: Any) -> None:
        if value is None:
            if hasattr(instance, self.private_name):
                delattr(instance, self.private_name)
        else:
            weak_ref = weakref.ref(value)
            setattr(instance, self.private_name, weak_ref)

# 6. Example classes using descriptors
class Person:
    """Example class using various descriptors"""
    
    # Logged attribute
    name = LoggedAttribute()
    
    # Validated attributes
    age = ValidatedAttribute(
        lambda x: isinstance(x, int) and 0 <= x <= 150,
        "Age must be an integer between 0 and 150"
    )
    
    email = ValidatedAttribute(
        lambda x: isinstance(x, str) and '@' in x,
        "Email must be a string containing '@'"
    )
    
    # Typed attributes
    salary = TypedAttribute(float, allow_none=True)
    
    def __init__(self, name: str, age: int, email: str, salary: float = None):
        self.name = name
        self.age = age
        self.email = email
        self.salary = salary
    
    # Computed property
    @ComputedProperty
    def display_name(self):
        """Computed property that combines name and age"""
        print("Computing display_name...")
        return f"{self.name} ({self.age} years old)"
    
    # Traditional property for comparison
    @property
    def age_group(self) -> str:
        """Traditional property using @property decorator"""
        if self.age < 18:
            return "Minor"
        elif self.age < 65:
            return "Adult"
        else:
            return "Senior"

# 7. Descriptor for database-like fields
class DatabaseField:
    """Descriptor that simulates database field behavior"""
    
    def __init__(self, field_type: Type, primary_key=False, nullable=True, default=None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.name = None
        self.private_name = None
    
    def __set_name__(self, owner: Type, name: str):
        self.name = name
        self.private_name = f'_{name}'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        if instance is None:
            return self
        
        value = getattr(instance, self.private_name, self.default)
        return value
    
    def __set__(self, instance: Any, value: Any) -> None:
        # Validation
        if value is None:
            if not self.nullable and not self.primary_key:
                raise ValueError(f"{self.name} cannot be None")
        elif not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be {self.field_type.__name__}")
        
        setattr(instance, self.private_name, value)

class User:
    """Example model class using database-like descriptors"""
    
    id = DatabaseField(int, primary_key=True, nullable=False)
    username = DatabaseField(str, nullable=False)
    email = DatabaseField(str, nullable=False)
    age = DatabaseField(int, nullable=True, default=0)
    is_active = DatabaseField(bool, default=True)
    
    def __init__(self, id: int, username: str, email: str, age: int = None, is_active: bool = True):
        self.id = id
        self.username = username
        self.email = email
        if age is not None:
            self.age = age
        self.is_active = is_active

# 8. Descriptor for method binding simulation
class BoundMethod:
    """Descriptor that simulates method binding"""
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, instance: Any, owner: Type):
        if instance is None:
            return self.func
        
        # Return bound method
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)
    
    def __set_name__(self, owner: Type, name: str):
        self.name = name

# 9. Properties with lazy loading
class LazyProperty:
    """Property descriptor with lazy loading"""
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.cache_name = f'_cached_{self.name}'
    
    def __get__(self, instance: Any, owner: Type) -> Any:
        if instance is None:
            return self
        
        # Check if already computed
        if hasattr(instance, self.cache_name):
            return getattr(instance, self.cache_name)
        
        # Compute and cache
        print(f"Lazy loading {self.name}...")
        value = self.func(instance)
        setattr(instance, self.cache_name, value)
        return value
    
    def __set__(self, instance: Any, value: Any) -> None:
        # Allow setting to override lazy computation
        setattr(instance, self.cache_name, value)
    
    def __delete__(self, instance: Any) -> None:
        # Clear cache
        if hasattr(instance, self.cache_name):
            delattr(instance, self.cache_name)

class ExpensiveResource:
    """Example class with lazy properties"""
    
    def __init__(self, size: int):
        self.size = size
    
    @LazyProperty
    def expensive_computation(self):
        """Expensive computation that's only done when needed"""
        import time
        time.sleep(0.1)  # Simulate expensive operation
        return sum(i ** 2 for i in range(self.size))
    
    @LazyProperty
    def cached_data(self):
        """Another expensive property"""
        return [i * 2 for i in range(self.size)]

# Usage examples
def descriptor_examples():
    print("=== Basic Descriptor Usage ===")
    person = Person("Alice", 30, "alice@example.com", 75000.0)
    
    print(f"Person created: {person.name}")
    print(f"Age group: {person.age_group}")
    print(f"Display name: {person.display_name}")  # First call - computed
    print(f"Display name: {person.display_name}")  # Second call - cached
    
    print("\n=== Validation Examples ===")
    try:
        person.age = -5  # Should raise error
    except ValueError as e:
        print(f"Validation error: {e}")
    
    try:
        person.email = "invalid_email"  # Should raise error
    except ValueError as e:
        print(f"Validation error: {e}")
    
    try:
        person.salary = "not a number"  # Should raise error
    except TypeError as e:
        print(f"Type error: {e}")
    
    print("\n=== Database Field Examples ===")
    user = User(1, "john_doe", "john@example.com", 25)
    print(f"User: {user.username}, Age: {user.age}, Active: {user.is_active}")
    
    try:
        user.username = None  # Should raise error
    except ValueError as e:
        print(f"Database validation error: {e}")
    
    print("\n=== Lazy Property Examples ===")
    resource = ExpensiveResource(1000)
    print("Resource created (no expensive computation yet)")
    
    # First access triggers computation
    print(f"Expensive result: {resource.expensive_computation}")
    
    # Second access uses cached value
    print(f"Cached result: {resource.expensive_computation}")
    
    print("\n=== WeakRef Descriptor Example ===")
    
    class Parent:
        child = WeakRefAttribute()
    
    class Child:
        def __init__(self, name):
            self.name = name
    
    parent = Parent()
    child = Child("test_child")
    parent.child = child
    
    print(f"Child name: {parent.child.name}")
    
    # Delete child reference
    del child
    print(f"Child after deletion: {parent.child}")  # Should be None

### Magic methods

Magic methods (dunder methods) enable operator overloading and customization of Python's built-in behavior.

```python
import functools
import operator
from typing import Any, Iterator, Union
from collections.abc import Sequence

# 1. Comprehensive magic methods example
class Vector:
    """A mathematical vector with comprehensive magic method support"""
    
    def __init__(self, *components):
        """Initialize vector with components"""
        self._components = tuple(components)
    
    # String representation
    def __repr__(self) -> str:
        """Unambiguous string representation"""
        return f"Vector{self._components}"
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        return f"({', '.join(map(str, self._components))})"
    
    def __format__(self, format_spec: str) -> str:
        """Custom formatting support"""
        if format_spec == 'polar':
            # Convert to polar coordinates (for 2D vector)
            if len(self._components) == 2:
                import math
                x, y = self._components
                r = math.sqrt(x**2 + y**2)
                theta = math.atan2(y, x)
                return f"r={r:.2f}, θ={math.degrees(theta):.1f}°"
        
        # Default formatting
        formatted_components = [format(c, format_spec) for c in self._components]
        return f"({', '.join(formatted_components)})"
    
    # Arithmetic operators
    def __add__(self, other):
        """Vector addition"""
        if isinstance(other, Vector):
            if len(self._components) != len(other._components):
                raise ValueError("Vectors must have same dimensions")
            return Vector(*(a + b for a, b in zip(self._components, other._components)))
        elif isinstance(other, (int, float)):
            return Vector(*(c + other for c in self._components))
        return NotImplemented
    
    def __radd__(self, other):
        """Reverse addition"""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Vector subtraction"""
        if isinstance(other, Vector):
            if len(self._components) != len(other._components):
                raise ValueError("Vectors must have same dimensions")
            return Vector(*(a - b for a, b in zip(self._components, other._components)))
        elif isinstance(other, (int, float)):
            return Vector(*(c - other for c in self._components))
        return NotImplemented
    
    def __rsub__(self, other):
        """Reverse subtraction"""
        if isinstance(other, (int, float)):
            return Vector(*(other - c for c in self._components))
        return NotImplemented
    
    def __mul__(self, other):
        """Scalar multiplication or dot product"""
        if isinstance(other, (int, float)):
            return Vector(*(c * other for c in self._components))
        elif isinstance(other, Vector):
            # Dot product
            if len(self._components) != len(other._components):
                raise ValueError("Vectors must have same dimensions")
            return sum(a * b for a, b in zip(self._components, other._components))
        return NotImplemented
    
    def __rmul__(self, other):
        """Reverse multiplication"""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Division by scalar"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Vector(*(c / other for c in self._components))
        return NotImplemented
    
    def __neg__(self):
        """Unary negation"""
        return Vector(*(-c for c in self._components))
    
    def __pos__(self):
        """Unary positive"""
        return Vector(*self._components)
    
    def __abs__(self):
        """Magnitude of vector"""
        return sum(c ** 2 for c in self._components) ** 0.5
    
    # Comparison operators
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if isinstance(other, Vector):
            return self._components == other._components
        return False
    
    def __lt__(self, other) -> bool:
        """Less than comparison (by magnitude)"""
        if isinstance(other, Vector):
            return abs(self) < abs(other)
        return NotImplemented
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison"""
        if isinstance(other, Vector):
            return abs(self) <= abs(other)
        return NotImplemented
    
    def __gt__(self, other) -> bool:
        """Greater than comparison"""
        if isinstance(other, Vector):
            return abs(self) > abs(other)
        return NotImplemented
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison"""
        if isinstance(other, Vector):
            return abs(self) >= abs(other)
        return NotImplemented
    
    # Container-like behavior
    def __len__(self) -> int:
        """Number of components"""
        return len(self._components)
    
    def __getitem__(self, index: int) -> float:
        """Get component by index"""
        return self._components[index]
    
    def __setitem__(self, index: int, value: float) -> None:
        """Set component by index (creates new vector since tuples are immutable)"""
        components = list(self._components)
        components[index] = value
        self._components = tuple(components)
    
    def __iter__(self) -> Iterator[float]:
        """Make vector iterable"""
        return iter(self._components)
    
    def __contains__(self, value) -> bool:
        """Check if value is a component"""
        return value in self._components
    
    # Hashing
    def __hash__(self) -> int:
        """Hash function"""
        return hash(self._components)
    
    # Boolean context
    def __bool__(self) -> bool:
        """Boolean evaluation (True if not zero vector)"""
        return any(c != 0 for c in self._components)

# 2. Custom sequence class
class ImmutableList:
    """Immutable list-like container with full sequence protocol"""
    
    def __init__(self, items=()):
        self._items = tuple(items)
    
    def __repr__(self) -> str:
        return f"ImmutableList({list(self._items)})"
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return ImmutableList(self._items[index])
        return self._items[index]
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)
    
    def __contains__(self, item) -> bool:
        return item in self._items
    
    def __add__(self, other):
        if isinstance(other, ImmutableList):
            return ImmutableList(self._items + other._items)
        elif isinstance(other, (list, tuple)):
            return ImmutableList(self._items + tuple(other))
        return NotImplemented
    
    def __mul__(self, other: int):
        if isinstance(other, int):
            return ImmutableList(self._items * other)
        return NotImplemented
    
    def __rmul__(self, other: int):
        return self.__mul__(other)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ImmutableList):
            return self._items == other._items
        elif isinstance(other, (list, tuple)):
            return self._items == tuple(other)
        return False
    
    # Additional sequence methods
    def count(self, value) -> int:
        return self._items.count(value)
    
    def index(self, value, start=0, stop=None) -> int:
        return self._items.index(value, start, stop or len(self._items))

# 3. Context manager using magic methods
class ManagedResource:
    """Resource manager using context manager protocol"""
    
    def __init__(self, name: str):
        self.name = name
        self.acquired = False
    
    def __enter__(self):
        print(f"Acquiring resource: {self.name}")
        self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Releasing resource: {self.name}")
        self.acquired = False
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

# 4. Descriptor-like behavior with magic methods
class AutoProperty:
    """Automatic property creation using __getattr__ and __setattr__"""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._properties = set(kwargs.keys())
    
    def __getattr__(self, name: str) -> Any:
        """Called when attribute is not found"""
        if name.startswith('get_'):
            prop_name = name[4:]  # Remove 'get_' prefix
            if prop_name in self._properties:
                return lambda: getattr(self, prop_name)
        elif name.startswith('set_'):
            prop_name = name[4:]  # Remove 'set_' prefix
            if prop_name in self._properties:
                return lambda value: setattr(self, prop_name, value)
        
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute setting"""
        if name.startswith('_') or name in ['_properties']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, '_properties'):
                self._properties.add(name)
            super().__setattr__(name, value)

# 5. Callable object
class Multiplier:
    """Callable object that multiplies by a factor"""
    
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, value: float) -> float:
        """Make object callable"""
        return value * self.factor
    
    def __repr__(self) -> str:
        return f"Multiplier({self.factor})"

# 6. Metaclass behavior with magic methods
class AttributeLogger:
    """Class that logs all attribute access"""
    
    def __init__(self, name: str):
        self.name = name
        self._access_log = []
    
    def __getattribute__(self, name: str) -> Any:
        """Intercept all attribute access"""
        if name.startswith('_') or name in ['name', 'log_access']:
            return super().__getattribute__(name)
        
        # Log access
        self._access_log.append(f"GET: {name}")
        return super().__getattribute__(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept all attribute setting"""
        if hasattr(self, '_access_log') and not name.startswith('_'):
            self._access_log.append(f"SET: {name} = {value}")
        super().__setattr__(name, value)
    
    def get_access_log(self) -> list:
        """Get the access log"""
        return self._access_log.copy()

# 7. Custom numeric type
class Fraction:
    """Fraction class with complete numeric protocol"""
    
    def __init__(self, numerator: int, denominator: int = 1):
        if denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero")
        
        # Simplify fraction
        gcd = self._gcd(abs(numerator), abs(denominator))
        self.numerator = numerator // gcd
        self.denominator = denominator // gcd
        
        # Ensure denominator is positive
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
    
    def _gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def __repr__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __add__(self, other):
        if isinstance(other, Fraction):
            num = self.numerator * other.denominator + other.numerator * self.denominator
            den = self.denominator * other.denominator
            return Fraction(num, den)
        elif isinstance(other, int):
            return Fraction(self.numerator + other * self.denominator, self.denominator)
        return NotImplemented
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Fraction):
            num = self.numerator * other.denominator - other.numerator * self.denominator
            den = self.denominator * other.denominator
            return Fraction(num, den)
        elif isinstance(other, int):
            return Fraction(self.numerator - other * self.denominator, self.denominator)
        return NotImplemented
    
    def __rsub__(self, other):
        if isinstance(other, int):
            return Fraction(other * self.denominator - self.numerator, self.denominator)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Fraction):
            return Fraction(self.numerator * other.numerator, 
                          self.denominator * other.denominator)
        elif isinstance(other, int):
            return Fraction(self.numerator * other, self.denominator)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Fraction):
            return Fraction(self.numerator * other.denominator, 
                          self.denominator * other.numerator)
        elif isinstance(other, int):
            return Fraction(self.numerator, self.denominator * other)
        return NotImplemented
    
    def __rtruediv__(self, other):
        if isinstance(other, int):
            return Fraction(other * self.denominator, self.numerator)
        return NotImplemented
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Fraction):
            return (self.numerator * other.denominator == 
                   other.numerator * self.denominator)
        elif isinstance(other, int):
            return self.numerator == other * self.denominator
        return False
    
    def __lt__(self, other) -> bool:
        if isinstance(other, Fraction):
            return (self.numerator * other.denominator < 
                   other.numerator * self.denominator)
        elif isinstance(other, int):
            return self.numerator < other * self.denominator
        return NotImplemented
    
    def __float__(self) -> float:
        """Convert to float"""
        return self.numerator / self.denominator
    
    def __int__(self) -> int:
        """Convert to int"""
        return self.numerator // self.denominator
    
    def __abs__(self):
        """Absolute value"""
        return Fraction(abs(self.numerator), self.denominator)
    
    def __neg__(self):
        """Negation"""
        return Fraction(-self.numerator, self.denominator)

# Usage examples
def magic_methods_examples():
    print("=== Vector Examples ===")
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 + v2: {v1 + v2}")
    print(f"v1 * 2: {v1 * 2}")
    print(f"v1 · v2 (dot product): {v1 * v2}")
    print(f"|v1| (magnitude): {abs(v1)}")
    print(f"v1 > v2: {v1 > v2}")
    print(f"Formatted v1: {format(v1, 'polar')}")
    
    print("\n=== ImmutableList Examples ===")
    il1 = ImmutableList([1, 2, 3])
    il2 = ImmutableList([4, 5, 6])
    
    print(f"il1: {il1}")
    print(f"il1 + il2: {il1 + il2}")
    print(f"il1 * 2: {il1 * 2}")
    print(f"2 in il1: {2 in il1}")
    print(f"il1[1:]: {il1[1:]}")
    
    print("\n=== Context Manager Example ===")
    with ManagedResource("database") as resource:
        print(f"Using {resource.name}")
    
    print("\n=== AutoProperty Example ===")
    obj = AutoProperty(name="test", value=42)
    print(f"obj.name: {obj.name}")
    
    # Dynamic getter/setter
    get_name = obj.get_name
    set_value = obj.set_value
    print(f"Dynamic get_name(): {get_name()}")
    set_value(100)
    print(f"obj.value after set_value(100): {obj.value}")
    
    print("\n=== Callable Object Example ===")
    double = Multiplier(2)
    triple = Multiplier(3)
    
    print(f"double(5): {double(5)}")
    print(f"triple(4): {triple(4)}")
    
    print("\n=== Attribute Logger Example ===")
    logger = AttributeLogger("test_object")
    logger.data = "some data"
    logger.number = 42
    value = logger.data
    
    print(f"Access log: {logger.get_access_log()}")
    
    print("\n=== Fraction Examples ===")
    f1 = Fraction(3, 4)
    f2 = Fraction(1, 2)
    
    print(f"f1: {f1}")
    print(f"f2: {f2}")
    print(f"f1 + f2: {f1 + f2}")
    print(f"f1 * f2: {f1 * f2}")
    print(f"f1 / f2: {f1 / f2}")
    print(f"f1 == 0.75: {f1 == Fraction(3, 4)}")
    print(f"float(f1): {float(f1)}")
    print(f"int(f1): {int(f1)}")

## Python Memory Management

### Reference counting

Python's primary memory management mechanism is reference counting, supplemented by a cycle detector for garbage collection.

```python
import sys
import gc
import weakref
from typing import Any, Optional

# 1. Basic reference counting demonstration
def reference_counting_basics():
    """Demonstrate basic reference counting behavior"""
    
    # Create an object
    data = [1, 2, 3, 4, 5]
    print(f"Initial reference count: {sys.getrefcount(data)}")
    
    # Create additional references
    ref1 = data
    print(f"After creating ref1: {sys.getrefcount(data)}")
    
    ref2 = data
    print(f"After creating ref2: {sys.getrefcount(data)}")
    
    # Delete references
    del ref1
    print(f"After deleting ref1: {sys.getrefcount(data)}")
    
    del ref2
    print(f"After deleting ref2: {sys.getrefcount(data)}")
    
    # Note: sys.getrefcount() itself creates a temporary reference,
    # so the count is always 1 higher than expected

def reference_counting_edge_cases():
    """Show edge cases in reference counting"""
    
    # Container holding references
    container = []
    data = "hello world"
    
    print(f"String reference count: {sys.getrefcount(data)}")
    
    container.append(data)
    print(f"After adding to container: {sys.getrefcount(data)}")
    
    # Multiple containers
    another_container = [data, data, data]
    print(f"After adding to multiple containers: {sys.getrefcount(data)}")
    
    # Function arguments create temporary references
    def print_refcount(obj):
        print(f"Inside function: {sys.getrefcount(obj)}")
    
    print_refcount(data)
    print(f"After function call: {sys.getrefcount(data)}")

# 2. Circular references and garbage collection
class Node:
    """Simple node class to demonstrate circular references"""
    
    def __init__(self, value: Any):
        self.value = value
        self.children: list['Node'] = []
        self.parent: Optional['Node'] = None
    
    def add_child(self, child: 'Node'):
        child.parent = self
        self.children.append(child)
    
    def __del__(self):
        print(f"Node {self.value} is being deleted")

def demonstrate_circular_references():
    """Show how circular references work and are collected"""
    
    print("=== Creating circular references ===")
    
    # Create nodes with circular references
    root = Node("root")
    child1 = Node("child1")
    child2 = Node("child2")
    
    root.add_child(child1)
    root.add_child(child2)
    
    # Create a cycle: child1 points back to root
    child1.parent = root
    
    print(f"Root references: {sys.getrefcount(root)}")
    print(f"Child1 references: {sys.getrefcount(child1)}")
    
    # Delete our references
    del root, child1, child2
    
    print("After deleting variables...")
    
    # Force garbage collection
    collected = gc.collect()
    print(f"Garbage collector collected {collected} objects")

# 3. Weak references to avoid cycles
class WeakRefExample:
    """Example using weak references to avoid cycles"""
    
    def __init__(self, name: str):
        self.name = name
        self.children: list['WeakRefExample'] = []
        self._parent_ref: Optional[weakref.ref] = None
    
    @property
    def parent(self) -> Optional['WeakRefExample']:
        if self._parent_ref is None:
            return None
        return self._parent_ref()
    
    @parent.setter
    def parent(self, parent: Optional['WeakRefExample']):
        if parent is None:
            self._parent_ref = None
        else:
            self._parent_ref = weakref.ref(parent)
    
    def add_child(self, child: 'WeakRefExample'):
        child.parent = self
        self.children.append(child)
    
    def __del__(self):
        print(f"WeakRefExample {self.name} is being deleted")

def demonstrate_weak_references():
    """Show weak references in action"""
    
    print("=== Using weak references ===")
    
    root = WeakRefExample("root")
    child = WeakRefExample("child")
    
    root.add_child(child)
    
    print(f"Child's parent: {child.parent.name if child.parent else None}")
    
    # Delete root
    del root
    
    print(f"Child's parent after deleting root: {child.parent}")
    
    # Clean up
    del child

# 4. Memory profiling and optimization
import tracemalloc

def memory_profiling_example():
    """Demonstrate memory profiling with tracemalloc"""
    
    # Start tracing
    tracemalloc.start()
    
    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Allocate some memory
    data = []
    for i in range(10000):
        data.append([i] * 100)
    
    # Take second snapshot
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare snapshots
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 5 memory allocations:")
    for stat in top_stats[:5]:
        print(stat)
    
    # Clean up
    del data
    tracemalloc.stop()

# 5. Memory-efficient data structures
def memory_efficient_structures():
    """Compare memory usage of different data structures"""
    
    # Regular list vs tuple
    list_data = [i for i in range(1000)]
    tuple_data = tuple(i for i in range(1000))
    
    print(f"List size: {sys.getsizeof(list_data)} bytes")
    print(f"Tuple size: {sys.getsizeof(tuple_data)} bytes")
    
    # String vs bytes
    text = "Hello, World!" * 1000
    byte_data = text.encode('utf-8')
    
    print(f"String size: {sys.getsizeof(text)} bytes")
    print(f"Bytes size: {sys.getsizeof(byte_data)} bytes")
    
    # Dict vs namedtuple
    from collections import namedtuple
    
    # Regular dict
    person_dict = {'name': 'John', 'age': 30, 'city': 'NYC'}
    print(f"Dict size: {sys.getsizeof(person_dict)} bytes")
    
    # Namedtuple
    Person = namedtuple('Person', ['name', 'age', 'city'])
    person_tuple = Person('John', 30, 'NYC')
    print(f"Namedtuple size: {sys.getsizeof(person_tuple)} bytes")
    
    # Slots for classes
    class RegularClass:
        def __init__(self, name, age, city):
            self.name = name
            self.age = age
            self.city = city
    
    class SlottedClass:
        __slots__ = ['name', 'age', 'city']
        
        def __init__(self, name, age, city):
            self.name = name
            self.age = age
            self.city = city
    
    regular_instance = RegularClass('John', 30, 'NYC')
    slotted_instance = SlottedClass('John', 30, 'NYC')
    
    print(f"Regular class instance: {sys.getsizeof(regular_instance)} + {sys.getsizeof(regular_instance.__dict__)} bytes")
    print(f"Slotted class instance: {sys.getsizeof(slotted_instance)} bytes")

# 6. Object pooling for memory optimization
class ObjectPool:
    """Simple object pool implementation"""
    
    def __init__(self, factory_func, max_size=10):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
    
    def get(self):
        """Get object from pool or create new one"""
        if self.pool:
            obj = self.pool.pop()
        else:
            obj = self.factory_func()
        
        self.in_use.add(id(obj))
        return obj
    
    def return_object(self, obj):
        """Return object to pool"""
        obj_id = id(obj)
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
            
            if len(self.pool) < self.max_size:
                # Reset object state if needed
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
            # If pool is full, object will be garbage collected
    
    def pool_info(self):
        """Get pool statistics"""
        return {
            'pool_size': len(self.pool),
            'in_use': len(self.in_use),
            'total_created': len(self.pool) + len(self.in_use)
        }

class PooledResource:
    """Example resource that can be pooled"""
    
    def __init__(self):
        self.data = []
        print("Creating new PooledResource")
    
    def reset(self):
        """Reset resource state for reuse"""
        self.data.clear()
        print("Resetting PooledResource")
    
    def add_data(self, item):
        self.data.append(item)

def demonstrate_object_pooling():
    """Show object pooling in action"""
    
    print("=== Object Pooling ===")
    
    # Create pool
    pool = ObjectPool(PooledResource, max_size=3)
    
    # Get objects from pool
    obj1 = pool.get()
    obj2 = pool.get()
    obj3 = pool.get()
    
    print(f"Pool info after getting 3 objects: {pool.pool_info()}")
    
    # Use objects
    obj1.add_data("data1")
    obj2.add_data("data2")
    
    # Return objects to pool
    pool.return_object(obj1)
    pool.return_object(obj2)
    
    print(f"Pool info after returning 2 objects: {pool.pool_info()}")
    
    # Get object again (should reuse existing)
    obj4 = pool.get()
    print(f"obj4 data (should be empty): {obj4.data}")
    
    pool.return_object(obj3)
    pool.return_object(obj4)

# 7. Memory leak detection
def find_memory_leaks():
    """Simple memory leak detection"""
    
    # Get initial object counts
    initial_objects = len(gc.get_objects())
    
    def potentially_leaky_function():
        # Create objects that might leak
        data = []
        for i in range(1000):
            item = {'id': i, 'data': [j for j in range(100)]}
            data.append(item)
        
        # Simulate keeping references accidentally
        global leaked_data
        leaked_data = data[:10]  # Oops, keeping references
        
        return len(data)
    
    # Call function multiple times
    for _ in range(5):
        potentially_leaky_function()
    
    # Force garbage collection
    gc.collect()
    
    # Check object counts
    final_objects = len(gc.get_objects())
    
    print(f"Initial objects: {initial_objects}")
    print(f"Final objects: {final_objects}")
    print(f"Object increase: {final_objects - initial_objects}")
    
    # Find objects by type
    object_types = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
    
    # Show most common object types
    common_types = sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nMost common object types:")
    for obj_type, count in common_types:
        print(f"  {obj_type}: {count}")
    
    # Clean up the leak
    del leaked_data

# Usage examples
def memory_management_examples():
    print("=== Reference Counting Basics ===")
    reference_counting_basics()
    
    print("\n=== Reference Counting Edge Cases ===")
    reference_counting_edge_cases()
    
    print("\n=== Circular References ===")
    demonstrate_circular_references()
    
    print("\n=== Weak References ===")
    demonstrate_weak_references()
    
    print("\n=== Memory Profiling ===")
    memory_profiling_example()
    
    print("\n=== Memory-Efficient Structures ===")
    memory_efficient_structures()
    
    print("\n=== Object Pooling ===")
    demonstrate_object_pooling()
    
    print("\n=== Memory Leak Detection ===")
    find_memory_leaks()

### Garbage collection

Python's garbage collector handles cyclic references that reference counting cannot resolve.

```python
import gc
import weakref
import time
from typing import List, Optional, Any

# 1. Understanding garbage collection
def gc_basics():
    """Demonstrate basic garbage collection concepts"""
    
    print("=== Garbage Collection Basics ===")
    
    # Check if garbage collection is enabled
    print(f"GC enabled: {gc.isenabled()}")
    
    # Get GC thresholds
    print(f"GC thresholds: {gc.get_threshold()}")
    
    # Get current GC counts
    print(f"GC counts: {gc.get_count()}")
    
    # Get GC statistics
    print(f"GC stats: {gc.get_stats()}")

class GCTestNode:
    """Node class for testing garbage collection"""
    
    def __init__(self, name: str):
        self.name = name
        self.refs: List['GCTestNode'] = []
        self.data = list(range(1000))  # Some data to make collection visible
    
    def add_ref(self, other: 'GCTestNode'):
        self.refs.append(other)
    
    def __del__(self):
        print(f"GCTestNode {self.name} collected")

def create_cycles():
    """Create circular references for GC testing"""
    
    # Create a cycle
    node1 = GCTestNode("Node1")
    node2 = GCTestNode("Node2")
    node3 = GCTestNode("Node3")
    
    # Create circular references
    node1.add_ref(node2)
    node2.add_ref(node3)
    node3.add_ref(node1)
    
    return node1, node2, node3

def demonstrate_gc_collection():
    """Show garbage collection in action"""
    
    print("\n=== Creating Cycles ===")
    
    # Get initial stats
    initial_objects = len(gc.get_objects())
    initial_count = gc.get_count()
    
    print(f"Initial objects: {initial_objects}")
    print(f"Initial GC count: {initial_count}")
    
    # Create cycles
    cycles = []
    for i in range(100):
        cycle = create_cycles()
        cycles.append(cycle)
    
    after_creation = len(gc.get_objects())
    print(f"Objects after creation: {after_creation}")
    print(f"Objects created: {after_creation - initial_objects}")
    
    # Delete references (but cycles remain)
    del cycles
    
    after_deletion = len(gc.get_objects())
    print(f"Objects after deletion: {after_deletion}")
    
    # Manual garbage collection
    collected = gc.collect()
    print(f"Objects collected: {collected}")
    
    final_objects = len(gc.get_objects())
    print(f"Final objects: {final_objects}")

# 2. Generational garbage collection
def demonstrate_generations():
    """Show how generational GC works"""
    
    print("\n=== Generational GC ===")
    
    # Objects start in generation 0
    def create_short_lived_objects():
        temp_objects = []
        for i in range(1000):
            temp_objects.append([i] * 10)
        return len(temp_objects)
    
    def create_long_lived_objects():
        long_lived = []
        for i in range(100):
            long_lived.append({'id': i, 'data': list(range(100))})
        return long_lived
    
    # Before creating objects
    print(f"Initial GC counts: {gc.get_count()}")
    
    # Create short-lived objects
    create_short_lived_objects()
    print(f"After short-lived objects: {gc.get_count()}")
    
    # Create long-lived objects
    long_lived = create_long_lived_objects()
    print(f"After long-lived objects: {gc.get_count()}")
    
    # Force collection of different generations
    print(f"Gen 0 collected: {gc.collect(0)}")
    print(f"Gen 1 collected: {gc.collect(1)}")
    print(f"Gen 2 collected: {gc.collect(2)}")
    
    print(f"Final GC counts: {gc.get_count()}")

# 3. GC callbacks and debugging
def gc_callback(phase, info):
    """Callback function for GC events"""
    print(f"GC {phase}: {info}")

def demonstrate_gc_debugging():
    """Show GC debugging capabilities"""
    
    print("\n=== GC Debugging ===")
    
    # Set debug flags
    old_debug = gc.get_debug()
    gc.set_debug(gc.DEBUG_STATS)
    
    # Set callback
    gc.callbacks.append(gc_callback)
    
    try:
        # Create some objects to trigger GC
        data = []
        for i in range(1000):
            node = GCTestNode(f"debug_{i}")
            if i > 0:
                node.add_ref(data[i-1])
            data.append(node)
        
        # Create cycles
        if len(data) > 10:
            data[0].add_ref(data[-1])
        
        # Force collection
        collected = gc.collect()
        print(f"Debug collection collected: {collected}")
        
    finally:
        # Restore debug settings
        gc.set_debug(old_debug)
        gc.callbacks.clear()

# 4. Weak references for breaking cycles
class WeakRefManager:
    """Manager that uses weak references to avoid cycles"""
    
    def __init__(self):
        self._objects = weakref.WeakSet()
        self._callbacks = weakref.WeakKeyDictionary()
    
    def register(self, obj, callback=None):
        """Register an object with optional cleanup callback"""
        self._objects.add(obj)
        if callback:
            self._callbacks[obj] = callback
    
    def unregister(self, obj):
        """Unregister an object"""
        self._objects.discard(obj)
        self._callbacks.pop(obj, None)
    
    def get_objects(self):
        """Get list of currently registered objects"""
        return list(self._objects)
    
    def cleanup_dead_refs(self):
        """Clean up any dead references"""
        # WeakSet automatically handles this
        return len(self._objects)

class ManagedObject:
    """Object that registers itself with WeakRefManager"""
    
    _manager = WeakRefManager()
    
    def __init__(self, name: str):
        self.name = name
        self._manager.register(self, self._cleanup_callback)
        print(f"Created ManagedObject {name}")
    
    def _cleanup_callback(self):
        print(f"Cleanup callback for {self.name}")
    
    def __del__(self):
        print(f"ManagedObject {self.name} deleted")
    
    @classmethod
    def get_all_objects(cls):
        return cls._manager.get_objects()

def demonstrate_weak_ref_management():
    """Show weak reference management"""
    
    print("\n=== Weak Reference Management ===")
    
    # Create managed objects
    obj1 = ManagedObject("obj1")
    obj2 = ManagedObject("obj2")
    obj3 = ManagedObject("obj3")
    
    print(f"Managed objects: {len(ManagedObject.get_all_objects())}")
    
    # Delete some objects
    del obj2
    
    print(f"After deleting obj2: {len(ManagedObject.get_all_objects())}")
    
    # Force garbage collection
    gc.collect()
    
    print(f"After GC: {len(ManagedObject.get_all_objects())}")
    
    # Clean up
    del obj1, obj3
    gc.collect()
    
    print(f"Final count: {len(ManagedObject.get_all_objects())}")

# 5. Custom finalizers
class CustomFinalizer:
    """Object with custom finalization"""
    
    def __init__(self, name: str, resource: Any):
        self.name = name
        self.resource = resource
        self._finalizer = weakref.finalize(self, self._cleanup, resource, name)
    
    @staticmethod
    def _cleanup(resource, name):
        """Cleanup function that runs when object is collected"""
        print(f"Finalizing {name} and cleaning up resource: {resource}")
        # Cleanup resource here
    
    def close(self):
        """Explicit cleanup"""
        if self._finalizer.detach():
            self._cleanup(self.resource, self.name)
    
    def __del__(self):
        print(f"CustomFinalizer {self.name} __del__ called")

def demonstrate_finalizers():
    """Show custom finalizers in action"""
    
    print("\n=== Custom Finalizers ===")
    
    # Create objects with finalizers
    obj1 = CustomFinalizer("obj1", "database_connection")
    obj2 = CustomFinalizer("obj2", "file_handle")
    
    # Explicit cleanup
    obj1.close()
    
    # Implicit cleanup through GC
    del obj2
    gc.collect()
    
    print("Finalizer demo completed")

# 6. GC optimization strategies
def gc_optimization_examples():
    """Show GC optimization strategies"""
    
    print("\n=== GC Optimization ===")
    
    # Strategy 1: Disable GC during critical sections
    def critical_operation():
        """Operation that should not be interrupted by GC"""
        gc.disable()
        try:
            # Critical code here
            result = sum(i ** 2 for i in range(100000))
            return result
        finally:
            gc.enable()
    
    # Strategy 2: Manual GC at appropriate times
    def batch_processing():
        """Process data in batches with manual GC"""
        batch_size = 1000
        total_processed = 0
        
        for batch_num in range(10):
            # Process batch
            batch_data = []
            for i in range(batch_size):
                batch_data.append({'id': total_processed + i, 'data': list(range(10))})
            
            # Process the batch
            processed = len(batch_data)
            total_processed += processed
            
            # Manual GC between batches
            if batch_num % 3 == 0:
                collected = gc.collect()
                print(f"After batch {batch_num}: collected {collected} objects")
        
        return total_processed
    
    # Strategy 3: Adjust GC thresholds
    def optimize_gc_thresholds():
        """Optimize GC thresholds for specific workload"""
        
        # Get current thresholds
        current = gc.get_threshold()
        print(f"Current thresholds: {current}")
        
        # Set more aggressive thresholds
        gc.set_threshold(100, 10, 10)
        print(f"Set aggressive thresholds: {gc.get_threshold()}")
        
        # Create some objects
        data = [list(range(100)) for _ in range(1000)]
        
        # Reset to default
        gc.set_threshold(*current)
        print(f"Restored thresholds: {gc.get_threshold()}")
        
        return len(data)
    
    # Run optimization examples
    result1 = critical_operation()
    print(f"Critical operation result: {result1}")
    
    result2 = batch_processing()
    print(f"Batch processing result: {result2}")
    
    result3 = optimize_gc_thresholds()
    print(f"Threshold optimization result: {result3}")

# Usage examples
def garbage_collection_examples():
    gc_basics()
    demonstrate_gc_collection()
    demonstrate_generations()
    demonstrate_gc_debugging()
    demonstrate_weak_ref_management()
    demonstrate_finalizers()
    gc_optimization_examples()

### Memory optimization

Optimize Python memory usage through various techniques and data structures.

```python
import sys
import array
import struct
from collections import deque, namedtuple
from typing import Iterator, List
import mmap

# 1. Using __slots__ to reduce memory overhead
class RegularClass:
    """Regular class with __dict__"""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class SlottedClass:
    """Class with __slots__ to save memory"""
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def compare_slots_memory():
    """Compare memory usage of regular vs slotted classes"""
    
    # Create instances
    regular = RegularClass(1, 2, 3)
    slotted = SlottedClass(1, 2, 3)
    
    print("=== __slots__ Memory Comparison ===")
    print(f"Regular class instance: {sys.getsizeof(regular)} bytes")
    print(f"Regular class __dict__: {sys.getsizeof(regular.__dict__)} bytes")
    print(f"Total regular: {sys.getsizeof(regular) + sys.getsizeof(regular.__dict__)} bytes")
    print(f"Slotted class instance: {sys.getsizeof(slotted)} bytes")
    
    # Memory savings calculation
    regular_total = sys.getsizeof(regular) + sys.getsizeof(regular.__dict__)
    slotted_total = sys.getsizeof(slotted)
    savings = regular_total - slotted_total
    print(f"Memory savings: {savings} bytes ({savings/regular_total*100:.1f}%)")

# 2. Efficient data structures
def efficient_data_structures():
    """Compare memory efficiency of different data structures"""
    
    print("\n=== Efficient Data Structures ===")
    
    # Arrays vs lists for numeric data
    size = 10000
    
    # Python list
    py_list = list(range(size))
    
    # Array module
    int_array = array.array('i', range(size))  # 'i' = signed int
    
    # Bytes for small integers
    byte_data = bytes(range(256))
    
    print(f"Python list ({size} ints): {sys.getsizeof(py_list)} bytes")
    print(f"Array module ({size} ints): {sys.getsizeof(int_array)} bytes")
    print(f"Bytes (256 values): {sys.getsizeof(byte_data)} bytes")
    
    # String vs bytes
    text = "Hello, World! " * 1000
    byte_text = text.encode('utf-8')
    
    print(f"String: {sys.getsizeof(text)} bytes")
    print(f"Bytes: {sys.getsizeof(byte_text)} bytes")
    
    # Tuple vs list for immutable data
    tuple_data = tuple(range(1000))
    list_data = list(range(1000))
    
    print(f"Tuple: {sys.getsizeof(tuple_data)} bytes")
    print(f"List: {sys.getsizeof(list_data)} bytes")

# 3. Generator-based memory optimization
class MemoryEfficientRange:
    """Memory-efficient range-like class"""
    
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step
    
    def __iter__(self):
        current = self.start
        while current < self.stop:
            yield current
            current += self.step
    
    def __len__(self):
        return max(0, (self.stop - self.start + self.step - 1) // self.step)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return MemoryEfficientRange(
                self.start + start * self.step,
                self.start + stop * self.step,
                self.step * step
            )
        
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        
        return self.start + index * self.step

def compare_range_memory():
    """Compare memory usage of different range implementations"""
    
    print("\n=== Range Memory Comparison ===")
    
    size = 1000000
    
    # Built-in range (efficient)
    range_obj = range(size)
    
    # List (memory-intensive)
    list_range = list(range(size))
    
    # Custom efficient range
    efficient_range = MemoryEfficientRange(0, size)
    
    print(f"Built-in range: {sys.getsizeof(range_obj)} bytes")
    print(f"List range: {sys.getsizeof(list_range)} bytes")
    print(f"Efficient range: {sys.getsizeof(efficient_range)} bytes")

# 4. Lazy loading and caching
class LazyLoader:
    """Lazy loading container"""
    
    def __init__(self, data_source):
        self.data_source = data_source
        self._cache = {}
        self._loaded = set()
    
    def __getitem__(self, key):
        if key not in self._loaded:
            self._cache[key] = self.data_source(key)
            self._loaded.add(key)
        return self._cache[key]
    
    def __contains__(self, key):
        return True  # Assume all keys are valid
    
    def memory_usage(self):
        return sys.getsizeof(self._cache) + sys.getsizeof(self._loaded)

def expensive_data_generator(key):
    """Simulate expensive data generation"""
    return f"expensive_data_{key}" * 100

def demonstrate_lazy_loading():
    """Show lazy loading in action"""
    
    print("\n=== Lazy Loading ===")
    
    loader = LazyLoader(expensive_data_generator)
    
    print(f"Initial memory: {loader.memory_usage()} bytes")
    
    # Access some data
    data1 = loader[1]
    data2 = loader[5]
    data3 = loader[10]
    
    print(f"After loading 3 items: {loader.memory_usage()} bytes")
    
    # Access same data (should use cache)
    data1_again = loader[1]
    
    print(f"After cache hit: {loader.memory_usage()} bytes")

# 5. Memory mapping for large files
def memory_mapping_example():
    """Demonstrate memory mapping for large file processing"""
    
    print("\n=== Memory Mapping ===")
    
    # Create a large test file
    filename = 'large_test_file.txt'
    
    # Write test data
    with open(filename, 'w') as f:
        for i in range(100000):
            f.write(f"Line {i}: This is test data for memory mapping.\n")
    
    # Regular file reading (loads all into memory)
    with open(filename, 'r') as f:
        regular_content = f.read()
    
    print(f"Regular file content size: {sys.getsizeof(regular_content)} bytes")
    
    # Memory mapping (doesn't load all into memory)
    with open(filename, 'r+b') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            print(f"Memory mapped object size: {sys.getsizeof(mmapped_file)} bytes")
            
            # Read specific parts
            first_100_bytes = mmapped_file[:100]
            print(f"First 100 bytes: {first_100_bytes[:50]}...")
    
    # Clean up
    import os
    os.remove(filename)

# 6. Interning for string optimization
def string_interning_demo():
    """Demonstrate string interning for memory optimization"""
    
    print("\n=== String Interning ===")
    
    # Small strings are automatically interned
    a = "hello"
    b = "hello"
    print(f"Small strings same object: {a is b}")
    
    # Larger strings are not automatically interned
    large_a = "hello" * 1000
    large_b = "hello" * 1000
    print(f"Large strings same object: {large_a is large_b}")
    
    # Manual interning
    interned_a = sys.intern(large_a)
    interned_b = sys.intern(large_b)
    print(f"Manually interned same object: {interned_a is interned_b}")
    
    # Memory savings from interning
    original_size = sys.getsizeof(large_a) + sys.getsizeof(large_b)
    interned_size = sys.getsizeof(interned_a) + sys.getsizeof(interned_b)
    
    print(f"Original size: {original_size} bytes")
    print(f"Interned size: {interned_size} bytes")
    print(f"Savings: {original_size - interned_size} bytes")

# 7. Circular references
def circular_reference_optimization():
    """Show how to avoid circular references"""
    
    print("\n=== Circular Reference Optimization ===")
    
    # Problem: Circular references
    class Parent:
        def __init__(self, name):
            self.name = name
            self.children = []
        
        def add_child(self, child):
            child.parent = self
            self.children.append(child)
    
    class Child:
        def __init__(self, name):
            self.name = name
            self.parent = None
    
    # Solution: Use weak references
    import weakref
    
    class OptimizedParent:
        def __init__(self, name):
            self.name = name
            self.children = []
        
        def add_child(self, child):
            child._parent_ref = weakref.ref(self)
            self.children.append(child)
    
    class OptimizedChild:
        def __init__(self, name):
            self.name = name
            self._parent_ref = None
        
        @property
        def parent(self):
            if self._parent_ref is None:
                return None
            return self._parent_ref()
    
    # Create circular structure
    parent = OptimizedParent("parent")
    child = OptimizedChild("child")
    parent.add_child(child)
    
    print(f"Child's parent: {child.parent.name if child.parent else None}")
    
    # Delete parent
    del parent
    print(f"Child's parent after deletion: {child.parent}")

# 8. Memory profiling tools
def memory_profiling_tools():
    """Demonstrate memory profiling techniques"""
    
    print("\n=== Memory Profiling Tools ===")
    
    import tracemalloc
    
    # Start memory tracking
    tracemalloc.start()
    
    # Take initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Allocate memory
    big_list = []
    for i in range(10000):
        big_list.append([j for j in range(100)])
    
    # Take second snapshot
    snapshot2 = tracemalloc.take_snapshot()
    
    # Analyze differences
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 3 memory allocations:")
    for stat in top_stats[:3]:
        print(f"  {stat}")
    
    # Memory usage statistics
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    # Stop tracking
    tracemalloc.stop()

# Usage examples
def memory_optimization_examples():
    compare_slots_memory()
    efficient_data_structures()
    compare_range_memory()
    demonstrate_lazy_loading()
    memory_mapping_example()
    string_interning_demo()
    circular_reference_optimization()
    memory_profiling_tools()

## Advanced Python Concepts

### Asyncio and coroutines

Modern Python's approach to asynchronous programming using async/await syntax and the asyncio library.

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Coroutine, AsyncIterator
from concurrent.futures import ThreadPoolExecutor
import threading

# 1. Basic async/await concepts
async def basic_async_function():
    """Basic async function example"""
    print("Starting async function")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("Async function completed")
    return "result"

async def demonstrate_basic_async():
    """Demonstrate basic async concepts"""
    print("=== Basic Async/Await ===")
    
    # Sequential execution
    start_time = time.time()
    result = await basic_async_function()
    end_time = time.time()
    
    print(f"Result: {result}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

# 2. Concurrent execution
async def fetch_data(url: str, delay: float) -> Dict[str, Any]:
    """Simulate fetching data from a URL"""
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)  # Simulate network delay
    return {"url": url, "data": f"Data from {url}", "delay": delay}

async def demonstrate_concurrency():
    """Show concurrent execution of async tasks"""
    print("\n=== Concurrent Execution ===")
    
    urls = [
        ("https://api1.example.com", 0.5),
        ("https://api2.example.com", 1.0),
        ("https://api3.example.com", 0.3),
        ("https://api4.example.com", 0.8),
    ]
    
    # Sequential execution
    start_time = time.time()
    sequential_results = []
    for url, delay in urls:
        result = await fetch_data(url, delay)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential execution: {sequential_time:.2f} seconds")
    
    # Concurrent execution using gather
    start_time = time.time()
    concurrent_results = await asyncio.gather(*[
        fetch_data(url, delay) for url, delay in urls
    ])
    concurrent_time = time.time() - start_time
    
    print(f"Concurrent execution: {concurrent_time:.2f} seconds")
    print(f"Speedup: {sequential_time / concurrent_time:.2f}x")

# 3. Task management
async def task_management_examples():
    """Demonstrate task creation and management"""
    print("\n=== Task Management ===")
    
    # Create tasks
    task1 = asyncio.create_task(fetch_data("task1", 0.5))
    task2 = asyncio.create_task(fetch_data("task2", 1.0))
    task3 = asyncio.create_task(fetch_data("task3", 0.3))
    
    # Wait for all tasks
    results = await asyncio.gather(task1, task2, task3)
    
    for result in results:
        print(f"Task result: {result}")
    
    # Task with timeout
    try:
        result = await asyncio.wait_for(
            fetch_data("slow_service", 2.0), 
            timeout=1.0
        )
    except asyncio.TimeoutError:
        print("Task timed out!")
    
    # Task cancellation
    long_task = asyncio.create_task(fetch_data("long_task", 5.0))
    
    # Cancel after short delay
    await asyncio.sleep(0.1)
    long_task.cancel()
    
    try:
        await long_task
    except asyncio.CancelledError:
        print("Long task was cancelled")

# 4. Async generators and iteration
async def async_number_generator(start: int, end: int, delay: float) -> AsyncIterator[int]:
    """Async generator that yields numbers with delay"""
    for i in range(start, end):
        await asyncio.sleep(delay)
        yield i

async def async_file_reader(lines: List[str]) -> AsyncIterator[str]:
    """Simulate async file reading"""
    for line in lines:
        await asyncio.sleep(0.1)  # Simulate I/O delay
        yield line.strip()

async def demonstrate_async_generators():
    """Show async generators and iteration"""
    print("\n=== Async Generators ===")
    
    # Async generator
    print("Async number generator:")
    async for number in async_number_generator(1, 5, 0.2):
        print(f"  Generated: {number}")
    
    # Async file reading simulation
    print("Async file reader:")
    file_lines = ["line 1\n", "line 2\n", "line 3\n", "line 4\n"]
    async for line in async_file_reader(file_lines):
        print(f"  Read: {line}")

# 5. Async context managers
class AsyncDatabase:
    """Async database connection simulation"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
    
    async def __aenter__(self):
        print(f"Connecting to {self.connection_string}")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connected = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Disconnecting from database")
        await asyncio.sleep(0.1)  # Simulate cleanup time
        self.connected = False
        return False
    
    async def query(self, sql: str) -> List[Dict[str, Any]]:
        if not self.connected:
            raise RuntimeError("Not connected to database")
        
        print(f"Executing query: {sql}")
        await asyncio.sleep(0.2)  # Simulate query time
        return [{"id": 1, "name": "test"}, {"id": 2, "name": "example"}]

async def demonstrate_async_context_manager():
    """Show async context managers"""
    print("\n=== Async Context Managers ===")
    
    async with AsyncDatabase("postgresql://localhost:5432/test") as db:
        results = await db.query("SELECT * FROM users")
        print(f"Query results: {results}")

# 6. Synchronization primitives
async def worker(name: str, semaphore: asyncio.Semaphore, shared_resource: List[str]):
    """Worker that accesses shared resource"""
    async with semaphore:  # Limit concurrent access
        print(f"Worker {name} acquired semaphore")
        await asyncio.sleep(0.5)  # Simulate work
        shared_resource.append(f"Work by {name}")
        print(f"Worker {name} releasing semaphore")

async def demonstrate_synchronization():
    """Show async synchronization primitives"""
    print("\n=== Async Synchronization ===")
    
    # Semaphore to limit concurrent access
    semaphore = asyncio.Semaphore(2)  # Only 2 workers at a time
    shared_resource = []
    
    # Create multiple workers
    workers = [
        worker(f"Worker-{i}", semaphore, shared_resource)
        for i in range(5)
    ]
    
    await asyncio.gather(*workers)
    
    print(f"Shared resource: {shared_resource}")
    
    # Event for coordination
    event = asyncio.Event()
    
    async def waiter(name: str):
        print(f"{name} waiting for event")
        await event.wait()
        print(f"{name} received event")
    
    async def setter():
        await asyncio.sleep(1)
        print("Setting event")
        event.set()
    
    # Run waiters and setter
    await asyncio.gather(
        waiter("Waiter-1"),
        waiter("Waiter-2"),
        setter()
    )

# 7. Integration with blocking code
def cpu_bound_task(n: int) -> int:
    """CPU-intensive task that blocks the event loop"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

async def demonstrate_blocking_integration():
    """Show how to handle blocking code in async context"""
    print("\n=== Blocking Code Integration ===")
    
    # Bad: This blocks the event loop
    # result = cpu_bound_task(1000000)
    
    # Good: Run in thread pool
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Run blocking task in thread
        result = await loop.run_in_executor(executor, cpu_bound_task, 1000000)
        print(f"CPU-bound result: {result}")
        
        # Multiple blocking tasks concurrently
        tasks = [
            loop.run_in_executor(executor, cpu_bound_task, 100000)
            for _ in range(4)
        ]
        
        results = await asyncio.gather(*tasks)
        print(f"Multiple CPU-bound results: {results}")

# 8. Error handling in async code
async def failing_task(delay: float, should_fail: bool = True):
    """Task that may fail"""
    await asyncio.sleep(delay)
    if should_fail:
        raise ValueError(f"Task failed after {delay} seconds")
    return f"Success after {delay} seconds"

async def demonstrate_error_handling():
    """Show error handling in async code"""
    print("\n=== Async Error Handling ===")
    
    # Individual task error handling
    try:
        result = await failing_task(0.5, should_fail=True)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # Error handling with gather
    tasks = [
        failing_task(0.2, should_fail=False),
        failing_task(0.3, should_fail=True),
        failing_task(0.1, should_fail=False),
    ]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed: {result}")
            else:
                print(f"Task {i} succeeded: {result}")
    
    except Exception as e:
        print(f"Gather failed: {e}")

# 9. Async queue for producer-consumer
async def producer(queue: asyncio.Queue, name: str, count: int):
    """Producer that puts items in queue"""
    for i in range(count):
        item = f"{name}-item-{i}"
        await queue.put(item)
        print(f"Producer {name} put: {item}")
        await asyncio.sleep(0.1)
    
    # Signal completion
    await queue.put(None)

async def consumer(queue: asyncio.Queue, name: str):
    """Consumer that processes items from queue"""
    while True:
        item = await queue.get()
        if item is None:
            # Signal to stop
            await queue.put(None)  # Pass signal to other consumers
            break
        
        print(f"Consumer {name} processing: {item}")
        await asyncio.sleep(0.2)  # Simulate processing time
        queue.task_done()

async def demonstrate_async_queue():
    """Show async queue for producer-consumer pattern"""
    print("\n=== Async Queue ===")
    
    queue = asyncio.Queue(maxsize=5)
    
    # Create producers and consumers
    producers = [
        producer(queue, f"P{i}", 3) for i in range(2)
    ]
    
    consumers = [
        consumer(queue, f"C{i}") for i in range(3)
    ]
    
    # Run all concurrently
    await asyncio.gather(*producers, *consumers)

# 10. Main async example runner
async def main():
    """Main function to run all async examples"""
    await demonstrate_basic_async()
    await demonstrate_concurrency()
    await task_management_examples()
    await demonstrate_async_generators()
    await demonstrate_async_context_manager()
    await demonstrate_synchronization()
    await demonstrate_blocking_integration()
    await demonstrate_error_handling()
    await demonstrate_async_queue()

def run_asyncio_examples():
    """Run all asyncio examples"""
    print("=== AsyncIO and Coroutines Examples ===")
    asyncio.run(main())

### Type hints and static typing

Python's type system helps catch errors early and improves code documentation.

```python
from typing import (
    List, Dict, Tuple, Set, Optional, Union, Any, Callable,
    TypeVar, Generic, Protocol, Literal, Final, ClassVar,
    Annotated, get_type_hints, TYPE_CHECKING
)
from typing_extensions import NotRequired, Required
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect

# 1. Basic type hints
def basic_type_examples():
    """Demonstrate basic type annotations"""
    
    # Simple types
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    def calculate_average(numbers: List[float]) -> float:
        return sum(numbers) / len(numbers)
    
    # Collections
    def process_data(data: Dict[str, List[int]]) -> Dict[str, float]:
        return {key: sum(values) / len(values) for key, values in data.items()}
    
    # Optional types
    def find_user(user_id: int) -> Optional[Dict[str, Any]]:
        # Simulate database lookup
        users = {1: {"name": "Alice", "age": 30}, 2: {"name": "Bob", "age": 25}}
        return users.get(user_id)
    
    # Union types
    def parse_value(value: Union[str, int, float]) -> float:
        if isinstance(value, str):
            return float(value)
        return float(value)
    
    # Examples
    print("=== Basic Type Examples ===")
    print(greet("World"))
    print(add_numbers(5, 3))
    print(calculate_average([1.0, 2.0, 3.0, 4.0]))
    
    data = {"group1": [1, 2, 3], "group2": [4, 5, 6]}
    print(process_data(data))
    
    user = find_user(1)
    print(f"User: {user}")

# 2. Generic types
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class Stack(Generic[T]):
    """Generic stack implementation"""
    
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)

class Cache(Generic[K, V]):
    """Generic cache with key-value pairs"""
    
    def __init__(self, max_size: int = 100) -> None:
        self._cache: Dict[K, V] = {}
        self._max_size = max_size
    
    def get(self, key: K) -> Optional[V]:
        return self._cache.get(key)
    
    def set(self, key: K, value: V) -> None:
        if len(self._cache) >= self._max_size and key not in self._cache:
            # Simple eviction: remove first item
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value
    
    def keys(self) -> List[K]:
        return list(self._cache.keys())

def generic_examples():
    """Demonstrate generic types"""
    print("\n=== Generic Types ===")
    
    # String stack
    string_stack: Stack[str] = Stack()
    string_stack.push("hello")
    string_stack.push("world")
    print(f"String stack pop: {string_stack.pop()}")
    
    # Integer stack
    int_stack: Stack[int] = Stack()
    int_stack.push(1)
    int_stack.push(2)
    print(f"Int stack peek: {int_stack.peek()}")
    
    # Cache examples
    str_cache: Cache[str, int] = Cache(max_size=3)
    str_cache.set("one", 1)
    str_cache.set("two", 2)
    print(f"Cache get 'one': {str_cache.get('one')}")

# 3. Protocols (structural typing)
class Drawable(Protocol):
    """Protocol for drawable objects"""
    
    def draw(self) -> str:
        """Draw the object"""
        ...
    
    def get_area(self) -> float:
        """Get the area of the object"""
        ...

class Circle:
    """Circle class that implements Drawable protocol"""
    
    def __init__(self, radius: float) -> None:
        self.radius = radius
    
    def draw(self) -> str:
        return f"Drawing circle with radius {self.radius}"
    
    def get_area(self) -> float:
        return 3.14159 * self.radius ** 2

class Rectangle:
    """Rectangle class that implements Drawable protocol"""
    
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
    
    def draw(self) -> str:
        return f"Drawing rectangle {self.width}x{self.height}"
    
    def get_area(self) -> float:
        return self.width * self.height

def render_shape(shape: Drawable) -> str:
    """Function that accepts any drawable object"""
    return f"{shape.draw()}, Area: {shape.get_area():.2f}"

def protocol_examples():
    """Demonstrate protocols"""
    print("\n=== Protocols ===")
    
    circle = Circle(5.0)
    rectangle = Rectangle(4.0, 6.0)
    
    print(render_shape(circle))
    print(render_shape(rectangle))

# 4. Literal types and Final
class Status:
    PENDING: Final[str] = "pending"
    COMPLETED: Final[str] = "completed"
    FAILED: Final[str] = "failed"

def process_task(status: Literal["pending", "completed", "failed"]) -> str:
    """Function that only accepts specific string literals"""
    if status == "pending":
        return "Task is waiting to be processed"
    elif status == "completed":
        return "Task has been completed successfully"
    elif status == "failed":
        return "Task has failed"
    else:
        # This should never happen with proper typing
        return "Unknown status"

# 5. Advanced type annotations
@dataclass
class User:
    """User dataclass with type annotations"""
    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class UserRepository:
    """Repository with type annotations"""
    
    def __init__(self) -> None:
        self._users: Dict[int, User] = {}
        self._next_id: int = 1
    
    def create_user(self, name: str, age: int, email: Optional[str] = None) -> User:
        user = User(name=name, age=age, email=email)
        user_id = self._next_id
        self._users[user_id] = user
        self._next_id += 1
        return user
    
    def get_user(self, user_id: int) -> Optional[User]:
        return self._users.get(user_id)
    
    def get_all_users(self) -> List[User]:
        return list(self._users.values())
    
    def filter_users(self, predicate: Callable[[User], bool]) -> List[User]:
        return [user for user in self._users.values() if predicate(user)]

# 6. TypedDict for dictionary types
if TYPE_CHECKING:
    from typing_extensions import TypedDict

class PersonDict(TypedDict):
    """Typed dictionary for person data"""
    name: str
    age: int
    email: NotRequired[str]  # Optional field

class PersonDictRequired(TypedDict):
    """All fields required"""
    name: Required[str]
    age: Required[int]
    email: Required[str]

def process_person(person: PersonDict) -> str:
    """Process person dictionary with type checking"""
    name = person["name"]
    age = person["age"]
    email = person.get("email", "No email")
    return f"{name} ({age}): {email}"

# 7. Callable types
def higher_order_function_examples():
    """Demonstrate callable type annotations"""
    
    # Simple callable
    def apply_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
        return operation(x, y)
    
    # Callable with specific signature
    FilterFunction = Callable[[str], bool]
    
    def filter_strings(strings: List[str], filter_func: FilterFunction) -> List[str]:
        return [s for s in strings if filter_func(s)]
    
    # Generator function type
    def number_generator(start: int, end: int) -> Callable[[], int]:
        current = start
        
        def next_number() -> int:
            nonlocal current
            if current >= end:
                raise StopIteration
            result = current
            current += 1
            return result
        
        return next_number
    
    print("\n=== Callable Types ===")
    
    # Examples
    result = apply_operation(5, 3, lambda x, y: x + y)
    print(f"Apply operation result: {result}")
    
    strings = ["hello", "world", "python", "typing"]
    long_strings = filter_strings(strings, lambda s: len(s) > 5)
    print(f"Long strings: {long_strings}")

# 8. Type checking and runtime validation
def runtime_type_checking():
    """Demonstrate runtime type checking"""
    
    def validate_types(func: Callable) -> Callable:
        """Decorator to validate function arguments at runtime"""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        def wrapper(*args, **kwargs):
            # Bind arguments to parameters
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Check each argument
            for param_name, value in bound.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]
                    
                    # Simple type checking (doesn't handle complex types)
                    if hasattr(expected_type, '__origin__'):
                        # Skip complex generic types for this example
                        continue
                    
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Argument '{param_name}' must be {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    @validate_types
    def typed_function(name: str, age: int) -> str:
        return f"{name} is {age} years old"
    
    print("\n=== Runtime Type Checking ===")
    
    try:
        result = typed_function("Alice", 30)
        print(f"Valid call: {result}")
        
        # This should raise a TypeError
        typed_function("Bob", "thirty")
    except TypeError as e:
        print(f"Type error caught: {e}")

# 9. Advanced generic constraints
class Comparable(Protocol):
    """Protocol for comparable objects"""
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...

CT = TypeVar('CT', bound=Comparable)

def find_maximum(items: List[CT]) -> CT:
    """Find maximum item in a list of comparable items"""
    if not items:
        raise ValueError("Cannot find maximum of empty list")
    
    maximum = items[0]
    for item in items[1:]:
        if item > maximum:
            maximum = item
    return maximum

def advanced_generics_examples():
    """Demonstrate advanced generic constraints"""
    print("\n=== Advanced Generics ===")
    
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    max_number = find_maximum(numbers)
    print(f"Maximum number: {max_number}")
    
    words = ["apple", "banana", "cherry", "date"]
    max_word = find_maximum(words)
    print(f"Maximum word: {max_word}")

# Usage examples
def type_hints_examples():
    basic_type_examples()
    generic_examples()
    protocol_examples()
    
    print("\n=== Literal Types ===")
    print(process_task("pending"))
    print(process_task("completed"))
    
    print("\n=== Advanced Annotations ===")
    repo = UserRepository()
    user1 = repo.create_user("Alice", 30, "alice@example.com")
    user2 = repo.create_user("Bob", 25)
    
    all_users = repo.get_all_users()
    print(f"All users: {[u.name for u in all_users]}")
    
    adults = repo.filter_users(lambda u: u.age >= 18)
    print(f"Adult users: {[u.name for u in adults]}")
    
    print("\n=== TypedDict ===")
    person: PersonDict = {"name": "Charlie", "age": 35}
    print(process_person(person))
    
    higher_order_function_examples()
    runtime_type_checking()
    advanced_generics_examples()

## Web Frameworks

### Django architecture

Django follows the Model-View-Template (MVT) pattern and provides a full-featured web framework.

```python
# Note: This is demonstration code showing Django concepts
# In a real Django project, these would be in separate files

# 1. Models (django/models.py)
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.validators import MinValueValidator, MaxValueValidator

class Category(models.Model):
    """Product category model"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name_plural = "categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('category_detail', kwargs={'pk': self.pk})

class Product(models.Model):
    """Product model with relationships"""
    name = models.CharField(max_length=200)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    in_stock = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Many-to-many relationship
    tags = models.ManyToManyField('Tag', blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['category', 'in_stock']),
        ]
    
    def __str__(self):
        return self.name
    
    def get_absolute_url(self):
        return reverse('product_detail', kwargs={'pk': self.pk})
    
    @property
    def is_expensive(self):
        return self.price > 100

class Tag(models.Model):
    """Tag model for products"""
    name = models.CharField(max_length=50, unique=True)
    
    def __str__(self):
        return self.name

class Review(models.Model):
    """Product review model"""
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['product', 'author']  # One review per user per product
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Review by {self.author.username} for {self.product.name}"

# 2. Views (django/views.py)
from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, CreateView, UpdateView
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, HttpResponse
from django.db.models import Q, Avg, Count
from django.core.paginator import Paginator
from django.contrib import messages
from django.urls import reverse_lazy

# Function-based views
def product_list(request):
    """List products with filtering and pagination"""
    products = Product.objects.select_related('category').prefetch_related('tags')
    
    # Filtering
    category_id = request.GET.get('category')
    if category_id:
        products = products.filter(category_id=category_id)
    
    search_query = request.GET.get('search')
    if search_query:
        products = products.filter(
            Q(name__icontains=search_query) | 
            Q(description__icontains=search_query)
        )
    
    # Pagination
    paginator = Paginator(products, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'categories': Category.objects.all(),
        'search_query': search_query,
        'selected_category': category_id,
    }
    
    return render(request, 'products/list.html', context)

def product_detail(request, pk):
    """Product detail with reviews"""
    product = get_object_or_404(
        Product.objects.select_related('category')
        .prefetch_related('tags', 'reviews__author'),
        pk=pk
    )
    
    # Calculate average rating
    avg_rating = product.reviews.aggregate(Avg('rating'))['rating__avg']
    
    context = {
        'product': product,
        'avg_rating': avg_rating,
        'reviews': product.reviews.all()[:5],  # Latest 5 reviews
    }
    
    return render(request, 'products/detail.html', context)

@login_required
def add_review(request, product_pk):
    """Add review for a product"""
    product = get_object_or_404(Product, pk=product_pk)
    
    if request.method == 'POST':
        rating = request.POST.get('rating')
        comment = request.POST.get('comment')
        
        if rating and comment:
            review, created = Review.objects.get_or_create(
                product=product,
                author=request.user,
                defaults={'rating': int(rating), 'comment': comment}
            )
            
            if created:
                messages.success(request, 'Review added successfully!')
            else:
                messages.warning(request, 'You have already reviewed this product.')
        
        return redirect('product_detail', pk=product_pk)
    
    return redirect('product_detail', pk=product_pk)

# Class-based views
class ProductListView(ListView):
    """Class-based view for product listing"""
    model = Product
    template_name = 'products/list.html'
    context_object_name = 'products'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Product.objects.select_related('category').prefetch_related('tags')
        
        # Apply filters
        category_id = self.request.GET.get('category')
        if category_id:
            queryset = queryset.filter(category_id=category_id)
        
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(
                Q(name__icontains=search_query) | 
                Q(description__icontains=search_query)
            )
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = Category.objects.all()
        context['search_query'] = self.request.GET.get('search', '')
        return context

class ProductDetailView(DetailView):
    """Class-based view for product detail"""
    model = Product
    template_name = 'products/detail.html'
    context_object_name = 'product'
    
    def get_queryset(self):
        return Product.objects.select_related('category').prefetch_related('tags', 'reviews__author')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        product = self.get_object()
        context['avg_rating'] = product.reviews.aggregate(Avg('rating'))['rating__avg']
        return context

class ProductCreateView(LoginRequiredMixin, CreateView):
    """Create new product (admin only)"""
    model = Product
    fields = ['name', 'description', 'price', 'category', 'in_stock', 'tags']
    template_name = 'products/create.html'
    success_url = reverse_lazy('product_list')

# API Views (Django REST Framework style)
def api_products(request):
    """JSON API for products"""
    products = Product.objects.select_related('category').values(
        'id', 'name', 'price', 'category__name', 'in_stock'
    )
    
    # Convert to list for JSON serialization
    product_list = list(products)
    
    return JsonResponse({
        'products': product_list,
        'count': len(product_list)
    })

# 3. URLs (django/urls.py)
from django.urls import path, include

# App URLs
app_name = 'products'
urlpatterns = [
    path('', ProductListView.as_view(), name='product_list'),
    path('<int:pk>/', ProductDetailView.as_view(), name='product_detail'),
    path('create/', ProductCreateView.as_view(), name='product_create'),
    path('<int:product_pk>/review/', add_review, name='add_review'),
    path('api/', api_products, name='api_products'),
]

# Main project URLs
main_urlpatterns = [
    path('admin/', admin.site.urls),
    path('products/', include('products.urls')),
    path('api/', include('api.urls')),
]

# 4. Forms (django/forms.py)
from django import forms
from django.core.exceptions import ValidationError

class ProductForm(forms.ModelForm):
    """Form for creating/editing products"""
    
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'category', 'in_stock', 'tags']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'tags': forms.CheckboxSelectMultiple(),
        }
    
    def clean_price(self):
        """Custom validation for price"""
        price = self.cleaned_data['price']
        if price <= 0:
            raise ValidationError("Price must be positive")
        return price
    
    def clean_name(self):
        """Ensure product name is unique"""
        name = self.cleaned_data['name']
        
        # Check if product with this name already exists
        existing = Product.objects.filter(name=name)
        if self.instance:
            existing = existing.exclude(pk=self.instance.pk)
        
        if existing.exists():
            raise ValidationError("Product with this name already exists")
        
        return name

class ReviewForm(forms.ModelForm):
    """Form for product reviews"""
    
    class Meta:
        model = Review
        fields = ['rating', 'comment']
        widgets = {
            'rating': forms.Select(choices=[(i, i) for i in range(1, 6)]),
            'comment': forms.Textarea(attrs={'rows': 3}),
        }

class ProductSearchForm(forms.Form):
    """Search form for products"""
    search = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Search products...',
            'class': 'form-control'
        })
    )
    category = forms.ModelChoiceField(
        queryset=Category.objects.all(),
        required=False,
        empty_label="All Categories",
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    min_price = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    max_price = forms.DecimalField(
        max_digits=10,
        decimal_places=2,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

# 5. Admin (django/admin.py)
from django.contrib import admin
from django.utils.html import format_html

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    """Admin interface for Category"""
    list_display = ['name', 'product_count', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    
    def product_count(self, obj):
        return obj.products.count()
    product_count.short_description = 'Products'

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    """Admin interface for Product"""
    list_display = ['name', 'category', 'price', 'in_stock', 'created_at']
    list_filter = ['category', 'in_stock', 'created_at']
    search_fields = ['name', 'description']
    list_editable = ['price', 'in_stock']
    readonly_fields = ['created_at', 'updated_at']
    filter_horizontal = ['tags']
    
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'category')
        }),
        ('Pricing', {
            'fields': ('price', 'in_stock')
        }),
        ('Tags', {
            'fields': ('tags',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('category')

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    """Admin interface for Review"""
    list_display = ['product', 'author', 'rating', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['product__name', 'author__username']
    readonly_fields = ['created_at']
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('product', 'author')

# 6. Settings example (django/settings.py)
DJANGO_SETTINGS = {
    'DATABASES': {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'your_db_name',
            'USER': 'your_db_user',
            'PASSWORD': 'your_db_password',
            'HOST': 'localhost',
            'PORT': '5432',
        }
    },
    
    'INSTALLED_APPS': [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        'products',  # Your app
        'rest_framework',  # For API development
    ],
    
    'MIDDLEWARE': [
        'django.middleware.security.SecurityMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'django.contrib.messages.middleware.MessageMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ],
    
    'TEMPLATES': [
        {
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [BASE_DIR / 'templates'],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        },
    ],
    
    # Caching
    'CACHES': {
        'default': {
            'BACKEND': 'django.core.cache.backends.redis.RedisCache',
            'LOCATION': 'redis://127.0.0.1:6379/1',
        }
    },
    
    # Logging
    'LOGGING': {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': 'django.log',
            },
        },
        'loggers': {
            'django': {
                'handlers': ['file'],
                'level': 'INFO',
                'propagate': True,
            },
        },
    },
}

def django_architecture_examples():
    """Demonstrate Django architecture concepts"""
    print("=== Django Architecture Examples ===")
    print("Django follows Model-View-Template (MVT) pattern:")
    print("- Models: Define data structure and business logic")
    print("- Views: Handle HTTP requests and responses")
    print("- Templates: Handle presentation layer")
    print("- URLs: Route requests to appropriate views")
    print("- Forms: Handle user input and validation")
    print("- Admin: Built-in administrative interface")
    
    print("\nKey Django Features:")
    print("- ORM for database abstraction")
    print("- Built-in admin interface")
    print("- Authentication and authorization")
    print("- Caching framework")
    print("- Internationalization support")
    print("- Security features (CSRF, XSS protection)")
    print("- Scalable architecture with apps")

### Flask vs Django

```python
# Comparison of Flask and Django approaches

# FLASK EXAMPLE
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow import fields

# Flask application setup
flask_app = Flask(__name__)
flask_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(flask_app)
ma = Marshmallow(flask_app)

# Flask Models
class FlaskUser(db.Model):
    """Flask SQLAlchemy model"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    posts = db.relationship('FlaskPost', backref='author', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class FlaskPost(db.Model):
    """Flask SQLAlchemy model"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('flask_user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Post {self.title}>'

# Flask Schemas (for serialization)
class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = FlaskUser
        load_instance = True
    
    posts = ma.Nested('PostSchema', many=True, exclude=('author',))

class PostSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = FlaskPost
        load_instance = True
    
    author = ma.Nested(UserSchema, exclude=('posts',))

# Initialize schemas
user_schema = UserSchema()
users_schema = UserSchema(many=True)
post_schema = PostSchema()
posts_schema = PostSchema(many=True)

# Flask Routes
@flask_app.route('/')
def flask_home():
    """Flask home page"""
    return render_template('home.html', framework='Flask')

@flask_app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users (Flask)"""
    users = FlaskUser.query.all()
    return jsonify(users_schema.dump(users))

@flask_app.route('/api/users', methods=['POST'])
def create_user():
    """Create user (Flask)"""
    json_data = request.get_json()
    
    if not json_data:
        return jsonify({'error': 'No input data provided'}), 400
    
    try:
        user = user_schema.load(json_data)
        db.session.add(user)
        db.session.commit()
        return jsonify(user_schema.dump(user)), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@flask_app.route('/api/users/<int:user_id>/posts')
def get_user_posts(user_id):
    """Get posts by user (Flask)"""
    user = FlaskUser.query.get_or_404(user_id)
    return jsonify(posts_schema.dump(user.posts))

# Flask Application Factory Pattern
def create_flask_app(config_name='default'):
    """Application factory for Flask"""
    app = Flask(__name__)
    
    # Configuration
    configs = {
        'development': 'config.DevelopmentConfig',
        'production': 'config.ProductionConfig',
        'testing': 'config.TestingConfig'
    }
    
    app.config.from_object(configs.get(config_name, 'config.DevelopmentConfig'))
    
    # Initialize extensions
    db.init_app(app)
    ma.init_app(app)
    
    # Register blueprints
    from .auth import auth_bp
    from .api import api_bp
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

# Flask Blueprint example
from flask import Blueprint

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'framework': 'Flask'})

@api_bp.route('/posts')
def api_get_posts():
    """Get all posts via API"""
    posts = FlaskPost.query.all()
    return jsonify(posts_schema.dump(posts))

# DJANGO COMPARISON

# Django equivalent would be:
DJANGO_COMPARISON = """
# Django Models (more built-in features)
class DjangoUser(models.Model):
    username = models.CharField(max_length=80, unique=True)
    email = models.EmailField(unique=True)
    
    class Meta:
        verbose_name_plural = "users"
    
    def __str__(self):
        return self.username

class DjangoPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(DjangoUser, on_delete=models.CASCADE, related_name='posts')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

# Django Views (more structure)
from django.http import JsonResponse
from django.views.generic import ListView
from rest_framework.viewsets import ModelViewSet

class DjangoUserListView(ListView):
    model = DjangoUser
    template_name = 'users/list.html'
    context_object_name = 'users'

def django_get_users(request):
    users = DjangoUser.objects.prefetch_related('posts').all()
    data = [{'id': u.id, 'username': u.username, 'email': u.email} for u in users]
    return JsonResponse({'users': data})

# Django REST Framework (more powerful for APIs)
class UserViewSet(ModelViewSet):
    queryset = DjangoUser.objects.all()
    serializer_class = UserSerializer
    
    def get_posts(self, request, pk=None):
        user = self.get_object()
        posts = user.posts.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

# Django URLs (more explicit)
from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'users', UserViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('users/', DjangoUserListView.as_view(), name='user-list'),
]
"""

def flask_vs_django_comparison():
    """Compare Flask and Django approaches"""
    print("=== Flask vs Django Comparison ===")
    
    comparison = {
        "Philosophy": {
            "Flask": "Microframework - minimal core, add what you need",
            "Django": "Batteries included - comprehensive framework"
        },
        "Learning Curve": {
            "Flask": "Easier to start, simpler concepts",
            "Django": "Steeper learning curve, more concepts to learn"
        },
        "Flexibility": {
            "Flask": "Highly flexible, fewer conventions",
            "Django": "Opinionated, follows conventions"
        },
        "Database": {
            "Flask": "SQLAlchemy (optional), more manual setup",
            "Django": "Built-in ORM with migrations"
        },
        "Admin Interface": {
            "Flask": "Third-party solutions (Flask-Admin)",
            "Django": "Built-in powerful admin interface"
        },
        "Authentication": {
            "Flask": "Flask-Login, Flask-Security (extensions)",
            "Django": "Built-in authentication and authorization"
        },
        "Routing": {
            "Flask": "Decorator-based routing",
            "Django": "URL configuration files"
        },
        "Templates": {
            "Flask": "Jinja2 templates",
            "Django": "Django templates (similar to Jinja2)"
        },
        "Forms": {
            "Flask": "WTForms (extension)",
            "Django": "Built-in forms with validation"
        },
        "Testing": {
            "Flask": "Flask testing utilities + pytest",
            "Django": "Built-in testing framework"
        },
        "Deployment": {
            "Flask": "More deployment options, simpler",
            "Django": "More complex, but well-documented patterns"
        },
        "Performance": {
            "Flask": "Lighter, potentially faster for simple apps",
            "Django": "More overhead but optimized for complex apps"
        },
        "Use Cases": {
            "Flask": "APIs, microservices, simple web apps, prototypes",
            "Django": "Complex web applications, CMS, enterprise apps"
        }
    }
    
    for category, details in comparison.items():
        print(f"\n{category}:")
        for framework, description in details.items():
            print(f"  {framework}: {description}")

# Example of when to choose each framework
def framework_selection_guide():
    """Guide for choosing between Flask and Django"""
    print("\n=== When to Choose Flask ===")
    flask_scenarios = [
        "Building RESTful APIs or microservices",
        "Need maximum flexibility and control",
        "Small to medium-sized applications", 
        "Prototyping and experimentation",
        "Learning web development concepts",
        "Integration with existing systems",
        "Custom architecture requirements"
    ]
    
    for scenario in flask_scenarios:
        print(f"• {scenario}")
    
    print("\n=== When to Choose Django ===")
    django_scenarios = [
        "Building full-featured web applications",
        "Content management systems",
        "Applications requiring admin interface",
        "Rapid development with conventions",
        "Applications with complex data models",
        "E-commerce or social media platforms",
        "Enterprise applications with multiple features"
    ]
    
    for scenario in django_scenarios:
        print(f"• {scenario}")

# Code organization comparison
def code_organization_comparison():
    """Compare code organization patterns"""
    print("\n=== Code Organization ===")
    
    print("Flask Project Structure:")
    flask_structure = """
    my_flask_app/
    ├── app/
    │   ├── __init__.py          # Application factory
    │   ├── models.py            # Database models
    │   ├── views.py             # View functions
    │   ├── forms.py             # WTForms
    │   └── templates/           # Jinja2 templates
    ├── migrations/              # Database migrations
    ├── tests/                   # Test files
    ├── config.py               # Configuration
    ├── requirements.txt        # Dependencies
    └── run.py                  # Application entry point
    """
    print(flask_structure)
    
    print("Django Project Structure:")
    django_structure = """
    my_django_project/
    ├── my_django_project/
    │   ├── __init__.py
    │   ├── settings.py         # Configuration
    │   ├── urls.py            # Main URL configuration
    │   └── wsgi.py            # WSGI application
    ├── app1/                  # Django app
    │   ├── migrations/        # Database migrations
    │   ├── models.py         # Models
    │   ├── views.py          # Views
    │   ├── urls.py           # App URLs
    │   ├── forms.py          # Forms
    │   ├── admin.py          # Admin configuration
    │   └── tests.py          # Tests
    ├── app2/                 # Another Django app
    ├── templates/            # Templates
    ├── static/              # Static files
    ├── requirements.txt     # Dependencies
    └── manage.py           # Django management script
    """
    print(django_structure)

def web_frameworks_examples():
    flask_vs_django_comparison()
    framework_selection_guide()
    code_organization_comparison()
    django_architecture_examples()

## Python Performance

### Profiling tools

Python provides several tools for performance analysis and optimization.

```python
import cProfile
import pstats
import timeit
import time
import functools
from memory_profiler import profile
import line_profiler

# 1. Basic profiling with cProfile
def fibonacci_recursive(n):
    """Inefficient recursive Fibonacci"""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_iterative(n):
    """Efficient iterative Fibonacci"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def cProfile_example():
    """Demonstrate cProfile usage"""
    print("=== cProfile Example ===")
    
    # Profile recursive version
    print("Profiling recursive Fibonacci:")
    cProfile.run('fibonacci_recursive(30)')
    
    # Profile iterative version
    print("\nProfiling iterative Fibonacci:")
    cProfile.run('fibonacci_iterative(30)')

def advanced_profiling():
    """Advanced profiling with pstats"""
    print("\n=== Advanced Profiling ===")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Profile code
    profiler.enable()
    result = fibonacci_recursive(25)
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print(f"Fibonacci result: {result}")
    print("\nTop 10 functions by cumulative time:")
    stats.print_stats(10)

# 2. Memory profiling
@profile
def memory_intensive_function():
    """Function that uses a lot of memory"""
    # Create large lists
    data = []
    for i in range(100000):
        data.append([j for j in range(100)])
    
    # Process data
    result = []
    for sublist in data:
        result.append(sum(sublist))
    
    return result

# 3. Line profiling
@profile
def line_by_line_example():
    """Example for line-by-line profiling"""
    data = list(range(100000))
    
    # Various operations
    squares = [x ** 2 for x in data]
    evens = [x for x in squares if x % 2 == 0]
    total = sum(evens)
    
    return total

# 4. Timing utilities
class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name="Operation"):
        self.name = name
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        print(f"{self.name} took {self.duration:.6f} seconds")

def timing_decorator(func):
    """Decorator for timing function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

def timing_examples():
    """Demonstrate timing utilities"""
    print("\n=== Timing Examples ===")
    
    # Using context manager
    with Timer("List comprehension"):
        squares = [x ** 2 for x in range(100000)]
    
    # Using timeit
    setup = "data = list(range(1000))"
    list_comp_time = timeit.timeit(
        "[x ** 2 for x in data]",
        setup=setup,
        number=1000
    )
    
    map_time = timeit.timeit(
        "list(map(lambda x: x ** 2, data))",
        setup=setup,
        number=1000
    )
    
    print(f"List comprehension (1000 runs): {list_comp_time:.6f} seconds")
    print(f"Map function (1000 runs): {map_time:.6f} seconds")
    
    # Using decorator
    @timing_decorator
    def slow_function():
        time.sleep(0.1)
        return "Done"
    
    result = slow_function()

# 5. Performance comparison utilities
def compare_performance(*functions, iterations=1000, setup_data=None):
    """Compare performance of multiple functions"""
    results = {}
    
    for func in functions:
        if setup_data:
            times = []
            for _ in range(iterations):
                data = setup_data()
                start = time.perf_counter()
                func(data)
                end = time.perf_counter()
                times.append(end - start)
            avg_time = sum(times) / len(times)
        else:
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func()
                end = time.perf_counter()
                times.append(end - start)
            avg_time = sum(times) / len(times)
        
        results[func.__name__] = avg_time
    
    return results

def performance_comparison_example():
    """Example of comparing different implementations"""
    print("\n=== Performance Comparison ===")
    
    def list_append():
        result = []
        for i in range(1000):
            result.append(i ** 2)
        return result
    
    def list_comprehension():
        return [i ** 2 for i in range(1000)]
    
    def generator_expression():
        return list(i ** 2 for i in range(1000))
    
    results = compare_performance(
        list_append,
        list_comprehension,
        generator_expression,
        iterations=1000
    )
    
    print("Performance comparison results:")
    for func_name, avg_time in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {func_name}: {avg_time:.8f} seconds average")

### Cython usage

Cython allows writing C extensions using Python-like syntax for performance improvements.

```python
# Note: This is example Cython code
# In practice, this would be in .pyx files and compiled

CYTHON_EXAMPLES = '''
# 1. Basic Cython function (would be in a .pyx file)
def pure_python_sum(data):
    """Pure Python sum function"""
    total = 0
    for item in data:
        total += item
    return total

# Cython version with type declarations
def cython_sum(double[:] data):
    """Cython sum with memory view"""
    cdef double total = 0.0
    cdef int i
    cdef int n = data.shape[0]
    
    for i in range(n):
        total += data[i]
    
    return total

# 2. Cython class example
cdef class FastCalculator:
    """Cython class for fast calculations"""
    cdef double factor
    
    def __init__(self, double factor):
        self.factor = factor
    
    cdef double _multiply(self, double value):
        """Private Cython method"""
        return value * self.factor
    
    def multiply(self, double value):
        """Public method"""
        return self._multiply(value)
    
    def multiply_array(self, double[:] values):
        """Process array efficiently"""
        cdef int i
        cdef int n = values.shape[0]
        cdef double[:] result = values.copy()
        
        for i in range(n):
            result[i] = self._multiply(values[i])
        
        return result

# 3. Integration with NumPy
import numpy as np
cimport numpy as cnp

def cython_matrix_multiply(cnp.ndarray[double, ndim=2] A,
                          cnp.ndarray[double, ndim=2] B):
    """Cython matrix multiplication"""
    cdef int i, j, k
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int p = B.shape[1]
    
    cdef cnp.ndarray[double, ndim=2] C = np.zeros((m, p), dtype=np.float64)
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C

# 4. Setup.py for compilation
setup_py_content = """
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("fast_module.pyx"),
    include_dirs=[numpy.get_include()]
)
"""

# Usage example
def demonstrate_cython_benefits():
    """Show the benefits of using Cython"""
    # This would be the comparison if Cython module was compiled
    import numpy as np
    
    data = np.random.random(1000000)
    
    # Pure Python timing
    start = time.perf_counter()
    python_result = sum(data)
    python_time = time.perf_counter() - start
    
    # NumPy timing (for comparison)
    start = time.perf_counter()
    numpy_result = np.sum(data)
    numpy_time = time.perf_counter() - start
    
    print("Cython Performance Benefits:")
    print(f"Pure Python sum: {python_time:.6f} seconds")
    print(f"NumPy sum: {numpy_time:.6f} seconds")
    print(f"Speedup with NumPy: {python_time / numpy_time:.1f}x")
    print("Cython would typically give 10-100x speedup over pure Python")
'''

def cython_usage_examples():
    """Demonstrate Cython concepts"""
    print("=== Cython Usage ===")
    print("Cython allows writing C extensions with Python-like syntax")
    print("\nKey Cython features:")
    print("• Static type declarations (cdef)")
    print("• Memory views for efficient array access")
    print("• Direct C API access")
    print("• Automatic Python/C type conversion")
    print("• Optional GIL release for parallel processing")
    
    print("\nTypical performance improvements:")
    print("• 2-10x for general code")
    print("• 10-100x for numerical computations")
    print("• 100-1000x for tight loops with minimal Python interaction")

### Multiprocessing vs threading

```python
import threading
import multiprocessing
import time
import concurrent.futures
from queue import Queue
import os

# CPU-bound task for comparison
def cpu_intensive_task(n):
    """CPU-intensive task that benefits from multiprocessing"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# I/O-bound task for comparison
def io_intensive_task(duration):
    """I/O-intensive task that benefits from threading"""
    time.sleep(duration)
    return f"Task completed after {duration} seconds"

def network_simulation(url):
    """Simulate network request"""
    # Simulate varying network delays
    delay = 0.1 + (hash(url) % 10) / 100
    time.sleep(delay)
    return f"Data from {url}"

def threading_example():
    """Demonstrate threading for I/O-bound tasks"""
    print("=== Threading Example ===")
    
    urls = [f"https://api{i}.example.com" for i in range(10)]
    
    # Sequential execution
    start_time = time.time()
    sequential_results = []
    for url in urls:
        result = network_simulation(url)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Threaded execution
    start_time = time.time()
    threaded_results = []
    
    def worker(url, results, index):
        result = network_simulation(url)
        results[index] = result
    
    threads = []
    threaded_results = [None] * len(urls)
    
    for i, url in enumerate(urls):
        thread = threading.Thread(target=worker, args=(url, threaded_results, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    threaded_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Threaded time: {threaded_time:.2f} seconds")
    print(f"Threading speedup: {sequential_time / threaded_time:.1f}x")

def multiprocessing_example():
    """Demonstrate multiprocessing for CPU-bound tasks"""
    print("\n=== Multiprocessing Example ===")
    
    tasks = [1000000] * 4  # 4 CPU-intensive tasks
    
    # Sequential execution
    start_time = time.time()
    sequential_results = []
    for task in tasks:
        result = cpu_intensive_task(task)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Multiprocessing execution
    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        multiprocess_results = pool.map(cpu_intensive_task, tasks)
    multiprocess_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f} seconds")
    print(f"Multiprocess time: {multiprocess_time:.2f} seconds")
    print(f"Multiprocessing speedup: {sequential_time / multiprocess_time:.1f}x")

def concurrent_futures_examples():
    """Demonstrate concurrent.futures for both threading and multiprocessing"""
    print("\n=== Concurrent Futures ===")
    
    # Threading with ThreadPoolExecutor
    urls = [f"https://api{i}.example.com" for i in range(8)]
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_url = {executor.submit(network_simulation, url): url for url in urls}
        threading_results = []
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                threading_results.append(result)
            except Exception as exc:
                print(f"URL {url} generated exception: {exc}")
    
    threading_time = time.time() - start_time
    
    # Multiprocessing with ProcessPoolExecutor
    tasks = [500000] * 4
    
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        process_results = list(executor.map(cpu_intensive_task, tasks))
    process_time = time.time() - start_time
    
    print(f"ThreadPoolExecutor time: {threading_time:.2f} seconds")
    print(f"ProcessPoolExecutor time: {process_time:.2f} seconds")

def producer_consumer_examples():
    """Demonstrate producer-consumer patterns"""
    print("\n=== Producer-Consumer Patterns ===")
    
    # Threading version
    def threading_producer_consumer():
        """Producer-consumer with threading"""
        queue = Queue(maxsize=10)
        
        def producer():
            for i in range(20):
                item = f"item_{i}"
                queue.put(item)
                print(f"Produced {item}")
                time.sleep(0.1)
            
            # Signal completion
            queue.put(None)
        
        def consumer(consumer_id):
            while True:
                item = queue.get()
                if item is None:
                    queue.put(None)  # Pass signal to other consumers
                    break
                
                print(f"Consumer {consumer_id} consumed {item}")
                time.sleep(0.15)
                queue.task_done()
        
        # Start producer and consumers
        producer_thread = threading.Thread(target=producer)
        consumer_threads = [
            threading.Thread(target=consumer, args=(i,))
            for i in range(3)
        ]
        
        producer_thread.start()
        for thread in consumer_threads:
            thread.start()
        
        producer_thread.join()
        for thread in consumer_threads:
            thread.join()
    
    # Multiprocessing version
    def multiprocessing_producer_consumer():
        """Producer-consumer with multiprocessing"""
        queue = multiprocessing.Queue(maxsize=10)
        
        def producer(q):
            for i in range(20):
                item = f"item_{i}"
                q.put(item)
                print(f"Produced {item}")
                time.sleep(0.1)
            
            # Signal completion
            q.put(None)
        
        def consumer(q, consumer_id):
            while True:
                item = q.get()
                if item is None:
                    q.put(None)  # Pass signal to other consumers
                    break
                
                print(f"Consumer {consumer_id} consumed {item}")
                time.sleep(0.15)
        
        # Start processes
        producer_process = multiprocessing.Process(target=producer, args=(queue,))
        consumer_processes = [
            multiprocessing.Process(target=consumer, args=(queue, i))
            for i in range(3)
        ]
        
        producer_process.start()
        for process in consumer_processes:
            process.start()
        
        producer_process.join()
        for process in consumer_processes:
            process.join()
    
    print("Threading producer-consumer:")
    threading_producer_consumer()
    
    print("\nMultiprocessing producer-consumer:")
    multiprocessing_producer_consumer()

def comparison_guide():
    """Guide for choosing between threading and multiprocessing"""
    print("\n=== Threading vs Multiprocessing Guide ===")
    
    comparison = {
        "Use Threading For": [
            "I/O-bound tasks (file operations, network requests)",
            "Tasks that spend time waiting",
            "Shared memory access",
            "Lower overhead for task switching",
            "UI responsiveness"
        ],
        "Use Multiprocessing For": [
            "CPU-bound tasks (calculations, data processing)",
            "Tasks that can run completely independently",
            "Bypassing GIL limitations",
            "Fault isolation (process crashes don't affect others)",
            "True parallelism on multi-core systems"
        ],
        "Threading Limitations": [
            "GIL prevents true CPU parallelism",
            "Shared memory can cause race conditions",
            "Debugging can be complex",
            "Limited scalability for CPU-bound tasks"
        ],
        "Multiprocessing Limitations": [
            "Higher memory overhead",
            "Inter-process communication is more expensive",
            "Serialization overhead for data sharing",
            "More complex setup and coordination"
        ]
    }
    
    for category, items in comparison.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

def performance_examples():
    """Run all performance examples"""
    cProfile_example()
    advanced_profiling()
    timing_examples()
    performance_comparison_example()
    cython_usage_examples()
    threading_example()
    multiprocessing_example()
    concurrent_futures_examples()
    producer_consumer_examples()
    comparison_guide()

## Testing and Development

### Unit testing frameworks

```python
import unittest
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Example classes to test
class Calculator:
    """Simple calculator for testing examples"""
    
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base, exponent):
        return base ** exponent

class FileProcessor:
    """File processor for testing file operations"""
    
    def read_file(self, filename):
        with open(filename, 'r') as f:
            return f.read()
    
    def write_file(self, filename, content):
        with open(filename, 'w') as f:
            f.write(content)
    
    def process_data(self, data):
        """Process data and return statistics"""
        if not data:
            return {"count": 0, "sum": 0, "average": 0}
        
        numbers = [float(x) for x in data.split(',') if x.strip()]
        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers) if numbers else 0
        }

# 1. unittest examples
class TestCalculatorUnittest(unittest.TestCase):
    """unittest example for Calculator"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.calc = Calculator()
    
    def tearDown(self):
        """Clean up after each test method"""
        pass  # Nothing to clean up in this example
    
    def test_add(self):
        """Test addition operation"""
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(0, 0), 0)
    
    def test_subtract(self):
        """Test subtraction operation"""
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(1, 1), 0)
        self.assertEqual(self.calc.subtract(0, 5), -5)
    
    def test_multiply(self):
        """Test multiplication operation"""
        self.assertEqual(self.calc.multiply(3, 4), 12)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(0, 100), 0)
    
    def test_divide(self):
        """Test division operation"""
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertEqual(self.calc.divide(9, 3), 3)
        self.assertAlmostEqual(self.calc.divide(1, 3), 0.333333, places=5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises exception"""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
        
        with self.assertRaisesRegex(ValueError, "Cannot divide by zero"):
            self.calc.divide(5, 0)
    
    def test_power(self):
        """Test power operation"""
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(5, 0), 1)
        self.assertEqual(self.calc.power(10, 1), 10)
    
    @unittest.skip("Skipping this test for demonstration")
    def test_skipped(self):
        """This test will be skipped"""
        self.fail("This test should not run")
    
    @unittest.skipIf(os.name == 'nt', "Skip on Windows")
    def test_conditional_skip(self):
        """Test that may be skipped based on condition"""
        self.assertTrue(True)

# 2. pytest examples
class TestCalculatorPytest:
    """pytest example for Calculator"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition with pytest"""
        assert self.calc.add(2, 3) == 5
        assert self.calc.add(-1, 1) == 0
        assert self.calc.add(0, 0) == 0
    
    def test_subtract(self):
        """Test subtraction with pytest"""
        assert self.calc.subtract(5, 3) == 2
        assert self.calc.subtract(1, 1) == 0
    
    def test_divide_by_zero(self):
        """Test division by zero with pytest"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(10, 0)
    
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (-1, 1, 0),
        (0, 0, 0),
        (100, -50, 50),
    ])
    def test_add_parametrized(self, a, b, expected):
        """Parametrized test for addition"""
        assert self.calc.add(a, b) == expected
    
    @pytest.mark.slow
    def test_large_power(self):
        """Test marked as slow"""
        result = self.calc.power(2, 1000)
        assert result > 0
    
    @pytest.mark.skip(reason="Not implemented yet")
    def test_future_feature(self):
        """Test for future feature"""
        pass

# 3. Fixtures with pytest
@pytest.fixture
def calculator():
    """Pytest fixture for calculator"""
    return Calculator()

@pytest.fixture
def sample_data():
    """Pytest fixture for sample data"""
    return {
        "numbers": [1, 2, 3, 4, 5],
        "strings": ["hello", "world", "python"],
        "mixed": [1, "two", 3.0, True]
    }

@pytest.fixture(scope="session")
def temp_directory():
    """Session-scoped fixture for temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after session
    import shutil
    shutil.rmtree(temp_dir)

def test_with_fixtures(calculator, sample_data):
    """Test using fixtures"""
    numbers = sample_data["numbers"]
    total = sum(numbers)
    
    # Test that our calculator gives same result
    result = 0
    for num in numbers:
        result = calculator.add(result, num)
    
    assert result == total

# 4. Mocking examples
class TestFileProcessorMocking(unittest.TestCase):
    """Test FileProcessor with mocking"""
    
    def setUp(self):
        self.processor = FileProcessor()
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="test content")
    def test_read_file_mock(self, mock_file):
        """Test file reading with mock"""
        result = self.processor.read_file("test.txt")
        
        self.assertEqual(result, "test content")
        mock_file.assert_called_once_with("test.txt", 'r')
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_write_file_mock(self, mock_file):
        """Test file writing with mock"""
        content = "Hello, World!"
        self.processor.write_file("output.txt", content)
        
        mock_file.assert_called_once_with("output.txt", 'w')
        mock_file().write.assert_called_once_with(content)
    
    def test_process_data(self):
        """Test data processing without mocking"""
        data = "1,2,3,4,5"
        result = self.processor.process_data(data)
        
        expected = {"count": 5, "sum": 15.0, "average": 3.0}
        self.assertEqual(result, expected)
    
    def test_process_empty_data(self):
        """Test processing empty data"""
        result = self.processor.process_data("")
        
        expected = {"count": 0, "sum": 0, "average": 0}
        self.assertEqual(result, expected)

# 5. Advanced mocking with pytest
def test_mock_with_side_effect():
    """Test using mock with side effects"""
    mock_func = Mock(side_effect=[1, 2, ValueError("Error on third call")])
    
    assert mock_func() == 1
    assert mock_func() == 2
    
    with pytest.raises(ValueError):
        mock_func()

def test_mock_configuration():
    """Test mock configuration"""
    mock_obj = Mock()
    
    # Configure return values
    mock_obj.method1.return_value = "configured_result"
    mock_obj.method2.side_effect = lambda x: x * 2
    
    assert mock_obj.method1() == "configured_result"
    assert mock_obj.method2(5) == 10
    
    # Verify calls
    mock_obj.method1.assert_called_once()
    mock_obj.method2.assert_called_with(5)

@patch('requests.get')
def test_external_api_mock(mock_get):
    """Test mocking external API calls"""
    # Configure mock response
    mock_response = Mock()
    mock_response.json.return_value = {"status": "success", "data": [1, 2, 3]}
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    # Code that would make API call
    def fetch_data(url):
        import requests
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    
    result = fetch_data("https://api.example.com/data")
    
    assert result == {"status": "success", "data": [1, 2, 3]}
    mock_get.assert_called_once_with("https://api.example.com/data")

# 6. Test organization and discovery
def run_unittest_examples():
    """Run unittest examples"""
    print("=== unittest Examples ===")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculatorUnittest)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

def pytest_configuration_example():
    """Example pytest configuration"""
    pytest_ini_content = """
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
markers =
    slow: marks tests as slow
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    smoke: marks tests as smoke tests
"""
    
    print("=== pytest Configuration ===")
    print("Example pytest.ini configuration:")
    print(pytest_ini_content)

def testing_best_practices():
    """Testing best practices and guidelines"""
    print("\n=== Testing Best Practices ===")
    
    practices = {
        "Test Structure": [
            "Follow AAA pattern: Arrange, Act, Assert",
            "One assertion per test (when possible)",
            "Clear, descriptive test names",
            "Independent tests (no dependencies between tests)"
        ],
        "Test Coverage": [
            "Aim for high coverage but focus on critical paths",
            "Test edge cases and error conditions", 
            "Cover both positive and negative scenarios",
            "Don't forget integration tests"
        ],
        "Mocking Guidelines": [
            "Mock external dependencies",
            "Don't mock what you don't own (sometimes)",
            "Use mocks to isolate units under test",
            "Verify mock interactions when relevant"
        ],
        "Test Organization": [
            "Group related tests in classes",
            "Use fixtures for common setup",
            "Separate unit, integration, and e2e tests",
            "Keep tests close to the code they test"
        ],
        "Performance": [
            "Keep tests fast",
            "Use appropriate test database strategies",
            "Parallel test execution when possible",
            "Profile slow tests and optimize"
        ]
    }
    
    for category, items in practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

def testing_examples():
    """Run all testing examples"""
    run_unittest_examples()
    pytest_configuration_example()
    testing_best_practices()
    
    print("\n=== Summary ===")
    print("Python testing ecosystem provides:")
    print("• unittest: Built-in testing framework")
    print("• pytest: Popular third-party framework with many features")
    print("• Mock/patch: For isolating dependencies")
    print("• Fixtures: For test setup and teardown")
    print("• Parametrized tests: For testing multiple scenarios")
    print("• Coverage tools: For measuring test effectiveness")

**Interview Tips for Python Interview Guide:**

1. **Preparation Strategy:**
   - Understand Python's memory model and GIL
   - Practice with advanced features like decorators and metaclasses
   - Be familiar with async programming concepts
   - Know when to use different web frameworks

2. **Common Mistakes to Avoid:**
   - Circular imports and module design issues
   - Not understanding mutable default arguments
   - Inefficient use of data structures
   - Blocking the event loop in async code

3. **Best Practices to Mention:**
   - Use virtual environments
   - Follow PEP 8 style guidelines
   - Write comprehensive tests
   - Use type hints for better code documentation
   - Profile before optimizing

4. **Real-world Scenarios:**
   - API design and development
   - Data processing pipelines
   - Web application architecture
   - Performance optimization strategies

This comprehensive Python interview guide covers essential topics for senior Python developers, providing both theoretical knowledge and practical examples for interviews and professional development.
```
```
```
```