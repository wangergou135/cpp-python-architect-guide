# Python高级特性指南 / Python Advanced Features Guide

*最后更新 / Last Updated: 2025-08-10 10:53:08 UTC*
*作者 / Author: @wangergou135*

## 目录 / Table of Contents
1. [装饰器 / Decorators](#装饰器--decorators)
2. [元类 / Metaclasses](#元类--metaclasses)
3. [上下文管理器 / Context Managers](#上下文管理器--context-managers)
4. [迭代器和生成器 / Iterators and Generators](#迭代器和生成器--iterators-and-generators)
5. [描述符 / Descriptors](#描述符--descriptors)
6. [协程和异步编程 / Coroutines and Async Programming](#协程和异步编程--coroutines-and-async-programming)
7. [类型提示 / Type Hints](#类型提示--type-hints)
8. [内存管理和优化 / Memory Management and Optimization](#内存管理和优化--memory-management-and-optimization)

## 装饰器 / Decorators

### 基础装饰器 / Basic Decorators
```python
def timer(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

@timer
def example_function():
    pass
```

### 带参数的装饰器 / Parameterized Decorators
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def example():
    pass
```

## 元类 / Metaclasses

### 基本元类 / Basic Metaclass
```python
class MetaExample(type):
    def __new__(cls, name, bases, attrs):
        # Modify class attributes here
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=MetaExample):
    pass
```

### 元类应用 / Metaclass Applications
```python
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
```

## 上下文管理器 / Context Managers

### 使用类 / Using Class
```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
        
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
```

### 使用装饰器 / Using Decorator
```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()
```

## 迭代器和生成器 / Iterators and Generators

### 迭代器 / Iterators
```python
class CountUp:
    def __init__(self, start, end):
        self.current = start
        self.end = end
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        self.current += 1
        return self.current - 1
```

### 生成器 / Generators
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
```

## 描述符 / Descriptors

### 数据描述符 / Data Descriptors
```python
class Validated:
    def __init__(self, minvalue=None, maxvalue=None):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
        
    def __set__(self, instance, value):
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError
        instance.__dict__[self.name] = value
```

## 协程和异步编程 / Coroutines and Async Programming

### 基本协程 / Basic Coroutines
```python
async def main():
    print('开始 / Start')
    await asyncio.sleep(1)
    print('结束 / End')

# 运行协程 / Run coroutine
asyncio.run(main())
```

### 异步上下文管理器 / Async Context Managers
```python
class AsyncResource:
    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
```

## 类型提示 / Type Hints

### 基本类型提示 / Basic Type Hints
```python
from typing import List, Dict, Optional, Union

def process_data(items: List[int]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    return result

def optional_parameter(value: Optional[str] = None) -> Union[str, int]:
    pass
```

## 内存管理和优化 / Memory Management and Optimization

### 弱引用 / Weak References
```python
import weakref

class Cache:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
        
    def get(self, key):
        return self._cache.get(key)
        
    def set(self, key, value):
        self._cache[key] = value
```

### 性能优化技巧 / Performance Optimization Tips
```python
# 使用列表推导式 / Using List Comprehension
squares = [x**2 for x in range(10)]

# 使用生成器表达式 / Using Generator Expression
sum_squares = sum(x**2 for x in range(10))

# 使用__slots__ / Using __slots__
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

---

*注意：本文档会定期更新以反映Python的最新特性和最佳实践。*
*Note: This document will be updated periodically to reflect the latest Python features and best practices.*