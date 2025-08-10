# Python Interview Guide

A comprehensive guide for Python technical interviews covering fundamental concepts, advanced topics, common questions, and modern practices.

## Table of Contents

1. [Fundamental Concepts](#fundamental-concepts)
2. [Advanced Topics](#advanced-topics)
3. [Common Interview Questions](#common-interview-questions)
4. [Best Practices](#best-practices)
5. [Real-World Scenarios](#real-world-scenarios)
6. [Coding Patterns and Anti-Patterns](#coding-patterns-and-anti-patterns)
7. [Performance Considerations](#performance-considerations)
8. [Modern Python Practices and Trends](#modern-python-practices-and-trends)

## Fundamental Concepts

### 1. Python Basics

#### Data Types and Variables
```python
# Basic types
integer = 42
floating = 3.14
string = "Hello, World!"
boolean = True
none_type = None

# Collections
list_example = [1, 2, 3, 4]
tuple_example = (1, 2, 3, 4)
dict_example = {'key': 'value', 'age': 30}
set_example = {1, 2, 3, 4}

# Type hints (Python 3.5+)
from typing import List, Dict, Optional

def process_items(items: List[int]) -> Dict[str, int]:
    return {'count': len(items), 'sum': sum(items)}
```

#### Control Flow
```python
# Conditional statements
if condition:
    pass
elif another_condition:
    pass
else:
    pass

# Loops
for item in iterable:
    if item == target:
        break
    if item == skip:
        continue
    process(item)

# List comprehensions
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dictionary comprehensions
word_lengths = {word: len(word) for word in ['hello', 'world']}

# Set comprehensions
unique_lengths = {len(word) for word in words}
```

#### Functions
```python
# Basic function
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

# Variable arguments
def process_args(*args, **kwargs):
    print(f"Positional: {args}")
    print(f"Keyword: {kwargs}")

# Lambda functions
square = lambda x: x**2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))

# Higher-order functions
def apply_operation(operation, numbers):
    return [operation(x) for x in numbers]
```

### 2. Object-Oriented Programming

#### Classes and Objects
```python
class Person:
    class_variable = "Human"
    
    def __init__(self, name: str, age: int):
        self.name = name          # Public attribute
        self._age = age          # Protected attribute (convention)
        self.__id = id(self)     # Private attribute (name mangling)
    
    def __str__(self) -> str:
        return f"Person(name={self.name}, age={self._age})"
    
    def __repr__(self) -> str:
        return f"Person('{self.name}', {self._age})"
    
    def celebrate_birthday(self) -> None:
        self._age += 1
    
    @property
    def age(self) -> int:
        return self._age
    
    @age.setter
    def age(self, value: int) -> None:
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value
    
    @classmethod
    def from_string(cls, person_str: str):
        name, age = person_str.split('-')
        return cls(name, int(age))
    
    @staticmethod
    def is_adult(age: int) -> bool:
        return age >= 18
```

#### Inheritance and Polymorphism
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius
    
    def area(self) -> float:
        return 3.14159 * self.radius ** 2
    
    def perimeter(self) -> float:
        return 2 * 3.14159 * self.radius

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def area(self) -> float:
        return self.width * self.height
    
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

# Multiple inheritance
class Flyable:
    def fly(self):
        return "Flying!"

class Bird(Shape, Flyable):
    def __init__(self, wingspan: float):
        self.wingspan = wingspan
    
    def area(self) -> float:
        return self.wingspan ** 2 * 0.5
    
    def perimeter(self) -> float:
        return self.wingspan * 4
```

## Advanced Topics

### 1. Decorators

#### Function Decorators
```python
import functools
import time
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def cache(func: Callable) -> Callable:
    """Simple memoization decorator."""
    cache_dict = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache_dict:
            cache_dict[key] = func(*args, **kwargs)
        return cache_dict[key]
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator with parameters for retry logic."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Usage examples
@timer
@cache
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@retry(max_attempts=5, delay=0.5)
def unreliable_api_call():
    # Simulate API call that might fail
    import random
    if random.random() < 0.7:
        raise ConnectionError("API temporarily unavailable")
    return "Success!"
```

#### Class Decorators
```python
def singleton(cls):
    """Singleton decorator for classes."""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        self.connection = "Connected to database"

# Property decorators
class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return (self._celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self.celsius = (value - 32) * 5/9
```

### 2. Generators and Iterators

#### Generators
```python
def fibonacci_generator():
    """Generate Fibonacci sequence infinitely."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def read_large_file(filename: str):
    """Memory-efficient file reading."""
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()

def process_data_pipeline(data):
    """Data processing pipeline with generators."""
    # Filter step
    for item in data:
        if item % 2 == 0:
            yield item
    
    # Transform step
    for item in filter_even(data):
        yield item ** 2
    
    # Aggregate step
    def batch_generator(data, batch_size=10):
        batch = []
        for item in data:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Generator expressions
squares = (x**2 for x in range(1000000))  # Memory efficient
even_squares = (x for x in squares if x % 2 == 0)
```

#### Custom Iterators
```python
class CountDown:
    """Custom iterator for countdown."""
    
    def __init__(self, start: int):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Usage
for num in CountDown(5):
    print(num)  # 5, 4, 3, 2, 1
```

### 3. Context Managers

#### Built-in Context Managers
```python
# File handling
with open('file.txt', 'r') as file:
    content = file.read()

# Thread locks
import threading
lock = threading.Lock()

with lock:
    # Critical section
    shared_resource += 1
```

#### Custom Context Managers
```python
from contextlib import contextmanager
import time
import sqlite3

class Timer:
    """Context manager for timing operations."""
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start
        print(f"Operation took {self.duration:.4f} seconds")

@contextmanager
def database_transaction(db_path: str):
    """Context manager for database transactions."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

@contextmanager
def temporary_attribute(obj, attr_name: str, temp_value):
    """Temporarily change an object's attribute."""
    old_value = getattr(obj, attr_name, None)
    setattr(obj, attr_name, temp_value)
    try:
        yield obj
    finally:
        if old_value is not None:
            setattr(obj, attr_name, old_value)
        else:
            delattr(obj, attr_name)

# Usage examples
with Timer():
    time.sleep(1)  # Simulate work

with database_transaction('example.db') as conn:
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name) VALUES (?)", ("John",))
```

### 4. Metaclasses

#### Understanding Metaclasses
```python
class SingletonMeta(type):
    """Metaclass for implementing singleton pattern."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "Database connection"

# Validation metaclass
class ValidatedMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Validate class definition
        if 'validate' not in namespace:
            raise TypeError(f"Class {name} must implement validate method")
        return super().__new__(mcs, name, bases, namespace)

class User(metaclass=ValidatedMeta):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
    
    def validate(self) -> bool:
        return '@' in self.email and len(self.name) > 0
```

## Common Interview Questions

### 1. Python Fundamentals

**Q: What's the difference between list and tuple?**
A: 
- **List**: Mutable, ordered, allows duplicates, defined with `[]`
- **Tuple**: Immutable, ordered, allows duplicates, defined with `()`
- **Use cases**: Lists for changing data, tuples for fixed data like coordinates

**Q: Explain Python's GIL (Global Interpreter Lock).**
A: The GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This means:
- True parallelism is limited for CPU-bound tasks
- I/O-bound tasks can still benefit from threading
- Use multiprocessing for CPU-bound parallelism

**Q: What are *args and **kwargs?**
```python
def example_function(*args, **kwargs):
    print(f"Positional arguments: {args}")
    print(f"Keyword arguments: {kwargs}")

example_function(1, 2, 3, name="John", age=30)
# Output:
# Positional arguments: (1, 2, 3)
# Keyword arguments: {'name': 'John', 'age': 30}
```

### 2. Memory Management

**Q: How does Python's garbage collection work?**
A: Python uses:
1. **Reference counting**: Objects are deleted when reference count reaches zero
2. **Cycle detection**: Handles circular references using generational garbage collection
3. **Memory pools**: Efficient allocation for small objects

**Q: What's the difference between `==` and `is`?**
```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)  # True (same content)
print(a is b)  # False (different objects)
print(a is c)  # True (same object)
```

### 3. Functional Programming

**Q: Explain map, filter, and reduce.**
```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# map: apply function to each element
squared = list(map(lambda x: x**2, numbers))

# filter: select elements that meet condition
evens = list(filter(lambda x: x % 2 == 0, numbers))

# reduce: combine elements using function
sum_all = reduce(lambda x, y: x + y, numbers)

# Modern Pythonic alternatives
squared = [x**2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
sum_all = sum(numbers)
```

### 4. Advanced Concepts

**Q: What are Python descriptors?**
```python
class ValidatedAttribute:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.value = None
    
    def __get__(self, obj, objtype=None):
        return self.value
    
    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Value must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Value must be <= {self.max_value}")
        self.value = value

class Person:
    age = ValidatedAttribute(min_value=0, max_value=150)
    
    def __init__(self, age):
        self.age = age
```

## Best Practices

### 1. Code Style and PEP 8

#### Naming Conventions
```python
# Constants
MAX_CONNECTIONS = 100
API_KEY = "secret-key"

# Functions and variables
def calculate_total_price(items):
    total_price = 0
    for item in items:
        total_price += item.price
    return total_price

# Classes
class ShoppingCart:
    def __init__(self):
        self._items = []
    
    def add_item(self, item):
        self._items.append(item)

# Private methods (convention)
class APIClient:
    def __init__(self, api_key):
        self._api_key = api_key
    
    def _authenticate(self):
        # Private method
        pass
    
    def get_data(self):
        self._authenticate()
        # Public method
        pass
```

#### Documentation and Type Hints
```python
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float
    category: str
    in_stock: bool = True

def calculate_discount(
    products: List[Product], 
    discount_rate: float,
    category_filter: Optional[str] = None
) -> Dict[str, Union[float, int]]:
    """
    Calculate discount for products.
    
    Args:
        products: List of products to process
        discount_rate: Discount rate (0.0 to 1.0)
        category_filter: Optional category to filter by
    
    Returns:
        Dictionary with total_discount and affected_products count
    
    Raises:
        ValueError: If discount_rate is not between 0 and 1
    """
    if not 0 <= discount_rate <= 1:
        raise ValueError("Discount rate must be between 0 and 1")
    
    filtered_products = [
        p for p in products 
        if category_filter is None or p.category == category_filter
    ]
    
    total_discount = sum(p.price * discount_rate for p in filtered_products)
    
    return {
        'total_discount': total_discount,
        'affected_products': len(filtered_products)
    }
```

### 2. Error Handling

#### Exception Handling Best Practices
```python
import logging

# Specific exception handling
def safe_divide(a: float, b: float) -> Optional[float]:
    try:
        return a / b
    except ZeroDivisionError:
        logging.error(f"Division by zero attempted: {a} / {b}")
        return None
    except TypeError as e:
        logging.error(f"Type error in division: {e}")
        raise ValueError("Invalid input types for division")

# Custom exceptions
class ValidationError(Exception):
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error in {field}: {message}")

def validate_email(email: str) -> None:
    if '@' not in email:
        raise ValidationError('email', 'Must contain @ symbol')
    if '.' not in email.split('@')[1]:
        raise ValidationError('email', 'Invalid domain format')

# Context manager for exception handling
@contextmanager
def handle_api_errors():
    try:
        yield
    except requests.ConnectionError:
        logging.error("API connection failed")
        raise APIError("Unable to connect to service")
    except requests.Timeout:
        logging.error("API request timeout")
        raise APIError("Request timeout")
```

### 3. Testing

#### Unit Testing with pytest
```python
import pytest
from unittest.mock import Mock, patch

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
    
    def divide(self, a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

class TestCalculator:
    def setup_method(self):
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        result = self.calc.add(2, 3)
        assert result == 5
    
    def test_add_negative_numbers(self):
        result = self.calc.add(-2, -3)
        assert result == -5
    
    def test_divide_normal_case(self):
        result = self.calc.divide(10, 2)
        assert result == 5.0
    
    def test_divide_by_zero_raises_error(self):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(10, 0)
    
    @pytest.mark.parametrize("a,b,expected", [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300)
    ])
    def test_add_parametrized(self, a, b, expected):
        assert self.calc.add(a, b) == expected

# Mocking external dependencies
class APIService:
    def get_user_data(self, user_id: int) -> dict:
        # External API call
        pass

class UserService:
    def __init__(self, api_service: APIService):
        self.api_service = api_service
    
    def get_user_name(self, user_id: int) -> str:
        user_data = self.api_service.get_user_data(user_id)
        return user_data.get('name', 'Unknown')

class TestUserService:
    def test_get_user_name_success(self):
        # Mock the API service
        mock_api = Mock()
        mock_api.get_user_data.return_value = {'name': 'John Doe', 'age': 30}
        
        user_service = UserService(mock_api)
        name = user_service.get_user_name(123)
        
        assert name == 'John Doe'
        mock_api.get_user_data.assert_called_once_with(123)
    
    @patch('requests.get')
    def test_external_api_call(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'success'}
        mock_get.return_value = mock_response
        
        # Test code that makes HTTP requests
        pass
```

## Real-World Scenarios

### 1. Web Development

#### Flask Application Structure
```python
from flask import Flask, request, jsonify
from functools import wraps
import jwt
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['POST'])
def login():
    auth = request.get_json()
    
    if auth and auth['username'] == 'admin' and auth['password'] == 'password':
        token = jwt.encode({
            'user': auth['username'],
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/protected', methods=['GET'])
@token_required
def protected():
    return jsonify({'message': 'This is a protected endpoint'})

# Error handling
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
```

#### Database Operations with SQLAlchemy
```python
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from contextlib import contextmanager

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(String(1000))
    user_id = Column(Integer, ForeignKey('users.id'))
    author = relationship("User", back_populates="posts")

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_user(self, username: str, email: str) -> User:
        with self.get_session() as session:
            user = User(username=username, email=email)
            session.add(user)
            session.flush()  # Get the ID
            return user
    
    def get_user_posts(self, user_id: int) -> List[Post]:
        with self.get_session() as session:
            return session.query(Post).filter(Post.user_id == user_id).all()
```

### 2. Data Processing

#### ETL Pipeline
```python
import pandas as pd
import numpy as np
from typing import Iterator, Dict, Any
import logging

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_data(self, source: str) -> Iterator[Dict]:
        """Extract data from various sources."""
        if source.endswith('.csv'):
            for chunk in pd.read_csv(source, chunksize=1000):
                for _, row in chunk.iterrows():
                    yield row.to_dict()
        elif source.endswith('.json'):
            import json
            with open(source, 'r') as f:
                data = json.load(f)
                for item in data:
                    yield item
    
    def transform_data(self, data: Dict) -> Dict:
        """Transform individual data records."""
        try:
            # Data cleaning
            transformed = {}
            for key, value in data.items():
                # Handle missing values
                if pd.isna(value):
                    transformed[key] = None
                # Normalize strings
                elif isinstance(value, str):
                    transformed[key] = value.strip().lower()
                # Convert numeric strings
                elif isinstance(value, str) and value.isdigit():
                    transformed[key] = int(value)
                else:
                    transformed[key] = value
            
            # Business logic transformations
            if 'birth_date' in transformed:
                from datetime import datetime
                birth_date = datetime.strptime(transformed['birth_date'], '%Y-%m-%d')
                transformed['age'] = (datetime.now() - birth_date).days // 365
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            return None
    
    def load_data(self, data: Iterator[Dict], destination: str):
        """Load data to destination."""
        valid_records = []
        error_count = 0
        
        for record in data:
            transformed = self.transform_data(record)
            if transformed:
                valid_records.append(transformed)
            else:
                error_count += 1
            
            # Batch processing
            if len(valid_records) >= 100:
                self._write_batch(valid_records, destination)
                valid_records = []
        
        # Write remaining records
        if valid_records:
            self._write_batch(valid_records, destination)
        
        self.logger.info(f"Processed data with {error_count} errors")
    
    def _write_batch(self, records: List[Dict], destination: str):
        """Write batch of records to destination."""
        df = pd.DataFrame(records)
        if destination.endswith('.csv'):
            df.to_csv(destination, mode='a', header=False, index=False)
        elif destination.endswith('.parquet'):
            df.to_parquet(destination, engine='pyarrow')

# Usage
processor = DataProcessor({'batch_size': 100})
data_stream = processor.extract_data('input.csv')
processor.load_data(data_stream, 'output.parquet')
```

### 3. Asynchronous Programming

#### Async/Await Patterns
```python
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict
import time

class AsyncDataFetcher:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str) -> Dict:
        """Fetch data from a single URL with rate limiting."""
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'url': url, 'data': data, 'status': 'success'}
                    else:
                        return {'url': url, 'error': f'HTTP {response.status}', 'status': 'error'}
            except Exception as e:
                return {'url': url, 'error': str(e), 'status': 'error'}
    
    async def fetch_multiple_urls(self, urls: List[str]) -> List[Dict]:
        """Fetch data from multiple URLs concurrently."""
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result), 
                    'status': 'exception'
                })
            else:
                processed_results.append(result)
        
        return processed_results

async def process_data_async(data: List[Dict]) -> List[Dict]:
    """Async data processing function."""
    
    async def process_item(item: Dict) -> Dict:
        # Simulate async processing
        await asyncio.sleep(0.1)
        return {
            'id': item.get('id'),
            'processed_at': time.time(),
            'result': item.get('value', 0) * 2
        }
    
    tasks = [process_item(item) for item in data]
    return await asyncio.gather(*tasks)

async def file_operations():
    """Async file operations."""
    
    # Read file asynchronously
    async with aiofiles.open('large_file.txt', 'r') as f:
        content = await f.read()
    
    # Write file asynchronously
    async with aiofiles.open('output.txt', 'w') as f:
        await f.write(content.upper())

# Usage example
async def main():
    urls = [
        'https://api.example1.com/data',
        'https://api.example2.com/data',
        'https://api.example3.com/data'
    ]
    
    async with AsyncDataFetcher(max_concurrent=5) as fetcher:
        results = await fetcher.fetch_multiple_urls(urls)
        
        successful_results = [r for r in results if r.get('status') == 'success']
        print(f"Successfully fetched {len(successful_results)} out of {len(urls)} URLs")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
```

## Coding Patterns and Anti-Patterns

### Good Patterns

#### 1. Factory Pattern
```python
from abc import ABC, abstractmethod
from typing import Dict, Type

class Animal(ABC):
    @abstractmethod
    def make_sound(self) -> str:
        pass

class Dog(Animal):
    def make_sound(self) -> str:
        return "Woof!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "Meow!"

class AnimalFactory:
    _animals: Dict[str, Type[Animal]] = {
        'dog': Dog,
        'cat': Cat
    }
    
    @classmethod
    def create_animal(cls, animal_type: str) -> Animal:
        animal_class = cls._animals.get(animal_type.lower())
        if animal_class:
            return animal_class()
        raise ValueError(f"Unknown animal type: {animal_type}")
    
    @classmethod
    def register_animal(cls, name: str, animal_class: Type[Animal]):
        cls._animals[name.lower()] = animal_class

# Usage
dog = AnimalFactory.create_animal('dog')
print(dog.make_sound())  # Woof!
```

#### 2. Strategy Pattern
```python
from abc import ABC, abstractmethod

class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        n = len(data)
        for i in range(n):
            for j in range(0, n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data

class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: SortStrategy):
        self.strategy = strategy
    
    def sort(self, data: List[int]) -> List[int]:
        return self.strategy.sort(data.copy())
```

#### 3. Observer Pattern
```python
from typing import List, Protocol

class Observer(Protocol):
    def update(self, subject, event_data):
        pass

class Subject:
    def __init__(self):
        self._observers: List[Observer] = []
    
    def attach(self, observer: Observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer: Observer):
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event_data=None):
        for observer in self._observers:
            observer.update(self, event_data)

class EmailNotifier:
    def update(self, subject, event_data):
        print(f"Email: {event_data}")

class SMSNotifier:
    def update(self, subject, event_data):
        print(f"SMS: {event_data}")

# Usage
notification_system = Subject()
email_notifier = EmailNotifier()
sms_notifier = SMSNotifier()

notification_system.attach(email_notifier)
notification_system.attach(sms_notifier)
notification_system.notify("New message received!")
```

### Anti-Patterns to Avoid

#### 1. God Object Anti-Pattern
```python
# BAD: God object that does everything
class Application:
    def __init__(self):
        self.users = []
        self.products = []
        self.orders = []
        self.email_service = None
        self.payment_processor = None
    
    def create_user(self, user_data):
        # User creation logic
        pass
    
    def process_payment(self, payment_data):
        # Payment processing logic
        pass
    
    def send_email(self, email_data):
        # Email sending logic
        pass
    
    def manage_inventory(self, inventory_data):
        # Inventory management logic
        pass

# GOOD: Separated responsibilities
class UserService:
    def create_user(self, user_data):
        pass

class PaymentService:
    def process_payment(self, payment_data):
        pass

class EmailService:
    def send_email(self, email_data):
        pass

class InventoryService:
    def manage_inventory(self, inventory_data):
        pass
```

#### 2. Mutable Default Arguments
```python
# BAD: Mutable default argument
def add_item(item, target_list=[]):  # Dangerous!
    target_list.append(item)
    return target_list

# GOOD: Use None and create new list
def add_item(item, target_list=None):
    if target_list is None:
        target_list = []
    target_list.append(item)
    return target_list
```

#### 3. Bare Except Clauses
```python
# BAD: Catches all exceptions, including system exit
def risky_operation():
    try:
        dangerous_code()
    except:  # Catches everything!
        pass

# GOOD: Specific exception handling
def risky_operation():
    try:
        dangerous_code()
    except (ValueError, TypeError) as e:
        logging.error(f"Expected error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
```

## Performance Considerations

### 1. Memory Optimization

#### Using Generators for Large Data Sets
```python
# Memory efficient data processing
def process_large_file(filename):
    """Generator that processes file line by line."""
    with open(filename, 'r') as file:
        for line in file:
            # Process line without loading entire file into memory
            yield process_line(line)

def memory_efficient_sum(numbers):
    """Calculate sum without storing all numbers in memory."""
    total = 0
    for num in numbers:  # numbers can be a generator
        total += num
    return total

# Using slots for memory efficiency
class Point:
    __slots__ = ['x', 'y']  # Reduces memory overhead
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

#### Profiling Memory Usage
```python
import tracemalloc
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Start tracing
    tracemalloc.start()
    
    # Memory intensive operations
    large_list = [i for i in range(1000000)]
    large_dict = {i: str(i) for i in range(100000)}
    
    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

### 2. Algorithm Optimization

#### Time Complexity Considerations
```python
# O(n²) - Inefficient for large datasets
def find_duplicates_slow(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

# O(n) - Much more efficient
def find_duplicates_fast(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

# Using collections.Counter for frequency counting
from collections import Counter

def find_duplicates_pythonic(items):
    counts = Counter(items)
    return [item for item, count in counts.items() if count > 1]
```

#### Caching and Memoization
```python
from functools import lru_cache
import time

# LRU Cache for expensive function calls
@lru_cache(maxsize=128)
def expensive_operation(n):
    time.sleep(1)  # Simulate expensive operation
    return n * n

# Custom caching decorator
def custom_cache(max_size=None):
    def decorator(func):
        cache = {}
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                if max_size and len(cache) >= max_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                cache[key] = func(*args, **kwargs)
            return cache[key]
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'max_size': max_size}
        return wrapper
    return decorator
```

### 3. Concurrency and Parallelism

#### Threading vs Multiprocessing
```python
import threading
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound_task(n):
    """CPU intensive task - better with multiprocessing."""
    return sum(i * i for i in range(n))

def io_bound_task(url):
    """I/O intensive task - better with threading."""
    import requests
    response = requests.get(url)
    return len(response.content)

# Threading for I/O bound tasks
def test_threading():
    urls = ['http://example.com'] * 10
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        start = time.time()
        results = list(executor.map(io_bound_task, urls))
        end = time.time()
        print(f"Threading time: {end - start:.2f} seconds")

# Multiprocessing for CPU bound tasks
def test_multiprocessing():
    numbers = [100000] * 4
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        start = time.time()
        results = list(executor.map(cpu_bound_task, numbers))
        end = time.time()
        print(f"Multiprocessing time: {end - start:.2f} seconds")
```

## Modern Python Practices and Trends

### 1. Type Hints and Static Analysis

#### Advanced Type Hints
```python
from typing import (
    Union, Optional, List, Dict, Tuple, Callable, 
    TypeVar, Generic, Protocol, Literal, Final
)
from dataclasses import dataclass
from enum import Enum

# Type variables
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Generic classes
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Protocols for structural typing
class Drawable(Protocol):
    def draw(self) -> None: ...

def render_shape(shape: Drawable) -> None:
    shape.draw()

# Literal types
Mode = Literal['read', 'write', 'append']

def open_file(filename: str, mode: Mode) -> None:
    pass

# Final and immutable types
CONSTANT: Final = 42

@dataclass(frozen=True)  # Immutable dataclass
class Point:
    x: float
    y: float
```

#### Using mypy for Static Type Checking
```python
# mypy configuration in setup.cfg or mypy.ini
"""
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
"""

# Example with proper type annotations
def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    if not numbers:
        return {}
    
    return {
        'mean': sum(numbers) / len(numbers),
        'max': max(numbers),
        'min': min(numbers)
    }
```

### 2. Dataclasses and Pydantic

#### Dataclasses (Python 3.7+)
```python
from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class Person:
    name: str
    age: int
    email: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")
    
    def add_skill(self, skill: str) -> None:
        self.skills.append(skill)

@dataclass
class Company:
    name: str
    employees: List[Person] = field(default_factory=list)
    
    def hire(self, person: Person) -> None:
        self.employees.append(person)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'employees': [
                {
                    'name': emp.name,
                    'age': emp.age,
                    'email': emp.email,
                    'skills': emp.skills
                } for emp in self.employees
            ]
        }
```

#### Pydantic for Data Validation
```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
from datetime import datetime

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., gt=0, le=120)
    email: str
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('tags')
    def validate_tags(cls, v):
        return [tag.strip().lower() for tag in v]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Usage
user_data = {
    'name': 'John Doe',
    'age': 30,
    'email': 'JOHN@EXAMPLE.COM',
    'tags': [' Python ', ' Programming ']
}

user = User(**user_data)
print(user.json())  # Automatically serializes to JSON
```

### 3. Async/Await and Modern Concurrency

#### Advanced Async Patterns
```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

@asynccontextmanager
async def async_database_connection():
    """Async context manager for database connections."""
    connection = await create_connection()
    try:
        yield connection
    finally:
        await connection.close()

async def async_generator_example() -> AsyncGenerator[int, None]:
    """Async generator for streaming data."""
    for i in range(10):
        await asyncio.sleep(0.1)  # Simulate async operation
        yield i

class AsyncWorkerPool:
    def __init__(self, max_workers: int = 5):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.tasks: List[asyncio.Task] = []
    
    async def submit(self, coro):
        async def worker():
            async with self.semaphore:
                return await coro
        
        task = asyncio.create_task(worker())
        self.tasks.append(task)
        return task
    
    async def wait_all(self):
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()

# Usage
async def main():
    pool = AsyncWorkerPool(max_workers=3)
    
    # Submit multiple async tasks
    for i in range(10):
        await pool.submit(async_operation(i))
    
    # Wait for all tasks to complete
    await pool.wait_all()
```

### 4. Context Managers and Dependency Injection

#### Advanced Context Manager Patterns
```python
from contextlib import contextmanager, ExitStack
import logging

class ResourceManager:
    def __init__(self):
        self.resources = []
    
    @contextmanager
    def managed_resource(self, resource_factory, *args, **kwargs):
        resource = resource_factory(*args, **kwargs)
        self.resources.append(resource)
        try:
            yield resource
        finally:
            if hasattr(resource, 'close'):
                resource.close()
            self.resources.remove(resource)
    
    def close_all(self):
        for resource in self.resources[:]:
            if hasattr(resource, 'close'):
                resource.close()

# Dependency injection pattern
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def get_data(self) -> List[dict]:
        # Database operations
        return []

class EmailService:
    def __init__(self, smtp_server: str):
        self.smtp_server = smtp_server
    
    def send_email(self, to: str, subject: str, body: str):
        # Email sending logic
        pass

class UserService:
    def __init__(self, db_service: DatabaseService, email_service: EmailService):
        self.db_service = db_service
        self.email_service = email_service
    
    def create_user(self, user_data: dict):
        # Create user in database
        user = self.db_service.create_user(user_data)
        
        # Send welcome email
        self.email_service.send_email(
            user_data['email'],
            'Welcome!',
            'Welcome to our service!'
        )
        
        return user

# Simple dependency injection container
class Container:
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, interface, implementation, singleton=False):
        self._services[interface] = (implementation, singleton)
    
    def get(self, interface):
        if interface in self._singletons:
            return self._singletons[interface]
        
        implementation, is_singleton = self._services[interface]
        
        # Simple dependency resolution
        import inspect
        sig = inspect.signature(implementation.__init__)
        args = {}
        
        for param_name, param in sig.parameters.items():
            if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                args[param_name] = self.get(param.annotation)
        
        instance = implementation(**args)
        
        if is_singleton:
            self._singletons[interface] = instance
        
        return instance
```

### Current Industry Trends

#### 1. Data Science and Machine Learning
```python
# Modern data science workflows
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self):
        self.pipeline = None
        self.model = None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Data cleaning and feature engineering
        df = df.dropna()
        df['feature_engineered'] = df['feature1'] * df['feature2']
        return df
    
    def create_pipeline(self, model):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        X_processed = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        score = self.pipeline.score(X_test, y_test)
        return score
```

#### 2. API Development with FastAPI
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Modern API", version="1.0.0")

class UserCreate(BaseModel):
    name: str
    email: str

class User(BaseModel):
    id: int
    name: str
    email: str

# Dependency injection
async def get_database():
    # Return database connection
    pass

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate, db=Depends(get_database)):
    # Create user logic
    return User(id=1, name=user.name, email=user.email)

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int, db=Depends(get_database)):
    # Get user logic
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/", response_model=List[User])
async def list_users(skip: int = 0, limit: int = 100, db=Depends(get_database)):
    users = await db.get_users(skip=skip, limit=limit)
    return users
```

#### 3. Testing with Modern Tools
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from httpx import AsyncClient

# Async testing
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected_value

# Fixtures for dependency injection
@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_database():
    mock_db = AsyncMock()
    mock_db.get_user.return_value = User(id=1, name="Test", email="test@example.com")
    return mock_db

# Property-based testing with Hypothesis
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=100))
def test_with_hypothesis(value):
    result = process_value(value)
    assert result >= 0
```

Remember: Modern Python development emphasizes readability, type safety, performance, and maintainability. Stay updated with PEPs (Python Enhancement Proposals) and community best practices for continued growth as a Python developer.