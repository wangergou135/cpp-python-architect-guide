# Software Architecture Interview Guide

A comprehensive guide covering system design, distributed systems, and architecture patterns for senior software architects and engineers.

## Table of Contents
1. [System Design Fundamentals](#system-design-fundamentals)
2. [Microservices Architecture](#microservices-architecture)
3. [Distributed Systems](#distributed-systems)
4. [Cloud Architecture](#cloud-architecture)
5. [Performance and Scalability](#performance-and-scalability)
6. [Security Architecture](#security-architecture)
7. [DevOps Practices](#devops-practices)

## System Design Fundamentals

### Load balancing strategies

Load balancing distributes incoming requests across multiple servers to ensure optimal resource utilization and prevent system overload.

```yaml
# Load Balancer Configuration Example
upstream backend {
    # Round Robin (default)
    server web1.example.com:8080;
    server web2.example.com:8080;
    server web3.example.com:8080;
}

upstream weighted_backend {
    # Weighted Round Robin
    server web1.example.com:8080 weight=3;
    server web2.example.com:8080 weight=2;
    server web3.example.com:8080 weight=1;
}

upstream least_conn_backend {
    # Least Connections
    least_conn;
    server web1.example.com:8080;
    server web2.example.com:8080;
    server web3.example.com:8080;
}

upstream ip_hash_backend {
    # IP Hash (session affinity)
    ip_hash;
    server web1.example.com:8080;
    server web2.example.com:8080;
    server web3.example.com:8080;
}
```

**Load Balancing Algorithms:**

1. **Round Robin**
   - Distributes requests sequentially
   - Simple and fair for uniform servers
   - No consideration for server load

2. **Weighted Round Robin**
   - Assigns weights based on server capacity
   - Better for heterogeneous server configurations
   - More powerful servers get more requests

3. **Least Connections**
   - Routes to server with fewest active connections
   - Good for long-running sessions
   - Considers current server load

4. **IP Hash**
   - Routes based on client IP hash
   - Provides session affinity
   - Same client always hits same server

5. **Health Check Based**
   - Removes unhealthy servers from pool
   - Monitors server responsiveness
   - Automatic failover capabilities

**Layer 4 vs Layer 7 Load Balancing:**

```python
# Layer 4 (Transport Layer) Load Balancing
class L4LoadBalancer:
    """
    - Operates at TCP/UDP level
    - Routes based on IP and port
    - Faster, lower latency
    - No application awareness
    """
    def route_request(self, client_ip, client_port, destination_port):
        # Simple hash-based routing
        server_index = hash(f"{client_ip}:{client_port}") % len(self.servers)
        return self.servers[server_index]

# Layer 7 (Application Layer) Load Balancing  
class L7LoadBalancer:
    """
    - Operates at HTTP level
    - Routes based on URL, headers, cookies
    - Can terminate SSL
    - Application-aware decisions
    """
    def route_request(self, http_request):
        if http_request.url.startswith('/api/'):
            return self.api_servers
        elif http_request.url.startswith('/static/'):
            return self.static_servers
        else:
            return self.web_servers
```

### Caching mechanisms

Caching improves performance by storing frequently accessed data in faster storage layers.

**Caching Layers:**

```python
# 1. Application-Level Caching
import functools
import time
from typing import Dict, Any

class ApplicationCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.ttl: Dict[str, float] = {}
    
    def get(self, key: str) -> Any:
        if key in self.cache:
            if time.time() < self.ttl.get(key, 0):
                return self.cache[key]
            else:
                self.evict(key)
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        self.cache[key] = value
        self.ttl[key] = time.time() + ttl_seconds
    
    def evict(self, key: str):
        self.cache.pop(key, None)
        self.ttl.pop(key, None)

# Decorator for caching function results
def cached(ttl_seconds: int = 300):
    def decorator(func):
        cache = ApplicationCache()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            result = cache.get(cache_key)
            
            if result is None:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator

@cached(ttl_seconds=600)
def expensive_computation(x: int, y: int) -> int:
    time.sleep(1)  # Simulate expensive operation
    return x ** y

# 2. Database Query Caching
class QueryCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_ttl = 300
    
    def get_user_by_id(self, user_id: int):
        cache_key = f"user:{user_id}"
        cached_user = self.redis.get(cache_key)
        
        if cached_user:
            return json.loads(cached_user)
        
        # Cache miss - fetch from database
        user = self.fetch_user_from_db(user_id)
        if user:
            self.redis.setex(
                cache_key, 
                self.default_ttl, 
                json.dumps(user)
            )
        
        return user
    
    def invalidate_user(self, user_id: int):
        cache_key = f"user:{user_id}"
        self.redis.delete(cache_key)
```

**Cache Patterns:**

```python
# 1. Cache-Aside (Lazy Loading)
class CacheAside:
    def get_data(self, key):
        # Check cache first
        data = cache.get(key)
        if data is None:
            # Cache miss - fetch from database
            data = database.get(key)
            if data is not None:
                cache.set(key, data)
        return data
    
    def update_data(self, key, data):
        # Update database first
        database.set(key, data)
        # Then invalidate cache
        cache.delete(key)

# 2. Write-Through
class WriteThrough:
    def update_data(self, key, data):
        # Write to cache and database simultaneously
        database.set(key, data)
        cache.set(key, data)
    
    def get_data(self, key):
        # Always check cache first
        return cache.get(key) or database.get(key)

# 3. Write-Behind (Write-Back)
class WriteBehind:
    def __init__(self):
        self.write_queue = []
        self.batch_size = 100
    
    def update_data(self, key, data):
        # Write to cache immediately
        cache.set(key, data)
        
        # Queue for later database write
        self.write_queue.append((key, data))
        
        if len(self.write_queue) >= self.batch_size:
            self.flush_to_database()
    
    def flush_to_database(self):
        # Batch write to database
        database.batch_write(self.write_queue)
        self.write_queue.clear()

# 4. Refresh-Ahead
class RefreshAhead:
    def __init__(self):
        self.refresh_threshold = 0.8  # Refresh when 80% of TTL passed
    
    def get_data(self, key):
        data, ttl_remaining = cache.get_with_ttl(key)
        
        if data is None:
            # Cache miss
            data = database.get(key)
            cache.set(key, data)
        elif ttl_remaining < (cache.default_ttl * self.refresh_threshold):
            # Proactively refresh
            self.async_refresh(key)
        
        return data
    
    def async_refresh(self, key):
        # Refresh cache in background
        threading.Thread(
            target=lambda: cache.set(key, database.get(key))
        ).start()
```

**Cache Eviction Policies:**

```python
class LRUCache:
    """Least Recently Used cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)

class LFUCache:
    """Least Frequently Used cache implementation"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.frequencies = {}
        self.min_frequency = 0
        self.frequency_groups = {0: set()}
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        self._update_frequency(key)
        return self.cache[key]
    
    def set(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            self._update_frequency(key)
        else:
            if len(self.cache) >= self.capacity:
                self._evict_lfu()
            
            self.cache[key] = value
            self.frequencies[key] = 1
            self.frequency_groups[1].add(key)
            self.min_frequency = 1
    
    def _update_frequency(self, key):
        freq = self.frequencies[key]
        self.frequency_groups[freq].remove(key)
        
        if not self.frequency_groups[freq] and freq == self.min_frequency:
            self.min_frequency += 1
        
        new_freq = freq + 1
        self.frequencies[key] = new_freq
        
        if new_freq not in self.frequency_groups:
            self.frequency_groups[new_freq] = set()
        
        self.frequency_groups[new_freq].add(key)
    
    def _evict_lfu(self):
        lfu_key = self.frequency_groups[self.min_frequency].pop()
        del self.cache[lfu_key]
        del self.frequencies[lfu_key]
```

### Database sharding

Database sharding distributes data across multiple database instances to handle large-scale applications.

**Sharding Strategies:**

```python
import hashlib
from typing import List, Dict, Any

class HorizontalSharding:
    """Distribute rows across multiple database shards"""
    
    def __init__(self, shard_configs: List[Dict]):
        self.shards = shard_configs
        self.shard_count = len(shard_configs)
    
    def get_shard_by_hash(self, shard_key: str) -> Dict:
        """Hash-based sharding"""
        hash_value = int(hashlib.md5(shard_key.encode()).hexdigest(), 16)
        shard_index = hash_value % self.shard_count
        return self.shards[shard_index]
    
    def get_shard_by_range(self, shard_key: int) -> Dict:
        """Range-based sharding"""
        # Example: User IDs 1-1000 in shard 0, 1001-2000 in shard 1, etc.
        shard_index = min(shard_key // 1000, self.shard_count - 1)
        return self.shards[shard_index]
    
    def get_shard_by_directory(self, shard_key: str) -> Dict:
        """Directory-based sharding"""
        # Use lookup service to determine shard
        shard_mapping = self.lookup_service.get_shard_mapping(shard_key)
        return self.shards[shard_mapping['shard_index']]

class UserRepository:
    """Example repository with sharding"""
    
    def __init__(self, sharding_strategy):
        self.sharding = sharding_strategy
    
    def create_user(self, user_data: Dict) -> str:
        user_id = user_data['user_id']
        shard = self.sharding.get_shard_by_hash(str(user_id))
        
        connection = self.get_connection(shard)
        query = """
            INSERT INTO users (user_id, name, email, created_at)
            VALUES (%(user_id)s, %(name)s, %(email)s, %(created_at)s)
        """
        
        connection.execute(query, user_data)
        return user_id
    
    def get_user(self, user_id: str) -> Dict:
        shard = self.sharding.get_shard_by_hash(user_id)
        connection = self.get_connection(shard)
        
        query = "SELECT * FROM users WHERE user_id = %s"
        result = connection.execute(query, (user_id,))
        
        return result.fetchone()
    
    def get_users_by_criteria(self, criteria: Dict) -> List[Dict]:
        """Cross-shard query - requires querying all shards"""
        results = []
        
        for shard in self.sharding.shards:
            connection = self.get_connection(shard)
            query = self.build_query(criteria)
            shard_results = connection.execute(query, criteria).fetchall()
            results.extend(shard_results)
        
        return results

# Vertical Sharding Example
class VerticalSharding:
    """Split tables/columns across different databases"""
    
    def __init__(self):
        self.user_profile_db = "user_profile_shard"
        self.user_activity_db = "user_activity_shard"
        self.user_preferences_db = "user_preferences_shard"
    
    def get_user_profile(self, user_id: str) -> Dict:
        connection = self.get_connection(self.user_profile_db)
        query = """
            SELECT user_id, name, email, created_at
            FROM user_profiles 
            WHERE user_id = %s
        """
        return connection.execute(query, (user_id,)).fetchone()
    
    def get_user_activity(self, user_id: str) -> List[Dict]:
        connection = self.get_connection(self.user_activity_db)
        query = """
            SELECT activity_type, timestamp, metadata
            FROM user_activities 
            WHERE user_id = %s 
            ORDER BY timestamp DESC
        """
        return connection.execute(query, (user_id,)).fetchall()
    
    def get_complete_user_data(self, user_id: str) -> Dict:
        """Combine data from multiple shards"""
        profile = self.get_user_profile(user_id)
        activities = self.get_user_activity(user_id)
        preferences = self.get_user_preferences(user_id)
        
        return {
            'profile': profile,
            'activities': activities,
            'preferences': preferences
        }
```

**Sharding Challenges and Solutions:**

```python
class ShardingChallenges:
    """Common sharding challenges and solutions"""
    
    def handle_cross_shard_queries(self, query_criteria: Dict) -> List[Dict]:
        """
        Challenge: Queries spanning multiple shards
        Solution: Query all relevant shards and merge results
        """
        results = []
        
        # Determine which shards to query
        relevant_shards = self.get_relevant_shards(query_criteria)
        
        # Execute query on each shard
        for shard in relevant_shards:
            shard_results = self.execute_on_shard(shard, query_criteria)
            results.extend(shard_results)
        
        # Sort and paginate combined results
        return self.merge_and_sort_results(results, query_criteria)
    
    def handle_shard_rebalancing(self, new_shard_config: List[Dict]):
        """
        Challenge: Adding/removing shards
        Solution: Gradual data migration with dual writes
        """
        # Phase 1: Start dual writes to old and new shards
        self.enable_dual_writes(new_shard_config)
        
        # Phase 2: Migrate existing data in background
        self.migrate_data_async(new_shard_config)
        
        # Phase 3: Switch reads to new shards
        self.switch_reads_to_new_shards()
        
        # Phase 4: Stop dual writes and remove old shards
        self.disable_dual_writes()
    
    def handle_shard_failures(self, failed_shard: str):
        """
        Challenge: Shard unavailability
        Solution: Read replicas and graceful degradation
        """
        if self.has_read_replica(failed_shard):
            # Route reads to replica
            self.route_to_replica(failed_shard)
        else:
            # Graceful degradation
            self.enable_degraded_mode(failed_shard)
        
        # Alert operations team
        self.send_alert(f"Shard {failed_shard} failed")
    
    def handle_hot_shards(self, shard_metrics: Dict):
        """
        Challenge: Uneven load distribution
        Solution: Dynamic routing and load monitoring
        """
        for shard_id, metrics in shard_metrics.items():
            if metrics['cpu_usage'] > 80 or metrics['query_latency'] > 1000:
                # Temporarily route traffic to other shards
                self.reduce_traffic_to_shard(shard_id, reduction_factor=0.5)
                
                # Consider shard splitting if consistently hot
                if self.is_consistently_hot(shard_id):
                    self.schedule_shard_split(shard_id)
```

**Sharding Best Practices:**

```yaml
# Database Sharding Configuration Example
sharding_config:
  strategy: "hash_based"  # hash_based, range_based, directory_based
  
  shards:
    - name: "shard_0"
      host: "db-shard-0.example.com"
      port: 5432
      database: "app_shard_0"
      range: "0-999999"
      
    - name: "shard_1" 
      host: "db-shard-1.example.com"
      port: 5432
      database: "app_shard_1"
      range: "1000000-1999999"
      
    - name: "shard_2"
      host: "db-shard-2.example.com"
      port: 5432
      database: "app_shard_2"
      range: "2000000-2999999"
  
  replication:
    enabled: true
    replicas_per_shard: 2
    read_preference: "replica"
    
  monitoring:
    metrics:
      - cpu_usage
      - memory_usage
      - query_latency
      - connection_count
    alerts:
      cpu_threshold: 80
      latency_threshold: 1000
      
  failover:
    automatic: true
    timeout_seconds: 30
    retry_attempts: 3
```

### Message queues

Message queues enable asynchronous communication between services and help decouple system components.

**Message Queue Patterns:**

```python
import json
import threading
import time
from typing import Callable, Dict, Any
from queue import Queue, Empty
from abc import ABC, abstractmethod

class MessageQueue(ABC):
    """Abstract base class for message queues"""
    
    @abstractmethod
    def publish(self, topic: str, message: Dict[str, Any]):
        pass
    
    @abstractmethod
    def subscribe(self, topic: str, callback: Callable):
        pass
    
    @abstractmethod
    def unsubscribe(self, topic: str):
        pass

class InMemoryMessageQueue(MessageQueue):
    """Simple in-memory message queue implementation"""
    
    def __init__(self):
        self.topics: Dict[str, Queue] = {}
        self.subscribers: Dict[str, list] = {}
        self.running = True
    
    def publish(self, topic: str, message: Dict[str, Any]):
        if topic not in self.topics:
            self.topics[topic] = Queue()
        
        # Add message to queue
        self.topics[topic].put({
            'message': message,
            'timestamp': time.time(),
            'id': self._generate_message_id()
        })
        
        # Notify subscribers
        self._notify_subscribers(topic)
    
    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        
        self.subscribers[topic].append(callback)
        
        # Start processing thread for this topic
        if topic not in self.topics:
            self.topics[topic] = Queue()
            self._start_topic_processor(topic)
    
    def _start_topic_processor(self, topic: str):
        def process_messages():
            while self.running:
                try:
                    message_data = self.topics[topic].get(timeout=1)
                    
                    # Call all subscribers
                    for callback in self.subscribers.get(topic, []):
                        try:
                            callback(message_data['message'])
                        except Exception as e:
                            print(f"Error in subscriber callback: {e}")
                    
                    self.topics[topic].task_done()
                    
                except Empty:
                    continue
                except Exception as e:
                    print(f"Error processing message: {e}")
        
        thread = threading.Thread(target=process_messages, daemon=True)
        thread.start()

# Work Queue Pattern
class TaskQueue:
    """Work queue for distributing tasks among workers"""
    
    def __init__(self):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        self.running = False
    
    def start_workers(self, num_workers: int):
        self.running = True
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                args=(f"worker_{i}",),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def stop_workers(self):
        self.running = False
        # Send stop signals
        for _ in self.workers:
            self.task_queue.put(None)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        task_id = self._generate_task_id()
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': time.time()
        }
        
        self.task_queue.put(task)
        return task_id
    
    def get_result(self, task_id: str, timeout: float = None):
        start_time = time.time()
        
        while True:
            try:
                result = self.result_queue.get(timeout=1)
                if result['task_id'] == task_id:
                    return result
                else:
                    # Put back result for other consumers
                    self.result_queue.put(result)
                    
            except Empty:
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Task {task_id} timeout")
                continue
    
    def _worker_loop(self, worker_name: str):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Stop signal
                    break
                
                print(f"{worker_name} processing task {task['id']}")
                
                # Execute task
                try:
                    result = task['func'](*task['args'], **task['kwargs'])
                    
                    self.result_queue.put({
                        'task_id': task['id'],
                        'result': result,
                        'status': 'success',
                        'worker': worker_name,
                        'completed_at': time.time()
                    })
                    
                except Exception as e:
                    self.result_queue.put({
                        'task_id': task['id'],
                        'error': str(e),
                        'status': 'error',
                        'worker': worker_name,
                        'completed_at': time.time()
                    })
                
                self.task_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_name} error: {e}")

# Publish-Subscribe Pattern
class PubSubMessageQueue:
    """Publish-Subscribe message queue"""
    
    def __init__(self):
        self.channels: Dict[str, list] = {}
        self.message_history: Dict[str, list] = {}
        self.max_history = 1000
    
    def publish(self, channel: str, message: Any, persistent: bool = False):
        message_data = {
            'channel': channel,
            'message': message,
            'timestamp': time.time(),
            'persistent': persistent,
            'id': self._generate_message_id()
        }
        
        # Store in history if persistent
        if persistent:
            if channel not in self.message_history:
                self.message_history[channel] = []
            
            self.message_history[channel].append(message_data)
            
            # Limit history size
            if len(self.message_history[channel]) > self.max_history:
                self.message_history[channel].pop(0)
        
        # Send to current subscribers
        subscribers = self.channels.get(channel, [])
        for subscriber_callback in subscribers:
            try:
                subscriber_callback(message)
            except Exception as e:
                print(f"Error notifying subscriber: {e}")
    
    def subscribe(self, channel: str, callback: Callable, 
                 get_history: bool = False):
        if channel not in self.channels:
            self.channels[channel] = []
        
        self.channels[channel].append(callback)
        
        # Send historical messages if requested
        if get_history and channel in self.message_history:
            for msg_data in self.message_history[channel]:
                try:
                    callback(msg_data['message'])
                except Exception as e:
                    print(f"Error sending historical message: {e}")
    
    def unsubscribe(self, channel: str, callback: Callable):
        if channel in self.channels:
            try:
                self.channels[channel].remove(callback)
            except ValueError:
                pass

# Message Queue with Dead Letter Queue
class ReliableMessageQueue:
    """Message queue with retry logic and dead letter queue"""
    
    def __init__(self, max_retries: int = 3):
        self.main_queue = Queue()
        self.retry_queue = Queue()
        self.dead_letter_queue = Queue()
        self.max_retries = max_retries
        self.running = False
    
    def publish(self, message: Dict[str, Any], priority: int = 0):
        message_data = {
            'message': message,
            'priority': priority,
            'retry_count': 0,
            'first_attempt': time.time(),
            'id': self._generate_message_id()
        }
        
        self.main_queue.put(message_data)
    
    def start_processing(self):
        self.running = True
        
        # Main queue processor
        main_processor = threading.Thread(
            target=self._process_main_queue, daemon=True)
        main_processor.start()
        
        # Retry queue processor
        retry_processor = threading.Thread(
            target=self._process_retry_queue, daemon=True)
        retry_processor.start()
    
    def _process_main_queue(self):
        while self.running:
            try:
                message_data = self.main_queue.get(timeout=1)
                
                if self._process_message(message_data):
                    print(f"Successfully processed message {message_data['id']}")
                else:
                    self._handle_processing_failure(message_data)
                
                self.main_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in main queue processor: {e}")
    
    def _process_retry_queue(self):
        while self.running:
            try:
                message_data = self.retry_queue.get(timeout=1)
                
                # Wait before retry
                time.sleep(min(2 ** message_data['retry_count'], 60))
                
                if self._process_message(message_data):
                    print(f"Successfully processed message {message_data['id']} on retry")
                else:
                    self._handle_processing_failure(message_data)
                
                self.retry_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in retry queue processor: {e}")
    
    def _process_message(self, message_data: Dict) -> bool:
        try:
            # Simulate message processing
            message = message_data['message']
            
            # Your actual message processing logic here
            if message.get('should_fail', False):
                raise Exception("Simulated processing failure")
            
            return True
            
        except Exception as e:
            print(f"Message processing failed: {e}")
            return False
    
    def _handle_processing_failure(self, message_data: Dict):
        message_data['retry_count'] += 1
        
        if message_data['retry_count'] <= self.max_retries:
            print(f"Retrying message {message_data['id']} (attempt {message_data['retry_count']})")
            self.retry_queue.put(message_data)
        else:
            print(f"Moving message {message_data['id']} to dead letter queue")
            message_data['moved_to_dlq_at'] = time.time()
            self.dead_letter_queue.put(message_data)

# Usage Examples
def message_queue_examples():
    print("=== Message Queue Examples ===")
    
    # Simple pub-sub example
    pubsub = PubSubMessageQueue()
    
    def user_handler(message):
        print(f"User service received: {message}")
    
    def email_handler(message):
        print(f"Email service received: {message}")
    
    # Subscribe to user events
    pubsub.subscribe('user.created', user_handler)
    pubsub.subscribe('user.created', email_handler)
    
    # Publish event
    pubsub.publish('user.created', {
        'user_id': '123',
        'email': 'user@example.com',
        'name': 'John Doe'
    })
    
    # Task queue example
    task_queue = TaskQueue()
    task_queue.start_workers(3)
    
    def slow_task(duration):
        time.sleep(duration)
        return f"Task completed after {duration} seconds"
    
    # Submit tasks
    task_ids = []
    for i in range(5):
        task_id = task_queue.submit_task(slow_task, 0.1)
        task_ids.append(task_id)
    
    # Get results
    for task_id in task_ids:
        result = task_queue.get_result(task_id, timeout=5)
        print(f"Task {task_id}: {result}")
    
    task_queue.stop_workers()

### Service discovery

Service discovery enables services to find and communicate with each other dynamically in distributed systems.

**Service Discovery Patterns:**

```python
import json
import threading
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

@dataclass
class ServiceInstance:
    """Represents a service instance"""
    service_name: str
    instance_id: str
    host: str
    port: int
    health_check_url: str
    metadata: Dict[str, str] = None
    status: str = "healthy"  # healthy, unhealthy, unknown
    last_heartbeat: float = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()

class ServiceRegistry(ABC):
    """Abstract service registry interface"""
    
    @abstractmethod
    def register(self, instance: ServiceInstance) -> bool:
        """Register a service instance"""
        pass
    
    @abstractmethod
    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance"""
        pass
    
    @abstractmethod
    def discover(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy instances of a service"""
        pass
    
    @abstractmethod
    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Send heartbeat for an instance"""
        pass

class InMemoryServiceRegistry(ServiceRegistry):
    """Simple in-memory service registry"""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.services: Dict[str, Dict[str, ServiceInstance]] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.lock = threading.RLock()
        self.health_checker_running = False
        self._start_health_checker()
    
    def register(self, instance: ServiceInstance) -> bool:
        with self.lock:
            service_name = instance.service_name
            instance_id = instance.instance_id
            
            if service_name not in self.services:
                self.services[service_name] = {}
            
            instance.last_heartbeat = time.time()
            instance.status = "healthy"
            self.services[service_name][instance_id] = instance
            
            print(f"Registered {service_name}:{instance_id} at {instance.host}:{instance.port}")
            return True
    
    def deregister(self, service_name: str, instance_id: str) -> bool:
        with self.lock:
            if (service_name in self.services and 
                instance_id in self.services[service_name]):
                
                del self.services[service_name][instance_id]
                print(f"Deregistered {service_name}:{instance_id}")
                
                # Clean up empty service entries
                if not self.services[service_name]:
                    del self.services[service_name]
                
                return True
            return False
    
    def discover(self, service_name: str) -> List[ServiceInstance]:
        with self.lock:
            if service_name not in self.services:
                return []
            
            # Return only healthy instances
            healthy_instances = [
                instance for instance in self.services[service_name].values()
                if instance.status == "healthy"
            ]
            
            return healthy_instances
    
    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        with self.lock:
            if (service_name in self.services and 
                instance_id in self.services[service_name]):
                
                instance = self.services[service_name][instance_id]
                instance.last_heartbeat = time.time()
                instance.status = "healthy"
                return True
            return False
    
    def _start_health_checker(self):
        if self.health_checker_running:
            return
        
        self.health_checker_running = True
        
        def health_check_loop():
            while self.health_checker_running:
                current_time = time.time()
                
                with self.lock:
                    for service_name in list(self.services.keys()):
                        for instance_id in list(self.services[service_name].keys()):
                            instance = self.services[service_name][instance_id]
                            
                            time_since_heartbeat = current_time - instance.last_heartbeat
                            
                            if time_since_heartbeat > self.heartbeat_timeout:
                                print(f"Marking {service_name}:{instance_id} as unhealthy")
                                instance.status = "unhealthy"
                            
                            # Remove instances that have been unhealthy for too long
                            if (time_since_heartbeat > self.heartbeat_timeout * 2 and 
                                instance.status == "unhealthy"):
                                print(f"Removing stale instance {service_name}:{instance_id}")
                                del self.services[service_name][instance_id]
                                
                                if not self.services[service_name]:
                                    del self.services[service_name]
                
                time.sleep(10)  # Check every 10 seconds
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()

# Client-side service discovery
class ServiceDiscoveryClient:
    """Client for service discovery with load balancing"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.cache: Dict[str, List[ServiceInstance]] = {}
        self.cache_ttl: Dict[str, float] = {}
        self.cache_timeout = 60  # Cache for 60 seconds
    
    def get_service_instance(self, service_name: str, 
                           load_balancing_strategy: str = "round_robin") -> Optional[ServiceInstance]:
        """Get a service instance using specified load balancing strategy"""
        instances = self._get_cached_instances(service_name)
        
        if not instances:
            return None
        
        if load_balancing_strategy == "round_robin":
            return self._round_robin_select(service_name, instances)
        elif load_balancing_strategy == "random":
            import random
            return random.choice(instances)
        elif load_balancing_strategy == "least_connections":
            return self._least_connections_select(instances)
        else:
            return instances[0]  # Default to first instance
    
    def _get_cached_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get instances from cache or registry"""
        current_time = time.time()
        
        # Check cache
        if (service_name in self.cache and 
            service_name in self.cache_ttl and
            current_time - self.cache_ttl[service_name] < self.cache_timeout):
            return self.cache[service_name]
        
        # Cache miss or expired - fetch from registry
        instances = self.registry.discover(service_name)
        self.cache[service_name] = instances
        self.cache_ttl[service_name] = current_time
        
        return instances
    
    def _round_robin_select(self, service_name: str, 
                          instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing"""
        if not hasattr(self, '_round_robin_counters'):
            self._round_robin_counters = {}
        
        if service_name not in self._round_robin_counters:
            self._round_robin_counters[service_name] = 0
        
        index = self._round_robin_counters[service_name] % len(instances)
        self._round_robin_counters[service_name] += 1
        
        return instances[index]
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections (simplified)"""
        # In real implementation, you'd track actual connections
        # Here we'll use a simple heuristic based on instance metadata
        
        min_connections = float('inf')
        selected_instance = instances[0]
        
        for instance in instances:
            connections = int(instance.metadata.get('active_connections', '0'))
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance
        
        return selected_instance

# Health checking
class HealthChecker:
    """Health checker for service instances"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.running = False
    
    def start(self, check_interval: float = 30.0):
        """Start health checking"""
        if self.running:
            return
        
        self.running = True
        
        def health_check_loop():
            import requests
            
            while self.running:
                # Get all registered services
                with self.registry.lock:
                    all_services = dict(self.registry.services)
                
                for service_name, instances in all_services.items():
                    for instance_id, instance in instances.items():
                        if instance.health_check_url:
                            try:
                                response = requests.get(
                                    instance.health_check_url,
                                    timeout=5
                                )
                                
                                if response.status_code == 200:
                                    # Instance is healthy - send heartbeat
                                    self.registry.heartbeat(service_name, instance_id)
                                else:
                                    print(f"Health check failed for {service_name}:{instance_id}")
                                    
                            except Exception as e:
                                print(f"Health check error for {service_name}:{instance_id}: {e}")
                
                time.sleep(check_interval)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
    
    def stop(self):
        """Stop health checking"""
        self.running = False

# Service mesh integration
class ServiceMeshConfig:
    """Configuration for service mesh integration"""
    
    def __init__(self):
        self.sidecar_config = {
            "envoy": {
                "admin_port": 9901,
                "proxy_port": 15001,
                "metrics_port": 15000
            },
            "circuit_breaker": {
                "max_connections": 100,
                "max_pending_requests": 10,
                "max_requests": 100,
                "max_retries": 3
            },
            "retry_policy": {
                "retry_attempts": 3,
                "per_try_timeout": "5s",
                "retry_conditions": ["5xx", "gateway-error", "connect-failure"]
            },
            "load_balancing": {
                "policy": "round_robin",
                "health_check": {
                    "path": "/health",
                    "interval": "10s",
                    "timeout": "3s"
                }
            }
        }

# Usage example
def service_discovery_examples():
    """Demonstrate service discovery patterns"""
    print("=== Service Discovery Examples ===")
    
    # Create registry
    registry = InMemoryServiceRegistry(heartbeat_timeout=30.0)
    
    # Register some service instances
    user_service_1 = ServiceInstance(
        service_name="user-service",
        instance_id="user-1",
        host="10.0.1.10",
        port=8080,
        health_check_url="http://10.0.1.10:8080/health",
        metadata={"version": "1.2.0", "zone": "us-west-1a"}
    )
    
    user_service_2 = ServiceInstance(
        service_name="user-service", 
        instance_id="user-2",
        host="10.0.1.11",
        port=8080,
        health_check_url="http://10.0.1.11:8080/health",
        metadata={"version": "1.2.0", "zone": "us-west-1b"}
    )
    
    registry.register(user_service_1)
    registry.register(user_service_2)
    
    # Create discovery client
    client = ServiceDiscoveryClient(registry)
    
    # Discover services
    print("\nDiscovering user-service instances:")
    instances = registry.discover("user-service")
    for instance in instances:
        print(f"  {instance.instance_id}: {instance.host}:{instance.port}")
    
    # Load-balanced instance selection
    print("\nLoad-balanced instance selection:")
    for i in range(5):
        instance = client.get_service_instance("user-service", "round_robin")
        if instance:
            print(f"  Request {i+1}: {instance.instance_id}")
    
    # Simulate service failure
    print(f"\nSimulating service failure...")
    time.sleep(2)  # Let some time pass
    
    # Stop heartbeats for one instance (simulate failure)
    print("Instance user-1 stopped sending heartbeats...")
    
    # After health check timeout, only healthy instances should be returned
    time.sleep(5)
    healthy_instances = registry.discover("user-service")
    print(f"Healthy instances: {len(healthy_instances)}")

### API gateway patterns

API gateways provide a single entry point for client requests and handle cross-cutting concerns.

```yaml
# API Gateway Configuration Example
api_gateway:
  server:
    host: "0.0.0.0"
    port: 8080
    ssl:
      enabled: true
      cert_file: "/etc/ssl/gateway.crt"
      key_file: "/etc/ssl/gateway.key"
  
  # Route definitions
  routes:
    - name: "user_service"
      path: "/api/users/*"
      upstream: "http://user-service:8080"
      methods: ["GET", "POST", "PUT", "DELETE"]
      timeout: "30s"
      retry_attempts: 3
      
    - name: "order_service"
      path: "/api/orders/*"
      upstream: "http://order-service:8080"
      methods: ["GET", "POST", "PUT", "DELETE"]
      timeout: "45s"
      
    - name: "static_content"
      path: "/static/*"
      upstream: "http://cdn.example.com"
      cache_ttl: "1h"
  
  # Cross-cutting concerns
  middleware:
    - name: "authentication"
      type: "jwt"
      config:
        secret_key: "${JWT_SECRET}"
        token_header: "Authorization"
        skip_paths: ["/health", "/docs", "/static/*"]
        
    - name: "rate_limiting"
      type: "rate_limit"
      config:
        requests_per_minute: 100
        burst_size: 10
        key_generator: "ip"  # ip, user_id, api_key
        
    - name: "request_logging"
      type: "logging"
      config:
        log_level: "info"
        include_request_body: false
        include_response_body: false
        
    - name: "cors"
      type: "cors"
      config:
        allowed_origins: ["https://app.example.com"]
        allowed_methods: ["GET", "POST", "PUT", "DELETE"]
        allowed_headers: ["Content-Type", "Authorization"]
        
    - name: "circuit_breaker"
      type: "circuit_breaker"
      config:
        failure_threshold: 5
        timeout: "60s"
        reset_timeout: "30s"
  
  # Load balancing
  load_balancing:
    default_strategy: "round_robin"
    health_check:
      enabled: true
      path: "/health"
      interval: "10s"
      timeout: "3s"
      
  # Caching
  caching:
    enabled: true
    default_ttl: "5m"
    cache_headers: ["Cache-Control", "ETag"]
    cache_keys: ["url", "query_params", "user_id"]
    
  # Security
  security:
    api_key:
      enabled: true
      header_name: "X-API-Key"
      rate_limit_per_key: 1000
      
    oauth2:
      enabled: true
      authorization_url: "https://auth.example.com/oauth/authorize"
      token_url: "https://auth.example.com/oauth/token"
      
    ip_whitelist:
      - "10.0.0.0/8"
      - "192.168.0.0/16"
      
  # Monitoring and observability
  observability:
    metrics:
      enabled: true
      endpoint: "/metrics"
      port: 9090
      
    tracing:
      enabled: true
      jaeger_endpoint: "http://jaeger:14268/api/traces"
      sample_rate: 0.1
      
    logging:
      level: "info"
      format: "json"
      output: "/var/log/gateway.log"
```

**API Gateway Implementation Example:**

```python
import time
import json
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import requests

@dataclass
class Route:
    """API Gateway route configuration"""
    name: str
    path_pattern: str
    upstream_url: str
    methods: List[str]
    timeout: float = 30.0
    retry_attempts: int = 3
    middleware: List[str] = None
    
    def __post_init__(self):
        if self.middleware is None:
            self.middleware = []

class Middleware(ABC):
    """Abstract middleware interface"""
    
    @abstractmethod
    def process_request(self, request: Dict) -> Dict:
        """Process incoming request"""
        pass
    
    @abstractmethod
    def process_response(self, response: Dict) -> Dict:
        """Process outgoing response"""
        pass

class AuthenticationMiddleware(Middleware):
    """JWT authentication middleware"""
    
    def __init__(self, secret_key: str, skip_paths: List[str] = None):
        self.secret_key = secret_key
        self.skip_paths = skip_paths or []
    
    def process_request(self, request: Dict) -> Dict:
        path = request.get('path', '')
        
        # Skip authentication for certain paths
        if any(path.startswith(skip) for skip in self.skip_paths):
            return request
        
        # Check for JWT token
        auth_header = request.get('headers', {}).get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            request['error'] = {'code': 401, 'message': 'Missing or invalid authorization header'}
            return request
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            # Simplified JWT validation (use proper JWT library in production)
            user_info = self._validate_jwt(token)
            request['user'] = user_info
            
        except Exception as e:
            request['error'] = {'code': 401, 'message': f'Invalid token: {str(e)}'}
        
        return request
    
    def process_response(self, response: Dict) -> Dict:
        return response
    
    def _validate_jwt(self, token: str) -> Dict:
        # Simplified JWT validation - use proper library like PyJWT
        # This is just for demonstration
        import base64
        
        try:
            # Split token
            header, payload, signature = token.split('.')
            
            # Decode payload (simplified)
            payload_data = base64.b64decode(payload + '==')  # Add padding
            user_info = json.loads(payload_data)
            
            # Check expiration
            if user_info.get('exp', 0) < time.time():
                raise Exception("Token expired")
            
            return user_info
            
        except Exception:
            raise Exception("Invalid token format")

class RateLimitingMiddleware(Middleware):
    """Rate limiting middleware"""
    
    def __init__(self, requests_per_minute: int = 100, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.request_counts: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
    
    def process_request(self, request: Dict) -> Dict:
        client_ip = request.get('client_ip', 'unknown')
        current_time = time.time()
        
        with self.lock:
            # Clean old requests (older than 1 minute)
            if client_ip in self.request_counts:
                self.request_counts[client_ip] = [
                    req_time for req_time in self.request_counts[client_ip]
                    if current_time - req_time < 60
                ]
            else:
                self.request_counts[client_ip] = []
            
            # Check rate limit
            request_count = len(self.request_counts[client_ip])
            
            if request_count >= self.requests_per_minute:
                request['error'] = {
                    'code': 429,
                    'message': 'Rate limit exceeded'
                }
                return request
            
            # Add current request
            self.request_counts[client_ip].append(current_time)
        
        return request
    
    def process_response(self, response: Dict) -> Dict:
        # Add rate limit headers
        client_ip = response.get('request', {}).get('client_ip', 'unknown')
        
        with self.lock:
            remaining = max(0, self.requests_per_minute - len(self.request_counts.get(client_ip, [])))
            
        response['headers'] = response.get('headers', {})
        response['headers']['X-RateLimit-Limit'] = str(self.requests_per_minute)
        response['headers']['X-RateLimit-Remaining'] = str(remaining)
        
        return response

class CircuitBreakerMiddleware(Middleware):
    """Circuit breaker middleware"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, reset_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        
        # Circuit breaker state per upstream
        self.circuit_states: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def process_request(self, request: Dict) -> Dict:
        upstream = request.get('upstream_url', '')
        
        with self.lock:
            state = self.circuit_states.get(upstream, {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': 0,
                'next_attempt_time': 0
            })
            
            current_time = time.time()
            
            if state['state'] == 'open':
                if current_time >= state['next_attempt_time']:
                    # Try to transition to half-open
                    state['state'] = 'half_open'
                    print(f"Circuit breaker half-open for {upstream}")
                else:
                    # Circuit is still open
                    request['error'] = {
                        'code': 503,
                        'message': 'Service temporarily unavailable (circuit breaker open)'
                    }
                    return request
            
            self.circuit_states[upstream] = state
        
        return request
    
    def process_response(self, response: Dict) -> Dict:
        upstream = response.get('request', {}).get('upstream_url', '')
        status_code = response.get('status_code', 200)
        
        with self.lock:
            if upstream in self.circuit_states:
                state = self.circuit_states[upstream]
                current_time = time.time()
                
                if 500 <= status_code < 600:  # Server error
                    state['failure_count'] += 1
                    state['last_failure_time'] = current_time
                    
                    if state['failure_count'] >= self.failure_threshold:
                        state['state'] = 'open'
                        state['next_attempt_time'] = current_time + self.reset_timeout
                        print(f"Circuit breaker opened for {upstream}")
                
                else:  # Success
                    if state['state'] == 'half_open':
                        # Reset circuit breaker
                        state['state'] = 'closed'
                        state['failure_count'] = 0
                        print(f"Circuit breaker closed for {upstream}")
                    else:
                        # Reset failure count on success
                        state['failure_count'] = max(0, state['failure_count'] - 1)
        
        return response

class APIGateway:
    """Simple API Gateway implementation"""
    
    def __init__(self):
        self.routes: List[Route] = []
        self.middleware: Dict[str, Middleware] = {}
        self.global_middleware: List[str] = []
    
    def add_route(self, route: Route):
        """Add a route to the gateway"""
        self.routes.append(route)
    
    def add_middleware(self, name: str, middleware: Middleware):
        """Add middleware to the gateway"""
        self.middleware[name] = middleware
    
    def set_global_middleware(self, middleware_names: List[str]):
        """Set global middleware that applies to all routes"""
        self.global_middleware = middleware_names
    
    def handle_request(self, request: Dict) -> Dict:
        """Handle incoming request"""
        # Find matching route
        route = self._find_route(request)
        
        if not route:
            return {
                'status_code': 404,
                'body': {'error': 'Route not found'},
                'headers': {'Content-Type': 'application/json'}
            }
        
        # Apply middleware (global + route-specific)
        middleware_chain = self.global_middleware + route.middleware
        
        # Process request through middleware
        for middleware_name in middleware_chain:
            if middleware_name in self.middleware:
                request = self.middleware[middleware_name].process_request(request)
                
                # Check if middleware returned an error
                if 'error' in request:
                    error = request['error']
                    return {
                        'status_code': error['code'],
                        'body': {'error': error['message']},
                        'headers': {'Content-Type': 'application/json'}
                    }
        
        # Forward request to upstream service
        request['upstream_url'] = route.upstream_url
        response = self._forward_request(request, route)
        
        # Process response through middleware (in reverse order)
        response['request'] = request
        for middleware_name in reversed(middleware_chain):
            if middleware_name in self.middleware:
                response = self.middleware[middleware_name].process_response(response)
        
        return response
    
    def _find_route(self, request: Dict) -> Optional[Route]:
        """Find matching route for request"""
        path = request.get('path', '')
        method = request.get('method', 'GET')
        
        for route in self.routes:
            if self._path_matches(path, route.path_pattern) and method in route.methods:
                return route
        
        return None
    
    def _path_matches(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern (simplified)"""
        if pattern.endswith('/*'):
            prefix = pattern[:-2]
            return path.startswith(prefix)
        else:
            return path == pattern
    
    def _forward_request(self, request: Dict, route: Route) -> Dict:
        """Forward request to upstream service"""
        try:
            # Construct upstream URL
            upstream_path = request['path']
            if route.path_pattern.endswith('/*'):
                # Remove route prefix
                prefix = route.path_pattern[:-2]
                if upstream_path.startswith(prefix):
                    upstream_path = upstream_path[len(prefix):]
            
            url = route.upstream_url.rstrip('/') + upstream_path
            
            # Add query parameters
            if request.get('query_params'):
                url += '?' + '&'.join(f"{k}={v}" for k, v in request['query_params'].items())
            
            # Forward request with retry logic
            for attempt in range(route.retry_attempts):
                try:
                    response = requests.request(
                        method=request['method'],
                        url=url,
                        headers=request.get('headers', {}),
                        data=request.get('body'),
                        timeout=route.timeout
                    )
                    
                    return {
                        'status_code': response.status_code,
                        'body': response.text,
                        'headers': dict(response.headers)
                    }
                    
                except requests.exceptions.RequestException as e:
                    if attempt == route.retry_attempts - 1:
                        # Last attempt failed
                        return {
                            'status_code': 502,
                            'body': {'error': f'Upstream service unavailable: {str(e)}'},
                            'headers': {'Content-Type': 'application/json'}
                        }
                    
                    # Wait before retry
                    time.sleep(min(2 ** attempt, 10))
        
        except Exception as e:
            return {
                'status_code': 500,
                'body': {'error': f'Gateway error: {str(e)}'},
                'headers': {'Content-Type': 'application/json'}
            }

# Usage example
def api_gateway_examples():
    """Demonstrate API Gateway patterns"""
    print("=== API Gateway Examples ===")
    
    # Create gateway
    gateway = APIGateway()
    
    # Add middleware
    gateway.add_middleware('auth', AuthenticationMiddleware(
        secret_key='your-secret-key',
        skip_paths=['/health', '/public']
    ))
    
    gateway.add_middleware('rate_limit', RateLimitingMiddleware(
        requests_per_minute=100,
        burst_size=10
    ))
    
    gateway.add_middleware('circuit_breaker', CircuitBreakerMiddleware(
        failure_threshold=5,
        timeout=60.0
    ))
    
    # Set global middleware
    gateway.set_global_middleware(['rate_limit', 'circuit_breaker'])
    
    # Add routes
    gateway.add_route(Route(
        name='user_service',
        path_pattern='/api/users/*',
        upstream_url='http://user-service:8080',
        methods=['GET', 'POST', 'PUT', 'DELETE'],
        middleware=['auth']
    ))
    
    gateway.add_route(Route(
        name='public_api',
        path_pattern='/public/*',
        upstream_url='http://public-service:8080',
        methods=['GET']
    ))
    
    # Example requests
    requests_examples = [
        {
            'method': 'GET',
            'path': '/api/users/123',
            'headers': {'Authorization': 'Bearer valid-jwt-token'},
            'client_ip': '192.168.1.100'
        },
        {
            'method': 'GET',
            'path': '/public/info',
            'headers': {},
            'client_ip': '192.168.1.101'
        },
        {
            'method': 'POST',
            'path': '/api/users',
            'headers': {},  # Missing auth
            'client_ip': '192.168.1.102'
        }
    ]
    
    for i, request in enumerate(requests_examples):
        print(f"\nRequest {i+1}: {request['method']} {request['path']}")
        response = gateway.handle_request(request)
        print(f"Response: {response['status_code']} - {response.get('body', {}).get('error', 'Success')}")

## Microservices Architecture

### Service boundaries

Defining proper service boundaries is crucial for successful microservices architecture.

**Domain-Driven Design Approach:**

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Domain entities and value objects
@dataclass
class CustomerId:
    """Value object for customer ID"""
    value: str
    
    def __post_init__(self):
        if not self.value or len(self.value) < 3:
            raise ValueError("Customer ID must be at least 3 characters")

@dataclass
class ProductId:
    """Value object for product ID"""
    value: str

@dataclass
class Money:
    """Value object for money"""
    amount: float
    currency: str = "USD"
    
    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

# Domain entities
@dataclass
class Customer:
    """Customer entity in Customer Management service"""
    customer_id: CustomerId
    name: str
    email: str
    address: str
    credit_limit: Money
    
    def update_credit_limit(self, new_limit: Money):
        """Business logic for updating credit limit"""
        if new_limit.amount > self.credit_limit.amount * 2:
            raise ValueError("Cannot increase credit limit by more than 100%")
        self.credit_limit = new_limit

@dataclass
class Product:
    """Product entity in Product Catalog service"""
    product_id: ProductId
    name: str
    description: str
    price: Money
    stock_quantity: int
    category: str
    
    def update_stock(self, quantity_change: int):
        """Business logic for stock updates"""
        new_quantity = self.stock_quantity + quantity_change
        if new_quantity < 0:
            raise ValueError("Insufficient stock")
        self.stock_quantity = new_quantity

@dataclass
class OrderItem:
    product_id: ProductId
    quantity: int
    unit_price: Money

@dataclass
class Order:
    """Order entity in Order Management service"""
    order_id: str
    customer_id: CustomerId
    items: List[OrderItem]
    status: OrderStatus
    total_amount: Money
    created_at: str
    
    def add_item(self, item: OrderItem):
        """Business logic for adding items"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Cannot modify confirmed order")
        
        self.items.append(item)
        self._recalculate_total()
    
    def _recalculate_total(self):
        """Recalculate order total"""
        total = sum(item.quantity * item.unit_price.amount for item in self.items)
        self.total_amount = Money(total)

# Service interfaces (abstractions for cross-service communication)
class CustomerServiceInterface(ABC):
    """Interface for Customer service"""
    
    @abstractmethod
    def get_customer(self, customer_id: CustomerId) -> Optional[Customer]:
        pass
    
    @abstractmethod
    def validate_credit_limit(self, customer_id: CustomerId, amount: Money) -> bool:
        pass

class ProductServiceInterface(ABC):
    """Interface for Product service"""
    
    @abstractmethod
    def get_product(self, product_id: ProductId) -> Optional[Product]:
        pass
    
    @abstractmethod
    def reserve_stock(self, product_id: ProductId, quantity: int) -> bool:
        pass
    
    @abstractmethod
    def release_stock(self, product_id: ProductId, quantity: int) -> bool:
        pass

class PaymentServiceInterface(ABC):
    """Interface for Payment service"""
    
    @abstractmethod
    def process_payment(self, customer_id: CustomerId, amount: Money) -> bool:
        pass

# Bounded contexts and service boundaries
class CustomerManagementService:
    """Customer Management bounded context"""
    
    def __init__(self):
        self.customers: Dict[str, Customer] = {}
    
    def create_customer(self, name: str, email: str, address: str, 
                       credit_limit: Money) -> Customer:
        """Create new customer"""
        customer_id = CustomerId(f"CUST_{len(self.customers) + 1:06d}")
        
        customer = Customer(
            customer_id=customer_id,
            name=name,
            email=email,
            address=address,
            credit_limit=credit_limit
        )
        
        self.customers[customer_id.value] = customer
        
        # Publish domain event
        self._publish_event("CustomerCreated", {
            "customer_id": customer_id.value,
            "name": name,
            "email": email
        })
        
        return customer
    
    def get_customer(self, customer_id: CustomerId) -> Optional[Customer]:
        return self.customers.get(customer_id.value)
    
    def validate_credit_limit(self, customer_id: CustomerId, amount: Money) -> bool:
        customer = self.get_customer(customer_id)
        if not customer:
            return False
        
        return amount.amount <= customer.credit_limit.amount
    
    def _publish_event(self, event_type: str, data: Dict):
        """Publish domain event to message bus"""
        print(f"Publishing event: {event_type} - {data}")

class ProductCatalogService:
    """Product Catalog bounded context"""
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
    
    def add_product(self, name: str, description: str, price: Money, 
                   stock_quantity: int, category: str) -> Product:
        """Add new product to catalog"""
        product_id = ProductId(f"PROD_{len(self.products) + 1:06d}")
        
        product = Product(
            product_id=product_id,
            name=name,
            description=description,
            price=price,
            stock_quantity=stock_quantity,
            category=category
        )
        
        self.products[product_id.value] = product
        
        self._publish_event("ProductAdded", {
            "product_id": product_id.value,
            "name": name,
            "price": price.amount,
            "category": category
        })
        
        return product
    
    def get_product(self, product_id: ProductId) -> Optional[Product]:
        return self.products.get(product_id.value)
    
    def reserve_stock(self, product_id: ProductId, quantity: int) -> bool:
        """Reserve stock for an order"""
        product = self.get_product(product_id)
        if not product or product.stock_quantity < quantity:
            return False
        
        product.update_stock(-quantity)
        
        self._publish_event("StockReserved", {
            "product_id": product_id.value,
            "quantity": quantity,
            "remaining_stock": product.stock_quantity
        })
        
        return True
    
    def release_stock(self, product_id: ProductId, quantity: int) -> bool:
        """Release reserved stock"""
        product = self.get_product(product_id)
        if not product:
            return False
        
        product.update_stock(quantity)
        
        self._publish_event("StockReleased", {
            "product_id": product_id.value,
            "quantity": quantity,
            "new_stock": product.stock_quantity
        })
        
        return True
    
    def _publish_event(self, event_type: str, data: Dict):
        print(f"Publishing event: {event_type} - {data}")

class OrderManagementService:
    """Order Management bounded context"""
    
    def __init__(self, customer_service: CustomerServiceInterface,
                 product_service: ProductServiceInterface,
                 payment_service: PaymentServiceInterface):
        self.orders: Dict[str, Order] = {}
        self.customer_service = customer_service
        self.product_service = product_service
        self.payment_service = payment_service
    
    def create_order(self, customer_id: CustomerId, 
                    items: List[Dict]) -> Optional[Order]:
        """Create new order with validation across services"""
        
        # Validate customer exists
        customer = self.customer_service.get_customer(customer_id)
        if not customer:
            raise ValueError("Customer not found")
        
        order_id = f"ORDER_{len(self.orders) + 1:06d}"
        order_items = []
        total_amount = 0
        
        # Validate and reserve stock for each item
        reserved_items = []
        
        try:
            for item_data in items:
                product_id = ProductId(item_data['product_id'])
                quantity = item_data['quantity']
                
                # Get product details
                product = self.product_service.get_product(product_id)
                if not product:
                    raise ValueError(f"Product {product_id.value} not found")
                
                # Reserve stock
                if not self.product_service.reserve_stock(product_id, quantity):
                    raise ValueError(f"Insufficient stock for {product_id.value}")
                
                reserved_items.append((product_id, quantity))
                
                # Create order item
                order_item = OrderItem(
                    product_id=product_id,
                    quantity=quantity,
                    unit_price=product.price
                )
                order_items.append(order_item)
                total_amount += quantity * product.price.amount
            
            # Validate credit limit
            total_money = Money(total_amount)
            if not self.customer_service.validate_credit_limit(customer_id, total_money):
                raise ValueError("Order exceeds customer credit limit")
            
            # Create order
            order = Order(
                order_id=order_id,
                customer_id=customer_id,
                items=order_items,
                status=OrderStatus.PENDING,
                total_amount=total_money,
                created_at=str(time.time())
            )
            
            self.orders[order_id] = order
            
            self._publish_event("OrderCreated", {
                "order_id": order_id,
                "customer_id": customer_id.value,
                "total_amount": total_amount,
                "item_count": len(order_items)
            })
            
            return order
            
        except Exception as e:
            # Rollback reserved stock
            for product_id, quantity in reserved_items:
                self.product_service.release_stock(product_id, quantity)
            
            raise e
    
    def confirm_order(self, order_id: str) -> bool:
        """Confirm order and process payment"""
        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.PENDING:
            return False
        
        # Process payment
        if self.payment_service.process_payment(order.customer_id, order.total_amount):
            order.status = OrderStatus.CONFIRMED
            
            self._publish_event("OrderConfirmed", {
                "order_id": order_id,
                "customer_id": order.customer_id.value,
                "amount": order.total_amount.amount
            })
            
            return True
        else:
            # Release reserved stock on payment failure
            for item in order.items:
                self.product_service.release_stock(item.product_id, item.quantity)
            
            order.status = OrderStatus.CANCELLED
            return False
    
    def _publish_event(self, event_type: str, data: Dict):
        print(f"Publishing event: {event_type} - {data}")

# Service boundary analysis
def analyze_service_boundaries():
    """Analyze and demonstrate proper service boundaries"""
    
    print("=== Service Boundary Analysis ===")
    
    principles = {
        "Single Responsibility": [
            "Each service owns a specific business capability",
            "Customer service manages customer data and validation",
            "Product service manages catalog and inventory",
            "Order service orchestrates the ordering process"
        ],
        
        "Data Ownership": [
            "Each service owns its data exclusively",
            "No direct database access between services",
            "Data consistency via events and eventual consistency",
            "Clear APIs for data access"
        ],
        
        "Domain Cohesion": [
            "Services align with business domains",
            "High cohesion within service boundaries",
            "Loose coupling between services",
            "Business rules encapsulated within services"
        ],
        
        "Communication Patterns": [
            "Synchronous calls for immediate validation",
            "Asynchronous events for state changes",
            "API contracts define service interfaces",
            "Backward compatibility maintained"
        ],
        
        "Autonomy": [
            "Services can be developed independently",
            "Different technologies per service",
            "Independent deployment cycles",
            "Separate development teams"
        ]
    }
    
    for principle, details in principles.items():
        print(f"\n{principle}:")
        for detail in details:
            print(f"  • {detail}")
    
    print("\n=== Anti-patterns to Avoid ===")
    antipatterns = [
        "Shared databases between services",
        "Chatty inter-service communication",
        "Services that are too fine-grained",
        "Distributed monoliths with tight coupling",
        "Synchronous chains of service calls",
        "Services organized by technical layers"
    ]
    
    for antipattern in antipatterns:
        print(f"  ✗ {antipattern}")

# Example usage
def service_boundaries_examples():
    """Demonstrate service boundaries in practice"""
    
    # Create service implementations
    customer_service = CustomerManagementService()
    product_service = ProductCatalogService()
    
    # Mock payment service
    class MockPaymentService:
        def process_payment(self, customer_id: CustomerId, amount: Money) -> bool:
            print(f"Processing payment for {customer_id.value}: ${amount.amount}")
            return True
    
    payment_service = MockPaymentService()
    
    # Create order service with dependencies
    order_service = OrderManagementService(
        customer_service=customer_service,
        product_service=product_service,
        payment_service=payment_service
    )
    
    # Create test data
    customer = customer_service.create_customer(
        name="John Doe",
        email="john@example.com",
        address="123 Main St",
        credit_limit=Money(1000.0)
    )
    
    product = product_service.add_product(
        name="Laptop",
        description="High-performance laptop",
        price=Money(800.0),
        stock_quantity=10,
        category="Electronics"
    )
    
    # Create order (cross-service orchestration)
    try:
        order = order_service.create_order(
            customer_id=customer.customer_id,
            items=[{
                'product_id': product.product_id.value,
                'quantity': 1
            }]
        )
        
        print(f"\nOrder created: {order.order_id}")
        
        # Confirm order
        success = order_service.confirm_order(order.order_id)
        print(f"Order confirmation: {'Success' if success else 'Failed'}")
        
    except Exception as e:
        print(f"Order creation failed: {e}")
    
    analyze_service_boundaries()
```
```