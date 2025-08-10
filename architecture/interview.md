# Software Architecture Interview Guide

A comprehensive guide for software architecture interviews covering design principles, system design, scalability, and real-world architectural decisions.

## Table of Contents

1. [Fundamental Design Principles](#fundamental-design-principles)
2. [System Architecture Patterns](#system-architecture-patterns)
3. [Scalability and Performance](#scalability-and-performance)
4. [Database Design and Data Management](#database-design-and-data-management)
5. [Common Interview Questions](#common-interview-questions)
6. [Real-World System Design Scenarios](#real-world-system-design-scenarios)
7. [Best Practices and Anti-Patterns](#best-practices-and-anti-patterns)
8. [Modern Architecture Trends](#modern-architecture-trends)

## Fundamental Design Principles

### 1. SOLID Principles

#### Single Responsibility Principle (SRP)
- **Definition**: A class should have only one reason to change
- **Example**: Separate user authentication from user data management

```java
// Bad: Multiple responsibilities
class User {
    private String name;
    private String email;
    
    public void save() { /* database logic */ }
    public void sendEmail() { /* email logic */ }
    public boolean authenticate(String password) { /* auth logic */ }
}

// Good: Single responsibilities
class User {
    private String name;
    private String email;
    // Only user data management
}

class UserRepository {
    public void save(User user) { /* database logic */ }
}

class EmailService {
    public void sendEmail(User user, String message) { /* email logic */ }
}

class AuthenticationService {
    public boolean authenticate(User user, String password) { /* auth logic */ }
}
```

#### Open-Closed Principle (OCP)
- **Definition**: Software entities should be open for extension but closed for modification

```java
// Good: Uses strategy pattern for extensibility
interface PaymentProcessor {
    void processPayment(double amount);
}

class CreditCardProcessor implements PaymentProcessor {
    public void processPayment(double amount) { /* credit card logic */ }
}

class PayPalProcessor implements PaymentProcessor {
    public void processPayment(double amount) { /* PayPal logic */ }
}

class PaymentService {
    private PaymentProcessor processor;
    
    public PaymentService(PaymentProcessor processor) {
        this.processor = processor;
    }
    
    public void processPayment(double amount) {
        processor.processPayment(amount);
    }
}
```

#### Liskov Substitution Principle (LSP)
- **Definition**: Objects of a superclass should be replaceable with objects of its subclasses

#### Interface Segregation Principle (ISP)
- **Definition**: Clients should not be forced to depend on interfaces they do not use

#### Dependency Inversion Principle (DIP)
- **Definition**: High-level modules should not depend on low-level modules; both should depend on abstractions

### 2. Additional Design Principles

#### DRY (Don't Repeat Yourself)
```python
# Bad: Code duplication
def calculate_circle_area(radius):
    return 3.14159 * radius * radius

def calculate_circle_circumference(radius):
    return 2 * 3.14159 * radius

# Good: Extract constants and common logic
import math

class CircleCalculator:
    PI = math.pi
    
    @staticmethod
    def area(radius):
        return CircleCalculator.PI * radius ** 2
    
    @staticmethod
    def circumference(radius):
        return 2 * CircleCalculator.PI * radius
```

#### KISS (Keep It Simple, Stupid)
- Prefer simple solutions over complex ones
- Avoid over-engineering
- Write code that's easy to understand and maintain

#### YAGNI (You Aren't Gonna Need It)
- Don't implement features until you actually need them
- Avoid speculative development
- Focus on current requirements

### 3. Design Patterns

#### Creational Patterns

**Singleton Pattern**
```java
public class DatabaseConnection {
    private static volatile DatabaseConnection instance;
    private Connection connection;
    
    private DatabaseConnection() {
        // Initialize connection
    }
    
    public static DatabaseConnection getInstance() {
        if (instance == null) {
            synchronized (DatabaseConnection.class) {
                if (instance == null) {
                    instance = new DatabaseConnection();
                }
            }
        }
        return instance;
    }
}
```

**Factory Pattern**
```java
interface Vehicle {
    void start();
}

class Car implements Vehicle {
    public void start() { System.out.println("Car started"); }
}

class Motorcycle implements Vehicle {
    public void start() { System.out.println("Motorcycle started"); }
}

class VehicleFactory {
    public static Vehicle createVehicle(String type) {
        switch (type.toLowerCase()) {
            case "car": return new Car();
            case "motorcycle": return new Motorcycle();
            default: throw new IllegalArgumentException("Unknown vehicle type");
        }
    }
}
```

#### Structural Patterns

**Adapter Pattern**
```java
// Legacy interface
interface OldPrinter {
    void oldPrint(String text);
}

// New interface
interface ModernPrinter {
    void print(String text);
}

// Adapter
class PrinterAdapter implements ModernPrinter {
    private OldPrinter oldPrinter;
    
    public PrinterAdapter(OldPrinter oldPrinter) {
        this.oldPrinter = oldPrinter;
    }
    
    public void print(String text) {
        oldPrinter.oldPrint(text);
    }
}
```

#### Behavioral Patterns

**Observer Pattern**
```java
interface Observer {
    void update(String message);
}

class Subject {
    private List<Observer> observers = new ArrayList<>();
    
    public void addObserver(Observer observer) {
        observers.add(observer);
    }
    
    public void removeObserver(Observer observer) {
        observers.remove(observer);
    }
    
    public void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}
```

## System Architecture Patterns

### 1. Monolithic Architecture

#### Characteristics
- Single deployable unit
- Shared database
- Inter-module communication through method calls
- Single technology stack

#### Pros and Cons
**Advantages:**
- Simple to develop and test initially
- Easy deployment
- Good performance for small applications
- Strong consistency

**Disadvantages:**
- Difficult to scale specific components
- Technology lock-in
- Large codebase becomes hard to maintain
- Single point of failure

#### When to Use
- Small to medium applications
- Simple business domains
- Limited team size
- Proof of concepts

### 2. Microservices Architecture

#### Characteristics
- Multiple small, independent services
- Service-specific databases
- Communication through APIs (REST, gRPC, messaging)
- Technology diversity

```yaml
# Example microservices architecture
services:
  user-service:
    responsibility: User management
    database: PostgreSQL
    technology: Java Spring Boot
    
  order-service:
    responsibility: Order processing
    database: MongoDB
    technology: Node.js
    
  payment-service:
    responsibility: Payment processing
    database: PostgreSQL
    technology: Python Django
    
  notification-service:
    responsibility: Notifications
    database: Redis
    technology: Go
```

#### Microservices Design Patterns

**API Gateway Pattern**
```
Client → API Gateway → [User Service, Order Service, Payment Service]
```

**Service Mesh Pattern**
- Istio, Linkerd for service-to-service communication
- Traffic management, security, observability

**Circuit Breaker Pattern**
```java
@Component
public class ExternalServiceClient {
    
    @CircuitBreaker(name = "external-service", fallbackMethod = "fallbackMethod")
    public String callExternalService() {
        // Call to external service
        return restTemplate.getForObject("/api/data", String.class);
    }
    
    public String fallbackMethod(Exception ex) {
        return "Fallback response";
    }
}
```

#### Pros and Cons
**Advantages:**
- Independent scaling
- Technology diversity
- Team autonomy
- Fault isolation

**Disadvantages:**
- Increased complexity
- Network latency
- Data consistency challenges
- Testing complexity

### 3. Serverless Architecture

#### Characteristics
- Function as a Service (FaaS)
- Event-driven execution
- Automatic scaling
- Pay-per-execution model

#### AWS Lambda Example
```python
import json
import boto3

def lambda_handler(event, context):
    # Process the event
    user_id = event['user_id']
    
    # Business logic
    result = process_user_data(user_id)
    
    # Return response
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Success',
            'result': result
        })
    }

def process_user_data(user_id):
    # Your business logic here
    return f"Processed user {user_id}"
```

#### Serverless Patterns

**Function Composition**
```yaml
# AWS Step Functions example
StateMachine:
  StartAt: ValidateInput
  States:
    ValidateInput:
      Type: Task
      Resource: arn:aws:lambda:region:account:function:validate-input
      Next: ProcessData
    ProcessData:
      Type: Task
      Resource: arn:aws:lambda:region:account:function:process-data
      Next: SaveResult
    SaveResult:
      Type: Task
      Resource: arn:aws:lambda:region:account:function:save-result
      End: true
```

### 4. Event-Driven Architecture

#### Components
- Event Producers
- Event Routers (Message Brokers)
- Event Consumers

#### Message Patterns

**Publish-Subscribe**
```python
# Using Redis pub/sub
import redis

class EventPublisher:
    def __init__(self):
        self.redis_client = redis.Redis()
    
    def publish_event(self, channel, event_data):
        self.redis_client.publish(channel, json.dumps(event_data))

class EventSubscriber:
    def __init__(self, channels):
        self.redis_client = redis.Redis()
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(channels)
    
    def listen_for_events(self):
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                self.handle_event(message['data'])
    
    def handle_event(self, event_data):
        # Process the event
        pass
```

**Event Sourcing**
```python
class Event:
    def __init__(self, event_type, data, timestamp=None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()

class EventStore:
    def __init__(self):
        self.events = []
    
    def append_event(self, stream_id, event):
        event_record = {
            'stream_id': stream_id,
            'event': event,
            'version': len(self.get_events(stream_id)) + 1
        }
        self.events.append(event_record)
    
    def get_events(self, stream_id):
        return [e for e in self.events if e['stream_id'] == stream_id]

class UserAggregate:
    def __init__(self, user_id):
        self.user_id = user_id
        self.name = None
        self.email = None
        self.version = 0
    
    def apply_event(self, event):
        if event.event_type == 'UserCreated':
            self.name = event.data['name']
            self.email = event.data['email']
        elif event.event_type == 'UserUpdated':
            if 'name' in event.data:
                self.name = event.data['name']
            if 'email' in event.data:
                self.email = event.data['email']
        self.version += 1
    
    @classmethod
    def from_events(cls, user_id, events):
        user = cls(user_id)
        for event in events:
            user.apply_event(event)
        return user
```

## Scalability and Performance

### 1. Horizontal vs Vertical Scaling

#### Vertical Scaling (Scale Up)
- **Definition**: Adding more power to existing machines
- **Examples**: More CPU, RAM, storage
- **Pros**: Simple to implement, no application changes
- **Cons**: Limited by hardware, single point of failure
- **Best for**: Databases, legacy applications

#### Horizontal Scaling (Scale Out)
- **Definition**: Adding more machines to the pool
- **Examples**: Load balancers, multiple server instances
- **Pros**: Theoretically unlimited scaling, fault tolerance
- **Cons**: Application complexity, data consistency challenges
- **Best for**: Web applications, microservices

### 2. Load Balancing

#### Load Balancing Algorithms

**Round Robin**
```python
class RoundRobinLoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
```

**Weighted Round Robin**
```python
class WeightedRoundRobinLoadBalancer:
    def __init__(self, servers_with_weights):
        self.servers = []
        for server, weight in servers_with_weights:
            self.servers.extend([server] * weight)
        self.current = 0
    
    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
```

**Least Connections**
```python
class LeastConnectionsLoadBalancer:
    def __init__(self, servers):
        self.servers = {server: 0 for server in servers}
    
    def get_server(self):
        return min(self.servers, key=self.servers.get)
    
    def add_connection(self, server):
        self.servers[server] += 1
    
    def remove_connection(self, server):
        self.servers[server] -= 1
```

#### Load Balancer Types
- **Layer 4 (Transport)**: Routes based on IP and port
- **Layer 7 (Application)**: Routes based on application data (HTTP headers, URLs)

### 3. Caching Strategies

#### Cache Levels
1. **Browser Cache**: Client-side caching
2. **CDN (Content Delivery Network)**: Geographic distribution
3. **Reverse Proxy Cache**: Nginx, Varnish
4. **Application Cache**: In-memory caching
5. **Database Cache**: Query result caching

#### Cache Patterns

**Cache-Aside (Lazy Loading)**
```python
class CacheAsideExample:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
    
    def get_user(self, user_id):
        # Try cache first
        user = self.cache.get(f"user:{user_id}")
        if user is None:
            # Cache miss - fetch from database
            user = self.database.get_user(user_id)
            if user:
                # Store in cache
                self.cache.set(f"user:{user_id}", user, ttl=3600)
        return user
    
    def update_user(self, user_id, user_data):
        # Update database
        self.database.update_user(user_id, user_data)
        # Invalidate cache
        self.cache.delete(f"user:{user_id}")
```

**Write-Through**
```python
class WriteThroughCache:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
    
    def update_user(self, user_id, user_data):
        # Update database first
        self.database.update_user(user_id, user_data)
        # Update cache
        self.cache.set(f"user:{user_id}", user_data, ttl=3600)
```

**Write-Behind (Write-Back)**
```python
import asyncio
from queue import Queue

class WriteBehindCache:
    def __init__(self, cache, database):
        self.cache = cache
        self.database = database
        self.write_queue = Queue()
        self.start_background_writer()
    
    def update_user(self, user_id, user_data):
        # Update cache immediately
        self.cache.set(f"user:{user_id}", user_data, ttl=3600)
        # Queue for database write
        self.write_queue.put(('update_user', user_id, user_data))
    
    async def background_writer(self):
        while True:
            if not self.write_queue.empty():
                operation, user_id, user_data = self.write_queue.get()
                if operation == 'update_user':
                    self.database.update_user(user_id, user_data)
            await asyncio.sleep(1)  # Write interval
    
    def start_background_writer(self):
        asyncio.create_task(self.background_writer())
```

### 4. Content Delivery Networks (CDN)

#### CDN Benefits
- Reduced latency
- Decreased server load
- Improved availability
- DDoS protection

#### CDN Configuration Example
```yaml
# CloudFlare configuration example
cache_rules:
  - pattern: "*.css"
    cache_ttl: 86400  # 1 day
  - pattern: "*.js"
    cache_ttl: 86400
  - pattern: "*.jpg"
    cache_ttl: 604800  # 1 week
  - pattern: "/api/*"
    cache_ttl: 0  # No cache
```

## Database Design and Data Management

### 1. SQL vs NoSQL

#### SQL Databases (RDBMS)
**Characteristics:**
- ACID properties
- Structured schema
- SQL query language
- Vertical scaling

**Use Cases:**
- Financial applications
- Complex queries and joins
- Strong consistency requirements

**Examples:** PostgreSQL, MySQL, Oracle

#### NoSQL Databases

**Document Stores**
```javascript
// MongoDB example
{
  "_id": ObjectId("..."),
  "name": "John Doe",
  "email": "john@example.com",
  "addresses": [
    {
      "type": "home",
      "street": "123 Main St",
      "city": "Anytown"
    }
  ],
  "preferences": {
    "theme": "dark",
    "notifications": true
  }
}
```

**Key-Value Stores**
```python
# Redis example
redis_client.set("user:1001", json.dumps(user_data))
redis_client.setex("session:abc123", 3600, session_data)  # With TTL
```

**Column-Family**
```sql
-- Cassandra example
CREATE TABLE user_activity (
    user_id UUID,
    activity_date DATE,
    activity_time TIMESTAMP,
    activity_type TEXT,
    details TEXT,
    PRIMARY KEY (user_id, activity_date, activity_time)
) WITH CLUSTERING ORDER BY (activity_date DESC, activity_time DESC);
```

**Graph Databases**
```cypher
// Neo4j example
CREATE (john:Person {name: 'John', age: 30})
CREATE (mary:Person {name: 'Mary', age: 25})
CREATE (john)-[:FRIEND_OF]->(mary)
CREATE (company:Company {name: 'TechCorp'})
CREATE (john)-[:WORKS_FOR]->(company)
```

### 2. Database Scaling Patterns

#### Read Replicas
```python
class DatabaseRouter:
    def __init__(self, master_db, read_replicas):
        self.master_db = master_db
        self.read_replicas = read_replicas
        self.replica_index = 0
    
    def write_query(self, query, params):
        return self.master_db.execute(query, params)
    
    def read_query(self, query, params):
        replica = self.read_replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.read_replicas)
        return replica.execute(query, params)
```

#### Database Sharding
```python
class ShardedDatabase:
    def __init__(self, shards):
        self.shards = shards
    
    def get_shard(self, user_id):
        shard_key = hash(user_id) % len(self.shards)
        return self.shards[shard_key]
    
    def get_user(self, user_id):
        shard = self.get_shard(user_id)
        return shard.get_user(user_id)
    
    def create_user(self, user_data):
        user_id = user_data['id']
        shard = self.get_shard(user_id)
        return shard.create_user(user_data)
```

#### Database Partitioning

**Horizontal Partitioning (Sharding)**
- Split rows across multiple databases
- Based on partition key (user_id, date, etc.)

**Vertical Partitioning**
- Split columns across multiple databases
- Separate frequently vs rarely accessed data

**Functional Partitioning**
- Split by feature/domain
- Users service, Orders service, etc.

### 3. Data Consistency Patterns

#### CAP Theorem
- **Consistency**: All nodes see the same data simultaneously
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures

**You can only guarantee 2 out of 3 properties**

#### Consistency Models

**Strong Consistency**
```python
# Example: Bank transfer with strong consistency
def transfer_money(from_account, to_account, amount):
    with database.transaction():
        # Both operations succeed or both fail
        from_balance = get_balance(from_account)
        if from_balance >= amount:
            update_balance(from_account, from_balance - amount)
            to_balance = get_balance(to_account)
            update_balance(to_account, to_balance + amount)
        else:
            raise InsufficientFundsError()
```

**Eventual Consistency**
```python
# Example: Social media likes with eventual consistency
def like_post(user_id, post_id):
    # Update local cache immediately
    cache.increment(f"post_likes:{post_id}")
    
    # Queue for asynchronous database update
    message_queue.send({
        'action': 'increment_likes',
        'post_id': post_id,
        'user_id': user_id
    })
```

**SAGA Pattern for Distributed Transactions**
```python
class OrderSaga:
    def __init__(self):
        self.steps = []
        self.compensations = []
    
    def execute(self, order_data):
        try:
            # Step 1: Reserve inventory
            reservation_id = self.inventory_service.reserve_items(order_data['items'])
            self.steps.append(('reserve_items', reservation_id))
            self.compensations.append(('release_items', reservation_id))
            
            # Step 2: Process payment
            payment_id = self.payment_service.charge(order_data['payment'])
            self.steps.append(('charge_payment', payment_id))
            self.compensations.append(('refund_payment', payment_id))
            
            # Step 3: Create order
            order_id = self.order_service.create_order(order_data)
            self.steps.append(('create_order', order_id))
            
            return order_id
            
        except Exception as e:
            # Compensate in reverse order
            self.compensate()
            raise e
    
    def compensate(self):
        for compensation_action, param in reversed(self.compensations):
            try:
                if compensation_action == 'release_items':
                    self.inventory_service.release_reservation(param)
                elif compensation_action == 'refund_payment':
                    self.payment_service.refund(param)
            except Exception as e:
                # Log compensation failure
                logger.error(f"Compensation failed: {e}")
```

## Common Interview Questions

### 1. System Design Questions

**Q: Design a URL shortener (like bit.ly)**

**Requirements:**
- Shorten long URLs
- Redirect to original URL
- Handle 100M URLs per day
- Custom aliases (optional)
- Analytics (optional)

**Solution Approach:**
```
1. Capacity Estimation:
   - 100M URLs/day = ~1160 URLs/second
   - Read:Write ratio = 100:1 (people click more than create)
   - Storage: ~500 bytes per URL × 100M = 50GB per day

2. Database Design:
   URL Table:
   - short_url (PK)
   - long_url
   - user_id
   - created_at
   - expires_at

3. Algorithm for Short URL Generation:
   - Base62 encoding (a-z, A-Z, 0-9)
   - Counter-based: encode incrementing counter
   - Hash-based: MD5/SHA256 + truncate
   - Random: generate random string + check collision

4. Architecture:
   [Client] → [Load Balancer] → [App Servers] → [Cache] → [Database]
                                    ↓
                               [Analytics DB]

5. Caching Strategy:
   - Cache popular URLs in Redis
   - LRU eviction policy
   - Cache hit ratio: ~80%
```

**Q: Design a chat application (like WhatsApp)**

**Solution Components:**
```
1. Real-time Messaging:
   - WebSocket connections
   - Message queues (Kafka/RabbitMQ)
   - Presence service

2. Data Models:
   Users: user_id, name, phone, last_seen
   Conversations: conv_id, participants, created_at
   Messages: message_id, conv_id, sender_id, content, timestamp

3. Architecture:
   [Mobile Apps] ←WebSocket→ [Gateway] → [Message Service]
                                           ↓
   [Notification Service] ← [Message Queue] → [Database]

4. Scalability:
   - Shard by user_id or conversation_id
   - Read replicas for message history
   - CDN for media files
```

### 2. Architecture Decision Questions

**Q: When would you choose microservices over monolith?**

**Choose Microservices when:**
- Large, complex domain
- Multiple teams working independently
- Different scaling requirements for components
- Technology diversity needed
- Organizational readiness (DevOps maturity)

**Choose Monolith when:**
- Small to medium applications
- Single team or small teams
- Simple domain
- Rapid prototyping
- Limited operational overhead capacity

**Q: How do you handle data consistency in microservices?**

**Strategies:**
1. **Shared Database** (anti-pattern but sometimes necessary)
2. **Database per Service** with eventual consistency
3. **Saga Pattern** for distributed transactions
4. **Event Sourcing** for audit trail and consistency
5. **Two-Phase Commit** (not recommended for high-scale)

### 3. Performance and Scalability Questions

**Q: How would you optimize database performance?**

**Database Optimization Strategies:**
```sql
-- 1. Indexing
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_date ON orders(created_at);

-- 2. Query optimization
-- Bad
SELECT * FROM users WHERE name LIKE '%john%';

-- Good
SELECT id, name, email FROM users WHERE name = 'john';

-- 3. Pagination
SELECT * FROM products 
ORDER BY created_at DESC 
LIMIT 20 OFFSET 0;

-- 4. Connection pooling
-- Configure connection pool size based on:
-- Pool size = ((core_count × 2) + effective_spindle_count)
```

**Q: How do you handle high traffic spikes?**

**Traffic Spike Strategies:**
1. **Auto-scaling**: Horizontal pod autoscaling in Kubernetes
2. **Circuit Breaker**: Prevent cascade failures
3. **Rate Limiting**: Protect against abuse
4. **Graceful Degradation**: Disable non-critical features
5. **Content Caching**: CDN and application-level caching

## Real-World System Design Scenarios

### 1. E-commerce Platform Architecture

#### System Components
```yaml
services:
  user-service:
    responsibility: User management, authentication
    database: PostgreSQL
    scaling: Read replicas
    
  product-catalog:
    responsibility: Product information, search
    database: Elasticsearch
    scaling: Horizontal scaling
    
  inventory-service:
    responsibility: Stock management
    database: PostgreSQL
    patterns: CQRS, Event Sourcing
    
  order-service:
    responsibility: Order processing
    database: PostgreSQL
    patterns: Saga pattern
    
  payment-service:
    responsibility: Payment processing
    database: PostgreSQL
    external: Stripe, PayPal APIs
    
  notification-service:
    responsibility: Email, SMS, push notifications
    database: Redis
    external: SendGrid, Twilio
```

#### Order Processing Flow
```python
class OrderProcessingOrchestrator:
    def __init__(self):
        self.inventory_service = InventoryService()
        self.payment_service = PaymentService()
        self.order_service = OrderService()
        self.notification_service = NotificationService()
    
    async def process_order(self, order_request):
        saga = OrderSaga()
        
        try:
            # Step 1: Validate and reserve inventory
            reservation = await saga.execute_step(
                self.inventory_service.reserve_items,
                order_request.items,
                compensation=self.inventory_service.release_reservation
            )
            
            # Step 2: Process payment
            payment = await saga.execute_step(
                self.payment_service.charge,
                order_request.payment_info,
                compensation=self.payment_service.refund
            )
            
            # Step 3: Create order
            order = await saga.execute_step(
                self.order_service.create_order,
                order_request,
                compensation=self.order_service.cancel_order
            )
            
            # Step 4: Confirm inventory
            await self.inventory_service.confirm_reservation(reservation.id)
            
            # Step 5: Send confirmation
            await self.notification_service.send_order_confirmation(
                order.customer_email, order
            )
            
            return order
            
        except Exception as e:
            await saga.compensate()
            raise OrderProcessingException(f"Order processing failed: {e}")
```

### 2. Social Media Platform Architecture

#### Feed Generation System
```python
class FeedGenerator:
    def __init__(self):
        self.redis_client = redis.Redis()
        self.graph_db = Neo4jClient()
        self.post_service = PostService()
    
    def generate_feed(self, user_id, limit=50):
        # Hybrid approach: precomputed + real-time
        
        # 1. Get precomputed feed (for heavy users)
        precomputed_feed = self.get_precomputed_feed(user_id, limit // 2)
        
        # 2. Get recent posts from close friends (real-time)
        close_friends = self.graph_db.get_close_friends(user_id)
        recent_posts = self.post_service.get_recent_posts(
            close_friends, 
            since=datetime.now() - timedelta(hours=6)
        )
        
        # 3. Merge and rank
        all_posts = precomputed_feed + recent_posts
        ranked_posts = self.rank_posts(all_posts, user_id)
        
        return ranked_posts[:limit]
    
    def get_precomputed_feed(self, user_id, limit):
        # Fan-out on write: precomputed feeds in Redis
        feed_key = f"feed:{user_id}"
        post_ids = self.redis_client.lrange(feed_key, 0, limit-1)
        return self.post_service.get_posts_by_ids(post_ids)
    
    def rank_posts(self, posts, user_id):
        # Machine learning ranking algorithm
        user_preferences = self.get_user_preferences(user_id)
        
        for post in posts:
            post.score = self.calculate_relevance_score(post, user_preferences)
        
        return sorted(posts, key=lambda p: p.score, reverse=True)
```

#### Real-time Notifications
```python
class NotificationSystem:
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.push_service = PushNotificationService()
        self.email_service = EmailService()
    
    async def send_notification(self, user_id, notification):
        user_preferences = await self.get_user_preferences(user_id)
        
        # Real-time notification (if user is online)
        if await self.websocket_manager.is_user_online(user_id):
            await self.websocket_manager.send_to_user(user_id, notification)
        
        # Push notification (if enabled and user is offline)
        elif user_preferences.push_notifications_enabled:
            await self.push_service.send_push(user_id, notification)
        
        # Email notification (for important notifications)
        if notification.priority == 'high' and user_preferences.email_notifications_enabled:
            await self.email_service.send_notification_email(user_id, notification)
        
        # Store notification for later retrieval
        await self.store_notification(user_id, notification)
```

### 3. Video Streaming Platform

#### Video Processing Pipeline
```python
class VideoProcessingPipeline:
    def __init__(self):
        self.storage_service = CloudStorageService()
        self.transcoding_service = VideoTranscodingService()
        self.cdn_service = CDNService()
        self.metadata_service = MetadataService()
    
    async def process_video_upload(self, video_file, metadata):
        pipeline_id = str(uuid.uuid4())
        
        try:
            # Step 1: Upload raw video to storage
            raw_video_url = await self.storage_service.upload(
                video_file, 
                f"raw/{pipeline_id}/{video_file.filename}"
            )
            
            # Step 2: Extract metadata and generate thumbnail
            video_info = await self.analyze_video(raw_video_url)
            thumbnail_url = await self.generate_thumbnail(raw_video_url)
            
            # Step 3: Transcode to multiple qualities
            transcoding_jobs = []
            for quality in ['240p', '480p', '720p', '1080p']:
                if video_info.resolution >= quality:
                    job = await self.transcoding_service.submit_job(
                        raw_video_url, 
                        quality,
                        output_path=f"transcoded/{pipeline_id}/{quality}/"
                    )
                    transcoding_jobs.append(job)
            
            # Step 4: Wait for transcoding completion
            transcoded_urls = {}
            for job in transcoding_jobs:
                result = await job.wait_for_completion()
                transcoded_urls[result.quality] = result.output_url
            
            # Step 5: Upload to CDN
            cdn_urls = {}
            for quality, url in transcoded_urls.items():
                cdn_url = await self.cdn_service.upload(url, quality)
                cdn_urls[quality] = cdn_url
            
            # Step 6: Store metadata
            video_record = await self.metadata_service.create_video({
                'title': metadata.title,
                'description': metadata.description,
                'duration': video_info.duration,
                'thumbnail_url': thumbnail_url,
                'video_urls': cdn_urls,
                'upload_date': datetime.utcnow()
            })
            
            return video_record
            
        except Exception as e:
            # Cleanup on failure
            await self.cleanup_failed_upload(pipeline_id)
            raise VideoProcessingException(f"Video processing failed: {e}")
```

#### Adaptive Bitrate Streaming
```python
class AdaptiveBitrateStreaming:
    def __init__(self):
        self.quality_levels = [
            {'quality': '240p', 'bitrate': 400, 'resolution': '426x240'},
            {'quality': '480p', 'bitrate': 1000, 'resolution': '854x480'},
            {'quality': '720p', 'bitrate': 2500, 'resolution': '1280x720'},
            {'quality': '1080p', 'bitrate': 5000, 'resolution': '1920x1080'}
        ]
    
    def generate_playlist(self, video_id, base_url):
        # Generate HLS (HTTP Live Streaming) playlist
        playlist = "#EXTM3U\n#EXT-X-VERSION:3\n"
        
        for level in self.quality_levels:
            playlist += f"#EXT-X-STREAM-INF:BANDWIDTH={level['bitrate']}000,"
            playlist += f"RESOLUTION={level['resolution']}\n"
            playlist += f"{base_url}/{video_id}/{level['quality']}/playlist.m3u8\n"
        
        return playlist
    
    def get_recommended_quality(self, user_bandwidth, device_type):
        # Recommend quality based on bandwidth and device
        if device_type == 'mobile' and user_bandwidth < 1000:
            return '240p'
        elif user_bandwidth < 2000:
            return '480p'
        elif user_bandwidth < 4000:
            return '720p'
        else:
            return '1080p'
```

## Best Practices and Anti-Patterns

### Best Practices

#### 1. API Design Best Practices

**RESTful API Design**
```python
# Good REST API design
class UserAPI:
    # GET /api/v1/users - List users
    def list_users(self, page=1, limit=20, filters=None):
        pass
    
    # GET /api/v1/users/{user_id} - Get specific user
    def get_user(self, user_id):
        pass
    
    # POST /api/v1/users - Create user
    def create_user(self, user_data):
        pass
    
    # PUT /api/v1/users/{user_id} - Update user
    def update_user(self, user_id, user_data):
        pass
    
    # DELETE /api/v1/users/{user_id} - Delete user
    def delete_user(self, user_id):
        pass
    
    # GET /api/v1/users/{user_id}/orders - Get user's orders
    def get_user_orders(self, user_id):
        pass
```

**API Versioning**
```python
# URL versioning
@app.route('/api/v1/users')
def users_v1():
    return jsonify({'version': 'v1', 'users': []})

@app.route('/api/v2/users')
def users_v2():
    return jsonify({'version': 'v2', 'users': [], 'metadata': {}})

# Header versioning
@app.route('/api/users')
def users():
    version = request.headers.get('API-Version', 'v1')
    if version == 'v2':
        return users_v2_logic()
    else:
        return users_v1_logic()
```

#### 2. Error Handling Best Practices

**Consistent Error Response Format**
```python
class APIResponse:
    @staticmethod
    def success(data, message="Success"):
        return {
            'success': True,
            'message': message,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def error(error_code, message, details=None):
        return {
            'success': False,
            'error': {
                'code': error_code,
                'message': message,
                'details': details
            },
            'timestamp': datetime.utcnow().isoformat()
        }

# Usage
@app.route('/api/users/<int:user_id>')
def get_user(user_id):
    try:
        user = user_service.get_user(user_id)
        if user:
            return APIResponse.success(user)
        else:
            return APIResponse.error('USER_NOT_FOUND', 'User not found'), 404
    except ValidationError as e:
        return APIResponse.error('VALIDATION_ERROR', str(e), e.details), 400
    except Exception as e:
        logger.exception(f"Unexpected error getting user {user_id}")
        return APIResponse.error('INTERNAL_ERROR', 'Internal server error'), 500
```

#### 3. Security Best Practices

**Authentication and Authorization**
```python
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return APIResponse.error('MISSING_TOKEN', 'Authorization token required'), 401
        
        try:
            token = token.replace('Bearer ', '')
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.get_by_id(payload['user_id'])
            request.current_user = current_user
        except jwt.ExpiredSignatureError:
            return APIResponse.error('TOKEN_EXPIRED', 'Token has expired'), 401
        except jwt.InvalidTokenError:
            return APIResponse.error('INVALID_TOKEN', 'Invalid token'), 401
        
        return f(*args, **kwargs)
    return decorated_function

def require_role(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return APIResponse.error('UNAUTHORIZED', 'Authentication required'), 401
            
            if request.current_user.role != role:
                return APIResponse.error('FORBIDDEN', 'Insufficient permissions'), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Usage
@app.route('/api/admin/users')
@require_auth
@require_role('admin')
def admin_list_users():
    return APIResponse.success(user_service.get_all_users())
```

**Input Validation and Sanitization**
```python
from marshmallow import Schema, fields, validate

class UserCreateSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    email = fields.Email(required=True)
    age = fields.Int(validate=validate.Range(min=0, max=150))
    password = fields.Str(required=True, validate=validate.Length(min=8))

@app.route('/api/users', methods=['POST'])
def create_user():
    schema = UserCreateSchema()
    try:
        data = schema.load(request.json)
    except ValidationError as e:
        return APIResponse.error('VALIDATION_ERROR', 'Invalid input', e.messages), 400
    
    # Hash password before storing
    data['password'] = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    
    user = user_service.create_user(data)
    return APIResponse.success(user), 201
```

### Anti-Patterns to Avoid

#### 1. God Object Anti-Pattern
```python
# BAD: Single class doing everything
class Application:
    def __init__(self):
        self.users = []
        self.orders = []
        self.products = []
    
    def authenticate_user(self, credentials):
        pass
    
    def process_order(self, order_data):
        pass
    
    def manage_inventory(self, product_id, quantity):
        pass
    
    def send_notifications(self, message):
        pass
    
    def generate_reports(self, report_type):
        pass

# GOOD: Separated responsibilities
class AuthenticationService:
    def authenticate_user(self, credentials):
        pass

class OrderService:
    def process_order(self, order_data):
        pass

class InventoryService:
    def manage_inventory(self, product_id, quantity):
        pass
```

#### 2. Chatty Interface Anti-Pattern
```python
# BAD: Multiple API calls for related data
def get_user_profile(user_id):
    user = api.get_user(user_id)  # API call 1
    orders = api.get_user_orders(user_id)  # API call 2
    preferences = api.get_user_preferences(user_id)  # API call 3
    recommendations = api.get_user_recommendations(user_id)  # API call 4
    
    return {
        'user': user,
        'orders': orders,
        'preferences': preferences,
        'recommendations': recommendations
    }

# GOOD: Single API call for related data
def get_user_profile(user_id):
    profile = api.get_user_profile(user_id, include=['orders', 'preferences', 'recommendations'])
    return profile
```

#### 3. Shared Database Anti-Pattern
```python
# BAD: Multiple services sharing same database
class UserService:
    def get_user(self, user_id):
        return shared_db.execute("SELECT * FROM users WHERE id = ?", user_id)

class OrderService:
    def get_user_orders(self, user_id):
        # Direct access to users table from different service
        user = shared_db.execute("SELECT * FROM users WHERE id = ?", user_id)
        orders = shared_db.execute("SELECT * FROM orders WHERE user_id = ?", user_id)
        return orders

# GOOD: Service-specific databases with API communication
class UserService:
    def __init__(self):
        self.user_db = UserDatabase()
    
    def get_user(self, user_id):
        return self.user_db.get_user(user_id)

class OrderService:
    def __init__(self):
        self.order_db = OrderDatabase()
        self.user_service = UserServiceClient()
    
    def get_user_orders(self, user_id):
        user = self.user_service.get_user(user_id)  # API call
        orders = self.order_db.get_orders_by_user(user_id)
        return orders
```

## Modern Architecture Trends

### 1. Serverless and Function-as-a-Service

#### Serverless Benefits
- **Cost efficiency**: Pay per execution
- **Auto-scaling**: Handles traffic spikes automatically
- **Reduced ops overhead**: No server management
- **Event-driven**: Natural fit for reactive architectures

#### Serverless Patterns
```python
# AWS Lambda with API Gateway
import json
import boto3

def lambda_handler(event, context):
    # Parse input
    body = json.loads(event['body'])
    user_id = body.get('user_id')
    
    # Business logic
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('Users')
    
    response = table.get_item(Key={'user_id': user_id})
    
    # Return response
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'user': response.get('Item', {})
        })
    }

# Serverless workflow with Step Functions
{
  "Comment": "Order processing workflow",
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:validate-order",
      "Next": "ProcessPayment"
    },
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:process-payment",
      "Next": "UpdateInventory"
    },
    "UpdateInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:update-inventory",
      "End": true
    }
  }
}
```

### 2. Container Orchestration

#### Kubernetes Deployment Example
```yaml
# Deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  labels:
    app: user-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
    spec:
      containers:
      - name: user-service
        image: user-service:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# Service configuration
apiVersion: v1
kind: Service
metadata:
  name: user-service
spec:
  selector:
    app: user-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: user-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: user-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. DevOps and CI/CD

#### CI/CD Pipeline Example (GitHub Actions)
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=app tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif

  build-and-deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: |
        docker build -t myapp:${{ github.sha }} .
        docker tag myapp:${{ github.sha }} myapp:latest
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push myapp:${{ github.sha }}
        docker push myapp:latest
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
        kubectl rollout status deployment/myapp
```

### 4. Observability and Monitoring

#### Three Pillars of Observability

**Metrics**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

def monitor_endpoint(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
        
        try:
            result = f(*args, **kwargs)
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper

@app.route('/api/users')
@monitor_endpoint
def get_users():
    return jsonify(user_service.get_all_users())
```

**Logging**
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, service_name):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        
        # Configure structured logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log(self, level, message, **kwargs):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name,
            'level': level,
            'message': message,
            **kwargs
        }
        
        if hasattr(request, 'trace_id'):
            log_data['trace_id'] = request.trace_id
        
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_data))

# Usage
logger = StructuredLogger('user-service')

@app.route('/api/users/<int:user_id>')
def get_user(user_id):
    logger.log('info', 'Getting user', user_id=user_id)
    
    try:
        user = user_service.get_user(user_id)
        logger.log('info', 'User retrieved successfully', user_id=user_id)
        return jsonify(user)
    except UserNotFoundError:
        logger.log('warning', 'User not found', user_id=user_id)
        return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.log('error', 'Failed to get user', user_id=user_id, error=str(e))
        return jsonify({'error': 'Internal server error'}), 500
```

**Distributed Tracing**
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

def trace_function(operation_name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(operation_name) as span:
                span.set_attribute("function.name", f.__name__)
                try:
                    result = f(*args, **kwargs)
                    span.set_attribute("function.result", "success")
                    return result
                except Exception as e:
                    span.set_attribute("function.result", "error")
                    span.set_attribute("error.message", str(e))
                    raise
        return wrapper
    return decorator

@trace_function("get_user_profile")
def get_user_profile(user_id):
    with tracer.start_as_current_span("fetch_user_data") as span:
        user = user_service.get_user(user_id)
        span.set_attribute("user.id", user_id)
    
    with tracer.start_as_current_span("fetch_user_orders"):
        orders = order_service.get_user_orders(user_id)
    
    return {"user": user, "orders": orders}
```

### 5. Event-Driven Architecture Evolution

#### Event Streaming with Apache Kafka
```python
from kafka import KafkaProducer, KafkaConsumer
import json

class EventStore:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def publish_event(self, topic, event_data):
        future = self.producer.send(topic, event_data)
        return future.get(timeout=10)  # Synchronous send

class EventProcessor:
    def __init__(self, bootstrap_servers, group_id):
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
    
    def subscribe_to_events(self, topics, handler):
        self.consumer.subscribe(topics)
        
        for message in self.consumer:
            try:
                handler(message.topic, message.value)
            except Exception as e:
                logger.error(f"Error processing event: {e}")

# Usage
event_store = EventStore(['kafka1:9092', 'kafka2:9092'])

# Publish events
user_created_event = {
    'event_type': 'UserCreated',
    'user_id': 12345,
    'timestamp': datetime.utcnow().isoformat(),
    'data': {
        'name': 'John Doe',
        'email': 'john@example.com'
    }
}

event_store.publish_event('user-events', user_created_event)

# Consume events
def handle_user_event(topic, event_data):
    if event_data['event_type'] == 'UserCreated':
        # Update search index
        search_service.index_user(event_data['data'])
        
        # Send welcome email
        email_service.send_welcome_email(event_data['data']['email'])

processor = EventProcessor(['kafka1:9092'], 'user-event-processor')
processor.subscribe_to_events(['user-events'], handle_user_event)
```

Remember: Software architecture is about making trade-offs. There's no one-size-fits-all solution. Always consider your specific requirements, team capabilities, timeline, and constraints when making architectural decisions. Stay updated with industry trends, but don't adopt new technologies just because they're popular—adopt them because they solve real problems in your context.