# Distributed Systems Architecture

## Theoretical Concepts

### Definition and Characteristics
Distributed systems are collections of independent computers that appear as a single coherent system to users. Key characteristics include:

- **Transparency**: Hide the complexity of distribution from users
- **Scalability**: Ability to handle increased load by adding resources
- **Fault Tolerance**: Continue operating despite component failures
- **Concurrency**: Handle multiple operations simultaneously
- **Consistency**: Maintain data integrity across nodes

### Fundamental Challenges

#### CAP Theorem
In any distributed system, you can only guarantee two of three properties:
- **Consistency**: All nodes see the same data simultaneously
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures

```python
# Example: Choosing consistency over availability
class ConsistentDataStore:
    def __init__(self):
        self.nodes = []
        self.quorum_size = len(self.nodes) // 2 + 1
    
    def write(self, key, value):
        successful_writes = 0
        for node in self.nodes:
            try:
                if node.write(key, value):
                    successful_writes += 1
            except NetworkException:
                continue
        
        if successful_writes >= self.quorum_size:
            return True
        else:
            # Rollback all writes to maintain consistency
            self.rollback_writes(key)
            raise ConsistencyException("Unable to achieve quorum")
```

#### ACID vs BASE
**ACID Properties** (Traditional databases):
- Atomicity, Consistency, Isolation, Durability

**BASE Properties** (NoSQL/Distributed systems):
- Basically Available, Soft state, Eventual consistency

```java
// ACID Example - Traditional Transaction
@Transactional
public class BankingService {
    public void transferMoney(Account from, Account to, BigDecimal amount) {
        from.withdraw(amount);  // Atomic operation
        to.deposit(amount);     // Either both succeed or both fail
    }
}

// BASE Example - Eventual Consistency
public class DistributedBankingService {
    public void transferMoney(String fromId, String toId, BigDecimal amount) {
        // Create saga for distributed transaction
        TransferSaga saga = new TransferSaga(fromId, toId, amount);
        sagaManager.start(saga);
        
        // Operations may complete at different times
        // System remains available but temporarily inconsistent
    }
}
```

## Communication Patterns

### Synchronous Communication

#### HTTP/REST APIs
```typescript
// Client-side service communication
class UserService {
    async getUser(id: string): Promise<User> {
        const response = await fetch(`/api/users/${id}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            timeout: 5000  // Handle network delays
        });
        
        if (!response.ok) {
            throw new ServiceException(`Failed to fetch user: ${response.status}`);
        }
        
        return response.json();
    }
}
```

#### gRPC Communication
```protobuf
// user.proto
syntax = "proto3";

service UserService {
    rpc GetUser(GetUserRequest) returns (GetUserResponse);
    rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
}

message GetUserRequest {
    string user_id = 1;
}

message GetUserResponse {
    User user = 1;
    bool found = 2;
}
```

```python
# Python gRPC client
import grpc
import user_pb2_grpc
import user_pb2

class UserClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = user_pb2_grpc.UserServiceStub(self.channel)
    
    def get_user(self, user_id):
        request = user_pb2.GetUserRequest(user_id=user_id)
        try:
            response = self.stub.GetUser(request, timeout=5.0)
            return response.user if response.found else None
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()}: {e.details()}")
            return None
```

### Asynchronous Communication

#### Message Queues
```python
# Producer using RabbitMQ
import pika
import json

class OrderEventProducer:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='order_events', durable=True)
    
    def publish_order_created(self, order):
        message = {
            'event_type': 'ORDER_CREATED',
            'order_id': order.id,
            'customer_id': order.customer_id,
            'timestamp': order.created_at.isoformat()
        }
        
        self.channel.basic_publish(
            exchange='',
            routing_key='order_events',
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
            )
        )

# Consumer
class OrderEventConsumer:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='order_events', durable=True)
        self.channel.basic_qos(prefetch_count=1)
    
    def process_order_event(self, ch, method, properties, body):
        try:
            event = json.loads(body)
            print(f"Processing event: {event['event_type']}")
            
            # Process the event
            if event['event_type'] == 'ORDER_CREATED':
                self.handle_order_created(event)
            
            # Acknowledge successful processing
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Error processing event: {e}")
            # Reject and requeue the message
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start_consuming(self):
        self.channel.basic_consume(
            queue='order_events',
            on_message_callback=self.process_order_event
        )
        self.channel.start_consuming()
```

#### Event Streaming with Apache Kafka
```java
// Kafka Producer
public class OrderEventProducer {
    private final KafkaProducer<String, String> producer;
    
    public OrderEventProducer() {
        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("acks", "all");  // Wait for all replicas
        props.put("retries", 3);
        
        this.producer = new KafkaProducer<>(props);
    }
    
    public void publishOrderCreated(Order order) {
        String event = JsonUtils.toJson(order);
        ProducerRecord<String, String> record = new ProducerRecord<>(
            "order-events", 
            order.getId(), 
            event
        );
        
        producer.send(record, (metadata, exception) -> {
            if (exception != null) {
                logger.error("Failed to send event", exception);
            } else {
                logger.info("Event sent to partition {} at offset {}", 
                    metadata.partition(), metadata.offset());
            }
        });
    }
}
```

## Distributed Data Management

### Data Partitioning Strategies

#### Horizontal Partitioning (Sharding)
```python
class ShardedDatabase:
    def __init__(self, shard_configs):
        self.shards = {}
        for config in shard_configs:
            self.shards[config['id']] = DatabaseConnection(config)
    
    def get_shard(self, key):
        # Hash-based sharding
        shard_id = hash(key) % len(self.shards)
        return self.shards[shard_id]
    
    def save_user(self, user):
        shard = self.get_shard(user.id)
        return shard.execute(
            "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
            (user.id, user.name, user.email)
        )
    
    def get_user(self, user_id):
        shard = self.get_shard(user_id)
        return shard.query_one(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )
```

#### Vertical Partitioning
```sql
-- Customer service database
CREATE TABLE customers (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMP
);

-- Order service database  
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    customer_id UUID,  -- Reference to customer service
    total_amount DECIMAL(10,2),
    status VARCHAR(50),
    created_at TIMESTAMP
);

-- Inventory service database
CREATE TABLE products (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL(10,2),
    stock_quantity INTEGER
);
```

### Distributed Transactions

#### Two-Phase Commit (2PC)
```java
public class TwoPhaseCommitCoordinator {
    private List<TransactionParticipant> participants;
    
    public boolean executeTransaction(DistributedTransaction transaction) {
        String transactionId = UUID.randomUUID().toString();
        
        // Phase 1: Prepare
        boolean allPrepared = true;
        for (TransactionParticipant participant : participants) {
            try {
                if (!participant.prepare(transactionId, transaction)) {
                    allPrepared = false;
                    break;
                }
            } catch (Exception e) {
                allPrepared = false;
                break;
            }
        }
        
        // Phase 2: Commit or Abort
        if (allPrepared) {
            for (TransactionParticipant participant : participants) {
                participant.commit(transactionId);
            }
            return true;
        } else {
            for (TransactionParticipant participant : participants) {
                participant.abort(transactionId);
            }
            return false;
        }
    }
}
```

#### Saga Pattern
```python
class OrderSaga:
    def __init__(self, order_service, payment_service, inventory_service):
        self.order_service = order_service
        self.payment_service = payment_service
        self.inventory_service = inventory_service
        self.compensation_actions = []
    
    def execute(self, order_request):
        try:
            # Step 1: Reserve inventory
            reservation = self.inventory_service.reserve_items(order_request.items)
            self.compensation_actions.append(
                lambda: self.inventory_service.cancel_reservation(reservation.id)
            )
            
            # Step 2: Process payment
            payment = self.payment_service.charge_customer(
                order_request.customer_id, 
                order_request.total_amount
            )
            self.compensation_actions.append(
                lambda: self.payment_service.refund(payment.id)
            )
            
            # Step 3: Create order
            order = self.order_service.create_order(order_request)
            self.compensation_actions.append(
                lambda: self.order_service.cancel_order(order.id)
            )
            
            return order
            
        except Exception as e:
            # Execute compensation actions in reverse order
            for action in reversed(self.compensation_actions):
                try:
                    action()
                except Exception as comp_error:
                    logger.error(f"Compensation failed: {comp_error}")
            raise e
```

## Consensus Algorithms

### Raft Consensus
```go
// Simplified Raft implementation in Go
type RaftNode struct {
    id           int
    currentTerm  int
    votedFor     *int
    log          []LogEntry
    commitIndex  int
    lastApplied  int
    state        NodeState  // Follower, Candidate, Leader
    peers        []RaftNode
}

func (n *RaftNode) RequestVote(term int, candidateId int, lastLogIndex int, lastLogTerm int) VoteResponse {
    if term < n.currentTerm {
        return VoteResponse{Term: n.currentTerm, VoteGranted: false}
    }
    
    if term > n.currentTerm {
        n.currentTerm = term
        n.votedFor = nil
        n.state = Follower
    }
    
    if (n.votedFor == nil || *n.votedFor == candidateId) && 
       n.isLogUpToDate(lastLogIndex, lastLogTerm) {
        n.votedFor = &candidateId
        return VoteResponse{Term: n.currentTerm, VoteGranted: true}
    }
    
    return VoteResponse{Term: n.currentTerm, VoteGranted: false}
}

func (n *RaftNode) AppendEntries(term int, leaderId int, prevLogIndex int, 
                                prevLogTerm int, entries []LogEntry, leaderCommit int) AppendResponse {
    if term < n.currentTerm {
        return AppendResponse{Term: n.currentTerm, Success: false}
    }
    
    if term > n.currentTerm {
        n.currentTerm = term
        n.votedFor = nil
    }
    
    n.state = Follower
    
    // Check log consistency
    if prevLogIndex > 0 && (len(n.log) < prevLogIndex || 
                           n.log[prevLogIndex-1].Term != prevLogTerm) {
        return AppendResponse{Term: n.currentTerm, Success: false}
    }
    
    // Append new entries
    n.log = append(n.log[:prevLogIndex], entries...)
    
    // Update commit index
    if leaderCommit > n.commitIndex {
        n.commitIndex = min(leaderCommit, len(n.log))
    }
    
    return AppendResponse{Term: n.currentTerm, Success: true}
}
```

## Distributed Caching

### Redis Cluster Configuration
```yaml
# redis-cluster.yml
version: '3.8'
services:
  redis-node-1:
    image: redis:7-alpine
    command: redis-server --port 7001 --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    ports:
      - "7001:7001"
      - "17001:17001"
    volumes:
      - redis-1-data:/data

  redis-node-2:
    image: redis:7-alpine
    command: redis-server --port 7002 --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    ports:
      - "7002:7002"
      - "17002:17002"
    volumes:
      - redis-2-data:/data

  redis-node-3:
    image: redis:7-alpine
    command: redis-server --port 7003 --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
    ports:
      - "7003:7003"
      - "17003:17003"
    volumes:
      - redis-3-data:/data

volumes:
  redis-1-data:
  redis-2-data:
  redis-3-data:
```

```python
# Redis cluster client
import redis
from rediscluster import RedisCluster

class DistributedCache:
    def __init__(self):
        startup_nodes = [
            {"host": "127.0.0.1", "port": "7001"},
            {"host": "127.0.0.1", "port": "7002"},
            {"host": "127.0.0.1", "port": "7003"}
        ]
        
        self.client = RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=True,
            skip_full_coverage_check=True,
            health_check_interval=30
        )
    
    def set_with_ttl(self, key, value, ttl_seconds=3600):
        """Set value with automatic expiration"""
        return self.client.setex(key, ttl_seconds, value)
    
    def get_or_compute(self, key, compute_func, ttl_seconds=3600):
        """Cache-aside pattern implementation"""
        value = self.client.get(key)
        if value is not None:
            return json.loads(value)
        
        # Compute value if not in cache
        computed_value = compute_func()
        self.client.setex(key, ttl_seconds, json.dumps(computed_value))
        return computed_value
    
    def invalidate_pattern(self, pattern):
        """Invalidate all keys matching pattern"""
        for key in self.client.scan_iter(match=pattern):
            self.client.delete(key)
```

## Load Balancing and Service Discovery

### Load Balancer Configuration
```nginx
# nginx.conf - Round-robin load balancing
upstream backend_servers {
    server 192.168.1.10:8080 weight=3;
    server 192.168.1.11:8080 weight=2;
    server 192.168.1.12:8080 weight=1;
    
    # Health checks
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://backend_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Connection pooling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
```

### Service Discovery with Consul
```python
import consul

class ServiceRegistry:
    def __init__(self, consul_host='localhost', consul_port=8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
    
    def register_service(self, service_name, service_id, address, port, health_check_url=None):
        """Register a service with Consul"""
        service_def = {
            'name': service_name,
            'service_id': service_id,
            'address': address,
            'port': port,
            'tags': [f'version-1.0', f'environment-prod']
        }
        
        if health_check_url:
            service_def['check'] = {
                'http': health_check_url,
                'interval': '10s',
                'timeout': '3s'
            }
        
        return self.consul.agent.service.register(**service_def)
    
    def discover_service(self, service_name):
        """Discover healthy instances of a service"""
        services = self.consul.health.service(service_name, passing=True)[1]
        
        instances = []
        for service in services:
            instances.append({
                'address': service['Service']['Address'],
                'port': service['Service']['Port'],
                'id': service['Service']['ID']
            })
        
        return instances
    
    def deregister_service(self, service_id):
        """Deregister a service"""
        return self.consul.agent.service.deregister(service_id)

# Usage example
class MicroserviceClient:
    def __init__(self, service_registry):
        self.registry = service_registry
        self.service_cache = {}
    
    def call_service(self, service_name, endpoint, data=None):
        # Get service instances
        instances = self.registry.discover_service(service_name)
        if not instances:
            raise ServiceUnavailableException(f"No instances of {service_name} available")
        
        # Simple round-robin selection
        instance = instances[hash(service_name) % len(instances)]
        
        # Make the call
        url = f"http://{instance['address']}:{instance['port']}{endpoint}"
        response = requests.post(url, json=data, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise ServiceException(f"Service call failed: {response.status_code}")
```

## Monitoring and Observability

### Distributed Tracing
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

class OrderService:
    def process_order(self, order_request):
        with tracer.start_as_current_span("process_order") as span:
            span.set_attribute("order.customer_id", order_request.customer_id)
            span.set_attribute("order.total_amount", str(order_request.total_amount))
            
            try:
                # Validate order
                with tracer.start_as_current_span("validate_order"):
                    self.validate_order(order_request)
                
                # Check inventory
                with tracer.start_as_current_span("check_inventory") as inventory_span:
                    inventory_span.set_attribute("service.name", "inventory-service")
                    available = self.check_inventory(order_request.items)
                    inventory_span.set_attribute("inventory.available", available)
                
                # Process payment
                with tracer.start_as_current_span("process_payment") as payment_span:
                    payment_span.set_attribute("service.name", "payment-service")
                    payment_result = self.process_payment(order_request)
                    payment_span.set_attribute("payment.transaction_id", payment_result.transaction_id)
                
                span.set_attribute("order.status", "completed")
                return self.create_order(order_request)
                
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
QUEUE_SIZE = Gauge('message_queue_size', 'Size of message queue', ['queue_name'])

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        start_time = time.time()
        method = environ['REQUEST_METHOD']
        path = environ['PATH_INFO']
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        def response_wrapper(status, headers):
            # Record metrics
            duration = time.time() - start_time
            status_code = status.split(' ')[0]
            
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
            ACTIVE_CONNECTIONS.dec()
            
            return start_response(status, headers)
        
        return self.app(environ, response_wrapper)

# Start metrics server
start_http_server(8000)
```

## Security Considerations

### Authentication and Authorization
```python
import jwt
from functools import wraps

class JWTAuth:
    def __init__(self, secret_key, algorithm='HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def generate_token(self, user_id, permissions, expires_in=3600):
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': time.time() + expires_in,
            'iat': time.time()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthException("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthException("Invalid token")

def require_auth(required_permissions=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                raise AuthException("Missing or invalid authorization header")
            
            token = auth_header.split(' ')[1]
            payload = jwt_auth.verify_token(token)
            
            if required_permissions:
                user_permissions = set(payload.get('permissions', []))
                if not set(required_permissions).issubset(user_permissions):
                    raise AuthException("Insufficient permissions")
            
            request.user_id = payload['user_id']
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@require_auth(['orders:read', 'orders:write'])
def create_order():
    # Only users with order permissions can access this
    pass
```

### Network Security
```yaml
# Service mesh configuration with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - match:
    - headers:
        authorization:
          regex: "Bearer .*"
    route:
    - destination:
        host: order-service
        port:
          number: 8080
  - fault:
      abort:
        percentage:
          value: 100
        httpStatus: 401

---
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: order-service
spec:
  selector:
    matchLabels:
      app: order-service
  mtls:
    mode: STRICT

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: order-service-authz
spec:
  selector:
    matchLabels:
      app: order-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/api-gateway"]
  - to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/orders/*"]
```

## Best Practices

### Circuit Breaker Pattern
```java
public class CircuitBreaker {
    private enum State { CLOSED, OPEN, HALF_OPEN }
    
    private State state = State.CLOSED;
    private int failureCount = 0;
    private int successCount = 0;
    private long lastFailureTime = 0;
    
    private final int failureThreshold;
    private final int successThreshold;
    private final long timeout;
    
    public CircuitBreaker(int failureThreshold, int successThreshold, long timeout) {
        this.failureThreshold = failureThreshold;
        this.successThreshold = successThreshold;
        this.timeout = timeout;
    }
    
    public <T> T execute(Supplier<T> operation) throws Exception {
        if (state == State.OPEN) {
            if (System.currentTimeMillis() - lastFailureTime < timeout) {
                throw new CircuitBreakerOpenException("Circuit breaker is OPEN");
            } else {
                state = State.HALF_OPEN;
                successCount = 0;
            }
        }
        
        try {
            T result = operation.get();
            onSuccess();
            return result;
        } catch (Exception e) {
            onFailure();
            throw e;
        }
    }
    
    private void onSuccess() {
        if (state == State.HALF_OPEN) {
            successCount++;
            if (successCount >= successThreshold) {
                state = State.CLOSED;
                failureCount = 0;
            }
        } else {
            failureCount = 0;
        }
    }
    
    private void onFailure() {
        failureCount++;
        lastFailureTime = System.currentTimeMillis();
        
        if (failureCount >= failureThreshold) {
            state = State.OPEN;
        }
    }
}
```

### Retry with Exponential Backoff
```python
import random
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0, jitter=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, TemporaryException) as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                    
                    # Add jitter to avoid thundering herd
                    if jitter:
                        delay *= (0.5 + 0.5 * random.random())
                    
                    print(f"Retry {retries}/{max_retries} after {delay:.2f}s delay")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=5, base_delay=0.5, max_delay=30.0)
def call_external_service(data):
    response = requests.post("https://api.example.com/data", json=data, timeout=10)
    if response.status_code >= 500:
        raise TemporaryException(f"Server error: {response.status_code}")
    return response.json()
```

### Configuration Management
```yaml
# Kubernetes ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgresql://user:pass@db:5432/myapp"
  redis_cluster_nodes: "redis-1:7001,redis-2:7002,redis-3:7003"
  feature_flags: |
    {
      "new_payment_flow": true,
      "enhanced_search": false,
      "rate_limiting": true
    }
  rate_limits: |
    {
      "api_calls_per_minute": 1000,
      "concurrent_connections": 100
    }

---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  jwt_secret: <base64-encoded-secret>
  api_key: <base64-encoded-key>
```

```python
# Configuration loader
import os
import json
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        self.config = {}
        self.load_config()
    
    def load_config(self):
        # Load from environment variables
        self.config['database_url'] = os.getenv('DATABASE_URL')
        self.config['redis_nodes'] = os.getenv('REDIS_CLUSTER_NODES', '').split(',')
        
        # Load feature flags
        feature_flags_str = os.getenv('FEATURE_FLAGS', '{}')
        self.config['feature_flags'] = json.loads(feature_flags_str)
        
        # Load rate limits
        rate_limits_str = os.getenv('RATE_LIMITS', '{}')
        self.config['rate_limits'] = json.loads(rate_limits_str)
        
        # Load secrets
        self.config['jwt_secret'] = os.getenv('JWT_SECRET')
        self.config['api_key'] = os.getenv('API_KEY')
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        return self.config.get('feature_flags', {}).get(feature_name, False)
    
    def get_rate_limit(self, limit_type: str) -> int:
        return self.config.get('rate_limits', {}).get(limit_type, 0)

# Global config instance
config = ConfigManager()
```

## Use Cases and Examples

### E-commerce Platform Architecture
```python
# Microservices architecture for e-commerce
class EcommerceOrchestrator:
    def __init__(self):
        self.user_service = UserServiceClient()
        self.product_service = ProductServiceClient()
        self.inventory_service = InventoryServiceClient()
        self.payment_service = PaymentServiceClient()
        self.order_service = OrderServiceClient()
        self.notification_service = NotificationServiceClient()
    
    async def process_order(self, order_request):
        # Distributed transaction using saga pattern
        saga_id = generate_uuid()
        
        try:
            # Step 1: Validate user
            user = await self.user_service.get_user(order_request.user_id)
            if not user or not user.is_active:
                raise ValidationException("Invalid user")
            
            # Step 2: Check product availability and pricing
            products = await self.product_service.get_products(order_request.product_ids)
            total_amount = sum(p.price * order_request.quantities[p.id] for p in products)
            
            # Step 3: Reserve inventory
            reservation = await self.inventory_service.reserve_items(
                saga_id, order_request.items
            )
            
            # Step 4: Process payment
            payment = await self.payment_service.charge_customer(
                saga_id, user.id, total_amount, order_request.payment_method
            )
            
            # Step 5: Create order
            order = await self.order_service.create_order(
                saga_id, order_request, payment.id, reservation.id
            )
            
            # Step 6: Send confirmation
            await self.notification_service.send_order_confirmation(
                user.email, order.id
            )
            
            return order
            
        except Exception as e:
            # Compensate in reverse order
            await self.compensate_saga(saga_id, e)
            raise
```

### Real-time Analytics Pipeline
```python
# Streaming analytics using Apache Kafka and Apache Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class RealTimeAnalytics:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("RealTimeAnalytics") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
            .getOrCreate()
    
    def process_user_events(self):
        # Define schema for incoming events
        event_schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("event_type", StringType(), True),
            StructField("timestamp", LongType(), True),
            StructField("properties", MapType(StringType(), StringType()), True)
        ])
        
        # Read from Kafka
        events_df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "user-events") \
            .load() \
            .select(from_json(col("value").cast("string"), event_schema).alias("event")) \
            .select("event.*")
        
        # Real-time aggregations
        user_activity = events_df \
            .withWatermark("timestamp", "10 minutes") \
            .groupBy(
                window(col("timestamp"), "5 minutes"),
                col("user_id"),
                col("event_type")
            ) \
            .count() \
            .withColumnRenamed("count", "event_count")
        
        # Write to multiple sinks
        query = user_activity.writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", "false") \
            .start()
        
        return query
```

## Performance Optimization

### Database Optimization
```sql
-- Partitioning strategy for time-series data
CREATE TABLE user_events (
    id BIGSERIAL,
    user_id UUID,
    event_type VARCHAR(50),
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE user_events_2024_01 PARTITION OF user_events
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE user_events_2024_02 PARTITION OF user_events
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Indexes for performance
CREATE INDEX CONCURRENTLY idx_user_events_user_id_created_at 
ON user_events (user_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_user_events_event_type 
ON user_events USING HASH (event_type);

CREATE INDEX CONCURRENTLY idx_user_events_properties 
ON user_events USING GIN (properties);
```

### Caching Strategies
```python
# Multi-level caching
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = redis.Redis(host='localhost', port=6379, db=0)  # Redis
        self.l3_cache = redis.Redis(host='redis-cluster', port=6379, db=1)  # Redis cluster
    
    def get(self, key):
        # Level 1: In-memory
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Level 2: Local Redis
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = json.loads(value)
            return self.l1_cache[key]
        
        # Level 3: Distributed Redis
        value = self.l3_cache.get(key)
        if value:
            parsed_value = json.loads(value)
            self.l1_cache[key] = parsed_value
            self.l2_cache.setex(key, 3600, value)  # Cache for 1 hour
            return parsed_value
        
        return None
    
    def set(self, key, value, ttl=3600):
        # Set in all levels
        self.l1_cache[key] = value
        serialized = json.dumps(value)
        self.l2_cache.setex(key, ttl, serialized)
        self.l3_cache.setex(key, ttl, serialized)
```

*Last updated: 2025-01-10 11:20:00 UTC*