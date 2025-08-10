# Microservices Architecture

## Theoretical Concepts

### Definition and Principles
Microservices architecture is a distributed system approach where applications are built as a collection of loosely coupled, independently deployable services. Each service is:

- **Single Responsibility**: Focuses on one business capability
- **Autonomous**: Can be developed, deployed, and scaled independently
- **Decentralized**: Owns its data and business logic
- **Resilient**: Designed to handle failures gracefully
- **Observable**: Provides comprehensive monitoring and logging

### Core Characteristics

#### Service Independence
```python
# Each microservice has its own database and business logic
class UserService:
    def __init__(self):
        self.db = UserDatabase()  # Dedicated database
        self.cache = UserCache()
        self.event_publisher = EventPublisher('user-events')
    
    def create_user(self, user_data):
        # Validate business rules
        if not self.is_valid_email(user_data.email):
            raise ValidationException("Invalid email format")
        
        # Create user
        user = self.db.create_user(user_data)
        
        # Cache for quick access
        self.cache.set(f"user:{user.id}", user)
        
        # Publish domain event
        self.event_publisher.publish({
            'event_type': 'USER_CREATED',
            'user_id': user.id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return user
    
    def get_user(self, user_id):
        # Try cache first
        cached_user = self.cache.get(f"user:{user_id}")
        if cached_user:
            return cached_user
        
        # Fallback to database
        user = self.db.get_user(user_id)
        if user:
            self.cache.set(f"user:{user.id}", user, ttl=3600)
        
        return user
```

#### Technology Diversity
```yaml
# Different services can use different technologies
services:
  user-service:
    language: Python
    framework: FastAPI
    database: PostgreSQL
    cache: Redis
    
  inventory-service:
    language: Java
    framework: Spring Boot
    database: MongoDB
    cache: Hazelcast
    
  notification-service:
    language: Node.js
    framework: Express
    database: DynamoDB
    queue: RabbitMQ
    
  analytics-service:
    language: Go
    framework: Gin
    database: ClickHouse
    stream_processing: Apache Kafka
```

## Service Design Patterns

### Domain-Driven Design (DDD)

#### Bounded Context
```python
# Order Management Bounded Context
class OrderManagement:
    class Order:
        def __init__(self, customer_id, items):
            self.id = generate_uuid()
            self.customer_id = customer_id
            self.items = items
            self.status = OrderStatus.PENDING
            self.total_amount = self.calculate_total()
            self.created_at = datetime.utcnow()
        
        def calculate_total(self):
            return sum(item.price * item.quantity for item in self.items)
        
        def confirm(self):
            if self.status != OrderStatus.PENDING:
                raise InvalidOrderStateException("Order cannot be confirmed")
            self.status = OrderStatus.CONFIRMED
        
        def cancel(self):
            if self.status in [OrderStatus.SHIPPED, OrderStatus.DELIVERED]:
                raise InvalidOrderStateException("Order cannot be cancelled")
            self.status = OrderStatus.CANCELLED

    class OrderRepository:
        def save(self, order): pass
        def find_by_id(self, order_id): pass
        def find_by_customer(self, customer_id): pass

    class OrderService:
        def __init__(self, repository, event_publisher):
            self.repository = repository
            self.event_publisher = event_publisher
        
        def create_order(self, customer_id, items):
            order = self.Order(customer_id, items)
            self.repository.save(order)
            
            self.event_publisher.publish(OrderCreatedEvent(
                order_id=order.id,
                customer_id=customer_id,
                total_amount=order.total_amount
            ))
            
            return order
```

#### Aggregate Design
```java
// Customer Aggregate
@Entity
public class Customer {
    @Id
    private CustomerId id;
    private String name;
    private Email email;
    private Address address;
    private List<PaymentMethod> paymentMethods;
    
    // Aggregate root - controls access to internal entities
    public void addPaymentMethod(PaymentMethod paymentMethod) {
        validatePaymentMethod(paymentMethod);
        this.paymentMethods.add(paymentMethod);
        
        // Publish domain event
        DomainEventPublisher.instance().publish(
            new PaymentMethodAddedEvent(this.id, paymentMethod.getId())
        );
    }
    
    public void updateAddress(Address newAddress) {
        Address oldAddress = this.address;
        this.address = newAddress;
        
        DomainEventPublisher.instance().publish(
            new CustomerAddressChangedEvent(this.id, oldAddress, newAddress)
        );
    }
    
    private void validatePaymentMethod(PaymentMethod paymentMethod) {
        if (paymentMethods.size() >= MAX_PAYMENT_METHODS) {
            throw new TooManyPaymentMethodsException();
        }
        
        if (paymentMethods.stream().anyMatch(pm -> pm.equals(paymentMethod))) {
            throw new DuplicatePaymentMethodException();
        }
    }
}
```

### Service Decomposition Strategies

#### Decomposition by Business Capability
```typescript
// E-commerce business capabilities mapped to services
interface BusinessCapabilities {
  userManagement: {
    services: ['user-service', 'authentication-service'];
    responsibilities: ['user registration', 'user profile', 'authentication', 'authorization'];
  };
  
  catalogManagement: {
    services: ['product-service', 'category-service'];
    responsibilities: ['product catalog', 'categories', 'search', 'recommendations'];
  };
  
  orderManagement: {
    services: ['order-service', 'cart-service'];
    responsibilities: ['order creation', 'order tracking', 'shopping cart'];
  };
  
  paymentProcessing: {
    services: ['payment-service', 'billing-service'];
    responsibilities: ['payment processing', 'billing', 'refunds'];
  };
  
  inventoryManagement: {
    services: ['inventory-service', 'warehouse-service'];
    responsibilities: ['stock management', 'reservations', 'fulfillment'];
  };
  
  customerSupport: {
    services: ['support-service', 'notification-service'];
    responsibilities: ['customer support', 'notifications', 'communication'];
  };
}
```

#### Decomposition by Data Ownership
```sql
-- Each service owns its data
-- User Service Database
CREATE DATABASE user_service_db;
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    profile JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Order Service Database
CREATE DATABASE order_service_db;
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    customer_id UUID,  -- Reference, not foreign key
    status VARCHAR(50),
    total_amount DECIMAL(10,2),
    items JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Inventory Service Database
CREATE DATABASE inventory_service_db;
CREATE TABLE products (
    id UUID PRIMARY KEY,
    sku VARCHAR(100) UNIQUE,
    name VARCHAR(255),
    price DECIMAL(10,2),
    stock_quantity INTEGER,
    reserved_quantity INTEGER DEFAULT 0
);
```

## Communication Patterns

### Synchronous Communication

#### API Gateway Pattern
```python
# API Gateway implementation using FastAPI
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio

app = FastAPI(title="API Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ServiceRegistry:
    def __init__(self):
        self.services = {
            'user-service': ['http://user-service-1:8000', 'http://user-service-2:8000'],
            'order-service': ['http://order-service-1:8000', 'http://order-service-2:8000'],
            'product-service': ['http://product-service-1:8000']
        }
        self.current_instance = {}
    
    def get_service_url(self, service_name: str) -> str:
        instances = self.services.get(service_name, [])
        if not instances:
            raise HTTPException(status_code=503, detail=f"Service {service_name} not available")
        
        # Simple round-robin load balancing
        current = self.current_instance.get(service_name, 0)
        url = instances[current]
        self.current_instance[service_name] = (current + 1) % len(instances)
        
        return url

service_registry = ServiceRegistry()

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # 'closed', 'open', 'half-open'
    
    async def call_service(self, service_name: str, method: str, url: str, **kwargs):
        state = self.state.get(service_name, 'closed')
        
        if state == 'open':
            if time.time() - self.last_failure_time.get(service_name, 0) > self.recovery_timeout:
                self.state[service_name] = 'half-open'
            else:
                raise HTTPException(status_code=503, detail=f"Circuit breaker open for {service_name}")
        
        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == 'GET':
                    response = await client.get(url, **kwargs)
                elif method.upper() == 'POST':
                    response = await client.post(url, **kwargs)
                elif method.upper() == 'PUT':
                    response = await client.put(url, **kwargs)
                elif method.upper() == 'DELETE':
                    response = await client.delete(url, **kwargs)
                
                response.raise_for_status()
                
                # Reset failure count on success
                self.failure_count[service_name] = 0
                self.state[service_name] = 'closed'
                
                return response.json()
                
        except Exception as e:
            self.failure_count[service_name] = self.failure_count.get(service_name, 0) + 1
            self.last_failure_time[service_name] = time.time()
            
            if self.failure_count[service_name] >= self.failure_threshold:
                self.state[service_name] = 'open'
            
            raise HTTPException(status_code=503, detail=f"Service call failed: {str(e)}")

circuit_breaker = CircuitBreaker()

# Route definitions
@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    service_url = service_registry.get_service_url('user-service')
    url = f"{service_url}/users/{user_id}"
    return await circuit_breaker.call_service('user-service', 'GET', url)

@app.post("/api/orders")
async def create_order(order_data: dict):
    service_url = service_registry.get_service_url('order-service')
    url = f"{service_url}/orders"
    return await circuit_breaker.call_service('order-service', 'POST', url, json=order_data)

@app.get("/api/products")
async def get_products(category: str = None, limit: int = 10):
    service_url = service_registry.get_service_url('product-service')
    params = {'limit': limit}
    if category:
        params['category'] = category
    
    url = f"{service_url}/products"
    return await circuit_breaker.call_service('product-service', 'GET', url, params=params)
```

#### Service Mesh with Istio
```yaml
# Service mesh configuration
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: user-service
spec:
  hosts:
  - user-service
  http:
  - match:
    - headers:
        version:
          exact: v2
    route:
    - destination:
        host: user-service
        subset: v2
      weight: 100
  - route:
    - destination:
        host: user-service
        subset: v1
      weight: 80
    - destination:
        host: user-service
        subset: v2
      weight: 20

---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: user-service
spec:
  host: user-service
  trafficPolicy:
    circuitBreaker:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
    loadBalancer:
      simple: LEAST_CONN
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

### Asynchronous Communication

#### Event-Driven Architecture
```python
# Domain events and event handlers
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any
import json

@dataclass
class DomainEvent:
    event_type: str
    aggregate_id: str
    event_data: Dict[str, Any]
    timestamp: datetime
    version: int = 1

class EventPublisher:
    def __init__(self, message_broker):
        self.broker = message_broker
    
    def publish(self, event: DomainEvent):
        message = {
            'event_type': event.event_type,
            'aggregate_id': event.aggregate_id,
            'event_data': event.event_data,
            'timestamp': event.timestamp.isoformat(),
            'version': event.version
        }
        
        self.broker.publish(
            topic=f"events.{event.event_type.lower()}",
            message=json.dumps(message)
        )

class OrderService:
    def __init__(self, repository, event_publisher):
        self.repository = repository
        self.event_publisher = event_publisher
    
    def create_order(self, customer_id, items):
        order = Order(customer_id, items)
        self.repository.save(order)
        
        # Publish domain event
        event = DomainEvent(
            event_type='ORDER_CREATED',
            aggregate_id=order.id,
            event_data={
                'customer_id': customer_id,
                'items': [{'product_id': item.product_id, 'quantity': item.quantity} for item in items],
                'total_amount': str(order.total_amount)
            },
            timestamp=datetime.utcnow()
        )
        
        self.event_publisher.publish(event)
        return order

# Event handlers in different services
class InventoryEventHandler:
    def __init__(self, inventory_service):
        self.inventory_service = inventory_service
    
    def handle_order_created(self, event_data):
        order_id = event_data['aggregate_id']
        items = event_data['event_data']['items']
        
        # Reserve inventory for the order
        for item in items:
            self.inventory_service.reserve_stock(
                product_id=item['product_id'],
                quantity=item['quantity'],
                order_id=order_id
            )

class NotificationEventHandler:
    def __init__(self, notification_service):
        self.notification_service = notification_service
    
    def handle_order_created(self, event_data):
        customer_id = event_data['event_data']['customer_id']
        order_id = event_data['aggregate_id']
        
        # Send order confirmation
        self.notification_service.send_order_confirmation(
            customer_id=customer_id,
            order_id=order_id
        )
```

#### Command Query Responsibility Segregation (CQRS)
```python
# Command side - writes
class CreateUserCommand:
    def __init__(self, email, name, password):
        self.email = email
        self.name = name
        self.password = password

class UserCommandHandler:
    def __init__(self, user_repository, event_publisher):
        self.repository = user_repository
        self.event_publisher = event_publisher
    
    def handle_create_user(self, command: CreateUserCommand):
        # Validate command
        if self.repository.exists_by_email(command.email):
            raise UserAlreadyExistsException(command.email)
        
        # Create user
        user = User(
            id=generate_uuid(),
            email=command.email,
            name=command.name,
            password_hash=hash_password(command.password)
        )
        
        self.repository.save(user)
        
        # Publish event
        event = UserCreatedEvent(
            user_id=user.id,
            email=user.email,
            name=user.name
        )
        self.event_publisher.publish(event)
        
        return user.id

# Query side - reads
class UserQueryModel:
    def __init__(self):
        self.id = None
        self.email = None
        self.name = None
        self.profile = None
        self.last_login = None
        self.order_count = 0
        self.total_spent = 0.0

class UserProjectionHandler:
    def __init__(self, query_repository):
        self.query_repository = query_repository
    
    def handle_user_created(self, event):
        query_model = UserQueryModel()
        query_model.id = event.user_id
        query_model.email = event.email
        query_model.name = event.name
        query_model.order_count = 0
        query_model.total_spent = 0.0
        
        self.query_repository.save(query_model)
    
    def handle_order_completed(self, event):
        query_model = self.query_repository.get_by_id(event.customer_id)
        if query_model:
            query_model.order_count += 1
            query_model.total_spent += float(event.total_amount)
            self.query_repository.save(query_model)

class UserQueryService:
    def __init__(self, query_repository):
        self.repository = query_repository
    
    def get_user_profile(self, user_id):
        return self.repository.get_by_id(user_id)
    
    def get_top_customers(self, limit=10):
        return self.repository.get_top_by_total_spent(limit)
```

## Data Management Patterns

### Database per Service
```python
# Each service manages its own database
class UserService:
    def __init__(self):
        # User service has its own PostgreSQL database
        self.db = PostgreSQLConnection('postgresql://user:pass@user-db:5432/users')
    
    def get_user(self, user_id):
        return self.db.query_one("SELECT * FROM users WHERE id = %s", (user_id,))

class OrderService:
    def __init__(self):
        # Order service uses MongoDB
        self.db = MongoClient('mongodb://order-db:27017')
        self.collection = self.db.orders.orders
    
    def get_order(self, order_id):
        return self.collection.find_one({"_id": order_id})

class AnalyticsService:
    def __init__(self):
        # Analytics service uses ClickHouse for time-series data
        self.db = ClickHouseConnection('clickhouse://analytics-db:9000/analytics')
    
    def track_event(self, event):
        self.db.execute(
            "INSERT INTO events (user_id, event_type, timestamp, properties) VALUES",
            (event.user_id, event.event_type, event.timestamp, event.properties)
        )
```

### Saga Pattern for Distributed Transactions
```java
// Orchestrator-based Saga
@Component
public class OrderSagaOrchestrator {
    
    @Autowired
    private InventoryService inventoryService;
    
    @Autowired
    private PaymentService paymentService;
    
    @Autowired
    private OrderService orderService;
    
    @Autowired
    private SagaManager sagaManager;
    
    public void processOrder(OrderRequest request) {
        SagaTransaction saga = SagaTransaction.builder()
            .sagaId(UUID.randomUUID().toString())
            .build();
        
        // Step 1: Reserve inventory
        saga.addStep(
            () -> inventoryService.reserveItems(request.getItems()),
            (reservationId) -> inventoryService.cancelReservation(reservationId)
        );
        
        // Step 2: Process payment
        saga.addStep(
            () -> paymentService.chargeCustomer(request.getCustomerId(), request.getAmount()),
            (paymentId) -> paymentService.refundPayment(paymentId)
        );
        
        // Step 3: Create order
        saga.addStep(
            () -> orderService.createOrder(request),
            (orderId) -> orderService.cancelOrder(orderId)
        );
        
        sagaManager.execute(saga);
    }
}

// Choreography-based Saga
@EventHandler
public class InventoryEventHandler {
    
    @Autowired
    private InventoryService inventoryService;
    
    @Autowired
    private EventPublisher eventPublisher;
    
    @EventListener
    public void handle(OrderCreatedEvent event) {
        try {
            ReservationResult result = inventoryService.reserveItems(
                event.getOrderId(), 
                event.getItems()
            );
            
            if (result.isSuccessful()) {
                eventPublisher.publish(new InventoryReservedEvent(
                    event.getOrderId(),
                    result.getReservationId()
                ));
            } else {
                eventPublisher.publish(new InventoryReservationFailedEvent(
                    event.getOrderId(),
                    result.getErrorMessage()
                ));
            }
        } catch (Exception e) {
            eventPublisher.publish(new InventoryReservationFailedEvent(
                event.getOrderId(),
                e.getMessage()
            ));
        }
    }
    
    @EventListener
    public void handle(OrderCancelledEvent event) {
        inventoryService.cancelReservation(event.getOrderId());
    }
}
```

## Service Discovery and Configuration

### Service Registration with Consul
```go
// Service registration in Go
package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
    "strconv"
    
    "github.com/hashicorp/consul/api"
    "github.com/gorilla/mux"
)

type UserService struct {
    consul *api.Client
    serviceID string
}

func NewUserService() (*UserService, error) {
    // Create Consul client
    config := api.DefaultConfig()
    config.Address = os.Getenv("CONSUL_ADDRESS")
    if config.Address == "" {
        config.Address = "localhost:8500"
    }
    
    client, err := api.NewClient(config)
    if err != nil {
        return nil, err
    }
    
    return &UserService{
        consul: client,
        serviceID: fmt.Sprintf("user-service-%s", os.Getenv("HOSTNAME")),
    }, nil
}

func (s *UserService) RegisterService() error {
    port, _ := strconv.Atoi(os.Getenv("PORT"))
    if port == 0 {
        port = 8080
    }
    
    registration := &api.AgentServiceRegistration{
        ID:      s.serviceID,
        Name:    "user-service",
        Port:    port,
        Address: os.Getenv("SERVICE_IP"),
        Tags:    []string{"user", "v1.0", "production"},
        Check: &api.AgentServiceCheck{
            HTTP:                           fmt.Sprintf("http://%s:%d/health", os.Getenv("SERVICE_IP"), port),
            Interval:                       "10s",
            Timeout:                        "3s",
            DeregisterCriticalServiceAfter: "30s",
        },
    }
    
    return s.consul.Agent().ServiceRegister(registration)
}

func (s *UserService) DeregisterService() error {
    return s.consul.Agent().ServiceDeregister(s.serviceID)
}

func (s *UserService) DiscoverService(serviceName string) ([]*api.ServiceEntry, error) {
    services, _, err := s.consul.Health().Service(serviceName, "", true, nil)
    return services, err
}

// Health check endpoint
func (s *UserService) healthHandler(w http.ResponseWriter, r *http.Request) {
    w.WriteHeader(http.StatusOK)
    w.Write([]byte("OK"))
}

// User service endpoints
func (s *UserService) getUserHandler(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    userID := vars["id"]
    
    // Simulate user retrieval
    user := map[string]interface{}{
        "id":    userID,
        "name":  "John Doe",
        "email": "john@example.com",
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func main() {
    service, err := NewUserService()
    if err != nil {
        log.Fatal("Failed to create service:", err)
    }
    
    // Register service with Consul
    if err := service.RegisterService(); err != nil {
        log.Fatal("Failed to register service:", err)
    }
    
    // Deregister on shutdown
    defer service.DeregisterService()
    
    // Setup routes
    r := mux.NewRouter()
    r.HandleFunc("/health", service.healthHandler).Methods("GET")
    r.HandleFunc("/users/{id}", service.getUserHandler).Methods("GET")
    
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }
    
    log.Printf("User service starting on port %s", port)
    log.Fatal(http.ListenAndServe(":"+port, r))
}
```

### Configuration Management
```yaml
# Kubernetes ConfigMap for microservice configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: user-service-config
  namespace: production
data:
  database_url: "postgresql://user:password@postgres:5432/users"
  redis_url: "redis://redis:6379/0"
  log_level: "INFO"
  features.json: |
    {
      "user_registration": true,
      "email_verification": true,
      "social_login": false,
      "advanced_search": true
    }
  rate_limits.json: |
    {
      "requests_per_minute": 1000,
      "concurrent_connections": 100,
      "burst_size": 50
    }

---
apiVersion: v1
kind: Secret
metadata:
  name: user-service-secrets
  namespace: production
type: Opaque
data:
  jwt_secret: <base64-encoded-secret>
  email_api_key: <base64-encoded-key>
  database_password: <base64-encoded-password>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-service
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: user-service
  template:
    metadata:
      labels:
        app: user-service
        version: v1
    spec:
      containers:
      - name: user-service
        image: user-service:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: user-service-config
              key: database_url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: user-service-secrets
              key: jwt_secret
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: secrets-volume
          mountPath: /app/secrets
          readOnly: true
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
      volumes:
      - name: config-volume
        configMap:
          name: user-service-config
      - name: secrets-volume
        secret:
          secretName: user-service-secrets
```

## Testing Strategies

### Unit Testing
```python
# Unit tests for microservice
import pytest
from unittest.mock import Mock, patch
from user_service import UserService, User, ValidationException

class TestUserService:
    def setup_method(self):
        self.mock_repository = Mock()
        self.mock_event_publisher = Mock()
        self.user_service = UserService(self.mock_repository, self.mock_event_publisher)
    
    def test_create_user_success(self):
        # Arrange
        user_data = {
            'email': 'test@example.com',
            'name': 'Test User',
            'password': 'password123'
        }
        self.mock_repository.exists_by_email.return_value = False
        
        # Act
        user_id = self.user_service.create_user(user_data)
        
        # Assert
        assert user_id is not None
        self.mock_repository.save.assert_called_once()
        self.mock_event_publisher.publish.assert_called_once()
    
    def test_create_user_duplicate_email(self):
        # Arrange
        user_data = {
            'email': 'test@example.com',
            'name': 'Test User',
            'password': 'password123'
        }
        self.mock_repository.exists_by_email.return_value = True
        
        # Act & Assert
        with pytest.raises(ValidationException):
            self.user_service.create_user(user_data)
        
        self.mock_repository.save.assert_not_called()
        self.mock_event_publisher.publish.assert_not_called()
    
    @patch('user_service.hash_password')
    def test_password_hashing(self, mock_hash):
        # Arrange
        mock_hash.return_value = 'hashed_password'
        user_data = {
            'email': 'test@example.com',
            'name': 'Test User',
            'password': 'password123'
        }
        self.mock_repository.exists_by_email.return_value = False
        
        # Act
        self.user_service.create_user(user_data)
        
        # Assert
        mock_hash.assert_called_once_with('password123')
```

### Integration Testing
```python
# Integration tests
import pytest
import requests
import docker
import time
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

class TestUserServiceIntegration:
    @pytest.fixture(scope="class")
    def setup_containers(self):
        # Start test containers
        postgres = PostgresContainer("postgres:13")
        redis = RedisContainer("redis:6")
        
        postgres.start()
        redis.start()
        
        # Start user service container
        client = docker.from_env()
        user_service = client.containers.run(
            "user-service:test",
            environment={
                "DATABASE_URL": postgres.get_connection_url(),
                "REDIS_URL": redis.get_connection_url(),
                "PORT": "8080"
            },
            ports={'8080/tcp': 8080},
            detach=True
        )
        
        # Wait for service to be ready
        time.sleep(10)
        
        yield {
            'postgres': postgres,
            'redis': redis,
            'user_service': user_service
        }
        
        # Cleanup
        user_service.stop()
        user_service.remove()
        postgres.stop()
        redis.stop()
    
    def test_create_and_get_user(self, setup_containers):
        base_url = "http://localhost:8080"
        
        # Create user
        user_data = {
            "email": "integration@test.com",
            "name": "Integration Test",
            "password": "password123"
        }
        
        response = requests.post(f"{base_url}/users", json=user_data)
        assert response.status_code == 201
        
        user_id = response.json()["id"]
        
        # Get user
        response = requests.get(f"{base_url}/users/{user_id}")
        assert response.status_code == 200
        
        user = response.json()
        assert user["email"] == user_data["email"]
        assert user["name"] == user_data["name"]
        assert "password" not in user  # Password should not be returned
    
    def test_user_service_health(self, setup_containers):
        response = requests.get("http://localhost:8080/health")
        assert response.status_code == 200
        assert response.text == "OK"
```

### Contract Testing with Pact
```python
# Consumer test (API Gateway)
import atexit
from pact import Consumer, Provider, Like, Term
import requests

pact = Consumer('api-gateway').has_pact_with(Provider('user-service'))

def test_get_user():
    expected = {
        'id': Like('123e4567-e89b-12d3-a456-426614174000'),
        'email': Like('user@example.com'),
        'name': Like('John Doe'),
        'created_at': Term(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '2023-01-01T00:00:00Z')
    }
    
    (pact
     .given('User 123e4567-e89b-12d3-a456-426614174000 exists')
     .upon_receiving('a request for user 123e4567-e89b-12d3-a456-426614174000')
     .with_request('GET', '/users/123e4567-e89b-12d3-a456-426614174000')
     .will_respond_with(200, body=expected))
    
    with pact:
        response = requests.get(
            'http://localhost:1234/users/123e4567-e89b-12d3-a456-426614174000'
        )
        assert response.status_code == 200
        user = response.json()
        assert user['id'] == '123e4567-e89b-12d3-a456-426614174000'

atexit.register(pact.stop)

# Provider verification (User Service)
from pact import Verifier

def test_user_service_provider():
    verifier = Verifier(provider='user-service',
                       provider_base_url='http://localhost:8080')
    
    # Verify against all consumer pacts
    output, logs = verifier.verify_pacts('./pacts/')
    
    assert output == 0  # Verification successful
```

## Monitoring and Observability

### Distributed Tracing
```python
# OpenTelemetry tracing for microservices
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrumentation
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()
SQLAlchemyInstrumentor().instrument(engine=database_engine)

class UserService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
    
    async def create_user(self, user_data):
        with self.tracer.start_as_current_span("create_user") as span:
            span.set_attribute("user.email", user_data.email)
            span.set_attribute("service.name", "user-service")
            
            try:
                # Validate user data
                with self.tracer.start_as_current_span("validate_user_data"):
                    self.validate_user_data(user_data)
                
                # Check if user exists
                with self.tracer.start_as_current_span("check_user_exists") as check_span:
                    exists = await self.repository.exists_by_email(user_data.email)
                    check_span.set_attribute("user.exists", exists)
                    
                    if exists:
                        span.set_status(trace.Status(trace.StatusCode.ERROR, "User already exists"))
                        raise UserAlreadyExistsException()
                
                # Create user
                with self.tracer.start_as_current_span("save_user_to_db") as save_span:
                    user = await self.repository.save(user_data)
                    save_span.set_attribute("user.id", user.id)
                
                # Publish event
                with self.tracer.start_as_current_span("publish_user_created_event"):
                    await self.event_publisher.publish(UserCreatedEvent(user.id))
                
                span.set_attribute("user.id", user.id)
                span.set_status(trace.Status(trace.StatusCode.OK))
                
                return user
                
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
```

### Metrics Collection
```python
# Prometheus metrics for microservices
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code', 'service']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'service']
)

ACTIVE_USERS = Gauge(
    'active_users_total',
    'Number of currently active users'
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

QUEUE_SIZE = Gauge(
    'message_queue_size',
    'Size of message queue',
    ['queue_name']
)

def track_metrics(service_name):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status_code = getattr(result, 'status_code', 200)
                
                REQUEST_COUNT.labels(
                    method='POST',
                    endpoint=func.__name__,
                    status_code=status_code,
                    service=service_name
                ).inc()
                
                return result
                
            except Exception as e:
                REQUEST_COUNT.labels(
                    method='POST',
                    endpoint=func.__name__,
                    status_code=500,
                    service=service_name
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    method='POST',
                    endpoint=func.__name__,
                    service=service_name
                ).observe(duration)
        
        return wrapper
    return decorator

class UserService:
    def __init__(self):
        self.active_connections = 0
    
    @track_metrics('user-service')
    async def create_user(self, user_data):
        # Business logic here
        ACTIVE_USERS.inc()
        return user
    
    @track_metrics('user-service')
    async def delete_user(self, user_id):
        # Business logic here
        ACTIVE_USERS.dec()
        return True
    
    def get_metrics(self):
        return generate_latest()

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Centralized Logging
```python
# Structured logging with correlation IDs
import logging
import json
import uuid
from contextvars import ContextVar
from fastapi import Request, Response
import structlog

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id')

def configure_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Middleware to add correlation ID
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id_value = request.headers.get('x-correlation-id', str(uuid.uuid4()))
    correlation_id.set(correlation_id_value)
    
    response = await call_next(request)
    response.headers['x-correlation-id'] = correlation_id_value
    
    return response

# Logger with correlation ID
class CorrelatedLogger:
    def __init__(self, service_name):
        self.service_name = service_name
        self.logger = structlog.get_logger()
    
    def _get_context(self):
        return {
            'service': self.service_name,
            'correlation_id': correlation_id.get(None),
            'timestamp': time.time()
        }
    
    def info(self, message, **kwargs):
        context = self._get_context()
        context.update(kwargs)
        self.logger.info(message, **context)
    
    def error(self, message, **kwargs):
        context = self._get_context()
        context.update(kwargs)
        self.logger.error(message, **context)
    
    def warn(self, message, **kwargs):
        context = self._get_context()
        context.update(kwargs)
        self.logger.warning(message, **context)

# Usage in service
class UserService:
    def __init__(self):
        self.logger = CorrelatedLogger('user-service')
    
    async def create_user(self, user_data):
        self.logger.info(
            "Creating user",
            email=user_data.email,
            operation="create_user"
        )
        
        try:
            user = await self.repository.save(user_data)
            
            self.logger.info(
                "User created successfully",
                user_id=user.id,
                email=user.email,
                operation="create_user"
            )
            
            return user
            
        except Exception as e:
            self.logger.error(
                "Failed to create user",
                email=user_data.email,
                error=str(e),
                operation="create_user"
            )
            raise
```

## Security Considerations

### Authentication and Authorization
```python
# JWT-based authentication for microservices
import jwt
import bcrypt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

class AuthService:
    def __init__(self, secret_key, algorithm='HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, user_id: str, permissions: list, expires_delta: timedelta = None):
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=1)
        
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': expire,
            'iat': datetime.utcnow(),
            'type': 'access_token'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Dependency for authentication
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    return payload

# Permission-based authorization
def require_permissions(required_permissions: list):
    def permission_checker(current_user: dict = Depends(get_current_user)):
        user_permissions = set(current_user.get('permissions', []))
        required_set = set(required_permissions)
        
        if not required_set.issubset(user_permissions):
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions"
            )
        
        return current_user
    
    return permission_checker

# Usage in endpoints
@app.post("/users")
async def create_user(
    user_data: UserCreateRequest,
    current_user: dict = Depends(require_permissions(['user:create']))
):
    return await user_service.create_user(user_data)

@app.get("/users/{user_id}")
async def get_user(
    user_id: str,
    current_user: dict = Depends(require_permissions(['user:read']))
):
    return await user_service.get_user(user_id)
```

### API Security
```python
# Rate limiting and API security
import redis
import time
from functools import wraps
from fastapi import HTTPException, Request

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        pipe = self.redis.pipeline()
        now = time.time()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, now - window_seconds)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiration
        pipe.expire(key, window_seconds)
        
        results = pipe.execute()
        current_requests = results[1]
        
        return current_requests < max_requests

rate_limiter = RateLimiter(redis.Redis(host='redis', port=6379, db=0))

def rate_limit(max_requests: int, window_seconds: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}:{func.__name__}"
            
            if not rate_limiter.is_allowed(key, max_requests, window_seconds):
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests"
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

# Input validation and sanitization
from pydantic import BaseModel, validator
import re

class UserCreateRequest(BaseModel):
    email: str
    name: str
    password: str
    
    @validator('email')
    def validate_email(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain digit')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        # Sanitize name - remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', v.strip())
        if len(sanitized) < 2:
            raise ValueError('Name must be at least 2 characters long')
        return sanitized

# Usage with rate limiting
@app.post("/users")
@rate_limit(max_requests=10, window_seconds=60)  # 10 requests per minute
async def create_user(
    request: Request,
    user_data: UserCreateRequest,
    current_user: dict = Depends(require_permissions(['user:create']))
):
    return await user_service.create_user(user_data)
```

## Best Practices

### Error Handling
```python
# Centralized error handling
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import traceback

class BusinessException(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationException(BusinessException):
    pass

class NotFoundError(BusinessException):
    pass

class ConflictError(BusinessException):
    pass

@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "business_error",
                "message": exc.message,
                "code": exc.error_code,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "type": "not_found",
                "message": exc.message,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception",
        error=str(exc),
        traceback=traceback.format_exc(),
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An internal error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    )
```

### Configuration and Deployment
```yaml
# Helm chart for microservice deployment
apiVersion: v2
name: user-service
description: User Service Helm Chart
version: 1.0.0
appVersion: "1.0.0"

---
# values.yaml
replicaCount: 3

image:
  repository: user-service
  tag: "1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.example.com
      paths:
        - path: /users
          pathType: Prefix
  tls:
    - secretName: api-tls
      hosts:
        - api.example.com

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

env:
  DATABASE_URL: postgresql://user:password@postgres:5432/users
  REDIS_URL: redis://redis:6379/0
  LOG_LEVEL: INFO

secrets:
  JWT_SECRET: jwt-secret-key
  DATABASE_PASSWORD: db-password

probes:
  liveness:
    path: /health
    initialDelaySeconds: 30
    periodSeconds: 10
  readiness:
    path: /ready
    initialDelaySeconds: 5
    periodSeconds: 5
```

### Performance Optimization
```python
# Connection pooling and caching
import asyncpg
import aioredis
from sqlalchemy.pool import QueuePool

class DatabaseManager:
    def __init__(self, database_url, pool_size=20, max_overflow=30):
        self.pool = None
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
    
    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=self.pool_size,
            command_timeout=30,
            server_settings={
                'jit': 'off',  # Disable JIT for faster connection
                'application_name': 'user_service'
            }
        )
    
    async def execute_query(self, query, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def execute_transaction(self, queries):
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                results = []
                for query, args in queries:
                    result = await connection.fetch(query, *args)
                    results.append(result)
                return results

class CacheManager:
    def __init__(self, redis_url, cluster_mode=False):
        self.redis_url = redis_url
        self.cluster_mode = cluster_mode
        self.redis = None
    
    async def initialize(self):
        if self.cluster_mode:
            self.redis = await aioredis.create_redis_cluster(
                self.redis_url,
                encoding='utf-8'
            )
        else:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
    
    async def get_or_set(self, key, fetch_func, expire=3600):
        # Try to get from cache
        cached_value = await self.redis.get(key)
        if cached_value:
            return json.loads(cached_value)
        
        # Fetch from source
        value = await fetch_func()
        
        # Cache the result
        await self.redis.setex(key, expire, json.dumps(value))
        
        return value
    
    async def invalidate_pattern(self, pattern):
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

# Optimized service with caching
class UserService:
    def __init__(self, db_manager, cache_manager):
        self.db = db_manager
        self.cache = cache_manager
    
    async def get_user(self, user_id):
        cache_key = f"user:{user_id}"
        
        async def fetch_from_db():
            query = "SELECT * FROM users WHERE id = $1"
            result = await self.db.execute_query(query, user_id)
            return result[0] if result else None
        
        return await self.cache.get_or_set(cache_key, fetch_from_db)
    
    async def update_user(self, user_id, user_data):
        # Update in database
        query = """
            UPDATE users 
            SET name = $2, email = $3, updated_at = NOW() 
            WHERE id = $1 
            RETURNING *
        """
        result = await self.db.execute_query(query, user_id, user_data.name, user_data.email)
        
        # Invalidate cache
        await self.cache.invalidate_pattern(f"user:{user_id}*")
        
        return result[0] if result else None
```

## Use Cases and Examples

### E-commerce Microservices
```python
# Complete e-commerce microservice ecosystem
class EcommerceMicroservices:
    def __init__(self):
        self.services = {
            'user_service': UserService(),
            'product_service': ProductService(),
            'order_service': OrderService(),
            'inventory_service': InventoryService(),
            'payment_service': PaymentService(),
            'notification_service': NotificationService(),
            'recommendation_service': RecommendationService()
        }
    
    async def process_order_workflow(self, order_request):
        """
        Complete order processing workflow across multiple microservices
        """
        correlation_id = generate_correlation_id()
        
        try:
            # 1. Validate user
            user = await self.services['user_service'].get_user(order_request.user_id)
            if not user:
                raise UserNotFoundError()
            
            # 2. Validate products and calculate pricing
            products = await self.services['product_service'].get_products(
                order_request.product_ids
            )
            total_amount = calculate_total(products, order_request.quantities)
            
            # 3. Check inventory availability
            availability = await self.services['inventory_service'].check_availability(
                order_request.items
            )
            if not availability.all_available:
                raise InsufficientInventoryError(availability.unavailable_items)
            
            # 4. Reserve inventory
            reservation = await self.services['inventory_service'].reserve_items(
                correlation_id, order_request.items
            )
            
            # 5. Process payment
            payment = await self.services['payment_service'].process_payment(
                correlation_id,
                order_request.user_id,
                total_amount,
                order_request.payment_method
            )
            
            # 6. Create order
            order = await self.services['order_service'].create_order(
                correlation_id,
                order_request,
                payment.id,
                reservation.id
            )
            
            # 7. Send notifications
            await self.services['notification_service'].send_order_confirmation(
                user.email, order.id
            )
            
            # 8. Update recommendations
            await self.services['recommendation_service'].update_user_preferences(
                user.id, order_request.product_ids
            )
            
            return order
            
        except Exception as e:
            # Compensate in reverse order
            await self.compensate_order_workflow(correlation_id, e)
            raise
```

*Last updated: 2025-01-10 11:25:00 UTC*
