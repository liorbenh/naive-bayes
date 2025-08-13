# Architecture Guide

## 1. Overview
The Naive Bayes Classifier is built as a microservices architecture with two main services: a Classifier Service for real-time predictions and a Model Pipeline Manager for training and model lifecycle management. The system follows clean architecture principles with separation of concerns, dependency injection, and modular design patterns.

**Key Design Principles:**
- **Microservices Architecture** - Independent, scalable services
- **Single Responsibility** - Each service has a focused purpose
- **Configuration Management** - Centralized environment-based configuration
- **Performance Optimization** - LRU caching and log probability calculations
- **Container-First** - Docker ready deployment

---

## 2. High-Level Diagram

### System Overview
```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   Client Apps   │ ───────────────→ │  Classifier     │
│   (Consumers)   │                  │  Service        │
└─────────────────┘                  │  (Port 8000)    │
                                     └─────────────────┘
                                              │
                                              │ Model Fetch
                                              ▼
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   Admin/ML      │ ───────────────→ │  Model Pipeline │
│   Engineers     │                  │  Manager        │
└─────────────────┘                  │  (Port 8001)    │
                                     └─────────────────┘
                                              │
                                              │ File I/O
                                              ▼
                                     ┌─────────────────┐
                                     │   Data Layer    │
                                     │   (Files/DB)    │
                                     └─────────────────┘
```

### Model Pipeline Manager Internal Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                  Model Pipeline Manager                     │
│                       (Port 8001)                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                    ┌──────────────────────┐ │
│  │   FastAPI   │ POST /models/create│   Pipeline Service   │ │
│  │ Controller  │ ─────────────────→ │                      │ │
│  └─────────────┘                    └──────────────────────┘ │
│                                                │             │
│                                                ▼             │
│  ┌─────────────┐                    ┌──────────────────────┐ │
│  │ Data Source │ CSV/Inline/DB      │    Data Loader       │ │
│  │ (Multiple)  │ ─────────────────→ │ • Load from sources  │ │
│  └─────────────┘                    │ • Format validation  │ │
│                                     └──────────────────────┘ │
│                                                │             │
│                                                ▼             │
│                                     ┌──────────────────────┐ │
│                                     │  Data Preprocessor   │ │
│                                     │ • Data preparation   │ │
│                                     │ • Quality checks     │ │
│                                     └──────────────────────┘ │
│                                                │             │
│                                                ▼             │
│                                     ┌──────────────────────┐ │
│                                     │   Model Trainer      │ │
│                                     │ • Algorithm training │ │
│                                     │ • Model fitting      │ │
│                                     └──────────────────────┘ │
│                                                │             │
│                                                ▼             │
│                                     ┌──────────────────────┐ │
│                                     │  Model Evaluator     │ │
│                                     │ • Performance metrics│ │
│                                     │ • Quality assessment │ │
│                                     └──────────────────────┘ │
│                                                │             │
│                                                ▼             │
│  ┌─────────────┐                    ┌──────────────────────┐ │
│  │ File System │ Model Storage      │   Trained Model      │ │
│  │data/models/ │ ←───────────────── │ • Model persistence  │ │
│  └─────────────┘                    │ • Metadata storage   │ │
│                                     └──────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### Component: Classifier Service
- **Purpose:** Real-time classification of incoming data using pre-trained models
- **Key Interfaces / APIs:** 
  - `POST /classify` - Batch classification
  - `GET /models` - List available models
  - `GET /health` - Health check
- **Dependencies:** Model Pipeline Manager (for model fetching), LRU Cache, Core Classifier

### Component: Model Pipeline Manager Service
- **Purpose:** Complete model lifecycle management including training, evaluation, and storage
- **Key Interfaces / APIs:**
  - `POST /models/create` - Train new models from various data sources
  - `GET /models/{model_id}` - Retrieve specific model details
  - `GET /models` - List all trained models
  - `GET /datasets` - List available datasets
- **Dependencies:** Core Training Pipeline, Data Loaders, File System

### Component: Core Business Logic
- **Purpose:** Domain-specific algorithms and data processing logic
- **Key Interfaces / APIs:**
  - `Classifier.classify()` - Single sample classification
  - `ModelTrainer.train()` - Model training workflow  
  - `DataLoader.load()` - Data ingestion from various sources
- **Dependencies:** NumPy, Pandas, Scikit-learn

### Component: Configuration Management
- **Purpose:** Centralized configuration using environment variables and .env files
- **Key Interfaces / APIs:**
  - `config.classifier_url` - Service endpoint URLs
  - `config.get_data_path()` - File path resolution
  - `config.get_available_datasets()` - Dynamic dataset discovery
- **Dependencies:** python-dotenv, PathLib

### Component: LRU Cache
- **Purpose:** Memory-efficient caching of trained models to reduce latency
- **Key Interfaces / APIs:**
  - `LRUCache.get()` - Retrieve cached model
  - `LRUCache.put()` - Store model in cache
- **Dependencies:** None (custom implementation)

---

## 4. Design Patterns & Principles

- **Microservices Pattern** – Services are independently deployable and scalable with clear API boundaries
- **Repository Pattern** – Data access is abstracted through DataLoader interfaces supporting multiple sources (CSV, Database, Inline)
- **Factory Pattern** – TrainedModel.from_dict() creates classifier instances from serialized data
- **Singleton Pattern** – Global config instance ensures single source of truth for configuration
- **Strategy Pattern** – Different data sources (CSV, Database, Inline) implement common interfaces
- **Dependency Injection** – Services receive dependencies through constructors for better testability
- **Clean Architecture** – Core business logic is independent of frameworks and external concerns

---

## 5. Data Flow

**Training Flow:**
1. **Data Ingestion** → Model Pipeline Manager receives data (CSV file/inline/database)
2. **Preprocessing** → DataPreprocessor cleans and validates data
3. **Model Training** → ModelTrainer fits Naive Bayes algorithm with Laplace smoothing
4. **Model Evaluation** → ModelEvaluator calculates accuracy, precision, recall metrics
5. **Model Storage** → Trained model serialized to disk with metadata

**Classification Flow:**
1. **Request Reception** → Classifier Service receives classification request
2. **Model Loading** → Service fetches model from cache or Pipeline Manager
3. **Classification** → Core Classifier applies Naive Bayes with log probabilities
4. **Response** → Results returned to client with confidence scores

**Inter-Service Communication:**
- Services communicate via HTTP/REST using httpx async client
- Model data exchanged as JSON payloads
- Error handling with proper HTTP status codes

---

## 6. Scalability & Performance Notes

**Performance Optimizations:**
- **Log Probability Calculations** - Prevents numerical underflow in classification
- **LRU Cache** - Configurable model caching (default: 3 models) reduces disk I/O
- **Async HTTP Clients** - Non-blocking inter-service communication
- **Memory Efficiency** - Models don't store training data after fitting

**Scalability Considerations:**
- **Horizontal Scaling** - Each service can be scaled independently
- **Stateless Design** - Services maintain no session state enabling load balancing
- **Container Ready** - Docker images support orchestration platforms
- **Database Integration** - Ready for external databases when file storage becomes insufficient

**Monitoring & Observability:**
- Health check endpoints for service discovery
- Structured logging directory (`logs/`)
- Configuration validation on startup
- Error handling with descriptive messages

---