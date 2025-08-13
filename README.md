# Naive Bayes Classifier

A modern microservices implementation of the Naive Bayes classifier with FastAPI, Docker support, and clean architecture for production-ready machine learning workflows.

---

## 🚀 Features

- **Microservices Architecture** - Separate services for classification and model training
- **RESTful APIs** - FastAPI with automatic documentation and type safety
- **Model Caching** - LRU cache for efficient model loading and reuse
- **Multiple Data Sources** - Support for CSV files, inline data, and database sources
- **Container Ready** - Docker deployment configurations
- **Production Optimized** - Log probability calculations, Laplace smoothing, and numerical stability

---

## 📦 Installation

```bash
# Clone the repository
git clone <repo-url>

# Go into the project directory
cd naive-bayes

# Install dependencies
pip install -r requirements.txt
```

**Docker Installation:**
```bash
# Copy environment configuration
cp .env.example .env

# Build and run with Docker Compose
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env up --build
```

---

## ▶️ Usage

**Local Development:**
```bash
# Run classifier service (port 8000)
python services/classifier/server/app.py

# Run model pipeline manager service (port 8001)
python services/model_pipeline_manager/server/app.py
```

**Docker Deployment:**
```bash
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env up
```

**API Examples:**
```python
import requests

# Train a model
response = requests.post("http://localhost:8001/models/create", json={
    "target_column": "buys_computer",
    "source": {
        "source_type": "file",
        "file_path": "data/raw/buy_computer_data.csv",
        "file_type": "csv"
    }
})

# Make predictions
predictions = requests.post("http://localhost:8000/classify", json={
    "model_id": "model_12345678",
    "data": [
        {"age": "<=30", "income": "high", "student": "no", "credit_rating": "fair"}
    ]
})
```

---

## ⚙️ Configuration

**Environment Variables:**
- `CLASSIFIER_HOST` – Classifier service hostname (default: 127.0.0.1)
- `CLASSIFIER_PORT` – Classifier service port (default: 8000)
- `MODEL_PIPELINE_MANAGER_HOST` – Model manager hostname (default: 127.0.0.1)
- `MODEL_PIPELINE_MANAGER_PORT` – Model manager port (default: 8001)
- `TEST_SIZE` – Train/test split ratio (default: 0.3)
- `LAPLACE_SMOOTHING` – Smoothing parameter (default: 1.0)
- `MODEL_CACHE_CAPACITY` – Number of models to cache (default: 3)

**Configuration Files:**
- `.env` – Environment variables (copy from `.env.example`)
- `config/general.py` – Application configuration

---

## 🗂 Project Structure

```
naive-bayes/
├── core/                    # Core business logic
│   ├── classifier.py        # Naive Bayes implementation
│   ├── data_loader.py       # Data loading utilities
│   ├── model_trainer.py     # Model training logic
│   └── trained_model.py     # Model serialization
├── services/                # Microservices
│   ├── classifier/          # Classification API (port 8000)
│   └── model_pipeline_manager/  # Model training API (port 8001)
├── utils/                   # Utilities
│   └── lru_cache.py         # LRU cache implementation
├── infrastructure/          # Deployment configs
│   └── docker/
│       └── docker-compose.yml # Docker Compose setup
├── data/                    # Data storage
│   ├── raw/                 # Sample datasets
│   └── models/              # Trained models
├── config/                  # Configuration management
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System design and patterns
│   └── CONFIGURATION.md     # Environment setup guide
├── tests/                   # Unit tests
└── requirements.txt         # Dependencies
```

---

## 🌐 API Documentation

**Classifier Service (localhost:8000):**
- `POST /classify` – Make predictions with trained model
- `GET /models` – List available models
- `GET /health` – Service health check

**Model Pipeline Manager (localhost:8001):**
- `POST /models/create` – Train new models from data sources
- `GET /models/{model_id}` – Get specific model details
- `GET /models` – List all trained models
- `GET /datasets` – List available datasets

Interactive API docs available at:
- http://localhost:8000/docs
- http://localhost:8001/docs

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙌 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to branch (`git push origin feature-name`)
5. Create a Pull Request

---

## 📚 Additional Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** – Detailed system design and architecture patterns
- **[Configuration Guide](docs/CONFIGURATION.md)** – Environment setup and deployment configurations