# Configuration Guide

## 1. Prerequisites
- **OS:** Linux (Ubuntu 22.04+), macOS, or Windows 11
- **Dependencies:** 
  - Python 3.11+
  - Docker 24.0+ (for containerized deployment)
- **Other:** Git for repository cloning

---

## 2. Environment Variables

| Variable Name                     | Description                              | Example Value          |
|-----------------------------------|------------------------------------------|------------------------|
| `SCHEME`                          | HTTP scheme for service URLs             | `http`                 |
| `CLASSIFIER_HOST`                 | Classifier service hostname              | `0.0.0.0`              |
| `CLASSIFIER_PORT`                 | Classifier service port                  | `8000`                 |
| `MODEL_PIPELINE_MANAGER_HOST`     | Model manager service hostname           | `0.0.0.0`              |
| `MODEL_PIPELINE_MANAGER_PORT`     | Model manager service port               | `8001`                 |
| `TEST_SIZE`                       | Train/test split ratio                   | `0.3`                  |
| `RANDOM_STATE`                    | Random seed for reproducible results     | `42`                   |
| `LAPLACE_SMOOTHING`               | Smoothing parameter for classification   | `1.0`                  |
| `MODEL_CACHE_CAPACITY`            | Number of models to cache in memory      | `3`                    |
| `CSV_ENCODING`                    | Character encoding for CSV files         | `utf-8`                |

**Environment-Specific Overrides:**

| Environment     | Host Override Pattern                    |
|-----------------|------------------------------------------|
| **Docker**      | Service names (`classifier`, `model-pipeline-manager`) |
| **Local Dev**   | `127.0.0.1` or `localhost`              |

---

## 3. Installation

```bash
# Clone repository
git clone https://github.com/liorbenh/naive-naive-bayes.git
cd naive-bayes

# Copy environment configuration
cp .env.example .env

# Install Python dependencies
pip install -r requirements.txt
```

**Directory Structure After Installation:**
```
naive-bayes/
├── data/
│   ├── raw/          # CSV datasets (auto-created)
│   └── models/       # Trained models (auto-created)
├── logs/             # Application logs (auto-created)
└── .env              # Your environment configuration
```

---

## 4. Local Development Setup

**Start Services Individually:**

```bash
# Terminal 1: Start Model Pipeline Manager (Port 8001)
python services/model_pipeline_manager/server/app.py

# Terminal 2: Start Classifier Service (Port 8000)  
python services/classifier/server/app.py
```

**Verify Services:**
```bash
# Check health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health

# Access interactive API documentation
open http://localhost:8000/docs  # Classifier API
open http://localhost:8001/docs  # Model Manager API
```

**Local Development Configuration (.env):**
```bash
SCHEME=http
CLASSIFIER_HOST=127.0.0.1
CLASSIFIER_PORT=8000
MODEL_PIPELINE_MANAGER_HOST=127.0.0.1
MODEL_PIPELINE_MANAGER_PORT=8001
TEST_SIZE=0.3
RANDOM_STATE=42
LAPLACE_SMOOTHING=1.0
MODEL_CACHE_CAPACITY=3
CSV_ENCODING=utf-8
```

---

## 5. Deployment

### Development (Docker Compose)

```bash
# Ensure Docker is running
sudo systemctl start docker

# Build and start services
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env up --build

# Run in background
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env up --build -d

# View logs
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env logs -f

# Stop services
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env down
```

**Docker Environment Variables:**
- Service hostnames automatically set to container names
- Volumes mounted for data persistence: `./data:/app/data`, `./logs:/app/logs`

### Environment Configuration Files

**Development (.env):**
```bash
CLASSIFIER_HOST=127.0.0.1
MODEL_PIPELINE_MANAGER_HOST=127.0.0.1
```

**Docker (infrastructure/docker/docker-compose.yml overrides):**
```yaml
environment:
  - CLASSIFIER_HOST=classifier
  - MODEL_PIPELINE_MANAGER_HOST=model-pipeline-manager
```

---

## 6. Troubleshooting

| Issue                              | Cause                          | Fix                                    |
| ---------------------------------- | ------------------------------ | -------------------------------------- |
| Services won't start               | Missing `.env` file            | Copy `.env.example` to `.env`          |
| Connection refused between services| Wrong hostnames                | Check host variables in environment    |
| Models not loading                 | Missing data directory         | Ensure `data/models/` exists           |
| Docker permission denied          | User not in docker group      | `sudo usermod -aG docker $USER`       |
| Port already in use               | Services running on same ports | Check for existing processes: `lsof -i :8000` |
| Import errors in Python           | PYTHONPATH not set             | Set `PYTHONPATH=/path/to/project`      |
| CSV encoding errors               | Wrong encoding setting        | Update `CSV_ENCODING` in `.env`        |
| Model cache memory issues         | Cache capacity too high        | Reduce `MODEL_CACHE_CAPACITY`          |
| Classification accuracy low       | Insufficient training data     | Add more training samples              |

**Debug Commands:**
```bash
# Check service logs
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env logs classifier
docker-compose -f infrastructure/docker/docker-compose.yml --env-file .env logs model-pipeline-manager

# Check file permissions
ls -la data/
ls -la logs/

# Validate environment variables
python -c "from config.general import config; print(config.classifier_url)"

# Test inter-service communication
curl -X POST http://localhost:8001/models/create \
  -H "Content-Type: application/json" \
  -d '{"target_column": "class", "source": {"source_type": "file", "file_path": "data/raw/buy_computer_data.csv", "file_type": "csv"}}'
```

**Performance Tuning:**
- Increase `MODEL_CACHE_CAPACITY` for frequently used models
- Adjust `TEST_SIZE` based on dataset size
- Tune `LAPLACE_SMOOTHING` for better classification accuracy
- Monitor memory usage and adjust Docker resource limits

---