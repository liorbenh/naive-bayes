FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY services/model_pipeline_manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (needed for core modules)
COPY . .

# Set Python path to include core modules
ENV PYTHONPATH=/app

# Use ARG to pass build-time variables, ENV to set runtime defaults
ARG MODEL_PIPELINE_MANAGER_PORT=8001
ENV MODEL_PIPELINE_MANAGER_PORT=${MODEL_PIPELINE_MANAGER_PORT}

EXPOSE ${MODEL_PIPELINE_MANAGER_PORT}

CMD ["python", "services/model_pipeline_manager/server/app.py"]
