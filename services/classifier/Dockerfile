FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY services/classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (needed for core modules)
COPY . .

# Set Python path to include core modules
ENV PYTHONPATH=/app

# Use ARG to pass build-time variables, ENV to set runtime defaults
ARG CLASSIFIER_PORT=8000
ENV CLASSIFIER_PORT=${CLASSIFIER_PORT}

EXPOSE ${CLASSIFIER_PORT}

CMD ["python", "services/classifier/server/app.py"]