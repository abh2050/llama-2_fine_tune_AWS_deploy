FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code and models into the container
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Set Python path for imports
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "--chdir", "/app/src", "app:app"]
