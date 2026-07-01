# Use an explicit, stable Python slim image
FROM python:3.11-slim

# Set system environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies required for building vector client binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker's layer caching mechanism
COPY requirements.txt .

# Install pinned dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY config/ ./config/
COPY src/ ./src/
COPY ui.py .
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Expose Streamlit's native port
EXPOSE 8501

# Run healthcheck to ensure application container safety
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the application
ENTRYPOINT ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
