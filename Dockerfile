# ==============================================================================
# Stage 1: Builder - Install Dependencies
# ==============================================================================
# Use a slim Python image as the base for building our dependencies
FROM python:3.10-slim as builder

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to the terminal
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Final Image - Create the Runtime Container
# ==============================================================================
# Use the same slim Python image for the final, lean container
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment with installed packages from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the path to use the virtual environment's Python and packages
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code and the ChromaDB database into the container
# This assumes 'chroma_db' and 'main.py' are in the same directory as the Dockerfile
COPY ./main.py .
COPY ./chroma_db ./chroma_db

# Expose the port the app runs on
EXPOSE 8080

# The command to run your application when the container starts
# We use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]