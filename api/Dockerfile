# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the app with Gunicorn
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8080", "app:app"]