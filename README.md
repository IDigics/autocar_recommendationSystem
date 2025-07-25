
---

# Car Recommendation ML Service - Deployment Guide

just put the files inside the project 

## 1. Requirements File (`requirements.txt`)

```
fastapi==0.95.2
uvicorn==0.22.0
pandas==2.0.2
scikit-learn==1.2.2
joblib==1.2.0
python-multipart==0.0.6
```

## 2. Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 3. Build and Run Commands

### Build the Docker image:
```bash
docker build -t car_recommendation_ml .
```

### Run the container:
```bash
docker run -p 8000:8000 --name car_ml car_recommendation_ml
```

### (Optional) Run in detached mode:
```bash
docker run -d -p 8000:8000 --name car_ml car_recommendation_ml
```

## 4. Testing Commands

### Test with curl:
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"car_id": 7, "n": 3}'
```
### Test in Postman:
Open Postman and click on "New" > "HTTP Request" (or just start a new tab).

Set method to POST (from the dropdown).

Enter the URL:

```bash
http://localhost:8000/recommend

```

Go to the "Headers" tab:

Key: 

```bash
Content-Type
```

Value: 

```bash
application/json
```

Go to the "Body" tab:

Select "raw"

In the dropdown next to it, choose JSON

Paste the JSON body:

```json

{
  "car_id": 7,
  "n": 3
}

```
Click "Send".
### Expected successful response:
```json
[149, 156, 387]
```

### Test with invalid car ID:
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"car_id": 999999, "n": 3}'
```

### Expected error response:
```json
{
  "detail": "Car ID 999999 not found."
}
```

## 5. Management Commands

### Stop the container:
```bash
docker stop car_ml
```

### Start the container again:
```bash
docker start car_ml
```

### Remove the container:
```bash
docker rm car_ml
```

### Remove the image:
```bash
docker rmi car_recommendation_ml
```

## 6. Volume Mount (for development)

If you need to mount your local directory for development:
```bash
docker run -p 8000:8000 -v $(pwd):/app --name car_ml car_recommendation_ml
```

---

This documentation covers all the essential commands for building, running, and testing your containerized ML service. The service will be available at `http://localhost:8000` with the `/recommend` endpoint accepting POST requests as shown in the testing section.

# Recommendation system
This repository contains all the critical files used to create and develop a recommendation model for autocar
It contains :
  - A note book that preprocesses the data in creates the learning algorithm
  - Joblibs that imports and extracts the model
  - A python main code that consumes the function that uses the algorithm and passes it threw an API
