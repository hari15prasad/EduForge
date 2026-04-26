# Use a multi-stage build to keep the image small
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# Final stage: Python for the FastAPI backend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn gunicorn

# Copy backend code
COPY . .

# Copy built frontend to a public directory
COPY --from=frontend-builder /app/frontend/out /app/frontend_static

# Create a small script to serve the frontend and backend together
RUN echo '#!/bin/bash\n\
python -m uvicorn app:app --host 0.0.0.0 --port 7860\n\
' > /app/start.sh
RUN chmod +x /app/start.sh

# HF Spaces uses 7860 by default
EXPOSE 7860

CMD ["/app/start.sh"]
