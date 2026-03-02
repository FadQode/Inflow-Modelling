FROM python:3.11-slim

# Avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and model
COPY ./app ./app
COPY ./EkmanClassifier ./EkmanClassifier

# Expose FastAPI port
EXPOSE 8135

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
