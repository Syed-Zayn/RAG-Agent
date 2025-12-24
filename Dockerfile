# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Railway dynamic port use karta hai)
EXPOSE 8000

# Start Command (Runs FastAPI Backend)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]