# Use official Python 3.11 image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the app using uvicorn
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
