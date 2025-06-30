FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose the port your FastAPI app will run on
EXPOSE 8001

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
