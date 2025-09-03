# Use official Python runtime as base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer output for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire project directory contents into the container
COPY . /app/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]
