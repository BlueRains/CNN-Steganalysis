# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Use pipenv for installing the packages
RUN pip install pipenv
# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install any dependencies
RUN pipenv install --system --deploy

# Copy the rest of the application code into the container
COPY src/ /app/

# Ensure the main script has execution permissions
RUN chmod +x /app/main.py

# Command to run the application
CMD ["python", "/app/main.py"]