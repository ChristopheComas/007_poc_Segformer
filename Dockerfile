# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
# COPY ./requirements.txt .

COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
	pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Copy the rest of the application code to the container
# COPY ./ .

COPY . .

# Expose the port FastAPI runs on
EXPOSE 8501

# Command to run the FastAPI application using uvicorn
CMD ["streamlit", "run", "Model_prediction_analysis.py", "--server.port=8501"]