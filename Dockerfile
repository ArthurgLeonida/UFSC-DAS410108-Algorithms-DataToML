# 1. Start with an official Python base image.
FROM python:3.13.7-slim

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy the requirements file first to leverage Docker's layer caching.
COPY requirements.txt .

# 4. Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container.
COPY . .

# 6. Expose the port Jupyter runs on.
EXPOSE 8888

# 7. The command to run when the container starts.
# This starts the Jupyter Notebook server.
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]