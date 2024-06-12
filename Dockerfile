# Use image from Docker Hub
FROM neopineapple/palm:v2.0

# Create a directory for the app
WORKDIR /server

# Copy the app to the container
COPY . .

# Change pip source to tsinghua
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install dependencies
RUN pip uninstall -y opencv-python
RUN pip install -r requirements-prod.txt

# Expose the port the app runs on
EXPOSE 5000

# Use gunicorn to run the app
CMD ["gunicorn","--certfile=cert.pem","--keyfile=key.pem","-b","0.0.0.0:5000","server:app"]