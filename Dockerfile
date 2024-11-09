FROM python:3.11.4-slim
WORKDIR /app
COPY config.yaml ./
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
# CMD ["python", "app.py"]