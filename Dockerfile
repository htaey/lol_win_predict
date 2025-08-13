FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["gunicorn", "mlserve.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "4"]
