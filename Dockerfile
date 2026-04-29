# 1. Gunakan slim image agar base layer kecil
FROM python:3.10-slim

WORKDIR /app

# 2. Instal library sistem yang hanya diperlukan saja
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements dulu (agar caching efisien)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. COPY file aplikasi (PASTIKAN sudah ada .dockerignore)
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]