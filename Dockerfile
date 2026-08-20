FROM python:3.10-slim

# Cài đặt ffmpeg và các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy mã nguồn và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render tự động cấp biến cổng qua môi trường PORT (mặc định 7860)
ENV PORT=7860
EXPOSE 7860

# Chạy ứng dụng
CMD ["python", "main.py"]
