FROM python:3.12-slim
WORKDIR /app

# Install redis and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends redis-server \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Create a user (HF uses UID 1000) and give them ownership of /app
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

RUN chmod +x /app/entrypoint.sh

# HF Spaces MUST use 7860
EXPOSE 7860
ENV PORT=7860

CMD ["/app/entrypoint.sh"]
