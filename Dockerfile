# ----------------------------------------
# BUILD
# ----------------------------------------

FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------
# RUNTIME
# ----------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "src/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]