FROM python:3.11-slim
WORKDIR /app
COPY Backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY Backend/ ./backend
WORKDIR /app/backend
ENV MODEL_PATH=./model/htgnn_transaction_balanced_1000.pt
EXPOSE 8000
CMD ["/usr/local/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
