FROM python:3.11-bullseye
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "get_signal_alpaca_embedded.py"]

