FROM python:3.11-bullseye
WORKDIR /app

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# Use streamlit CLI to run app on 0.0.0.0 so it's reachable from the container host
# Replace get_signal_alpaca_embedded.py with your app filename if different
CMD ["streamlit", "run", "get_signal_alpaca_embedded.py", "--server.port=8501", "--server.address=0.0.0.0"]
