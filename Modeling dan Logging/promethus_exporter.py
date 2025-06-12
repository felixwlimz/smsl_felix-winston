from flask import Flask, request, jsonify 
import requests 
import time 
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # Waktu respons API
THROUGHPUT = Counter('http_requests_throughput', 'Total number of requests per second')  # Throughput
 

CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage of the system')  # Penggunaan CPU
MEMORY_USAGE = Gauge('system_memory_usage', 'Memory Usage of the system')  # Penggunaan Memori

@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(generate_latest())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()
    
    api_url = 'http://127.0.0.1:5005/invocations'
    data = request.get_json()
    
    try : 
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        
        return jsonify(response.json())
    except Exception as e :
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)