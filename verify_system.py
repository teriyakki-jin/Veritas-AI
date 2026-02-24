import requests
import time
import sys
import subprocess
import os

def wait_for_server(url, timeout=60):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    print("\nServer timed out.")
    return False

def test_health():
    print("\nTesting /health...")
    try:
        resp = requests.get("http://127.0.0.1:8000/health")
        print(f"Status: {resp.status_code}")
        print(f"Body: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def test_verify():
    print("\nTesting /verify with 'The earth is flat'...")
    try:
        resp = requests.post("http://127.0.0.1:8000/verify", json={"claim": "The earth is flat"})
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Verdict: {data.get('verdict')}")
        print(f"Score: {data.get('credibility_score')}")
        print(f"Inference Time: {data.get('inference_time_ms')} ms")
        return resp.status_code == 200 and 'verdict' in data
    except Exception as e:
        print(f"Request failed: {e}")
        return False

def main():
    # Start server
    print("Starting API Server...")
    # Using specific python to ensure environment
    server_process = subprocess.Popen([sys.executable, "src/api_server.py"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
    
    try:
        if wait_for_server("http://127.0.0.1:8000/health"):
            h_ok = test_health()
            v_ok = test_verify()
            
            if h_ok and v_ok:
                print("\nSUCCESS: System Verified.")
            else:
                print("\nFAILURE: Verification failed.")
        else:
            print("Could not connect to server.")
            # Print stderr
            print("Server Stderr:")
            print(server_process.stderr.read())
            
    finally:
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main()
