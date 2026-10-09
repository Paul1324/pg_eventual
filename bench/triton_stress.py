import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v2/models/eventual_text_embedder/versions/1/infer"
results = {"ok": 0, "errors": {}}
lock = threading.Lock()


def one(i):
    body = {"inputs": [{"name": "TEXT", "shape": [1], "datatype": "BYTES", "data": [f"Sentence {i} about GPUs."]},
                       {"name": "NORMALIZE", "shape": [1], "datatype": "BOOL", "data": [True]}]}
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            json.load(response)
        with lock:
            results["ok"] += 1
    except Exception as error:
        key = type(error).__name__ + ": " + str(error)[:80]
        with lock:
            results["errors"][key] = results["errors"].get(key, 0) + 1


def worker(n, offset):
    for i in range(n):
        one(offset + i)


threads_n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
per_thread = int(sys.argv[2]) if len(sys.argv) > 2 else 100
start = time.time()
threads = [threading.Thread(target=worker, args=(per_thread, t * per_thread)) for t in range(threads_n)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(json.dumps(results, indent=1), f"{time.time() - start:.1f}s")
