import httpx
import json

base_url = "https://huggingface.co/jaredpalmer/kev-4b/raw/qwen3/"

print("Fetching adapter_config.json...")
r = httpx.get(base_url + "adapter_config.json")
if r.status_code == 200:
    print(json.dumps(r.json(), indent=2))

print("\nFetching provenance.json...")
r = httpx.get(base_url + "provenance.json")
if r.status_code == 200:
    print(json.dumps(r.json(), indent=2))

print("\nFetching training_config.json...")
r = httpx.get(base_url + "training_config.json")
if r.status_code == 200:
    print(json.dumps(r.json(), indent=2))
