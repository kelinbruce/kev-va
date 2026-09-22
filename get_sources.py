import httpx
import json

print("Fetching jaredpalmer/kev commit...")
try:
    r = httpx.get("https://api.github.com/repos/jaredpalmer/kev/commits/90990a5fac2995b9faa3190f7d437e84f2067768")
    if r.status_code == 200:
        print(f"KEV commit fetched: {r.json()['sha']}")
    else:
        print(f"Failed to fetch KEV commit: {r.status_code} {r.text}")
except Exception as e:
    print(e)

print("\nFetching HF model info...")
try:
    r = httpx.get("https://huggingface.co/api/models/jaredpalmer/kev-4b/revision/qwen3")
    if r.status_code == 200:
        data = r.json()
        print(f"HF model revision: {data.get('sha')}")
        print("Files:")
        for f in data.get('siblings', []):
            print(f"  {f['rfilename']}")
    else:
        print(f"Failed to fetch HF model: {r.status_code} {r.text}")
except Exception as e:
    print(e)
