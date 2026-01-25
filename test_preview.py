
import sys
import subprocess
import json
import os

def get_preview(url):
    key = url.strip()
    if not key:
        raise ValueError("URL kosong")

    print(f"Testing preview for: {key}")
    
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--skip-download",
        "-J",
        key,
    ]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        print(f"Subprocess failed: {e}")
        return

    if res.returncode != 0:
        print("yt-dlp returned non-zero exit code")
        print("Stderr:", res.stderr)
        return

    try:
        raw = json.loads(res.stdout)
        item = raw["entries"][0] if isinstance(raw, dict) and "entries" in raw and raw.get("entries") else raw
        
        preview = {
            "title": item.get("title"),
            "thumbnail": item.get("thumbnail"),
            "uploader": item.get("uploader"),
            "duration": item.get("duration"),
            "webpage_url": item.get("webpage_url") or key,
            "id": item.get("id"),
        }
        print("SUCCESS! Preview extracted:")
        print(json.dumps(preview, indent=2))
    except Exception as e:
        print(f"JSON parsing failed: {e}")

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    if len(sys.argv) > 1:
        url = sys.argv[1]
    get_preview(url)
