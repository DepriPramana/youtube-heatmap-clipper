import os
import json
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

from flask import Flask, jsonify, render_template, request, send_from_directory

import run as core

# --- Add this import ---
from pyngrok import ngrok
import google.generativeai as genai
import openai

app = Flask(__name__, static_folder="static", template_folder="templates")

jobs_lock = threading.Lock()
jobs = {}
preview_lock = threading.Lock()
preview_cache = {}


def now_ms():
    return int(time.time() * 1000)


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def parse_time_to_seconds(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + int(float(sec))
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + int(float(sec))
    return None


def set_job(job_id, **patch):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(patch)


def add_log(job_id, line):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["logs"].append(line)
        if len(job["logs"]) > 300:
            job["logs"] = job["logs"][-300:]


def list_outputs(job_dir):
    if not os.path.isdir(job_dir):
        return []
    items = []
    for name in os.listdir(job_dir):
        path = os.path.join(job_dir, name)
        if os.path.isfile(path):
            if name.lower().endswith(".mp4"):
               items.append({"name": name, "size": os.path.getsize(path), "type": "video"})
            elif name.lower().endswith(".srt"):
               items.append({"name": name, "size": os.path.getsize(path), "type": "subtitle"})
    
    items.sort(key=lambda x: x["name"])
    return items


def run_job(job_id, payload):
    started = now_ms()
    try:
        set_job(job_id, status="running", started_at=started)

        url = (payload.get("url") or "").strip()
        if not url:
            raise ValueError("URL kosong")

        crop = payload.get("crop") or "default"
        ratio = payload.get("ratio") or "9:16"
        subtitle = bool(payload.get("subtitle"))
        whisper_model = payload.get("whisper_model") or "small"
        subtitle_font = payload.get("subtitle_font") or "Arial"
        subtitle_location = payload.get("subtitle_location") or "bottom"
        subtitle_fontsdir = payload.get("subtitle_fontsdir") or None
        if not subtitle_fontsdir and os.path.isdir("fonts"):
            subtitle_fontsdir = "fonts"
        padding = safe_int(payload.get("padding"), 10)
        max_clips = safe_int(payload.get("max_clips"), 10)
        mode = payload.get("mode") or "heatmap"
        custom_crop = payload.get("custom_crop")
        set_job(job_id, subtitle_enabled=subtitle)

        core.WHISPER_MODEL = whisper_model
        core.SUBTITLE_FONT = subtitle_font
        core.SUBTITLE_FONTS_DIR = subtitle_fontsdir
        core.SUBTITLE_LOCATION = subtitle_location
        core.PADDING = max(0, padding if padding is not None else 10)
        core.set_ratio_preset(ratio)

        job_dir = os.path.join("clips", job_id)
        os.makedirs(job_dir, exist_ok=True)
        core.OUTPUT_DIR = job_dir

        core.cek_dependensi._args = SimpleNamespace(no_update_ytdlp=True)
        ok = core.cek_dependensi(install_whisper=subtitle, fatal=False)
        if not ok:
            raise RuntimeError("FFmpeg tidak ketemu")

        video_id = core.extract_video_id(url)
        if not video_id:
            raise ValueError("URL YouTube invalid")

        total_duration = core.get_duration(video_id)

        targets = []
        picked = payload.get("segments")
        if isinstance(picked, list) and len(picked) > 0:
            add_log(job_id, f"Pakai {len(picked)} segment yang dipilih...")
            for seg in picked:
                try:
                    start = float(seg.get("start"))
                    dur = float(seg.get("duration"))
                    score = float(seg.get("score", 1.0))
                    item_custom_crop = seg.get("custom_crop") # Extract custom crop if exists
                except Exception:
                    continue
                if dur <= 0:
                    continue
                
                target_item = {"start": start, "duration": dur, "score": score}
                if item_custom_crop:
                    target_item["custom_crop"] = item_custom_crop
                targets.append(target_item)
            if not targets:
                raise ValueError("Segment pilihan invalid")
        elif mode == "custom":
            start_s = parse_time_to_seconds(payload.get("start"))
            end_s = parse_time_to_seconds(payload.get("end"))
            if start_s is None or end_s is None:
                raise ValueError("Start/End belum diisi")
            if end_s <= start_s:
                raise ValueError("End harus lebih besar dari Start")
            targets = [{"start": float(start_s), "duration": float(end_s - start_s), "score": 1.0}]
        else:
            add_log(job_id, "Scan heatmap...")
            segments = core.ambil_most_replayed(video_id)
            if not segments:
                raise RuntimeError("Tidak ada heatmap/Most Replayed data")
            targets = segments[: max(1, max_clips or 10)]

        watermark_text = payload.get("watermark_text")
        watermark_pos = payload.get("watermark_pos")

        set_job(job_id, total=len(targets), done=0, status_text="processing")

        def event_hook(kind, data):
            if kind != "stage" or not isinstance(data, dict):
                return
            stage = data.get("stage") or ""
            clip_index = safe_int(data.get("clip_index"), 0) or 0
            set_job(job_id, stage=stage, stage_at=now_ms(), stage_clip=clip_index)

        success = 0
        for idx, item in enumerate(targets, start=1):
            set_job(job_id, current=idx, status_text=f"clip {idx}/{len(targets)}")
            
            # Determine crop mode for this specific clip
            this_crop_mode = crop
            this_custom_crop = custom_crop  # global setting

            # If segment has specific custom_crop, use it
            if item.get("custom_crop"):
                this_crop_mode = "custom_manual"
                this_custom_crop = item.get("custom_crop")
                
            ok = core.proses_satu_clip(
                video_id, item, idx, total_duration, this_crop_mode, subtitle, 
                watermark_text=watermark_text, watermark_pos=watermark_pos, 
                custom_crop=this_custom_crop,
                event_hook=event_hook
            )
            if ok:
                success += 1
            set_job(job_id, done=idx, success=success, outputs=list_outputs(job_dir))

        set_job(job_id, status="done", finished_at=now_ms(), outputs=list_outputs(job_dir))
    except Exception as e:
        set_job(job_id, status="error", error=str(e), finished_at=now_ms())


@app.get("/")
def index():
    return render_template("index.html")

@app.get("/assets/fonts/<path:filename>")
def serve_font(filename):
    return send_from_directory("fonts", filename, as_attachment=False)


def get_preview(url):
    key = url.strip()
    if not key:
        raise ValueError("URL kosong")

    with preview_lock:
        cached = preview_cache.get(key)
        if cached:
            return cached

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--skip-download",
        "-J",
        key,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "Gagal ambil metadata").strip())

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

    with preview_lock:
        preview_cache[key] = preview
        if len(preview_cache) > 200:
            preview_cache.clear()

    return preview


@app.post("/api/preview")
def api_preview():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    try:
        preview = get_preview(url)
        return jsonify({"ok": True, "preview": preview})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/scan")
def api_scan():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    video_id = core.extract_video_id(url)
    if not video_id:
        return jsonify({"ok": False, "error": "URL YouTube invalid"}), 400

    core.cek_dependensi._args = SimpleNamespace(no_update_ytdlp=True)
    ok = core.cek_dependensi(install_whisper=False, fatal=False)
    if not ok:
        return jsonify({"ok": False, "error": "FFmpeg tidak ketemu"}), 400

    segments = core.ambil_most_replayed(video_id)
    total = core.get_duration(video_id)
    return jsonify({"ok": True, "video_id": video_id, "duration": total, "segments": segments})


@app.post("/api/clip")
def api_clip():
    payload = request.get_json(silent=True) or {}
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": now_ms(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "total": 0,
            "done": 0,
            "success": 0,
            "current": 0,
            "status_text": "",
            "stage": "",
            "stage_at": None,
            "stage_clip": 0,
            "subtitle_enabled": False,
            "outputs": [],
            "logs": [],
        }

    t = threading.Thread(target=run_job, args=(job_id, payload), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route('/api/preview-frame', methods=['POST'])
def preview_frame():
    data = request.json or {}
    url = data.get("url")
    
    if not url:
        return jsonify({"ok": False, "error": "URL required"})
        
    try:
        # Create preview directory if not exists
        preview_dir = os.path.join("static", "previews")
        os.makedirs(preview_dir, exist_ok=True)
        
        # Hash URL to get consistent filename
        import hashlib
        url_hash = hashlib.md5(url.encode()).hexdigest()
        preview_filename = f"preview_{url_hash}.jpg"
        preview_path = os.path.join(preview_dir, preview_filename)
        
        # Check if preview already exists
        if os.path.exists(preview_path):
             return jsonify({"ok": True, "url": f"/static/previews/{preview_filename}"})
             
        # If not, we need to download a snippet or get direct URL
        # For simplicity, let's use yt-dlp to get the direct video URL
        # and then ffmpeg to grab a frame from the middle duration
        
        # 1. Get video info (direct URL and duration)
        # Force select a clear format (up to 720p mp4) to ensure we get a direct URL
        cmd_info = [
            "yt-dlp", "--dump-json", "--skip-download", 
            "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
            url
        ]
        info_json = subprocess.check_output(cmd_info).decode()
        info = json.loads(info_json)
        
        video_url = info.get("url")
        if not video_url:
             # Fallback: check requested_formats if it's a split stream
             if "requested_formats" in info:
                 # Use the first stream (usually video)
                 video_url = info["requested_formats"][0].get("url")
        
        duration = info.get("duration", 60)
        
        if not video_url:
            print(f"[ERROR] No video URL found in info. Keys: {info.keys()}")
            raise Exception("Could not get video URL")
            
        # 2. Extract frame from specific timestamp or 20% default
        ts_req = data.get("timestamp")
        timestamp = int(duration * 0.2)
        
        if ts_req is not None:
             try:
                 ts_val = int(float(ts_req))
                 timestamp = max(0, min(ts_val, int(duration - 1)))
                 preview_filename = f"preview_{url_hash}_t{timestamp}.jpg"
                 preview_path = os.path.join(preview_dir, preview_filename)
             except:
                 pass
        
        if os.path.exists(preview_path):
             return jsonify({"ok": True, "url": f"/static/previews/{preview_filename}"})
             
        # ROBUST STRATEGY: Download snippet using yt-dlp, then extract frame
        # This handles HLS, DASH, and Auth Cookies automatically
        temp_snippet = f"temp_snippet_{url_hash}_{timestamp}.mp4"
        start_dl = timestamp
        end_dl = timestamp + 3 # Download 3 seconds
        
        cmd_dl_snippet = [
            "yt-dlp", "--force-ipv4", "--quiet", "--no-warnings",
            "--download-sections", f"*{start_dl}-{end_dl}",
            "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
             "--force-keyframes-at-cuts",
            "-o", temp_snippet,
            url
        ]
        
        print(f"[PREVIEW] Downloading snippet: {start_dl}s - {end_dl}s")
        subprocess.check_call(cmd_dl_snippet)
        
        if not os.path.exists(temp_snippet):
             raise Exception("Failed to download video snippet")
             
        # Extract frame from the local snippet
        # We take the first frame of the snippet
        cmd_ffmpeg = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", temp_snippet,
            "-frames:v", "1",
            "-q:v", "2",
            preview_path
        ]
        subprocess.check_call(cmd_ffmpeg)
        
        # Cleanup
        try:
             os.remove(temp_snippet)
        except:
             pass
        
        return jsonify({"ok": True, "url": f"/static/previews/{preview_filename}"})
        
    except Exception as e:
        print(f"Preview Error: {e}")
        return jsonify({"ok": False, "error": str(e)})


@app.get("/api/job/<job_id>")
def api_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "Job not found"}), 404
        return jsonify({"ok": True, "job": job})


@app.get("/clips/<job_id>/<path:filename>")
def serve_clip(job_id, filename):
    job_dir = os.path.join("clips", job_id)
    return send_from_directory(job_dir, filename, as_attachment=True)


@app.post("/api/generate_metadata")
def api_generate_metadata():
    data = request.get_json(silent=True) or {}
    
    provider = data.get("provider", "gemini")
    api_key = data.get("api_key", "").strip()
    model_name = data.get("model", "").strip()
    transcript = data.get("transcript", "").strip()
    
    if not api_key:
        return jsonify({"ok": False, "error": "API Key is required"}), 400
        
    if not transcript:
        return jsonify({"ok": False, "error": "Transcript is empty"}), 400
        
    prompt = f"""
    Based on the following video transcript, generate:
    1. A catchy, viral Title (max 60 chars)
    2. A compelling Description (max 300 chars)
    3. A list of 5-10 comma-separated Keywords/Tags

    Transcript:
    {transcript[:2000]}
    
    Return the response as a valid JSON object with keys: "title", "description", "keywords".
    Do not wrap in markdown code blocks.
    """
    
    try:
        result_json = {}
        
        if provider == "gemini":
            genai.configure(api_key=api_key)
            # Use user provided model or default to gemini-1.5-flash
            target_model = model_name if model_name else "gemini-1.5-flash"
            model = genai.GenerativeModel(target_model)
            response = model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(text)
            
        elif provider == "grok":
            if not model_name: model_name = "grok-beta"
            client = openai.OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=api_key
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(text)
            
        elif provider == "openrouter":
            if not model_name: model_name = "mistralai/mixtral-8x7b-instruct"
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1", 
                api_key=api_key
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(text)
            
        else:
             return jsonify({"ok": False, "error": f"Unknown provider: {provider}"}), 400

        return jsonify({
            "ok": True,
            "data": result_json
        })
        
    except Exception as e:
        print(f"AI Generation Error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# --- Execution block ---
if __name__ == "__main__":
    import os
    
    # Get ngrok token from environment variable or use default
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "38doKfu9q7H8l6p1vlZelDtrYaG_6Wq7JxQNzd3JF1tZ64gFX")
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("🔍 Detected Google Colab environment")
    except ImportError:
        IN_COLAB = False
        print("🖥️  Running in local environment")
    
    port = 5000
    
    # Setup ngrok for public URL (useful for Colab)
    if ngrok_token:
        try:
            ngrok.set_auth_token(ngrok_token)
            public_url = ngrok.connect(port)
            print(f"\n{'='*60}")
            print(f"🌐 Public URL: {public_url}")
            print(f"{'='*60}\n")
            
            if IN_COLAB:
                # In Colab, display clickable link
                from IPython.display import display, HTML
                display(HTML(f'<h3><a href="{public_url}" target="_blank">🚀 Click here to open the app</a></h3>'))
        except Exception as e:
            print(f"⚠️  Ngrok setup failed: {e}")
            print(f"📍 App will be available at: http://127.0.0.1:{port}")
    else:
        print(f"📍 App running at: http://127.0.0.1:{port}")
    
    # Start the Flask app
    print(f"\n🎬 YouTube Heatmap Clipper is starting...")
    app.run(host='0.0.0.0', port=port, use_reloader=False)