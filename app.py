import threading
import time
import re
import shutil
import zipfile
import io
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import yt_dlp
import whisper

app = Flask(__name__)

DOWNLOAD_DIR = Path("downloads")
OUTPUT_DIR = Path("transcriptions")
VIDEO_DIR = Path("videos")
AUDIO_DIR = Path("audios")
for d in [DOWNLOAD_DIR, OUTPUT_DIR, VIDEO_DIR, AUDIO_DIR]:
    d.mkdir(exist_ok=True)

def find_ffmpeg():
    if shutil.which("ffmpeg"):
        return None
    winget_base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_base.exists():
        for p in winget_base.rglob("ffmpeg.exe"):
            return str(p.parent)
    return None

FFMPEG_LOCATION = find_ffmpeg()
jobs = {}

def base_opts():
    o = {'quiet': True, 'no_warnings': True}
    if FFMPEG_LOCATION:
        o['ffmpeg_location'] = FFMPEG_LOCATION
    return o

def safe_title(title, max_len=80):
    return "".join(c for c in title if c.isalnum() or c in " _-").strip()[:max_len]

def expand_playlist(url):
    opts = base_opts()
    opts['extract_flat'] = 'in_playlist'
    opts['skip_download'] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    urls = []
    if info is None:
        return urls
    if 'entries' in info:
        for entry in (info['entries'] or []):
            if entry and entry.get('id'):
                urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
    elif info.get('id'):
        urls.append(f"https://www.youtube.com/watch?v={info['id']}")
    return urls

def extract_plain_video_urls(text):
    pattern = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w\-]+)'
    ids = re.findall(pattern, text)
    seen = set()
    urls = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            urls.append(f"https://www.youtube.com/watch?v={vid}")
    return urls

# ── TRANSCRIPTION ────────────────────────────────────────────

def dl_audio_for_transcription(url, job_id, index):
    opts = base_opts()
    opts.update({
        'format': 'bestaudio/best',
        'outtmpl': str(DOWNLOAD_DIR / f"{job_id}_{index}.%(ext)s"),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', f'video_{index}')
    return DOWNLOAD_DIR / f"{job_id}_{index}.mp3", title

def transcribe_audio(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language="es")
    return result["text"]

def process_transcription(job_id, urls, model_name):
    jobs[job_id].update({"status": "running", "total": len(urls), "done": 0,
                         "results": [], "errors": [], "current": f"{len(urls)} videos. Iniciando..."})
    for i, url in enumerate(urls):
        try:
            jobs[job_id]["current"] = f"Descargando {i+1}/{len(urls)}..."
            audio_path, title = dl_audio_for_transcription(url, job_id, i)
            jobs[job_id]["current"] = f"Transcribiendo: {title[:50]}..."
            text = transcribe_audio(audio_path, model_name)
            st = safe_title(title)
            txt_path = OUTPUT_DIR / f"{st}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Título: {title}\nURL: {url}\n{'='*60}\n\n{text}")
            jobs[job_id]["results"].append({"title": title, "url": url, "file": str(txt_path), "filename": txt_path.name})
            if audio_path.exists():
                audio_path.unlink()
        except Exception as e:
            jobs[job_id]["errors"].append({"url": url, "error": str(e)})
        jobs[job_id]["done"] = i + 1
    combined = OUTPUT_DIR / f"COMPLETO_{time.strftime('%Y-%m-%d_%H-%M')}.txt"
    with open(combined, "w", encoding="utf-8") as f:
        f.write(f"TRANSCRIPCIONES - {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n\n")
        for r in jobs[job_id]["results"]:
            with open(r['file'], "r", encoding="utf-8") as rf:
                content = rf.read()
            parts = content.split("="*60)
            body = parts[-1].strip() if len(parts) > 1 else content
            f.write(f"## {r['title']}\nURL: {r['url']}\n\n{body}\n\n{'-'*60}\n\n")
    jobs[job_id].update({"combined_file": str(combined), "combined_filename": combined.name,
                         "status": "done", "current": "¡Completado!"})

# ── VIDEO DOWNLOAD ───────────────────────────────────────────

def process_video(job_id, urls):
    jobs[job_id].update({"status": "running", "total": len(urls), "done": 0,
                         "results": [], "errors": [], "current": f"{len(urls)} videos. Iniciando..."})
    for i, url in enumerate(urls):
        try:
            jobs[job_id]["current"] = f"Descargando video {i+1}/{len(urls)}..."
            opts = base_opts()
            opts.update({
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': str(VIDEO_DIR / f"{job_id}_{i}.%(ext)s"),
                'merge_output_format': 'mp4',
            })
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', f'video_{i}')
            st = safe_title(title)
            src = VIDEO_DIR / f"{job_id}_{i}.mp4"
            dst = VIDEO_DIR / f"{st}.mp4"
            if src.exists():
                src.rename(dst)
            jobs[job_id]["results"].append({"title": title, "url": url, "file": str(dst), "filename": dst.name})
        except Exception as e:
            jobs[job_id]["errors"].append({"url": url, "error": str(e)})
        jobs[job_id]["done"] = i + 1
    jobs[job_id].update({"status": "done", "current": "¡Completado!"})

# ── AUDIO DOWNLOAD ───────────────────────────────────────────

def process_audio(job_id, urls):
    jobs[job_id].update({"status": "running", "total": len(urls), "done": 0,
                         "results": [], "errors": [], "current": f"{len(urls)} audios. Iniciando..."})
    for i, url in enumerate(urls):
        try:
            jobs[job_id]["current"] = f"Descargando audio {i+1}/{len(urls)}..."
            opts = base_opts()
            opts.update({
                'format': 'bestaudio/best',
                'outtmpl': str(AUDIO_DIR / f"{job_id}_{i}.%(ext)s"),
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            })
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', f'audio_{i}')
            st = safe_title(title)
            src = AUDIO_DIR / f"{job_id}_{i}.mp3"
            dst = AUDIO_DIR / f"{st}.mp3"
            if src.exists():
                src.rename(dst)
            jobs[job_id]["results"].append({"title": title, "url": url, "file": str(dst), "filename": dst.name})
        except Exception as e:
            jobs[job_id]["errors"].append({"url": url, "error": str(e)})
        jobs[job_id]["done"] = i + 1
    jobs[job_id].update({"status": "done", "current": "¡Completado!"})

# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/expand_playlist", methods=["POST"])
def api_expand_playlist():
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL vacía"}), 400
    try:
        urls = expand_playlist(url)
        if not urls:
            return jsonify({"error": "No se encontraron videos"}), 400
        return jsonify({"urls": urls, "count": len(urls)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def make_job(urls, target, model_name=None):
    job_id = f"{target}_{int(time.time())}"
    jobs[job_id] = {"status": "pending", "current": "Iniciando...", "total": 0,
                    "done": 0, "results": [], "errors": [], "type": target}
    if target == "transcribe":
        t = threading.Thread(target=process_transcription, args=(job_id, urls, model_name))
    elif target == "video":
        t = threading.Thread(target=process_video, args=(job_id, urls))
    elif target == "audio":
        t = threading.Thread(target=process_audio, args=(job_id, urls))
    t.daemon = True
    t.start()
    return job_id

@app.route("/start", methods=["POST"])
def start_job():
    data = request.json
    urls = extract_plain_video_urls(data.get("urls_text", ""))
    if not urls:
        return jsonify({"error": "No se encontraron URLs válidas."}), 400
    job_id = make_job(urls, "transcribe", data.get("model", "base"))
    return jsonify({"job_id": job_id, "url_count": len(urls)})

@app.route("/start_video", methods=["POST"])
def start_video():
    urls = extract_plain_video_urls(request.json.get("urls_text", ""))
    if not urls:
        return jsonify({"error": "No se encontraron URLs válidas."}), 400
    job_id = make_job(urls, "video")
    return jsonify({"job_id": job_id, "url_count": len(urls)})

@app.route("/start_audio", methods=["POST"])
def start_audio():
    urls = extract_plain_video_urls(request.json.get("urls_text", ""))
    if not urls:
        return jsonify({"error": "No se encontraron URLs válidas."}), 400
    job_id = make_job(urls, "audio")
    return jsonify({"job_id": job_id, "url_count": len(urls)})

@app.route("/status/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify(jobs[job_id])

def cleanup_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return
    for r in job.get("results", []):
        p = Path(r["file"])
        if p.exists():
            p.unlink()
    cf = job.get("combined_file")
    if cf and Path(cf).exists():
        Path(cf).unlink()
    jobs.pop(job_id, None)

@app.route("/download/<job_id>")
def download_combined(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "No disponible"}), 400
    f = jobs[job_id].get("combined_file")
    if not f or not Path(f).exists():
        return jsonify({"error": "Archivo no encontrado"}), 404
    return send_file(f, as_attachment=True, download_name=jobs[job_id]["combined_filename"])

@app.route("/download_zip/<job_id>")
def download_zip(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "done":
        return jsonify({"error": "No disponible"}), 400
    results = jobs[job_id]["results"]
    job_type = jobs[job_id].get("type", "transcribe")
    buf = io.BytesIO()
    # For videos, use ZIP_STORED (no compression, they're already compressed)
    compress = zipfile.ZIP_STORED if job_type == "video" else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buf, "w", compress) as zf:
        for r in results:
            if Path(r["file"]).exists():
                zf.write(r["file"], arcname=r["filename"])
        cf = jobs[job_id].get("combined_file")
        if cf and Path(cf).exists():
            zf.write(cf, arcname=jobs[job_id]["combined_filename"])
    buf.seek(0)
    prefix = {"transcribe": "transcripciones", "video": "videos", "audio": "audios"}.get(job_type, "archivos")
    zip_name = f"{prefix}_{time.strftime('%Y-%m-%d_%H-%M')}.zip"
    response = send_file(buf, as_attachment=True, download_name=zip_name, mimetype="application/zip")
    cleanup_job(job_id)
    return response

@app.route("/download_single/<job_id>/<int:index>")
def download_single(job_id, index):
    if job_id not in jobs:
        return jsonify({"error": "Job no encontrado"}), 404
    results = jobs[job_id]["results"]
    if index >= len(results):
        return jsonify({"error": "Índice inválido"}), 400
    r = results[index]
    if not Path(r["file"]).exists():
        return jsonify({"error": "Archivo no encontrado en disco"}), 404
    return send_file(r["file"], as_attachment=True, download_name=r["filename"])

if __name__ == "__main__":
    msg = f"ffmpeg en: {FFMPEG_LOCATION}" if FFMPEG_LOCATION else "ffmpeg detectado en PATH"
    print(f"\n🎙️  YouTube Transcriber → http://localhost:5000\n   {msg}\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
