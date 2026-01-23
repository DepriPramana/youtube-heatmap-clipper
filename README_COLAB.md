# 🎬 YouTube Heatmap Clipper - Google Colab Guide

## 🚀 Quick Start (3 Minutes)

### Option 1: Using Colab Notebook (Recommended)

1. **Open the Colab Notebook**:
   - Upload `colab_setup.ipynb` to Google Colab
   - Or open directly: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DepriPramana/youtube-heatmap-clipper/blob/main/colab_setup.ipynb)

2. **Run All Cells** (Runtime → Run All)
   - Wait 2-3 minutes for setup
   - Click the ngrok link when it appears

3. **Start Clipping!**
   - Paste your YouTube URL
   - Scan or select custom time
   - Create clips!

---

## 🛠️ Manual Setup (Alternative)

If you prefer setting up manually in Colab:

### Step 1: Install FFmpeg
```bash
!apt-get update -qq
!apt-get install -y -qq ffmpeg
!ffmpeg -version
```

### Step 2: Clone Repository
```bash
!git clone https://github.com/DepriPramana/youtube-heatmap-clipper.git
%cd youtube-heatmap-clipper
```

### Step 3: Install Dependencies
```bash
!pip install -q -r requirements.txt
```

### Step 4: Run the App
```python
!python webapp.py
```

---

## 💡 Tips for Colab

### Performance Optimization

**For Faster Processing**:
- Use smaller Whisper model (`tiny` or `base`)
- Process fewer clips at once (max 3-5)
- Enable GPU runtime: Runtime → Change runtime type → GPU

**For Better Quality**:
- Use `small` or `medium` Whisper model
- Increase padding to 15-20 seconds
- Use higher resolution output

### Storage Management

Colab has limited disk space (~100GB). To manage:

```bash
# Check available space
!df -h

# Clean up clips after downloading
!rm -rf clips/*

# Remove temp files
!rm -f temp_*.mkv temp_*.mp4 temp_*.srt
```

### Session Limits

- **Free Colab**: 12 hours max per session
- **Colab Pro**: 24 hours max per session
- Save your clips before session expires!

---

## 🎯 Recommended Settings for Colab

### For Instagram Reels / TikTok:
```
Ratio: 9:16
Crop: Default
Padding: 10s
Subtitle: Yes
Whisper Model: small
Max Clips: 3-5
```

### For YouTube Shorts:
```
Ratio: 9:16
Crop: Split Left/Right (if gaming)
Padding: 15s
Subtitle: Yes
Whisper Model: small
Max Clips: 3-5
```

### For Quick Testing:
```
Ratio: 9:16
Crop: Default
Padding: 5s
Subtitle: No
Max Clips: 1-2
```

---

## ⚠️ Common Issues & Solutions

### Issue: "Ngrok tunnel disconnected"
**Solution**: Re-run the cell with `!python webapp.py`. Ngrok tunnels expire after some time.

### Issue: "FFmpeg not found"
**Solution**: Make sure you ran the FFmpeg installation cell first.

### Issue: "Out of memory"
**Solutions**:
- Use smaller Whisper model (tiny instead of large)
- Process fewer clips
- Restart runtime: Runtime → Restart runtime
- Upgrade to Colab Pro for more RAM

### Issue: "Video download failed"
**Solutions**:
- Check if video is available in your region
- Update yt-dlp: `!pip install -U yt-dlp`
- Try a different video

### Issue: "Session disconnected"
**Prevention**:
- Keep the Colab tab active
- Enable "Stay Connected" browser extension
- Use Colab Pro for longer sessions

---

## 📥 Downloading Your Clips

### Method 1: From Web Interface
1. Click the "Download" button next to your clip
2. Browser will download directly

### Method 2: From Colab Files
1. Click folder icon in left sidebar
2. Navigate to `clips/` folder
3. Right-click → Download

### Method 3: Bulk Download via Code
```python
from google.colab import files
import os

# Download all clips
for clip in os.listdir('clips'):
    if clip.endswith('.mp4'):
        files.download(f'clips/{clip}')
```

### Method 4: Save to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy clips to Drive
!cp clips/*.mp4 /content/drive/MyDrive/youtube_clips/
```

---

## 🎓 Advanced Usage

### Using Custom Ngrok Token

If you have your own ngrok account:

```python
import os
os.environ['NGROK_AUTH_TOKEN'] = 'your_token_here'
```

Then run the app normally.

### Batch Processing Multiple Videos

```python
videos = [
    'https://www.youtube.com/watch?v=VIDEO_ID_1',
    'https://www.youtube.com/watch?v=VIDEO_ID_2',
    'https://www.youtube.com/watch?v=VIDEO_ID_3',
]

# Process each via the web interface or CLI
for url in videos:
    # Use web interface for each URL
    print(f"Process this next: {url}")
```

### Using GPU for Faster Whisper

Enable GPU in Colab (Runtime → Change runtime type → GPU), then the app will automatically use GPU for faster transcription.

---

## 🔒 Security Notes

### Ngrok Token
- The default ngrok token is for demo purposes only
- For production use, get your own free token: https://ngrok.com
- Set via environment variable for security

### Video Privacy
- Downloaded video segments are temporary
- Clean up clips after downloading
- Don't process copyrighted content without permission

---

## 📊 Resource Monitoring

### Check GPU Status
```python
!nvidia-smi
```

### Monitor RAM Usage
```python
import psutil
print(f"RAM Usage: {psutil.virtual_memory().percent}%")
```

### Check Processing Time
The web interface shows real-time progress for each clip.

---

## 🆘 Getting Help

1. **Check the main README**: [English](README_EN.md) | [Bahasa](README.md)
2. **GitHub Issues**: Report bugs or request features
3. **Troubleshooting**: See sections above

---

## 🎁 Pro Tips

1. **Start Small**: Test with 1 clip first before batch processing
2. **Save Early**: Download clips as soon as they're ready
3. **Monitor Resources**: Keep an eye on RAM and disk space
4. **Clean Regularly**: Remove old clips to save space
5. **Use GPU**: Enable GPU for 3-4x faster subtitle generation

---

## 📝 Changelog

### Version 2.0 (Colab-Ready)
- ✅ Fixed critical bug (undefined port variable)
- ✅ Added Colab environment detection
- ✅ Improved ngrok integration
- ✅ Added complete Colab notebook
- ✅ Better error handling
- ✅ Enhanced documentation

---

**Happy Clipping! 🎬✨**
