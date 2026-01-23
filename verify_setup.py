#!/usr/bin/env python3
"""
Quick verification script to check if all dependencies are installed
and the app can run in Colab environment.
"""

import sys
import os

def check_imports():
    """Check if all required packages can be imported."""
    results = {}
    
    packages = {
        'flask': 'Flask',
        'requests': 'Requests',
        'pyngrok': 'Ngrok',
    }
    
    optional_packages = {
        'faster_whisper': 'Faster-Whisper (AI Subtitle)',
        'google.colab': 'Google Colab (only in Colab)',
    }
    
    print("=" * 60)
    print("[*] Checking Required Dependencies")
    print("=" * 60)
    
    # Check required packages
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            results[name] = "✅ OK"
            print(f"[OK] {name}: Installed")
        except ImportError:
            results[name] = "❌ Missing"
            print(f"[!!] {name}: NOT INSTALLED")
            all_ok = False
    
    print("\n" + "=" * 60)
    print("[*] Checking Optional Dependencies")
    print("=" * 60)
    
    # Check optional packages
    for module, name in optional_packages.items():
        try:
            __import__(module)
            results[name] = "✅ OK"
            print(f"[OK] {name}: Installed")
        except ImportError:
            results[name] = "⚠️  Not installed (optional)"
            print(f"[--] {name}: Not installed (optional)")
    
    return all_ok, results

def check_ffmpeg():
    """Check if FFmpeg is available."""
    import shutil
    print("\n" + "=" * 60)
    print("[*] Checking FFmpeg")
    print("=" * 60)
    
    if shutil.which("ffmpeg"):
        print("[OK] FFmpeg: Installed and accessible")
        # Get version
        import subprocess
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
            return True
        except Exception:
            pass
    else:
        print("[!!] FFmpeg: NOT FOUND")
        print("   Install with: !apt-get install -y ffmpeg (Colab)")
        return False
    
    return True

def check_environment():
    """Detect current environment."""
    print("\n" + "=" * 60)
    print("[*] Environment Detection")
    print("=" * 60)
    
    # Check if running in Colab
    try:
        import google.colab
        print("[OK] Running in: Google Colab")
        return "colab"
    except ImportError:
        print("[OK] Running in: Local/Other environment")
        return "local"

def check_webapp():
    """Check if webapp.py exists and is valid."""
    print("\n" + "=" * 60)
    print("[*] Checking Application Files")
    print("=" * 60)
    
    if os.path.exists("webapp.py"):
        print("[OK] webapp.py: Found")
        # Quick syntax check
        try:
            with open("webapp.py", "r", encoding="utf-8") as f:
                compile(f.read(), "webapp.py", "exec")
            print("[OK] webapp.py: Syntax OK")
            return True
        except SyntaxError as e:
            print(f"[!!] webapp.py: Syntax Error at line {e.lineno}")
            return False
    else:
        print("[!!] webapp.py: NOT FOUND")
        return False

def main():
    """Run all checks."""
    print("\n")
    print("YouTube Heatmap Clipper - Setup Verification")
    print("=" * 60)
    print()
    
    deps_ok, _ = check_imports()
    ffmpeg_ok = check_ffmpeg()
    env = check_environment()
    webapp_ok = check_webapp()
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_ok = deps_ok and ffmpeg_ok and webapp_ok
    
    if all_ok:
        print("\n[OK] ALL CHECKS PASSED!")
        print("\n[*] You're ready to run the app:")
        print("   python webapp.py")
        print("\n   Or use the Colab notebook: colab_setup.ipynb")
    else:
        print("\n[!] SOME CHECKS FAILED")
        print("\n[FIX] To fix:")
        if not deps_ok:
            print("   1. Install dependencies: pip install -r requirements.txt")
        if not ffmpeg_ok:
            print("   2. Install FFmpeg:")
            if env == "colab":
                print("      !apt-get install -y ffmpeg")
            else:
                print("      - Windows: winget install Gyan.FFmpeg")
                print("      - Mac: brew install ffmpeg")
                print("      - Linux: sudo apt-get install ffmpeg")
        if not webapp_ok:
            print("   3. Download webapp.py from repository")
    
    print("\n" + "=" * 60)
    print()
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
