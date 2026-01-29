import sys
import subprocess
import os
import shutil

def run_test():
    with open("repro_log.txt", "w", encoding="utf-8") as log:
        def log_print(s):
            print(s)
            log.write(s + "\n")

        log_print("--- Environment Check ---")
        ffmpeg_path = shutil.which("ffmpeg")
        log_print(f"ffmpeg path: {ffmpeg_path}")
        
        if ffmpeg_path:
            try:
                res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
                log_print(f"ffmpeg -version exit code: {res.returncode}")
                log_print(f"ffmpeg version output head: {res.stdout.splitlines()[0] if res.stdout else ''}")
            except Exception as e:
                log_print(f"Failed to run ffmpeg -version: {e}")
        else:
            log_print("ffmpeg NOT FOUND in PATH")
            # Try to look in expected locations
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            if local_app_data:
                winget_packages = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
                log_print(f"Checking WinGet dir: {winget_packages}")
                if os.path.exists(winget_packages):
                    files = os.listdir(winget_packages)
                    log_print(f"Found packages: {[f for f in files if 'ffmpeg' in f.lower()]}")

        log_print("\n--- Running yt-dlp Download Test ---")
        video_id = "dQw4w9WgXcQ"
        start = 10
        end = 20
        temp_file = "reproduce_temp.mkv"
        if os.path.exists(temp_file):
            os.remove(temp_file)

        cmd_download = [
            sys.executable, "-m", "yt_dlp",
            # "--force-ipv4", # Commented out just in case
            # "--quiet", "--no-warnings", # Commented out to see output
            "--verbose",
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "--merge-output-format", "mkv",
            "-f",
            "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
            "-o", temp_file,
            f"https://youtu.be/{video_id}"
        ]
        
        log_print(f"Command: {cmd_download}")

        try:
            result = subprocess.run(
                cmd_download,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            log_print("Success!")
            log_print("STDOUT:\n" + result.stdout)
            log_print("STDERR:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            log_print(f"Failed with CalledProcessError: {e.returncode}")
            log_print("STDOUT:\n" + (e.stdout or ""))
            log_print("STDERR:\n" + (e.stderr or ""))
        except Exception as e:
            log_print(f"Failed with Exception: {e}")

if __name__ == "__main__":
    run_test()
