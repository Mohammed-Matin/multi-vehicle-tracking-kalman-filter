import os, glob, time
from vehicle_tracker import run_pipeline

IN_DIR = "/mnt/d/WSL/Videos/AIR/Untracked"
OUT_DIR = "/mnt/d/WSL/Videos/AIR/tracked"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(IN_DIR, "*.avi")))
print(f"Found {len(files)} videos", flush=True)

for i, f in enumerate(files, 1):
    stem = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(OUT_DIR, stem + "_tracked.mp4")
    print(f"[{i}/{len(files)}] Processing: {os.path.basename(f)}", flush=True)
    t0 = time.time()
    try:
        run_pipeline(f, out, conf=0.4, trail=0, show=False)
        print(f"  -> done in {time.time()-t0:.1f}s : {out}", flush=True)
    except Exception as e:
        print(f"  !! ERROR: {e}", flush=True)

print("ALL DONE", flush=True)