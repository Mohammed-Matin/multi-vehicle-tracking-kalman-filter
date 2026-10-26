#!/bin/bash
export PATH="$HOME/.local/bin:$HOME/.deno/bin:$PATH"
mkdir -p /mnt/d/WSL/Videos/AIR
cd /mnt/d/WSL/Videos/AIR
yt-dlp --no-playlist --js-runtimes deno --remote-components ejs:github -f 'best[height<=720][ext=mp4]' -o 'traffic_highway.mp4' 'ytsearch1:highway traffic camera fixed view surveillance' 2>&1 | tail -3
yt-dlp --no-playlist --js-runtimes deno --remote-components ejs:github -f 'best[height<=720][ext=mp4]' -o 'traffic_intersection.mp4' 'ytsearch1:traffic intersection camera surveillance vehicles' 2>&1 | tail -3
yt-dlp --no-playlist --js-runtimes deno --remote-components ejs:github -f 'best[height<=720][ext=mp4]' -o 'traffic_night.mp4' 'ytsearch1:night traffic highway camera cars' 2>&1 | tail -3
echo "=== DONE ==="
ls -lh