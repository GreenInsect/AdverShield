#!/bin/bash
# AdverShield Startup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "   AdverShield · Adversarial Defense"
echo "========================================"

# Check Python
python3 --version || { echo "[ERROR] Python3 not found"; exit 1; }

# Check / create venv
if [ ! -d "venv" ]; then
  echo "[*] Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate
echo "[*] Installing / verifying dependencies..."
pip install -r requirements.txt

# Create overlay directory and add sample overlays
mkdir -p frontend/static/images/overlays

# Generate placeholder PNG overlays using Python
python3 - <<'PYEOF'
import os, struct, zlib

def make_png(w, h, color_rgba, filename):
    """Create minimal RGBA PNG"""
    def write_chunk(name, data):
        c = zlib.crc32(name + data) & 0xffffffff
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', c)
    
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)
    ihdr = write_chunk(b'IHDR', ihdr_data)
    
    raw = b''
    for _ in range(h):
        raw += b'\x00'
        for _ in range(w):
            raw += bytes(color_rgba)
    
    idat = write_chunk(b'IDAT', zlib.compress(raw, 9))
    iend = write_chunk(b'IEND', b'')
    
    with open(filename, 'wb') as f:
        f.write(sig + ihdr + idat + iend)

base = "frontend/static/images/overlays"

# Sample overlays (solid color with transparency)
samples = [
    ("hat_red.png",    [220, 30, 30, 200]),
    ("mask_blue.png",  [30, 80, 220, 180]),
    ("star_gold.png",  [255, 200, 0, 210]),
    ("patch_green.png",[0, 200, 80, 190]),
]

for fname, rgba in samples:
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        make_png(64, 64, rgba, path)
        print(f"  Created sample overlay: {fname}")
PYEOF

echo ""
echo "[✓] Setup complete"
echo ""
echo "  Starting server at: http://0.0.0.0:8000"
echo "  Press Ctrl+C to stop"
echo ""

python3 -m uvicorn main:socket_app --host 0.0.0.0 --port 8000 --workers 1
