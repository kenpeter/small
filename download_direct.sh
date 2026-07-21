#!/bin/bash
# Direct download using curl - bypasses slow hf_hub_download
set -e

TOKEN="${HF_TOKEN:-}"
STAGING="/home/kenpeter/work/data/_staging_multi"
mkdir -p "$STAGING/fineweb-edu"

# List files using HF API
echo "[fineweb-edu] Listing files..."
if [ -n "$TOKEN" ]; then
    FILES=$(curl -sL -H "Authorization: Bearer $TOKEN" "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main?recursive=true" | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = [item['path'] for item in data if item.get('type') == 'file' and item['path'].endswith('.parquet') and item['path'].startswith('data/CC-MAIN-') and 'train-' in item['path']]
for f in files[:200]:
    print(f)
")
else
    FILES=$(curl -sL "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main?recursive=true" | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = [item['path'] for item in data if item.get('type') == 'file' and item['path'].endswith('.parquet') and item['path'].startswith('data/CC-MAIN-') and 'train-' in item['path']]
for f in files[:200]:
    print(f)
")
fi

TOTAL=$(echo "$FILES" | wc -l)
echo "[fineweb-edu] $TOTAL files to download"

DONE=0
OK=0
SKIP=0
t0=$(date +%s)

# Download files one at a time (avoid router overload)
echo "$FILES" | while read -r remote_path; do
    [ -z "$remote_path" ] && continue
    
    filename=$(basename "$remote_path")
    local_path="$STAGING/fineweb-edu/$filename"
    
    if [ -f "$local_path" ] && [ $(stat -c%s "$local_path" 2>/dev/null || echo 0) -gt 100000 ]; then
        SKIP=$((SKIP + 1))
        DONE=$((DONE + 1))
        continue
    fi
    
    url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/$remote_path"
    
    if [ -n "$TOKEN" ]; then
        curl -sL --max-time 120 -o "$local_path" -H "Authorization: Bearer $TOKEN" "$url" 2>/dev/null || true
    else
        curl -sL --max-time 120 -o "$local_path" "$url" 2>/dev/null || true
    fi
    
    if [ -f "$local_path" ] && [ $(stat -c%s "$local_path" 2>/dev/null || echo 0) -gt 100000 ]; then
        OK=$((OK + 1))
    else
        SKIP=$((SKIP + 1))
        rm -f "$local_path"
    fi
    
    DONE=$((DONE + 1))
    
    if [ $((DONE % 5)) -eq 0 ] || [ $DONE -eq $TOTAL ]; then
        elapsed=$(($(date +%s) - t0))
        echo "  [fineweb-edu] $DONE/$TOTAL | ok=$OK skip=$SKIP | ${elapsed}s"
    fi
    
    sleep 2  # stagger to avoid router overload
done

elapsed=$(($(date +%s) - t0))
echo "[fineweb-edu] Done: $OK ok, $SKIP skip in ${elapsed}s"