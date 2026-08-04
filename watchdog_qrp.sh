#!/usr/bin/env bash
# Watchdog: run download_qrp_par.py; if staging stops growing for >4 min, kill & restart.
# wget -c resumes partial files, so each restart continues from the stall point.
set -u
STAGING=/mnt/file_drive/_qrp_staging
LOG=/mnt/file_drive/qrp_download.log
cd /home/kenpeter/work/small

while true; do
  # count target files already staged
  HAVE=$(ls "$STAGING"/*.parquet 2>/dev/null | wc -l)
  if [ "$HAVE" -ge 200 ]; then
    echo "[$(date +%H:%M:%S)] All 200 files present. Done." | tee -a "$LOG"
    break
  fi

  echo "[$(date +%H:%M:%S)] Launching download (have $HAVE/200)..." | tee -a "$LOG"
  source venv/bin/activate
  python3 -u download_qrp_par.py >> "$LOG" 2>&1 &
  DLPID=$!

  # monitor growth
  LAST_SIZE=$(du -sb "$STAGING" 2>/dev/null | cut -f1)
  STALL_COUNT=0
  while kill -0 "$DLPID" 2>/dev/null; do
    sleep 60
    CUR=$(du -sb "$STAGING" 2>/dev/null | cut -f1)
    if [ "$CUR" -gt "$LAST_SIZE" ]; then
      STALL_COUNT=0
    else
      STALL_COUNT=$((STALL_COUNT + 1))
    fi
    LAST_SIZE=$CUR
    if [ "$STALL_COUNT" -ge 4 ]; then  # 4 min no growth
      echo "[$(date +%H:%M:%S)] Stalled (no growth 4 min). Killing $DLPID..." | tee -a "$LOG"
      kill -9 "$DLPID" 2>/dev/null
      pkill -9 -f "python3 -u download_qrp_par.py" 2>/dev/null
      sleep 3
      break
    fi
  done

  # process exited on its own (finished or crashed)
  wait "$DLPID" 2>/dev/null
  HAVE=$(ls "$STAGING"/*.parquet 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] Download process ended (have $HAVE/200). Restarting..." | tee -a "$LOG"
  sleep 5
done
