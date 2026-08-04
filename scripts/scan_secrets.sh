#!/usr/bin/env bash
# Pre-commit secret scanner — blocks commits containing known token patterns.
# Installed at .git/hooks/pre-commit (see install command in README or run:
#   ln -sf ../../scripts/scan_secrets.sh .git/hooks/pre-commit
set -u

# Token patterns: HF, GitHub, OpenAI, AWS, generic "Bearer <sk->", private keys
PATTERNS='hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----'

FILES=$(git diff --cached --name-only --diff-filter=ACM)
HITS=0

for f in $FILES; do
  # Skip binary files
  if [ -f "$f" ] && file "$f" | grep -q text; then
    if grep -aE "$PATTERNS" "$f" >/dev/null 2>&1; then
      echo "🔴 SECRET DETECTED in $f:"
      grep -anE "$PATTERNS" "$f" | sed 's/^/    /'
      HITS=1
    fi
  fi
done

if [ "$HITS" -eq 1 ]; then
  echo ""
  echo "Commit blocked. Remove the secret from the file (use env vars / ~/.huggingface/token),"
  echo "then re-stage: git add <file> && git commit"
  exit 1
fi
exit 0
