python -m dialoop.quality_cli coordinator-trace \
  --annotations .dialoop/annotations.jsonl \
  --show-problems 10

python -m dialoop.quality_cli mismatch-attribution \
  --answers answers \
  --labels labeled_test.txt \
  --annotations .dialoop/annotations.jsonl \
  --novel novel.txt \
  --max-errors 200

python -m dialoop.quality_cli verifier-false-pass \
  --answers answers \
  --labels labeled_test.txt \
  --annotations .dialoop/annotations.jsonl \
  --novel novel.txt \
  --max-errors 200

python -m dialoop.quality_cli scan-terms \
  --term 罗伦斯 \
  --term 赫萝
