python -m dialoop.cli novel.txt \
--output labeled_test.txt \
--batch-size 1 \
--max-tool-steps 20 \
--max-iterations 2 \
--protocol auto \
--base-url http://172.31.102.237:11434/v1 \
--api-key ollama \
--model qwen3:32b \
--model-timeout 60