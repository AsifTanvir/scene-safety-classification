#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export LD_LIBRARY_PATH="$(pwd)/llama.cpp/build/bin"

MODEL="qwen3vl-2b-fp16.gguf"
MMPROJ="qwen3vl-2b-mmproj-fp16.gguf"
BIN="llama.cpp/build/bin/llama-server"

if [ ! -f "$MODEL" ]; then echo "Error: $MODEL not found!"; exit 1; fi
if [ ! -f "$MMPROJ" ]; then echo "Error: $MMPROJ not found!"; exit 1; fi

echo "Starting llama-server for $MODEL on port 8080..."

exec $BIN \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    --port 8080 \
    --host 127.0.0.1 \
    -ngl 99 \
    -c 4096 \
    -b 2048 \
    -ub 512 \
    -np 1 \
    --no-mmap