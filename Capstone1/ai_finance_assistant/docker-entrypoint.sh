#!/bin/sh
# Builds the knowledge base index on first start, then runs the given command.
# Output goes to stderr, because the MCP server uses stdout for its protocol.
set -e
mkdir -p /app/data/chroma_db
# The marker is written only after a full build, so a failed build is retried next start.
if [ ! -f /app/data/chroma_db/.built ]; then
  echo "Building the knowledge base index (one time; calls OpenAI embeddings)..." >&2
  python -m src.rag.build_index >&2
  touch /app/data/chroma_db/.built
fi
exec "$@"
