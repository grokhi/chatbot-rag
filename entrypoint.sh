#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

echo "🔴 Retrieve LLAMA3 model... \nPlease wait for the 🟢 mark while initializing."
ollama run ${LLAMA_MODEL}
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid