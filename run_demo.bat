@echo off
echo ========================================================
echo Starting OmniVoice Gradio Web UI...
echo Running on CPU (no-asr mode for faster startup)
echo First run will download model weights (~3.25 GB) from Hugging Face.
echo ========================================================
.\.venv\Scripts\omnivoice-demo.exe --model k2-fsa/OmniVoice --ip 127.0.0.1 --port 7860 --device cpu --no-asr
pause

