@echo off
if "%~1"=="" (
    echo ========================================================
    echo OmniVoice CLI Inference Helper
    echo ========================================================
    echo Usage: run_infer.bat "Your text to synthesize" [output.wav]
    echo Example: run_infer.bat "Hello, this is a test of OmniVoice." hello.wav
    exit /b 1
)
set TEXT=%~1
set OUT=%~2
if "%OUT%"=="" set OUT=output.wav
echo Synthesizing: "%TEXT%"
echo Output file: %OUT%
echo First run will download model weights (~3.25 GB) from Hugging Face if not cached.
.\.venv\Scripts\omnivoice-infer.exe --model k2-fsa/OmniVoice --text "%TEXT%" --output "%OUT%" --device cpu
pause

