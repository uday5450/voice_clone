@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo        OmniVoice Automated Story Synthesizer
echo ========================================================
echo.

set STORY_FILE=story.txt
if not "%~1"=="" (
    set STORY_FILE=%~1
)

if not exist "!STORY_FILE!" (
    echo Error: "!STORY_FILE!" not found!
    echo Please create "story.txt" with your story or drag and drop a .txt file onto this bat script.
    echo.
    pause
    exit /b 1
)

set ARGS=--file "!STORY_FILE!" --num_step 10

if exist "sample.wav" (
    echo [Voice Clone] Found sample.wav - Cloned voice will be applied!
    set ARGS=!ARGS! --ref_audio sample.wav
    if exist "sample_text.txt" (
        set /p REF_TEXT=<sample_text.txt
        set ARGS=!ARGS! --ref_text "!REF_TEXT!"
    )
) else (
    echo [Voice Mode] No sample.wav found. Using default voice design.
)

echo.
echo Processing story from: !STORY_FILE!
echo.
.\.venv\Scripts\python.exe story_synth.py !ARGS! --output story_final.wav

echo.
echo Done! Output saved as: story_final.wav
pause

