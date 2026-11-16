@echo off
echo ========================================================
echo OmniVoice Long Story Synthesizer
echo ========================================================
echo Automatically splits long text into sentences for fast CPU synthesis.
echo.
if "%~1"=="" (
    echo Enter your story in a file called "story.txt" or pass it as an argument:
    echo Usage: run_story.bat story.txt
    if exist story.txt (
        echo Found story.txt! Synthesizing...
        .\.venv\Scripts\python.exe story_synth.py --file story.txt --output story_output.wav --num_step 10
    ) else (
        echo Create a story.txt file in this folder and run again.
    )
) else (
    .\.venv\Scripts\python.exe story_synth.py --file "%~1" --output story_output.wav --num_step 10
)
pause

