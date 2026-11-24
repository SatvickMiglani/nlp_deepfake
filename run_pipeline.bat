@echo off
REM ==============================================================================
REM Multimodal Deepfake Detection - Full Benchmark Pipeline Runner
REM Evaluates 7 ablation configurations on the full FakeAVCeleb paper scale.
REM ==============================================================================

echo [1/2] Extracting full scale multimodal features (MediaPipe + CNN + MFCC + Emotion)...
python src/extract_features.py --target_clips 2144 --device cuda

if %errorlevel% neq 0 (
    echo [Error] Feature extraction failed!
    exit /b %errorlevel%
)

echo.
echo [2/2] Running Branch Training and 7-Ablation Evaluation...
python src/train_and_evaluate.py

echo.
echo ==============================================================================
echo [Complete] All experimental results saved to outputs\results\
echo Progress report updated at progress_report.md
echo ==============================================================================
pause
