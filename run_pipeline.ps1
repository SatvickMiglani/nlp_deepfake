# Multimodal Deepfake Detection - Full Benchmark Pipeline Runner
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "MULTIMODAL DEEPFAKE DETECTION — FULL BENCHMARK SCALE (FakeAVCeleb)" -ForegroundColor Cyan
Write-Host "Audio (A), Visual MediaPipe (V), Emotion Consistency (E)" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

# Step 1: Feature Extraction
Write-Host "`n[Step 1/2] Extracting Full Scale Multimodal Features (MediaPipe + CNN + MFCC + Emotion)..." -ForegroundColor Yellow
Write-Host "Hugging Face downloads routed to D:\deepfake\hf_cache (378 GB free)" -ForegroundColor DarkGray
python src/extract_features.py --target_clips 2144 --device cuda

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[Error] Feature extraction failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 2: Training & 7-Ablation Evaluation
Write-Host "`n[Step 2/2] Training Forensic Heads & Evaluating 7 Ablation Configurations..." -ForegroundColor Yellow
python src/train_and_evaluate.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[Error] Training and evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n==============================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] All experimental runs completed on full benchmark scale!" -ForegroundColor Green
Write-Host "  Results JSON: outputs\results\final_experimental_results.json" -ForegroundColor Green
Write-Host "  Progress Report: progress_report.md" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Green
