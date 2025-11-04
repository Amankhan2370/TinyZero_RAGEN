@echo off
echo Starting TinyZero Training on RTX 4090
echo =====================================
echo.

REM Activate conda environment if needed
call conda activate base

REM Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo.
echo Starting 100 iteration training...
echo.

REM Run the training
python run_rtx4090.py

echo.
echo Training complete! Check ./checkpoints_rtx4090/ for results.
pause