@echo off
echo Plant Disease Classifier - Inference Tool
echo ----------------------------------------
echo.

if "%~1"=="" (
    echo Usage: run_inference.bat [path_to_image]
    echo Example: run_inference.bat C:\path\to\your\image.jpg
    exit /b
)

echo Running inference on: %1
echo.

echo Using robust inference method...
python robust_inference.py --image "%~1"

echo.
echo Done!
