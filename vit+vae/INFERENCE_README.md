# Plant Disease Classification Inference Guide

This guide explains how to use the inference script to make predictions with your trained hybrid VAE-ViT model.

## Prerequisites

Make sure you have:
1. Trained the VAE model using `trainvae.py`
2. Trained the hybrid model using `hybrid_model.py`
3. All required dependencies installed (TensorFlow, NumPy, Matplotlib)

## Making Predictions

### Using the Batch File (Recommended)

The easiest way to make predictions is to use the provided batch file:

```bash
run_inference.bat path/to/your/image.jpg
```

For example:
```bash
run_inference.bat C:\Users\Prudvi\Downloads\archive\test\test\AppleCedarRust3.JPG
```

### Using the Python Script Directly

You can also run the Python script directly:

```bash
python robust_inference.py --image path/to/your/image.jpg
```

Optional arguments:
- `--save_dir`: Custom directory to save results (default: `inference_results`)

## How It Works

The inference script:
1. Rebuilds the complete model architecture (VAE encoder + ViT classifier)
2. Loads the trained weights from `saved_models/hybrid/hybrid_final.weights.h5`
3. Processes the input image
4. Displays the prediction results
5. Saves the visualization to the `inference_results` directory

## Understanding the Results

The prediction results include:

1. **Predicted Class**: The plant disease class predicted by the model
2. **Confidence**: The confidence score (0-100%) of the prediction
3. **Inference Time**: How long it took to make the prediction
4. **Visualization**: An image showing the input with the prediction overlaid

## Troubleshooting

If you encounter issues:

1. **Model Loading Errors**:
   - Make sure you've trained both the VAE and hybrid models
   - Check that the weight file exists at `saved_models/hybrid/hybrid_final.weights.h5`

2. **Image Loading Errors**:
   - Ensure the image format is supported (JPG, PNG, BMP, etc.)
   - Check that the image file exists and is not corrupted

3. **Memory Issues**:
   - If you're running out of memory, try closing other applications
   - The model requires approximately 1-2GB of RAM for inference

## Example Usage

```bash
# Using the batch file
run_inference.bat C:\path\to\your\image.jpg

# Using the Python script directly
python robust_inference.py --image C:\path\to\your\image.jpg --save_dir my_results
```
