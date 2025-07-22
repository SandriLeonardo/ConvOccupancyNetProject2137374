# TransONet: Lightweight Convolutional Occupancy Networks

PyTorch implementation of **TransONet** in a noteboook runnable on cloud computing environmets such as Kaggle or Colab. It is a lightweight convolutional network for efficient 3D occupancy prediction and scene generation. This repository documents systematic experiments in architecture optimization, training improvements, and model compression.

## Key Features

- **Complete TransONet Architecture:** PointNet encoder, dynamic plane predictor, U-Net, and positional encoding decoder
- **Optimized Training Pipeline:** Early stopping, learning rate scheduling, weighted sampling for class imbalance
- **Model Compression:** L1 unstructured pruning with fine-tuning workflows
- **Comprehensive Evaluation:** Occupancy metrics (F1/IoU) and mesh quality assessment

## Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-username/ConvOccupancyNetProject2137374.git
   cd ConvOccupancyNetProject2137374
   ```

2. **Install Dependencies**
   - PyTorch, numpy, scikit-learn, scikit-image, trimesh

3. **Download Data**
   - [Synthetic Rooms Dataset (4 Rooms)](https://www.kaggle.com/datasets/leonardosandri/synthetic-rooms-rooms-04050607)
   - [GT Synthetic Rooms (4 Rooms)](https://www.kaggle.com/datasets/leonardosandri/gt-synthetic-rooms-04050607)
   - [Synthetic Rooms Dataset (2 Rooms)](https://www.kaggle.com/datasets/leonardosandri/synthetic-rooms-04and05)
   - [GT Synthetic Rooms (2 Rooms)](https://www.kaggle.com/datasets/leonardosandri/gt-rooms-mesh-04and05)
   - Update paths in notebook **Globals** section

4. **Run Experiments**
   - Open `main.ipynb` and run cells sequentially
   - Models saved to `./models/` directory

## Key Findings

- **20x Speed Improvement:** Vectorized tensor operations replaced slow loops in planar projection
- **Hyperparameter Sensitivity:** Performance highly dependent on batch size, learning rate, and positive weight balance
- **Pruning Limitations:** Standard L1 pruning ineffective - model requires complete weight structure or different architecture/exploration of parameters.
- **Data Scale vs Quality Trade-off:** Larger datasets improve geometry (Chamfer/Hausdorff) but require hyperparameter retuning for occupancy metrics

## Architecture

The model implements the full TransONet pipeline with convolutional multi-plane feature extraction and transformer-based occupancy prediction, optimized for computational efficiency while maintaining reconstruction quality.