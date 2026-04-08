# Stock Market Prediction using Refined Relational Graph Convolutional Networks (REGCN)

A comprehensive framework for stock market prediction that leverages Graph Neural Networks (GNNs) with Variational Mode Decomposition (VMD) preprocessing and multiple relational adjacency matrices to capture complex inter-stock relationships.

## 🚀 Key Features

- **Advanced Graph Neural Networks**: Implements Relational Graph Convolutional Networks (REGCN) with multiple adjacency matrices
- **Signal Processing**: Integrates Variational Mode Decomposition (VMD) for noise reduction and feature extraction
- **Multi-Relational Graphs**: Uses Pearson correlation, Spearman correlation, and Dynamic Time Warping (DTW) distances
- **Comprehensive Baselines**: Compares against ARIMA, LSTM, GRU, CNN-LSTM, T-GCN, GCN-Attention, and more
- **Multiple Datasets**: Supports DJIA, NASDAQ, and SSE stock indices
- **Genetic Algorithm Optimization**: Optimizes VMD parameters using Genetic Algorithms
- **Extensive Evaluation**: Multiple metrics including Trend Accuracy, R², RMSE, MAE, and Relative Error

## 📊 Datasets

The project uses three major stock market indices:

- **DJIA (Dow Jones Industrial Average)**: 28 stocks, 752 trading days
- **NASDAQ**: Technology-heavy index with similar structure
- **SSE (Shanghai Stock Exchange)**: Chinese market data

Each dataset contains OHLCV (Open, High, Low, Close, Volume) data plus adjusted close prices.

## 🏗️ Architecture

### Data Processing Pipeline

1. **Raw Data Loading**: OHLCV stock price data
2. **Genetic Algorithm Optimization**: Optimizes VMD parameters (K, α) for each stock
3. **Variational Mode Decomposition**: Decomposes signals into intrinsic mode functions
4. **Adjacency Matrix Construction**: Creates multiple relational graphs
5. **Normalization**: Min-max scaling of features
6. **Sequence Generation**: Creates temporal sequences for time series prediction

### Model Architecture

The Refined REGCN model consists of:

- **Multi-Relational Graph Convolution**: Processes three adjacency matrices simultaneously
- **Graph Convolutional GRU (GCGRU)**: Custom layer combining graph convolution with GRU cells
- **Temporal Modeling**: Captures both spatial (inter-stock) and temporal dependencies
- **Trend-Aware Loss Function**: Balances MSE loss with trend prediction accuracy

## 📈 Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| ARIMA | Statistical | AutoRegressive Integrated Moving Average |
| LSTM | Deep Learning | Long Short-Term Memory Networks |
| GRU | Deep Learning | Gated Recurrent Units |
| CNN-LSTM | Hybrid | Convolutional Neural Network + LSTM |
| T-GCN | Graph Neural Network | Temporal Graph Convolutional Network |
| GCN-ATTN | Graph Neural Network | Graph Convolutional Network with Attention |
| REGCN | Graph Neural Network | Relational Graph Convolutional Network |
| **Proposed** | **Advanced GNN** | **Refined REGCN with VMD preprocessing** |

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- TensorFlow 1.15.0
- CUDA-compatible GPU (recommended for training)

### Dependencies

```bash
pip install numpy pandas tensorflow==1.15.0 scikit-learn matplotlib scipy vmdpy
```

### Clone the Repository

```bash
git clone https://github.com/HimanshuJain-2004/StockMarketModel.git
cd StockMarketModel
```

## 📖 Usage

### Configuration

Edit `Models/config.ini` to set hyperparameters and dataset paths:

```ini
[hyper]
datasets = DJIA          # Choose: DJIA, NASDAQ, SSE
seq_len = 30            # Sequence length
n_epochs = 100          # Training epochs
batch_size = 128        # Batch size
lr = 0.0008            # Learning rate
n_neurons = 128        # Number of neurons
```

### Data Preprocessing

1. **Load Raw Data**:
```bash
cd dataprecossing
python data.py
```

2. **Optimize VMD Parameters**:
```bash
python GA_VMD.py
```

3. **Apply VMD Decomposition**:
```bash
python data_VMD.py
```

4. **Process Adjacency Matrices**:
```bash
python adjprocessing.py
```

5. **Normalize Data**:
```bash
python normalization.py
```

### Training Models

Run individual models:

```bash
cd Models
python Proposed_model.py    # Run the proposed refined REGCN model
python REGCN.py            # Run the baseline REGCN model
python LSTM.py             # Run LSTM baseline
# ... etc for other models
```

### Evaluation

Results are automatically saved to `Final_Results_CSV/[DATASET]/` with metrics:
- Trend Accuracy
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Relative Error
- Training/Test Time

## 📊 Results

### Performance Comparison (DJIA Dataset)

| Model | Trend Accuracy | R² | RMSE | MAE |
|-------|----------------|----|------|-----|
| Proposed (REGCN + VMD) | **0.793** | **0.975** | **1.354** | **1.042** |
| REGCN | 0.752 | 0.986 | 2.590 | 2.083 |
| GCN-ATTN | 0.743 | 0.981 | 2.667 | 1.954 |
| T-GCN | 0.736 | 0.988 | 2.546 | 1.895 |
| CNN-LSTM | 0.653 | 0.957 | 1.845 | 1.459 |
| LSTM | 0.570 | 0.931 | 0.681 | 0.569 |
| GRU | 0.545 | 0.574 | 2.915 | 2.495 |

*Results averaged across all stocks in DJIA index. Higher Trend Accuracy and R², lower RMSE/MAE indicate better performance.*

## 🔧 Key Components

### Core Files

- `Models/Proposed_model.py`: Main proposed model implementation
- `Models/REGCN.py`: Baseline REGCN implementation
- `Models/dgcgru.py`: Graph Convolutional GRU layer
- `Models/input_data.py`: Data loading and preprocessing utilities
- `Models/utils.py`: Graph operations and evaluation metrics

### Data Processing

- `dataprecossing/GA_VMD.py`: Genetic Algorithm for VMD parameter optimization
- `dataprecossing/data_VMD.py`: VMD decomposition application
- `dataprecossing/normalization.py`: Data normalization

### Configuration

- `Models/config.ini`: Hyperparameter configuration
- `Models/tuner.py`: Hyperparameter tuning utilities

## 🎯 Methodology

### 1. Data Preparation
- Load historical stock prices (OHLCV)
- Apply Genetic Algorithm to find optimal VMD parameters for each stock
- Decompose price signals using VMD to extract intrinsic modes
- Construct multiple adjacency matrices based on different correlation measures

### 2. Graph Construction
- **Pearson Correlation**: Linear relationship between stocks
- **Spearman Correlation**: Rank-based monotonic relationship
- **DTW Distance**: Temporal alignment-based similarity

### 3. Model Training
- Create temporal sequences from decomposed signals
- Train GNN model with multi-relational adjacency matrices
- Optimize trend-aware loss function balancing accuracy and MSE

### 4. Evaluation
- Predict next-day stock prices
- Evaluate trend prediction accuracy
- Compare against multiple baseline models

## 📈 Visualization

Training results and predictions are visualized in `Proposed_Result_png/` directory, organized by dataset and model type.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
