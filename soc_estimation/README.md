# SOC Estimation Repository

Welcome to the SOC Estimation Repository! This project provides a collection of tools and notebooks for estimating the State of Charge (SOC) using both deep learning and machine learning techniques. It includes data analysis tools, estimation models, and testing frameworks to facilitate comprehensive SOC estimation and evaluation.

## Repository Structure

### 1. Data Analysis

- **`data_analysis.ipynb`**: Analyze and visualize the distribution of data in the datasets. This notebook helps you understand the characteristics of the data stored in:
  - `./data/train/`
  - `./data/val/`
  - `./data/test/`

### 2. SOC Estimation

- **`soc_estimation_dl.ipynb`**: Implements a deep learning approach for SOC estimation using features such as:
  - Voltage (V)
  - Current (I)
  - Temperature (Temp)
  - Average Voltage (V_avg)
  - Average Current (I_avg)
- **`soc_estimation_ml.ipynb`**: Implements a machine learning approach for SOC estimation using the same features as above.

### 3. Model Testing

- **`testing_dl.ipynb`**: Tests deep learning models saved in the `./models/dl/` directory against the test dataset.
- **`testing_ml.ipynb`**: Tests machine learning models saved in the `./models/ml/` directory against the test dataset.

## Directory Structure

- **`./data/`**: Contains datasets for training, validation, and testing.
  - `train/`
  - `val/`
  - `test/`
- **`./models/`**: Stores pre-trained models for both deep learning and machine learning.
  - `dl/`
  - `ml/`
