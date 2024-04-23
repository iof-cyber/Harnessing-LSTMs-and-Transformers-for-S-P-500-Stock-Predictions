# Deep Learning Dynamics: Harnessing LSTMs and Transformers for S&P 500 Stock Predictions

This project explores the application of deep learning models, specifically Long Short-Term Memory (LSTM) networks and Transformers, for predicting stock prices of companies in the S&P 500 index. By leveraging historical stock price data and incorporating technical indicators like Simple Moving Average (SMA), we aim to develop accurate and reliable predictive models.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Introduction
Stock price prediction is a challenging task due to the complex and dynamic nature of financial markets. This project aims to harness the power of deep learning models, specifically LSTMs and Transformers, to predict the stock prices of S&P 500 companies. By incorporating technical indicators and utilizing ensemble methods, we strive to develop accurate and reliable predictive models that can aid in investment decision-making.

## Dataset
The project utilizes the S&P 500 dataset, which contains historical stock prices of prominent companies. The dataset is stored in a CSV file named `all_stocks_5yr.csv` and can be downloaded from [here](https://www.kaggle.com/datasets/camnugent/sandp500). Please ensure that you have downloaded the dataset and placed it in the same directory as the notebook before running the code.

## Requirements
To run this project, you need the following dependencies:
- Python 3.x
- Jupyter Notebook or JupyterLab
- Required libraries: pandas, numpy, matplotlib, scikit-learn, TensorFlow, Keras, TA-Lib

## Installation
1. Clone this repository to your local machine using the following command:
git clone https://github.com/iof-cyber/Harnessing-LSTMs-and-Transformers-for-S-P-500-Stock-Predictions.git

2. Navigate to the project directory:
cd Harnessing-LSTMs-and-Transformers-for-S-P-500-Stock-Predictions

3. Install the required libraries using pip:
pip install -r requirements.txt

4. Download the S&P 500 dataset (`all_stocks_5yr.csv`) from [here](https://www.kaggle.com/datasets/camnugent/sandp500) and place it in the same directory as the notebook.

## Usage
1. Open the Jupyter Notebook or JupyterLab.

2. Navigate to the project directory and open the `Musa_Ibrahim_DS677_Finaltermproj.ipynb` notebook.

3. Run the notebook cells in sequential order to execute the code and reproduce the results.

4. Follow the instructions provided in the notebook to explore different sections of the project, such as data preprocessing, model training, evaluation, and stock price prediction.

5. Modify the code and experiment with different hyperparameters, feature engineering techniques, and model architectures to further enhance the performance of the models.

## Results
The project evaluates the performance of LSTM and Transformer models using various metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R^2). The results demonstrate the effectiveness of the LSTM model in predicting stock prices, while the Transformer model shows room for improvement. Detailed results and analysis can be found in the notebook.

## Contributing
Contributions to this project are welcome! If you have any ideas, suggestions, or improvements, please feel free to open an issue or submit a pull request. Let's collaborate and enhance the project together.

## Acknowledgements
This project is part of the coursework for the Deep Learning course DS677. The project utilizes open-source libraries and datasets, and we acknowledge their respective authors and contributors.

## References
- [Transformer models for stock price prediction](https://arxiv.org/abs/2003.12840)
- [Hugging Face Transformers Library](https://huggingface.co/transformers/)
- [Keras LSTM Tutorial](https://keras.io/examples/nlp/lstm_seq2seq/)
- [S&P 500 Dataset on Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500)
- [Project Repository](https://github.com/iof-cyber/Harnessing-LSTMs-and-Transformers-for-S-P-500-Stock-Predictions)
