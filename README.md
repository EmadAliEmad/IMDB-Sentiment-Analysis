# IMDB Sentiment Analysis

This repository contains a Convolutional Neural Network (CNN) model for classifying the sentiment of IMDB movie reviews as either positive or negative.

## Overview

This project leverages the IMDB movie review dataset to train a CNN model. The process includes:

*   **Data Loading and Exploration:** Loading the IMDB dataset and exploring its basic characteristics.
*   **Data Preprocessing:** Preparing text data by padding sequences to a uniform length.
*   **Model Building:** Constructing a CNN architecture using TensorFlow/Keras.
*   **Model Training:** Training the model on the prepared dataset.
*   **Model Evaluation:** Evaluating the trained model's performance.
*   **Results Visualization:** Plotting training history, confusion matrix and generating classification report to assess the model.

## Getting Started

### Prerequisites

*   Python 3.6+
*   TensorFlow 2.x
*   Keras
*   NumPy
*   Pandas
*   Matplotlib
*   Seaborn
*   Plotly
*   Scikit-learn

You can install these packages using pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn plotly scikit-learn
```

### Installation

1.  Clone the repository:

```bash
git clone https://github.com/your_username/your_repo_name.git
```
2. Navigate to the project directory:
```bash
cd your_repo_name
```

### Usage

1.  Run the Jupyter Notebook file `IMDB Sentiment Analysis.ipynb` to execute the complete analysis, training and visualization.
or
2. Run the python file `main.py` to execute the complete analysis, training and visualization.

## File Structure

```
IMDB-Sentiment-Analysis/
├── IMDB Sentiment Analysis.ipynb       # Jupyter notebook containing the code
├── README.md         # This file
├── main.py   # main python file to run the project
└── model_weights.h5  # pretrained model
```

## Key Files

*   `IMDB Sentiment Analysis.ipynb`: A Jupyter Notebook containing the full project workflow, including EDA, preprocessing, model definition, training, and evaluation.
*  `main.py`: A python file containing the full project workflow, including EDA, preprocessing, model definition, training, and evaluation.
*  `model_weights.h5`: contains a pretrained model

## Results

The model achieved a test accuracy of approximately 86%. The project includes plots of:
*   Training and validation loss/accuracy curves.
*   Confusion matrix.

A classification report is also provided to show model performance metrics.

## Contributing

Feel free to fork the repository and submit pull requests. Any contributions are welcome.

## Author
[Emad Ali Emad]

## License
This project is licensed under the MIT License.
