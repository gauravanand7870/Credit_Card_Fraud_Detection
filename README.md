# Credit_Card_Fraud_Detection

Detecting fraudulent transactions is critical for any credit card company. This repository contains a Credit Card Fraud Detection System built on top of various machine learning algorithms, which helps in identifying suspicious transactions. 

## Features:

1. Uses multiple algorithms including Logistic Regression, Decision Trees, Random Forest, K-NN, and Naive Bayes.
2. Handling of imbalanced dataset using undersampling and oversampling techniques.
3. Integrated with a GUI for a user-friendly experience.
4. The model can be trained on updated data or enhanced using advanced algorithms in the future.

## Data:

The data used for this project can be found in 'creditcard.csv'. Each transaction contains the following features:
- V1, V2, ... V29: Transformed features due to confidentiality issues.
- Amount: Transaction Amount
- Class: 0 for normal transaction and 1 for fraudulent transaction.

## Dependencies:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn
- tkinter

## Usage:

1. Once the GUI starts, input the values for V1 through V29.
2. Click on the "Predict" button.
3. The system will display if the transaction is Normal or Fraudulent.

## Contributing:

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License:

This project is licensed under the MIT License. 
