**DMML_GROUP_12: Used Car Resale Price Prediction and Image Classification**

**Project Goal**
This project implements and compares three different machine learning regression algorithms to accurately predict the resale price of used cars:
1. Linear Regression 
2. XGBoost (Extreme Gradient Boosting) 
3. Random Forest
Additionally, it includes an Image Classification Model (CNN) using transfer learning to extract core car attributes (Maker, Body Type, Colour, Model) directly from various angled-images of cars.

**Data Sources**
The models are trained using a dataset organized into the following components:

**Tabular Data** (/content/drive/MyDrive/DMML DATA/tables_V2.0/):

**Adv_table.csv:** Contains car advertisement data, including mileage, engine size, body type, fuel type, and the target variable: 'Price' (Resale Price).

**Price_table.csv:** Contains MSRP data ('Entry_price') used for feature enrichment.

**Basic_table.csv:** Used by the CNN to source the 'Automaker' label.

**Image_table.csv:** Links 'Adv_ID' to image file names and viewpoint data.

**Image Data** (/content/drive/MyDrive/DMML DATA/DVM_images/):

A root folder containing car images used for the classification tasks.For regression tasks, we mainly use 'Adv_table' and 'Price_table'

**Prerequisites & Setup**
This project is designed to run in a Google Colab environment connected to Google Drive.

**1. Dependencies**
The tabular regression models rely on standard scientific Python libraries. The image classification model requires deep learning libraries:

**Task Type:** Regression
**Key Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib

**2. Google Drive Connection**
Before running any modeling cells, you must execute the initial setup cell in the Colab notebook to mount your Google Drive:

from google.colab import drive
drive.mount('/content/drive')
