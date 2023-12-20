# Pumpkin Seed Binary Classification

## Problem Description

Dataset Abstract: Pumpkin seeds are frequently consumed as confection worldwide because of their adequate amount of protein, fat, carbohydrate, and mineral contents. This study was carried out on the two most important and quality types of pumpkin seeds, ‘‘Urgup_Sivrisi’’ and ‘‘Cercevelik’’, generally grown in Urgup and Karacaoren regions in Turkey. However, morphological measurements of 2500 pumpkin seeds of both varieties were made possible by using the gray and binary forms of threshold techniques.

The morphological measurements of the 2500 pumpkin seeds were used to develop a model to differentiate between the two pumpkin species. 

The list of features provided in the dataset:
- area
- perimeter
- major_axis_length
- minor_axis_length
- convex_area
- equiv_diameter
- eccentricity
- solidity
- extent
- roundness
- aspect_ration
- compactness

The objective of this project is to develop an application that returns the determination of pumpkin seed species based on provided morphological measurements.

## Dataset Source

Kaggle: https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset/data

Original Source: https://www.muratkoklu.com/datasets/

The dataset source provides .xlsx and .arff file formats. For ease of use, the data provided with this project includes a .csv file. In order for Pandas to interpret .xlsx files, the python package openpyxl is needed.

The data can also be downloaded direct from the source using the bash commands below, run the commands in the project directory. This code is also available within the notebook.

```bash
wget https://www.muratkoklu.com/datasets/vtdhnd05.php -O Pumpkin_Seeds_Dataset.zip
mkdir data
unzip Pumpkin_Seeds_Dataset.zip -d data
rm Pumpkin_Seeds_Dataset.zip
```

## Files

A) notebook.ipnyb: This notebook was used for preparing and cleaning data, exploratory data analysis, as well as model selection and parameter tuning.

B) train.py: The train script performs the essential data preparation steps needed to build the selected model. The target classes 'Çerçevelik' & 'Ürgüp Sivrisi' are mapped to classes 0 and 1 respectively as needed for the model training process. The prepared model is saved as a pickle output file for use by the prediction service.

C) predict.py: The predict file creates the flask web service used to recieve queries, preproccess the data, and provide a response prediction.

D) predict-test.py: The predict-test script is used to test if the predict.py service is functioning properly.

E) std.npy/mean.npy: Used to store sklearn StandardScaler() scaling data used for training dataset, which is utilized by the predict.py script for scaling data. These   files are generated by the train.py script.

## Dataset Analysis & Model Selection

### Data Preprocessing

- The target column 'Class' is mapped to binary values: {'Çerçevelik': 0, 'Ürgüp Sivrisi': 1}

- Dataset was split into training (80%) and test (20%) sets.

- X_train data is fit and scaled using StandardScaler, X_test data is scaled based on the fit from the X_train data.

### EDA

Çerçevelik: 0, Ürgüp Sivrisi: 1

- Çerçevelik generally had a tighter grouping of values across features, where Ürgüp Sivrisi generally had a higher standard deviation across features.
- Plotting various features against each other, particularly eccentricity vs aspect_ration or compactness, it is evident features within this dataset were created through relationships between other values within the dataset. Due to this, several features share a high correlation.
-Çerçevelik is generally shorter and wider, while Ürgüp Sivrisi generally longer and more narrow.

### Feature Engineering

- Features were both added and removed while testing the various models. In the end the models performed best with no additional features nor with features removed.

- AutoViz was used for graph generation and analysis tools. The FixDQ function (used for feature elimination) was tested to determine effects on model accuracy. The features removed by FixDQ decreased model performance for this dataset, so it was not used in training the final model.

### Models Used

#### Logistic Regression

Training Accuracy:  0.8855
Testing Accuracy:   0.856

#### Decision Tree Classifier

Training Accuracy:  0.9025
Testing Accuracy:   0.86

#### K Neighbors Classifier

Training Accuracy:  0.9085
Testing Accuracy:   0.854


#### Random Forest Classifier
The Random Forest Classifier model had a tendency to overtrain.

Training Accuracy:  1.0
Testing Accuracy:   0.864

#### Support Vector Classifier
The SVC model performed well, providing more consistent prediction accuracy under different feature selections.

Training Accuracy:  0.912
Testing Accuracy:   0.866

#### XGBoost Classifier
XGBClassifier had the best overall performace of the tested models. This is the model used for the train.py script.

Multiple parameters were tuned for this model, including learning_rate, n_estimators, and max_depth. Due to the slight class imbalance of the target variable in the dataset, scale_pos_weight was included as a parameter to account for the class imbalance effects on model training.

Training Accuracy:  0.902
Testing Accuracy:   0.872

### Model Parameter Selection & Tuning

GridSearchCV was used to test multiple parameters for several of the models. StratifiedKFold with 10 folds was used for cross validation within the parameter search. The parameter 'n_jobs=-1' was also passed into the GridSearchCV to utilize multiple processor cores in the calculations.

### Validation Methods

Model training was performed with 80% of the total available data, with a 10-fold StratifiedKFold validation method.

After training and parameter tuning, the model was tested on the test data (20%).

## Instructions to Run

Note: pipenv and Docker needed to execute all of the functions of this project.

1). Clone github repo:

```bash
git clone repo name
```

2). Navigate to the cloned repo directory in a terminal window.

3). Install dependencies to run files.

If running locally, use pipenv to ensure all required dependencies are installed.
```bash
pipenv install
```
4). Run the train.py script to generate the model used for predictions.

```bash
pipenv run python train.py
```

5). Run the predict.py application:

- Option A: Run locally with gunicorn
```bash
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```
- Option B: Run with Docker

```bash
# Build 
docker build -t seed-prediction .

# Run
docker run -it --rm -p 9696:9696 seed-prediction:latest
```

6). Run predict-test.py in another terminal window

```bash
python predict-test.py
```

### Commands for AWS Elastic Beanstalk CLI:

Run these from within the project directory.

Note: Replace 'us-east-1' with your local AWS region.
```bash
eb init -p "Docker running on 64bit Amazon Linux 2" -r us-east-1 seedtype-serving
```
Note: "Docker running on 64bit Amazon Linux 2" is used as a workaround, as using "docker" instead will result in the below error when running 'eb local run':

'ERROR: NotSupportedError - You can use "eb local" only with preconfigured, generic and multicontainer Docker platforms.'


```bash
eb local run --port 9696
```



```bash
eb create seedtype-serving-env
```

You will need to capture the elastic beanstalk service url host within the predict-test.py to make a request. You can find this within the logs after running eb create, or obtained on the AWS Elastic Beanstalk Environments page.

```bash
python predict-test.py
```


Don't forget to shut down the eb service when finished.

```bash
eb terminate seedtype-serving-env
```