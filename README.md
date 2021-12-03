# Logistic_Regression_Evaluation
Evaluate the effectiveness of a Logistic Regression Model for classification by generating and comparing the accuracy score, a confusion matrix, and the classification report for the original data and then for `oversampled`  minority class data.

---

## Technologies

This project leverages python 3.7 with the following packages:

**[Numpy Library](https://numpy.org/)** - NumPy offers comprehensive mathematical functions, random number generators, linear algebra routines, Fourier transforms, and more.<br>

**[Pandas Library](https://pandas.pydata.org/)** - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language.<br>

**[Pathlib Library](https://pathlib.readthedocs.io/en/pep428/)** - This module offers a set of classes featuring all the common operations on paths in an easy, object-oriented way.<br>

**[SkLearn.metrics Balanced Accuracy Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)** - The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.<br>

**[SkLearn.metrics Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)** - Compute confusion matrix to evaluate the accuracy of a classification.<br>

**[imblearn.metrics Classification Report Imbalanced](https://imbalanced-learn.org/dev/references/generated/imblearn.metrics.classification_report_imbalanced.html)** - Build a classification report based on metrics used with imbalanced dataset.<br>

**[SkLearn.preprocessing Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)** - Standardize features by removing the mean and scaling to unit variance.<br>

**[SkLearn.model_selection train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)** - Split arrays or matrices into random train and test subsets.<br>

**[SkLearn.linear_model Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)** - Logistic Regression (aka logit, MaxEnt) classifier.<br>

**[imblearn.over_sampling Random Over Sampler](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.RandomOverSampler.html)** - Class to perform random over-sampling.

---

## Installation Guide

Before running the application first install the following dependencies:

For this application, you need to add the imbalanced-learn and PyDotPlus libraries to your `dev` virtual environment. The imbalanced-learn library has models that were developed specifically to deal with class imbalance. You’ll use PyDotPlus to plot a decision tree.

### Install imbalance-learn

1. Open a terminal window, and then activate your `dev` virtual environment by running the following command:

```python
conda activate dev
```

2. Install imbalance-learn by running the following command:

```python
conda install -c conda-forge imbalanced-learn
```

3. Verify the installation by running the following command:

```python
conda list imbalanced-learn
```

4. Note that the result on your screen should resemble the following image:

![ImbalancedLearn Library](imblearn_list.png)


### Install PyDotPlus

1. If your `dev` virtual environment is still active, skip to Step 2. Otherwise, activate your `dev` virtual environment by running the following command:

```python
conda activate dev
```

2. Install PyDotPlus by running the following command:

```python
conda install -c conda-forge pydotplus
```

3. Verify the installation by running the following command:

```python
conda list pydotplus
```

Note that the result on your screen should resemble the following image:

![Pydotplus Library](pydot_list.png)

You're now all set up!

---

## Usage

To use the Logistic Regression Evaluation application, simply clone the repository and run the Jupyter Notebook **credit_risk_resampling.ipynb** either in VSC, or in Jupyter Lab.

Step 1: Import the required libraries and modules:

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import StandardScaler
```

Step 2: Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame:

```python
# Read the CSV file from the Resources folder into a Pandas DataFrame
lending_df = pd.read_csv(
  Path("./Resources/lending_data.csv")
)

# Review the DataFrame
lending_df.head()
```

Step 3: Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns:

```python
# Separate the data into labels and features

# Separate the y variable, the labels
y = lending_df["loan_status"]

# Separate the X variable, the features
X = lending_df.drop(columns="loan_status")
```

Verify that the individual dataframes are correctly divided:

![y_split DataFrame](y_split.png)

In the preceding image we can see that only the target `y` column is in this dataframe.

![X_split DataFrame](x_split.png)

In the image above, we can verify that the target `y` column has been removed from this DataFrame.
