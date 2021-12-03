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
