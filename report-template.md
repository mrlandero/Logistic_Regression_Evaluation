# Lending Data Report

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* The purpose of this application was to create a model that can predict, given certain financial data, whether a loan would be repaid, or if it would result in a default of the loan. 

* The finacial data that we used was loan size, interest rate, borrower income, borrower debt to income, borrower number of accounts, derogatory marks, and borrower total debt. We build and fit a Logistic Regression Model so that it can predict the `loan status`. In other words, we feed the financial data (features) to the model, and the model should predict whether that loan will be fully repaid, or if it will default. 

* We wanted to predict the `y`, or the target, in this case being `loan_status`. Before we could really use the model effectively, we had to oversample the smaller class. We first used `y.value_counts()` to see the class inbalance. Then we oversample the minority class, and run the `y_resampled.value_counts()`. We should see a balance between the classes now.

* The stages for selecting and using this Logistic Regression Model were declaring a model, in this case Logistic Regression. Next, we need to fit the model to the training data we want to work with. Then, we predict the targets with the testing data. Finally, we evaluate our model to make sure that we used the correct model based on the question we want to answer. Repeat if you need to fit a different model. 

* We used `train_test_split()` to separate the data into training and testing sets. We used `StandardScaler()` to scale the X (feature) data. We used `value_counts()` twice. First to verify the class inbalance in the original data. The other to verify the class parity after resampling. We also used `RandomOverSampler()` to help us balance the classes for more efficient predictions. We used `confusion_matrix()` twice. Once to generate the matrix for the original data. The other to generate the matrix for the resampled data. Another method used was `balanced_accuracy_score()` to generate the accuracy of the model. Finally, we utilized `classification_report_balanced()` to create a statistic table with the accuracy metrics for the model.    

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * This Logistic Regression Model was generated for the original data, before oversampling. 
  The balanced accuracy score is 98.9%. This means that the model identified 98.9% of all labels correctly. This could be misleading however, knowing that we have a class imbalance. 
  The recall of this model is 98%, which means that, for class 1, the model identified the correct label 98% of the time. However, the precision tells us how confident we are with the predicted labels. Of the labels picked as `class 1` by the model, we are 84% confident (precision) that those loans will default. 


* Machine Learning Model 2:
  * This Logistic Regression Model was generated for the oversampled data. The balanced accuracy score is 99.3%. This means that it performed better overall than Machine Learning Model 1. The recall also improved. For this model it is 99%. This means that, for class 1, or the defaulting loans, the model was able to correctly predict 99% of the time. The precision lets us know how confident we are with these predicted labels. As expected from Model 1 to Model 2, the precision dropped by 1%. This makes sense as recall and precision tend to, historically, have a negative correlation. This means that in Model 2, the increase in our recall also makes us less confident in those predicted labels. There is a trade-off between recall and precision and we need to use a model that helps us answer the problem at hand. Sometimes you will need better recall, but there are times when precision is what you're looking for.  

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
