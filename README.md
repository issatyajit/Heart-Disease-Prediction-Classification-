The dataset is from an ongoing cardiovascular study on residents of the town of Framingham,
Massachusetts. The classification goal is to predict whether the patient has a 10-year risk of
future coronary heart disease (CHD). The dataset provides the patients’ information. It includes
over 4,000 records and 15 attributes. Our aim is to predict whether a patient will get heart disease in the next 10 years. Basically, we try to maximize our true negatives where positives are 0s and negatives are 1s.
During EDA we see that our target variable is imbalanced, we have 15% as negatives and the rest are positive. Here we also see that a few categorical variables i.e., the variables having less than 4 unique values are also heavily imbalanced. We remove such categorical variables where more than 90% of observations have the same value. In doing so we lose 3 variables BPMeds, Diabetes, and prevalent stroke. Furthermore, we drop the column is_smoking because all of its information can be obtained by the variable Cigs per day. We remove the null values in the dataset by imputing median values because the columns contained outliers and the columns are skewed. Similarly, we impute outlier of continuous columns with the median of any value lying below and above the lower and upper quartile ranges.
Then we prepare three kinds of datasets for logistic regressions, Support Vector Classifiers, and Ensemble tree models according to the requirements. For logistic regression we scale the dataset and remove multicollinearity, this dataset is also used for the Naive Bayes Classifier. For SVCs we only standardize the variables and for Ensemble tree models we keep the data as it is.
Now, we fix the data imbalance of target variables using a combination of SMOTE and Tomeklinks, further, we also use penalized loss functions in some of our models.
Finally, we apply a number of models to our dataset, and the results are as follows:

1. Logistic Regression can predict 61% of the negative values long with 35% of False     Negative predictions.


2. Gaussian Naive Bayes can predict 55% of the negative values along with 34% of False Negative predictions.

3. Support Vector Classifier (without balanced loss function) can predict 68% of the negative values with 37% of False Negativepredictions.

4. Support Vector Classifier (with balanced loss function) can predict 70% of the negative values with 37% of False Negativepredictions.

5.  LGBM can predict 63% of the negative values with 40% of False Negativepredictions.

6. XGRFB can predict 69% of the negative values with 42% of False Negativepredictions.

7. Random Forest Model can predict 66% of the negative values with 39 % False Negative predictions.

