# **CardioVascular risk prediction**
## **Abstract**
We classify binaries as representing the case of whether a person will develop heart problems in the next ten years or not. We implement various Machine Learning models to predict the case.
## **Introduction**
The given dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. Our goal is to predict whether the patient has a 10-year risk of future coronary heart disease (CHD). 
The dataset provides the patients’ information. It includes over 4,000 records and 15 attributes.
Each attribute is a potential risk factor. There are both demographic, behavioral, and medical risk factors. Most of the variables are self-explanatory.
The variables involved are as follows:

Demographic:

• Sex: male or female("M" or "F")

• Age: Age of the patient;(Continuous - Although the recorded ages are whole numbers as they are only years, the concept of age is continuous)

• Education: Ordinal, high values represents highly educated.

Behavioral:

• is_smoking: whether or not the patient is a current smoker ("YES" or "NO")

• Cigs Per Day: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarettes, even half a cigarette.)

Medical History:

• BP Meds: whether or not the patient was on blood pressure medication (Nominal)

• Prevalent Stroke: whether or not the patient had previously had a stroke (Nominal)

• Prevalent Hyp: whether or not the patient was hypertensive (Nominal)

• Diabetes: whether or not the patient had diabetes (Nominal)

Current Medical Conditions:

• Tot Chol: total cholesterol level (Continuous)

• Sys BP: systolic blood pressure (Continuous)

• Dia BP: diastolic blood pressure (Continuous)

• BMI: Body Mass Index (Continuous)

• Heart Rate: Continuous, in medical research, variables such as heart rate though in fact discrete, are considered continuous because of a large number of possible values.

• Glucose: The glucose level in the blood.

• TenYearCHD: Abbreviation for Ten Year coronary heart disease, nominal and our target variable as well.

## **EDA**

On exploring the dataset we see that some variables have null values.


<img width="372" alt="Screenshot 2022-12-11 173115" src="https://user-images.githubusercontent.com/71693871/206902420-e8a0f099-ac44-4d2b-8f85-254960e6189e.png">

So, we have to deal with these null values further in the process.
Next, we divide the variables into numerical and categorical variables.
For the categorical variables, we see that the target variable TenYearCHD is highly imbalanced. So, we need to balance it as well. 

<img width="395" alt="Screenshot 2022-12-11 173437" src="https://user-images.githubusercontent.com/71693871/206902534-4522b378-af40-4ac6-8ee9-023339eb2433.png">

Furthermore, we remove BPMeds, diabetes, and prevalent stroke because of their heavy imbalance.
The distributions of continuous variables are as follows:

<img width="386" alt="Screenshot 2022-12-11 173607" src="https://user-images.githubusercontent.com/71693871/206902582-512f763c-f2fe-4a82-b5b4-c3472e2c87a8.png">

<img width="371" alt="Screenshot 2022-12-11 173718" src="https://user-images.githubusercontent.com/71693871/206902611-7f5e2509-7938-4e75-8e03-15e0190de8cf.png">

<img width="407" alt="Screenshot 2022-12-11 173757" src="https://user-images.githubusercontent.com/71693871/206902648-8fce98e0-ebba-4599-b461-9c64f6d2476c.png">

<img width="422" alt="Screenshot 2022-12-11 173959" src="https://user-images.githubusercontent.com/71693871/206902730-527296c8-eb4a-4e30-bed3-b165acd6fe03.png">

<img width="430" alt="Screenshot 2022-12-11 174033" src="https://user-images.githubusercontent.com/71693871/206902751-778aa253-d37e-45f6-b949-8d5ff9937298.png">

<img width="425" alt="Screenshot 2022-12-11 174115" src="https://user-images.githubusercontent.com/71693871/206902796-6ba3befe-628f-4898-b6da-debe8226340e.png">

<img width="425" alt="Screenshot 2022-12-11 174157" src="https://user-images.githubusercontent.com/71693871/206902818-48a0a258-e81d-4778-a85a-2ac526a368ed.png">

From here we can see that many of the variables are right-skewed. This may be because of outliers, which we will see later in outlier treatment.

## Null Value treatment
Null values are present in education, CigsPerDay, BPMeds, totChol, BMI, heartrate, and Glucose.
Our options to impute are mean, mode and median. To decide which to impute we first see the boxplot of the variables.

<img width="553" alt="image" src="https://user-images.githubusercontent.com/71693871/206902871-81625923-c478-495f-bcea-4a2285c9cb7d.png">

Since there are outliers in the variables we should not input the mean as the mean is heavily affected by outliers.
For all the variables we see that mode and median are equal. So, we can use either mode or median as these are not affected by outliers. 
Hence, there are no more null values in the dataset.

## Outlier Treatment
As seen before many of the aforementioned variables have outliers. Some of the values are absurd for example the blood glucose levels, in which we have values well above 400. So, our aim now is to decide what should be value with which we should replace the outliers. First, we define the upper and lower i.e.,

q1-1.5*(q3-q1) and
  
q3 + 1.5*(q3-q1) respectively.

All the values below the lower limit and above the upper limit are substituted with the median of the variables.

## Handling class imbalance
We see that the target variable i.e., ‘TenYearCHD’ is heavily imbalanced.

<img width="485" alt="image" src="https://user-images.githubusercontent.com/71693871/206903254-1cfde914-96bc-4564-9f01-7631d1f96602.png">

So, we have to fix it and in order to do that we use the following:
1. SMOTE (Synthetic Minority Oversampling Technique)
2. Tomeklinks
3. Penalizing misclassification.
Since we have a small dataset we try to get more data points using SMOTE then cut a few down using Tomek and then for some(RandomForest, SVC) of the classifier models we penalize the loss function. 

## Fixing Non-Numeric Variables:
In our model, we have only two variables containing non-numeric values sex and is_smoking.
We drop is_smoking because its information is obtained from the number of cigs per day column. 
Next, we replace Female with 0 and Male with 1.

## Preparing Data for logistic regression.
To make sure our data has no multicollinearity we make a column called glucose_age_chol which is a function of glucose, age, and chol. Where age and chol contribute the most.
In the end the VIF values for the variables are as follows:

<img width="374" alt="image" src="https://user-images.githubusercontent.com/71693871/206903332-acf56b28-8acd-478f-81a6-3afcd1614218.png">

## Train Test split and oversampling for logistic regression
For logistic regression, we need to remove multicollinearity which is done. Next, we need to scale the data by MinMaxScaler.
i.e., for each data point at each variable, we replace each value with

<img width="727" alt="image" src="https://user-images.githubusercontent.com/71693871/206903382-5172db5c-0b97-4368-aa2f-51f0fd793e47.png">

Then we split the data into train and test set, 80% of data goes to the train set and rest to test set. The test set contains 102 values as 1. The train set contains 2303 values as 0 and 409 values as 1.
Now, we have to balance the 1’s in the training dataset. After using SMOTE and Tomeklinks our training set has 1905 values of zeros and ones.
This same dataset is also used for Naive Bayes model.

## Applying Classification Model.

Before applying any model we need to decide our aim. Since, our aim mostly should be to predict whether some person is going to have heart diseasein next 10 years or not, we will prioritize predicting the 1s and the scoring metric chosen is ROC-AUC curve.

Logistic Regression:

Using GridSearchCV we get C value as 0.77 and penalty as l2. The confusion matrix formed is as follows.

<img width="518" alt="image" src="https://user-images.githubusercontent.com/71693871/206903477-5ac30fd1-d52a-45ce-bf04-ca0cc69db66b.png">

Here we are able to predict 60% of the people who may develop heart disease, our aim is to increase that.
The ROC-AUC curve and Shap values are also obtained.

<img width="659" alt="image" src="https://user-images.githubusercontent.com/71693871/206903502-be0ada4e-851b-4d64-83d1-43982a8348a9.png">

Gaussian Naive Bayes:

On applying Naive Bayes model. We canot get better results so we mostly neglect this model.

Support Vector Classifier 1:

Before applying SVC we prepare the dataset. For this we use the dataset before removing multicollinearity. On applying support vector classifier use GridSearchCV and obtain this confusion matrix.

<img width="550" alt="image" src="https://user-images.githubusercontent.com/71693871/206903573-4dc30997-4358-4c06-8dca-d93f82cd39a5.png">

Here we obtained a better result than Logistic Regression. Same as before we obtain the ROC-AUC curve and classification report as well.

Support Vector Classifier 2:

We use two methods to handle the class imbalance, we use smote and tomek links to generate new data points and then we use loss function where misclassifications are heavily penalized.
Then we apply SVC with GridSearchCV to get C = 40, gamma = 0.01 and kernel = rbf to get

<img width="467" alt="image" src="https://user-images.githubusercontent.com/71693871/206903640-2f6920a2-69ee-42bb-8f31-0a44ec34dd73.png">

 Ensemble Tree Models: 
 We prepare the data for ensemble tree models. Since these do not require scaling we create the dataset and train test split accordingly.

LGBM Implementation: 
We use GridSearchCV to get the optimum hyperparameters (maximum depth, number of estimators and learning rate). The confusion matrix obtained is

<img width="530" alt="image" src="https://user-images.githubusercontent.com/71693871/206903703-e326f23d-6b97-47cf-a1bb-9f69c7847258.png">

Here we could appoint for a bit more than 60% of the data. The ROC-AUC curve and SHAP plots are also obtained.

<img width="554" alt="image" src="https://user-images.githubusercontent.com/71693871/206903770-329a7d9b-7bba-4740-a7b4-735fce6eba51.png">

Extreme Gradiend Random Forest Boosting Model

We implement XGRFB model and tune the hyperparameters using GridSearchCV. The confusion matrix hence obtained is

<img width="520" alt="image" src="https://user-images.githubusercontent.com/71693871/206903826-448f4afb-6dd8-4bb7-84f8-60ae25158317.png">

Further we obtain the ROC-AUC curves and SHAP values as well.

<img width="623" alt="image" src="https://user-images.githubusercontent.com/71693871/206903858-3b997b69-b80a-4fdb-a9c3-544c56afb006.png">

Random Forest Implementation

Same as before we apply Random Forest Model and use GridSearchCV to tune the hyperparameters. The confusion matrix obtained is

<img width="547" alt="image" src="https://user-images.githubusercontent.com/71693871/206903893-1795a935-77ab-4486-82dc-d2ded989dc39.png">

Further, we plot ROC-AUC curve and plot the SHAP values.

<img width="526" alt="image" src="https://user-images.githubusercontent.com/71693871/206903915-f361194c-875b-4a32-8339-16fce5212e6f.png">

## **Conclusion.**
The confusion matrix, classification report and many other metrics are obtained and the best values are chosen. Thus we select SVM with a balanced loss function.

