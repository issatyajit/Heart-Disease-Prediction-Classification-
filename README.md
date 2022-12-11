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


