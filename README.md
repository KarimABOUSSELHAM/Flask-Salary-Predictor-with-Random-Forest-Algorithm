# Flask Salary/Medals Predictor
In this project, we have forked the repository as suggested in the 3rd exercise of the 1st edition of "Practical Mlops" book written by N. Gift and A. Deza.  

The purpose of the exercise is to deploy the same Flask project using AWS Elastic Beanstalk and AWS Codepipeline instead of GCloud. Creating a new environment and application will not b detailed futher as they are supposed to be learned from the previous exercise.

The datasets are from Kaggle. (e.g the salary dataset is called: ["Kaggle Years of experience and Salary dataset"](https://www.kaggle.com/rohankayan/years-of-experience-and-salary-dataset))

# Architecture
![image](https://github.com/user-attachments/assets/3b0e8055-0299-4d73-9abc-a4d804866ac2)

The above diagram is the cloud architecture of our application. The diagram shows how the system globally works: The project files are initially stored in this github profile. Each modification in this github triggers the pipeline created in codepipeline which in return builds the updated version of the application in an elastic beanstalk environment. This latter when ready can show the deployed application in a website.  

# Model
`model.py` trains and saves the model to disk.
`model.pkl` is the model compressed in pickle format.

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Train the model

# random forest model (or any other preferred algorithm)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving model using pickle
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[1.8]]))
```

# App
`main.py` has the main function and contains all the required functions for the flask app. In the code, we have created the instance of the Flask() and loaded the model. `model.predict()` method takes input from the json request and converts it into 2D numpy array. The results are stored and returned into the variable named `output`. Finally, we used port 8080 and have set debug=True to enable debugging when necessary.

```python
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

application = Flask(__name__)
# Load model from model.pkl
model = pickle.load(open('model.pkl', 'rb'))

# Homepage route
@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Salary is {}'.format(output))

if __name__ == "__main__":
    application.run(host='127.0.0.1', port=8080, debug=True)
```


## How to Run the App

1) Clone the repo
```python
git clone https://github.com/YisongZou/IDS721-Final-Project.git
```
2) Setup - Install the required packages
```python
make all
```
3) Train the model: This will resave the pickle model
```python
python3 model.py
```
4) Run the application
```python
python3 main.py
```
5) Test the application in your localhost to check it works correctly

## Set up AWS Elastic Beanstalk environment and application
Step 1: Log in to the AWS Management Console

Step 2: Click on `Create application`. Choose a name for your application and a platform (The platform we selected is `Python 3.12 running on 64bit Amazon Linux 2`)

Step 3: Create and launch a New Environment. Choose a name for the new environment and select the appropriate options (This will be a web server environment and we won't upload project files in this step as we'll let codepipeline do this job. The ec-2 instance type will be t2.micro)

Step 4: Check your environment is in ready state in aws elastic beanstalk environment dashboard


## Set up Continuous Deployment with AWS codepipeline (CD)
Step 1: Log in codepipeline and click on create pipeline  
![1b-log to codepipeline and click create pipeline](https://github.com/user-attachments/assets/968e9a32-b2a0-4cc1-b344-28a5be2b6a1a)

Step 2: Choose build custom pipeline  
![2b-choose build custom pipeline](https://github.com/user-attachments/assets/c90010a0-3b53-471e-83d2-b4733a6a1b93)

Step 3: Choose a name for the new pipeline and let aws create an automatic service role  
![3b-give the pipeline a name and let aws create an automatic service role](https://github.com/user-attachments/assets/b980d813-3927-4c2c-b562-0cc3ee70072d)

Step 4: Choose Github (via OAuth app) as a source code provider and click on connect
![4b-connect to your github repo](https://github.com/user-attachments/assets/e0b29403-a14d-4c17-922c-64f3f56dd784)

Step 5: Skip the build stage as building dependencies will be operated through requirements text file  
![5b-skip the build stage as building dependencies is already with requirements text file](https://github.com/user-attachments/assets/e7958ebf-c14b-4dbb-8517-3cd506294e8a)

Step 6: Choose elastice beanstalk as the deployment provider and the appropriate environment and application names which you created earlier  
![6b-choose elastice beanstalk as the deployment provider and the appropriate env and app names](https://github.com/user-attachments/assets/6003b338-0d80-49dc-903b-9c0101dd9b91)

Step 7: Create the pipeline after checking the review section  
![7b-create the pipeline after review of the entered data](https://github.com/user-attachments/assets/bb0160e4-45fd-4fe3-9ccf-7c465f920df6)

While creating the pipeline you can view the progress in codepipline page. Once correctly set, go check your environment in elastic beanstalk. The following picture shows the obtained result.  
![image](https://github.com/user-attachments/assets/ecf818ad-c46a-4fbb-a737-57e79fff55e5)
