
## Lung Cancer Detection
In this project, I have done EDA and plotted various graphs depicting whether a person has lung cancer or not. I have used methods like K means clustering, random forest classifier, decision tree, and support vector machine among others.

## K Means clustering
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/5a9f8559-7d4d-4cd0-8786-01a15f1ff677)

dataset into a predetermined number of clusters. It aims to group data points into clusters such that the points within the same cluster are more similar to each other than to those in other clusters.

Here's how the algorithm works:

Initialization: The algorithm starts by randomly selecting K centroids, where K is the number of clusters specified by the user.

Assignment: Each data point is assigned to the nearest centroid based on a distance metric, commonly Euclidean distance. This step forms K clusters.

Update centroids: After the assignment step, the centroids are recomputed as the mean of all data points assigned to each cluster.

Repeat: Steps 2 and 3 are repeated iteratively until convergence,which occurs when the centroids no longer change significantly or a maximum number of iterations is reached.

The algorithm's objective is to minimize the within-cluster sum of squared distances, which means it tries to make the data points within each cluster as close to the centroid as possible.

## Random Forest
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/f2c9e848-9987-4cc8-8195-393ab482cfb9)

Random Forest is a popular ensemble learning technique used in supervised machine learning for classification and regression tasks. It belongs to the family of bagging algorithms, which aim to improve the performance of individual models by combining multiple models.

Here's how Random Forest works:

Bootstrapped Sampling: Random Forest creates multiple decision trees by repeatedly sampling the training data with replacement (bootstrapping). Each sample is used to train a separate decision tree.

Feature Randomness: At each node of the decision tree, a random subset of features is considered for splitting. This introduces diversity among the trees and helps to reduce overfitting.

Decision Tree Building: Each decision tree is grown to its maximum depth without pruning, which means the trees are typically deep and can capture complex relationships in the data.

Voting: For classification tasks, the predictions from all the decision trees are combined using a majority voting mechanism. For regression tasks, the predictions are averaged across all the trees.

## Decision trees
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/13cd3427-b621-4b14-8209-34990d89752b)

Decision trees are a fundamental supervised learning technique used for both classification and regression tasks. They are popular due to their simplicity and interpretability. A decision tree breaks down a dataset into smaller subsets while progressively constructing a tree-like structure consisting of decision nodes and leaf nodes.

Here's how decision trees work:

Tree Structure: A decision tree is hierarchical, with each internal node representing a decision based on a feature (attribute) of the data. The edges leaving each node represent the possible outcomes of the decision, leading to child nodes.

Decision Rules: At each internal node, a decision is made based on the value of a certain feature. This process is repeated recursively for each subset of data until a leaf node is reached, which corresponds to the final decision or prediction.

Splitting Criteria: The decision of which feature to split on at each node and where to make the splits is determined based on a certain criterion. For classification tasks, common splitting criteria include Gini impurity and entropy, which measure the purity of the classes in the resulting subsets. For regression tasks, mean squared error or mean absolute error may be used.

Stopping Criteria: The process of creating the tree continues until a stopping criterion is met. This could be a predefined maximum depth of the tree, a minimum number of samples required to split a node, or when further splitting does not improve the model's performance.

## Support vector machine
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/62e5e51b-7c6d-43d7-9882-bb4191fa0e6b)

Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. They operate by finding the optimal hyperplane that best separates data points into different classes. The key concept behind SVMs is to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class, also known as support vectors. By maximizing this margin, SVMs aim to improve generalization and robustness of new data.

What sets SVMs apart from other classifiers is their ability to handle high-dimensional data efficiently and effectively, even when the number of features exceeds the number of samples. This is achieved through the use of kernel functions, which allow SVMs to implicitly map input data into higher-dimensional feature spaces where linear separation becomes possible.

Despite their effectiveness, SVMs come with some trade-offs. They can be sensitive to the choice of kernel and parameters, and their training process can be computationally intensive, particularly for large datasets. Additionally, SVMs are primarily designed for binary classification tasks, although they can be extended to handle multi-class classification by using strategies such as one-vs-one or one-vs-all.

## Bernoulli Naive Bayes
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/0e4278ba-55e9-44d1-8841-86b0cdddeaad)

Bernoulli Naive Bayes is a variant of the Naive Bayes classifier specifically designed for binary feature vectors. It operates on the same principles as the traditional Naive Bayes algorithm but is particularly suited for text classification tasks where features represent the presence or absence of certain words in a document.

In Bernoulli Naive Bayes, each feature is treated as a binary variable indicating whether a particular word occurs in the document or not. The classifier then estimates the probability of each class given the presence or absence of each feature, using the Bernoulli distribution. It assumes that the presence of a feature is generated by a Bernoulli distribution independently of each other feature.

Despite its simplicity, Bernoulli Naive Bayes can achieve good performance in text classification tasks, especially when dealing with sparse binary data typical in natural language processing applications. It is computationally efficient and can handle large-scale datasets with ease.

However, like other variants of Naive Bayes, Bernoulli Naive Bayes also relies on the assumption of feature independence, which may not always hold in practice. Additionally, it may not perform well when dealing with continuous or multi-modal data.

## Logistic Regression
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/1f68f617-ed59-4ebd-bec7-18c5ed2e20a1)
Logistic regression is a popular statistical method used for binary classification tasks, where the target variable or outcome is binary, meaning it can take only two possible values (e.g., 0 or 1, yes or no). Despite its name, logistic regression is a classification algorithm rather than a regression one.

The core idea behind logistic regression is to model the probability that a given instance belongs to a particular class. It does this by fitting a logistic function, also known as the sigmoid function, to the input data. The sigmoid function maps any real-valued input to a value between 0 and 1, representing the probability of the positive class.

During training, logistic regression estimates the parameters of the logistic function by maximizing the likelihood of observing the training data under the assumed model. This is typically done using optimization techniques such as gradient descent.

One of the key advantages of logistic regression is its simplicity and interpretability. The model coefficients can be directly interpreted as the impact of each feature on the log-odds of the outcome, providing insights into the relationship between the features and the target variable.

Logistic regression can also handle both linear and nonlinear relationships between the features and the target variable, by incorporating polynomial terms or interaction effects.

Despite its simplicity, logistic regression can perform well in many real-world classification tasks, especially when the decision boundary between classes is approximately linear. It is widely used in various fields such as medicine, finance, and marketing for tasks like disease diagnosis, credit risk assessment, and customer churn prediction.

## Gaussian Naive Bayes
Gaussian Naive Bayes is another variant of the Naive Bayes classifier, but it assumes that the features follow a Gaussian (normal) distribution. This makes it suitable for classification tasks where the features are continuous variables. Here's a brief passage on Gaussian Naive Bayes:

Gaussian Naive Bayes is a variant of the Naive Bayes classifier that is well-suited for classification tasks with continuous features. Unlike Bernoulli Naive Bayes, which works with binary features, Gaussian Naive Bayes assumes that the features follow a Gaussian distribution.

In Gaussian Naive Bayes, the probability density function of each feature in each class is estimated using the Gaussian (normal) distribution. Given a set of training data, the classifier calculates the mean and variance of each feature for each class. During prediction, it then uses these statistics to compute the likelihood of observing a particular feature value given each class and combines them with prior probabilities using Bayes' theorem to make predictions.

Despite its simplicity and the assumption of feature independence, Gaussian Naive Bayes can perform well in many real-world classification tasks, especially when dealing with continuous-valued features. It is computationally efficient and requires minimal tuning of parameters, making it particularly suitable for tasks with limited training data or when computational resources are constrained.

## Shap Explainer Model
SHAP values are a common way of getting a consistent and objective explanation of how each feature impacts the model's prediction.

SHAP values are based on game theory and assign an importance value to each feature in a model. Features with positive SHAP values positively impact the prediction, while those with negative values have a negative impact. The magnitude is a measure of how strong the effect is.

SHAP values are model-agnostic, meaning they can be used to interpret any machine learning model, including:

Linear regression
Decision trees
Random forests
Gradient boosting models
Neural networks


## Model used
The primary model which has been used is Logistic regression

## Libraries and Usage

```
#IMPORTING ALL THE LIBRARIES

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

```






## Accuracy
There was a very high Accuracy from the model as we were able to get the decision variables from requiredvarious models





## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is used in biomedical industries extensively. Prediction of a disease has a direct relation with synthetic data generation which is used in generative adversarial networks(GANs), and hence is very important in today's world.
## Appendix

A very crucial project in the realm of data science and bio-medical domain using visualization techniques as well as machine learning modelling.

## Tech Stack

**Client:** Python, Naive byes classifier, gaussian naive byes, support vector machine, random forest, decision tree classifier, logistic regression model, EDA analysis, machine learning, sequential model of ML, SHAP explainer model, data visualization libraries of python.


## Output
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/b7261509-050c-4a39-8a4d-6ce6923984d5)
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/a87b80b3-46b0-4661-afd8-eac490957016)
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/64d89268-34e1-41f4-9ea6-4538f4db64da)


![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/16830114-972d-400d-88ac-d27a073aa0b6)
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/f8d11fb1-f6c8-44b6-aead-38e02458bbf6)
![image](https://github.com/Prayag-Chawla/Lung-Cancer-Detection/assets/92213377/e7977640-ec5a-4a94-9b5e-219adaab7bea)

## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

