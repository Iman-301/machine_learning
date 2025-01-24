Task 1: Churn Prediction
Project Description: Built a churn prediction model using machine learning. The model identifies whether a customer will churn based on their characteristics and usage.
Technologies Used: Python, Pandas, Scikit-learn, SMOTE, Matplotlib, Seaborn
Steps Taken:
Preprocessed the dataset by cleaning null values, encoding categorical variables, and addressing class imbalance using SMOTE.
Trained a Random Forest Classifier model.
Evaluated the model using metrics like accuracy, precision, recall, and ROC-AUC.
Challenges Faced: Addressing class imbalance and tuning the model for better performance.
Outcomes:
Achieved accuracy of 84.29%.
Generated a confusion matrix for deeper analysis of misclassifications.


Task 2: Recommender System
Project Description: Developed a collaborative filtering recommender system using Surprise library to predict user-product ratings.
Technologies Used: Python, Pandas, Surprise, Scikit-learn
Steps Taken:
Preprocessed and cleaned the dataset by removing duplicates and invalid ratings.
Trained the SVD model on a split dataset.
Evaluated the model using RMSE and generated top-N recommendations for users.
Challenges Faced: Managing computational costs for a large dataset.
Outcomes:
RMSE score of 1.24 on test data.
Generated top-10 product recommendations for each user.


Task 3: Image Classification (CIFAR-10)
Project Description: Created an image classification model for the CIFAR-10 dataset using convolutional neural networks.
Technologies Used: TensorFlow, Keras, Matplotlib, Seaborn
Steps Taken:
Preprocessed image data by normalizing pixel values and performing data augmentation.
Designed a CNN architecture with Conv2D, MaxPooling, and Dropout layers.
Trained the model using data augmentation and evaluated it on the test set.
Challenges Faced: Tuning the hyperparameters and improving the model’s generalization ability.
Outcomes:
Test accuracy of 74% with epoch of 25 which increase per increase of epoch.
Visualized predictions and confusion matrix for a detailed analysis.



Task 4: Sentiment Analysis
Project Description: Built a text sentiment analysis model to classify tweets into positive, neutral, and negative sentiments.
Technologies Used: Python, Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn
Steps Taken:
Cleaned and preprocessed text data by removing noise, lemmatizing, and vectorizing using TF-IDF.
Built a logistic regression model for sentiment classification.
Evaluated the model using accuracy and confusion matrix.
Challenges Faced: Preprocessing noisy text data for high accuracy.
Outcomes:
Achieved accuracy of 79%.
Generated classification reports and confusion matrix for evaluation.

