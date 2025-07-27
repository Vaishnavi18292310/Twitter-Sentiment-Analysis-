# Twitter-Sentiment-Analysis

This project delves into sentiment analysis of tweets, leveraging classical machine learning models. It employs TF-IDF for robust feature extraction and trains Naive Bayes and Logistic Regression classifiers. These models are then harmonized using a Soft VotingClassifier to enhance predictive performance. The resulting model is capable of predicting sentiment and visualizing temporal trends in tweet data.
```
📁 Project Structure

├── train_model.ipynb           # Training notebook for TF-IDF + VotingClassifier 
├── predict_model.ipynb         # Load model and predict sentiment on new text
├── sentiment_trend_plot.ipynb  # Visualization of sentiment trends using matplotlib
├── requirements.txt            # Dependencies
├── model_pipeline.joblib       # Trained pipeline model (TF-IDF + Voting)
└── twitter_data.csv            # Input dataset (0 = Negative, 4 = Positive)

```
✅ **Key Features**

**TF-IDF Vectorization:** Transforms raw text into meaningful numerical features for machine learning models.
**Multinomial Naive Bayes:** A probabilistic classifier well-suited for text classification tasks due to its efficiency and performance on sparse data.
**Logistic Regression:** A powerful linear model used for its generalization capabilities and as an ensemble component.
**Soft VotingClassifier:** Combines the predictions of Naive Bayes and Logistic Regression by averaging their probabilities, leading to a more robust and often more accurate ensemble model.
**Machine Learning Pipeline:** An integrated pipeline that encapsulates both feature extraction (TF-IDF) and the ensemble model (VotingClassifier) for streamlined model saving and loading.
**Sentiment Prediction:** Enables the prediction of sentiment labels (0 = Negative, 4 = Positive) for unseen tweet data.
**Trend Visualization:** Generates informative plots using matplotlib to illustrate monthly sentiment shifts and patterns.

📊 Dataset
The project utilizes the twitter_data.csv dataset. This dataset comprises tweets annotated with:

0: Denoting Negative sentiment

4: Denoting Positive sentiment

Please ensure your dataset includes at least the following columns:

text: The actual tweet content.

targett: The corresponding sentiment label (0 or 4).

date: The date of the tweet, crucial for time-based trend analysis.

.

🚀 How to Run
Follow these steps to set up and run the project locally:

1. Install Dependencies
Bash

pip install -r requirements.txt
2. Train the Model
Open and execute the cells in train_model.ipynb. This notebook will:

Perform text vectorization using TF-IDF.

Train the MultinomialNB and LogisticRegression models.

Combine these models using a VotingClassifier.

Save the complete, trained pipeline to model_pipeline.joblib.

3. Predict Sentiment
Launch predict_model.ipynb. This notebook demonstrates how to:

Load the saved model_pipeline.joblib.

Make sentiment predictions on new textual inputs.

4. Visualize Trends
Execute sentiment_trend_plot.ipynb to:

Generate and display plots illustrating the monthly sentiment trends from the dataset using matplotlib.

📌 Requirements
The project dependencies are listed in requirements.txt and include:

numpy

pandas

scikit-learn

matplotlib

joblib

📷 Sample Output
📈 Monthly Sentiment Trends: A plot showcasing the positive versus negative sentiment distribution over time.

✅ Console Output: Displays model accuracy and a detailed classification report during training.

🧠 Model Insights
The ensemble approach, combining Naive Bayes and Logistic Regression with soft voting, offers distinct advantages:

Naive Bayes: Provides fast and robust performance, particularly effective on sparse text data.

Logistic Regression: Contributes strong generalization capabilities.

Soft Voting: Leverages the strengths of both classifiers by averaging their prediction probabilities, resulting in a more balanced and often more accurate overall prediction.









