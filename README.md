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
<hr>

✅**Key Features**

**TF-IDF Vectorization**: Converts text data into numerical features.  
**MultinomialNB**: Naive Bayes classifier for text classification.  
**LogisticRegression**: Another classical model for comparison and ensemble.  
**VotingClassifier (soft)**: Combines predictions using averaged probabilities.  
**Pipeline**: End-to-end model (TF-IDF + VotingClassifier) packed and saved.  
**Sentiment Prediction**: Predict sentiment (0 = Negative, 4 = Positive) for new tweets.  
**Trend Visualization**: Monthly sentiment trend plotted using matplotlib.



📊**Dataset**
The dataset used is `twitter_data.csv`, which contains tweets labeled with:

- `0` : Negative sentiment  
- `4` : Positive sentiment

Ensure the dataset contains at least:

- `text` : The tweet content  
- `target` : Sentiment label (0 or 4)  
- `date` : Date of tweet (used in trend analysis)
<hr>

🚀**How to Run**

Follow these steps to set up and run the project locally:

**1. Install Dependencies**

```pip install -r requirements.txt```

**2. Train the Model**
Open and execute the cells in train_model.ipynb. This notebook will:

- Perform text vectorization using **TF-IDF**.
- Train the **MultinomialNB** and **LogisticRegression** models.
- Combine these models using a **VotingClassifier**.
- Save the complete, trained pipeline to `model_pipeline.joblib`.


**3. Predict Sentiment**</br>
Launch `predict_model.ipynb`. This notebook demonstrates how to:

- Load the saved `model_pipeline.joblib`.
- Make sentiment predictions on new textual inputs.

**4. Visualize Trends**</br>
Execute `sentiment_trend_plot.ipynb` to:</br>
Generate and display plots illustrating the monthly sentiment trends from the dataset using matplotlib.
<hr>

📌**Requirements**</br>
The project dependencies are listed in requirements.txt and include:
- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
<hr>

📷**Sample Output**</br>
- 📈 Monthly Trends: Positive vs. negative sentiment over time.
- ✅ Console Output: Model accuracy & classification report.
<hr>

🧠**Model Insights**</br>
The ensemble approach, combining Naive Bayes and Logistic Regression with soft voting, offers distinct advantages:
- 🧠 Naive Bayes: Fast, robust, and effective on sparse text data.
- 📊 Logistic Regression: Offers strong generalization.
- 🤝 Soft Voting: Averages predictions for balanced, accurate results.








