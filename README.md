# Sentiment Analysis of Movie Reviews
Final project for the course *DD1418 Language Engineering with Introduction to Machine Learning* at KTH. Graded A, with full points on all criteria. Arvid Hedbäck and Lalo Saleh, December 17, 2023.

## Description
Movie reviews from IMDb (Internet Movie Database) have been classified as 1, 2, 3, 4 or 5, with 1 being the most negative sentiment. For this, the original reviews with a rating of 1-10 have been reclassified as 1-5 as follows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] => [1, 1, 2, 2, 3, 3, 4, 4, 5, 5].

The classification has been done two ways, using Naive Bayes and a Neural Network model, in order to compare the two methods. As expected, as can be seen in workbook.ipynb and in the report (written in Swedish), the neural network is significantly better at predicting the correct class. For example, the neural network has an overall accuracy of 78.6% compared to 53.3% for Naive Bayes. In addition, we experimented with using both unbalanced and balanced training data, concluding that balanced training data yields better and more reliable results.

## Usage
1. Download the movie review data from https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset. The extracted files are in JSON-format.
2. Run data_cleaning.ipynb from start to end, to remove non-word characters, stop words and lemmatize. The last cell of code saves the cleaned movie reviews and their ratings to a CSV file.
3. Run workbook.ipynb from start to end. Pay attention to comments and make changes as desired (for example if unbalanced training data is to be used instead of balanced, it model results are to be saved to external files, etcetera).
4. If a neural network model, vectorizer and the corresponding indices over the most important words have been saved, it is also possible to run demo.ipynb directly. Note that there is no additional functionality in demo.ipynb compared to workbook.ipynb.
