# Sentiment Analysis for Stance & Category Detection of Arabic COVID Tweets
This project involves two sentiment classifiers that based on a given tweet decides its vaccination stance (whether its with, against or indifferent towards vaccination) and its category (vaccination news, celebrities, government plans,...). The category classification model won second place 🥈 and the project overall was granted bonuses in the competition that corresponded to the project in fulfillment of the classwork requirements of the NLP course taught to computer engineering juniors in Cairo University.

## Datasets & Preprocessing 📊
Training was done over 7000 arabic COVID related tweets scraped from Twitter. The dataset suffered from severe imbalance and small subsets

<img width="774" alt="image" src="https://user-images.githubusercontent.com/49572294/215448707-a35df590-9663-43ad-b603-f077f1375b40.png">

We considered three approaches for preprocessing (Hand-made, NLTK-based & Snowball-based). Each of them involved Tweet Cleaning, Arabic Letters Normalization, Stop-word Removal, Stemming & Tokenization.

## Features Extracted 🗿
We have considered BoW, TF-IDF (n-gram), One-hot, Aravec, FastTEXT and AraBERT Contextual Embeddings for features. 

To illustrate the difficulty of the task at hand, here is an example for feature visualization over the stance task (TF-IDF after PCA): <br>

<img width="758" alt="image" src="https://user-images.githubusercontent.com/49572294/215450284-5ac03319-4666-44af-b4bd-db07c8dcc82b.png">

## Models Considered 🤖
For each of the two tasks, we used two sets of models. A classical set composing Naive Bayes, SVM and Gradient Boosting and a deep learning
set composing FFNN, LSTM, GRU and ARABERT.

## Metric & Results 📉
The metric set for the competition was Macro F1 score which too sensitive to the present imbalance issues in the sense that 545 errors for the 2nd class are equivalent to 2 errors for the 7th class considering category classification over the dev set. Nonetheless, after a lot of fine tuning as could be evidenced by the logs (which were automated) our best models for each task were:

<table>
<tr>
<th>Task</th>
<td>
Stance Classification 
</td>
<td>
Category Classification
</td>
</tr>

<tr>
<th>
Best Model
</th>
<td>
AraBERT with Class Weighting
</td>
<td>
Naive Bayes with Class Weighting
</td>
</tr>

<tr>
<th>
Macro F1 Score
</th>
<td>
0.61
</td>
<td>
0.59
</td>
</tr>
</table>

The figures severely drop if class weighting is ignored.


## Running the Project 🚀
If you are a developer then you know how to navigate to the corresponding model/feature extractor/preprocessing module and run it (everything is notebooks). Understanding the directory structure should help
```
1-Preprocessing
   |-- DataInsight
   |   |-- DataInsight.ipynb
   |-- Preprocessing
   |   |-- Preprocess.ipynb
2-FeatureExtraction
   |-- BagOfWords
   |   |-- BagOfWords.ipynb
   |-- BowTF
   |   |-- BowTF.ipynb
   |-- ContextualEmbeddings
   |   |-- ContextualEmbeddings.ipynb
   |-- OneHot
   |   |-- OneHot.ipynb
   |-- TF-IDF
   |   |-- TF-IDF.ipynb
   |-- TweetFeatures
   |   |-- TweetFeatures.ipynb
   |-- WordEmbeddings
   |   |-- FastText.ipynb
   |   |-- WordEmbeddings.ipynb
   |   |-- data.txt
3-Models
   |-- Arabert
   |   |-- Arabert-1.ipynb
   |   |-- Arabert-2.ipynb
   |   |-- runs.csv
   |-- GRU
   |   |-- GRU-1.ipynb
   |   |-- GRU-2.ipynb
   |   |-- data.txt
   |   |-- runs.csv
   |-- GradientBoost
   |   |-- GB-1.ipynb
   |   |-- GB-2.ipynb
   |   |-- runs.csv
   |-- LSTM
   |   |-- LSTM-1.ipynb
   |   |-- LSTM-2.ipynb
   |   |-- runs.csv
   |-- NaiveBayes
   |   |-- NB-1.ipynb
   |   |-- NB-2.csv
   |   |-- NB-2.ipynb
   |   |-- model.pkl
   |   |-- model2.pkl
   |   |-- runs copy.csv
   |   |-- runs.csv
   |-- SVM
   |   |-- SVM-1.ipynb
   |   |-- SVM-2.ipynb
   |   |-- runs.csv
Dataset
   |-- SavedFeatures
   |   |-- CONTEXT
   |   |   |-- X_test.npy
   |   |   |-- x.npy
   |   |-- ONEHOT
   |   |   |-- Preprocessing.npy
    ...
   ```

## Collaborators

<!-- readme: contributors -start -->
<!-- readme: contributors -end -->


