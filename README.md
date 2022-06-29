##### Adam Mickiewicz University, faculty of mathematics and computer science
##### author: Mateusz Czajka
##### class: recommendation systems

# Hotel room neural network recommender
The purpose of this project was to:
 - Prepare user features (if content based/hybrid recommender used)
 - Code neural network recommender (collaborative filtering in my case)
 - Use cuda to speed up training process
 - Tune recommender (optimizer (SGD, Adam, AdamW), optimizer params, recommender params, model (GMF, MLP, NeuMF))
 - Run the final evaluation of recommender and present its results against the Amazon and Netflix recommenders

# Requirements
    python 3
    numpy
    pandas
    matplotlib
    seaborn
    livelossplot
    hyperopt
    traceback


# Dataset
The original dataset was preprocessed to facilitate the work with data. Anonymized data from dataset comes from real hotel database.

## Original dataset
 - The original dataset: [hotel-recommender](https://github.com/C7A7A/hotel-recommender)/[data](https://github.com/C7A7A/hotel-recommender/tree/main/data)/[hotel_data](https://github.com/C7A7A/hotel-recommender/tree/main/data/hotel_data)/**hotel_data_original.csv**


## Preprocessed dataset
 - Preprocessed dataset: [hotel-recommender](https://github.com/C7A7A/hotel-recommender-neural-network)/[data](https://github.com/C7A7A/hotel-recommender-neural-network/tree/main/data)/[hotel_data](https://github.com/C7A7A/hotel-recommender-neural-network/tree/main/data/hotel_data)/**hotel_data_preprocessed.csv**

 - Preprocessed dataset with **interactions** between **users** and **items** and selected data: [hotel-recommender](https://github.com/C7A7A/hotel-recommender-neural-network)/[data](https://github.com/C7A7A/hotel-recommender-neural-network/tree/main/data)/[hotel_data](https://github.com/C7A7A/hotel-recommender-neural-network/tree/main/data/hotel_data)/**hotel_data_interactions_df.csv**

Part of **interactions preprocessed dataset** <br />
![Preprocessed dataset image](/screenshots/interactions_df.png?raw=true "Preprocessed dataset")


# User Features
There are three methods that prepare user features
### 1. probability method (was already given)
For **each user** for **each feature** it calculates the **probability** of its occurrence in data frame. <br />
![Probability method](/screenshots/users_df_prob.png?raw=true "Probability method dataset")

### 2. Average bucketing method
For **each numeric** feature it maps value to **average value** (for instance [50-100] -> 75).
After that rows are **grouped by user_id** and **mean** is applied on dataframe. <br />
![Avg bucketing method](/screenshots/users_df_avg_bucketing.png?raw=true "Avg bucketing method")

### 3. One hot encoding method
Vector of ones and zeroes.
Data frame is **grouped by user_id** and **1** is given if **feature exists**, otherwise 0. <br />
![One hot method](/screenshots/users_df_one_hot.png?raw=true "One hot method")


# Item Features (one hot encoding)
Part of **items_df** <br />
![Items dataframe image](/screenshots/items_df.png?raw=true "Items dataframe")

# Recommender
I decided to create collaborative filtering recommender. <br />
I tried to code hybrid recommender (content based + collaborative filterring) but with no avail.

# Tuning
### Models
I used 3 models during tuning (GMF, MLP, NeuMF). On average GMF < MLP < NeuMF during my tests, so I chose NeuMF as my main model and tried to get best results with it.

### Optimizers
- Adam/AdamW optimizer - "C" charts. Validation loss usually increase after few epochs and it is way bigger than training loss <br />
![Adam](/screenshots/NeuMF_Adam.png?raw=true "Adam chart")
- SGD - validation loss usually have the same shape as training loss and decrease similarly <br />
![SGD](/screenshots/MLP_SGD.png?raw=true "SGD chart")

At first glance, you could come to the conclusion that working with SGD optimizer would be more beneficial, but Adam gave me better results for some reason.

# Results
I managed to beat both recommenders (Amazon and Netflix) and got 0.2515 HR10 score.

**Best score** <br />
![Best results](/screenshots/best_result.png?raw=true "Best result")

**All results** <br />
![All results](/screenshots/results.png?raw=true "All results")

Best score parameters:
```
    batch_size = 56,
    embedding_dim = 32,
    lr = 0.0054,
    model_to_use = 'NeuMFModel',
    n_epochs = 13,
    n_neg_per_pos = 9,
    optimizer_to_use = 'Adam',
    weight_decay = 0.00005
```

