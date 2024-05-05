from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate, selection_rate, count, demographic_parity_ratio, equalized_odds_ratio
from fairlearn.adversarial import AdversarialFairnessClassifier
from sklearn.metrics import accuracy_score, precision_score

import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

DATA_PATH = "datasets/heart_2020_cleaned.csv"

def get_data():
    # Load the dataset
    data = pd.read_csv(DATA_PATH) 

    # First column is the target variable and the rest are features
    X = data.iloc[:,1:]
    y = data.iloc[:,0]

    X = X.drop('AgeCategory', axis=1) # Excluding Age

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    y = pd.get_dummies(y, drop_first=True)

    # Step 3: Ensure Numeric Data Types
    X = X.astype(int)
    y = y.astype(int)

    # Initialize RandomOverSampler
    sampler = RandomUnderSampler()

    # Undersample the majority class (No Heart Disease)
    X, y = sampler.fit_resample(X, y)

    # X['Is_White'] = (X['Race_Asian'] + X['Race_Black'] + 
    #                 X['Race_Hispanic'] + X['Race_Other']).eq(0).astype(int)
    # sf_indx = X.columns.get_loc("Is_White")

    sf_indx = X.columns.get_loc("Sex_Male")

    return X, y, sf_indx

def default_preprocess(X, y, sf_indx):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    sf_train = X_train[:, sf_indx]
    sf_test = X_test[:, sf_indx]

    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, sf_train, sf_test

def ballanced_preprocess(X, y, sf_indx):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    sf_train = X_train[:, sf_indx]
    sf_test = X_test[:, sf_indx]

    women_samples = X_train[sf_train == 0]
    women_labels = y_train[sf_train == 0]

    men_samples = X_train[sf_train == 1]
    men_labels = y_train[sf_train == 1]

    num_women_samples = women_samples.shape[0]
    num_men_samples = men_samples.shape[0]

    target_num = min(num_women_samples, num_men_samples)

    random_indices = np.random.choice(num_men_samples, target_num, replace=False)
    balanced_men_samples = men_samples[random_indices]
    balanced_men_labels = men_labels[random_indices]

    # Combine balanced subset of men samples with all women samples
    balanced_X = np.concatenate((balanced_men_samples, women_samples), axis=0)
    balanced_y = np.concatenate((balanced_men_labels, women_labels), axis=0)

    # Update sf_train to subsampled level
    sf_train = balanced_X[:, sf_indx]
    
    scaler = preprocessing.StandardScaler().fit(balanced_X)
    balanced_X = scaler.transform(balanced_X)
    X_test = scaler.transform(X_test)

    return balanced_X, X_test, balanced_y, y_test, sf_train, sf_test



def train_log_reg(X_train, y_train):
    y_train = y_train.flatten()
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


def model_analysis(model, X_test, y_test, sf_test, X_df):  
    y_test = y_test.flatten()
    predictions = model.predict(X_test)

    metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "false positive rate": false_positive_rate,
    "false negative rate": false_negative_rate,
    "selection rate": selection_rate,
    "count": count
    }

    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_test, y_pred=predictions, sensitive_features={"Sex" : sf_test})
    
    print(metric_frame.by_group)

    dpr = demographic_parity_ratio(y_test, predictions, sensitive_features={"Sex" : sf_test})
    print(f'Demographic Parity Ratio: {dpr:.4f}')

    eor = equalized_odds_ratio(y_test, predictions, sensitive_features={"Sex" : sf_test})
    print(f"Equalized Odds Ratio: {eor:.4f}")
    
    fig= metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[2,3],
        legend=False,
        figsize=[12, 8],
        title="Show all metrics",
    )
    plt.show()

    return 

def get_top_features(model, X_df):
    coef = np.absolute(model.coef_[0])
    top_five = np.argsort(coef)[::-1]
    column_names = X_df.columns[top_five].tolist()
    n = 10

    print(f"TOP {n} FEATURES by weighting")
    for i in range(n):
        print(f'Feature #{i+1}: {column_names[i]}')


def compare_original():
    '''
    RESULTS:

        accuracy  precision  false positive rate  false negative rate  selection rate   count
    Sex                                                                                       
    0    0.739911   0.732232             0.178327             0.366334        0.376349  8524.0
    1    0.713088   0.738615             0.325369             0.255783        0.556848  9543.0

    Demographic Parity Ratio:  0.6758561846760203
    Equalized Odds Ratio:  0.5480755477697553

    TOP 10 FEATURES
    Feature #1: GenHealth_Fair
    Feature #2: GenHealth_Good
    Feature #3: GenHealth_Poor
    Feature #4: Stroke_Yes
    Feature #5: Diabetic_Yes
    Feature #6: Sex_Male
    Feature #7: GenHealth_Very good
    Feature #8: DiffWalking_Yes
    Feature #9: SkinCancer_Yes
    Feature #10: Smoking_Yes
    '''
    X_df, y_df, sf_indx = get_data()
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X_train, X_test, y_train, y_test, sf_train, sf_test = default_preprocess(X, y, sf_indx)
    unballanced_model = train_log_reg(X_train, y_train)
    model_analysis(unballanced_model, X_test, y_test, sf_test, X_df)
    get_top_features(unballanced_model, X_df)


def compare_ballanced():
    '''
    RESULTS:

         accuracy  precision  false positive rate  false negative rate  selection rate   count
    Sex                                                                                       
    0    0.730148   0.729990             0.167663             0.404175        0.352659  8538.0
    1    0.717389   0.752288             0.310024             0.261098        0.550320  9529.0

    Demographic Parity Ratio:  0.6408247089915609
    Equalized Odds Ratio:  0.5408081569162376
    '''
    X_df, y_df, sf_indx = get_data()
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X_train, X_test, y_train, y_test, sf_train, sf_test = ballanced_preprocess(X, y, sf_indx)
    ballanced_model = train_log_reg(X_train, y_train)
    model_analysis(ballanced_model, X_test, y_test, sf_test, X_df)
    get_top_features(ballanced_model, X_df)


def compare_exclusion():
    '''
    RESULTS: 

         accuracy  precision  false positive rate  false negative rate  selection rate   count
    Sex                                                                                       
    0    0.730053   0.685929             0.256617             0.286907        0.457499  8435.0
    1    0.720619   0.782258             0.239664             0.311157        0.489203  9632.0

    Demographic Parity Ratio:  0.9351922169229793
    Equalized Odds Ratio:  0.9339367363505796

    TOP 10 FEATURES
    Feature #1: PhysicalActivity_Yes
    Feature #2: GenHealth_Good
    Feature #3: GenHealth_Fair
    Feature #4: Stroke_Yes
    Feature #5: Diabetic_No, borderline diabetes
    Feature #6: GenHealth_Poor
    Feature #7: Smoking_Yes
    Feature #8: KidneyDisease_Yes
    Feature #9: DiffWalking_Yes
    Feature #10: Asthma_Yes
    '''
    X_df, y_df, sf_indx = get_data()
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X_train, X_test, y_train, y_test, sf_train, sf_test = default_preprocess(X, y, sf_indx)

    nosf_X_train = np.delete(X_train, sf_indx, axis=1)
    nosf_X_test = np.delete(X_test, sf_indx, axis=1)
    excluded_model = train_log_reg(nosf_X_train, y_train)
    model_analysis(excluded_model, nosf_X_test, y_test, sf_test, X_df)
    get_top_features(excluded_model, X_df)


def compare_adversarial():
    '''
    RESULTS: 

         accuracy  precision  false positive rate  false negative rate  selection rate   count
    Sex                                                                                       
    0    0.644293   0.569438             0.393995             0.304610        0.523099  8507.0
    1    0.630021   0.679227             0.386591             0.356929        0.530230  9560.0

    Demographic Parity Ratio:  0.9865501778988226
    Equalized Odds Ratio:  0.9247629327437474
    '''

    X_df, y_df, sf_indx = get_data()
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X_train, X_test, y_train, y_test, sf_train, sf_test = default_preprocess(X, y, sf_indx)

    predictor_model = tf.keras.Sequential([
    tf.keras.layers.Dense(X_train.shape[1], activation='relu'), # Single layer Neural Network (acts the same as logistic regression)
    tf.keras.layers.Dense(y_train.shape[1])])

    adversary_model = tf.keras.Sequential([ 
        tf.keras.layers.Dense(32, activation='relu'), 
        tf.keras.layers.Dense(1)])
    
    mitigator = AdversarialFairnessClassifier(
        predictor_model=predictor_model,
        adversary_model=adversary_model
    )

    mitigator.fit(X_train, y_train, sensitive_features=sf_train)

    model_analysis(mitigator, X_test, y_test, sf_test, X_df)

def compare_combined():
    '''
    RESULTS:

         accuracy  precision  false positive rate  false negative rate  selection rate   count
    Sex                                                                                       
    0    0.625501   0.553843             0.433715             0.297202        0.550436  8486.0
    1    0.654942   0.684800             0.412293             0.291908        0.577497  9581.0

    Demographic Parity Ratio:  0.953140689213763
    Equalized Odds Ratio:  0.9506087130132541
    '''
    X_df, y_df, sf_indx = get_data()
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    X_train, X_test, y_train, y_test, sf_train, sf_test = ballanced_preprocess(X, y, sf_indx)

    predictor_model = tf.keras.Sequential([
    tf.keras.layers.Dense(X_train.shape[1], activation='relu'), # Acts the same as linear regression
    tf.keras.layers.Dense(y_train.shape[1])])

    adversary_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)])
    
    mitigator = AdversarialFairnessClassifier(
        predictor_model=predictor_model,
        adversary_model=adversary_model
    )

    mitigator.fit(X_train, y_train, sensitive_features=sf_train)

    model_analysis(mitigator, X_test, y_test, sf_test, X_df)


# compare_original()
# compare_ballanced()
# compare_exclusion()
# compare_adversarial()
# compare_combined()
