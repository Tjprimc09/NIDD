## Let's load our dataset and tools. 
import pandas as pd
import numpy as np
import joblib 

# sklearn: data splitting
from sklearn.model_selection import train_test_split, GridSearchCV

# sklearn: preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer

# sklearn: models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier, StackingClassifier, VotingClassifier

# sklearn: metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

import matplotlib.pyplot as mplt
import seaborn as sbn
import os
import json
from datetime import datetime

## Let's read our training dataset.set
## column names aquired from dataset documentation
columns = [
    'duration', 
    'protocol_type', 
    'service', 
    'flag', 
    'src_bytes', 
    'dst_bytes',
    'land', 
    'wrong_fragment', 
    'urgent', 
    'hot', 
    'num_failed_logins', 
    'logged_in',
    'num_compromised', 
    'root_shell', 
    'su_attempted', 
    'num_root',
    'num_file_creations', 
    'num_shells', 
    'num_access_files', 
    'num_outbound_cmds',
    'is_host_login', 
    'is_guest_login', 
    'count', 
    'srv_count', 
    'serror_rate',
    'srv_serror_rate', 
    'rerror_rate', 
    'srv_rerror_rate', 
    'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# creating environment agnostic code to read the datasets regardless of where the file is located.
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, "data", "KDDTrain+.txt")
    test_file = os.path.join(base_dir, "data", "KDDTest+.txt")

    ## Reading the datasets and creating dataframes with the column names defined above.
    df_train = pd.read_csv(train_file, header=None, names=columns)
    df_train.drop('difficulty', axis=1, inplace=True)
    df_test = pd.read_csv(test_file, header=None, names=columns)
    df_test.drop('difficulty', axis=1, inplace=True)

    ## Dropped the "difficulty" column. If left in as a feature, it would cause data leakage.
    ## The "difficulty" column is like a proxy for the label and would allow the model to achieve perfect accuracy.

except FileNotFoundError:
    print(f"Error: Could not find the dataset files.")
    print(f"Please download the KDD dataset and place it in the data folder.")
    print(f"https://www.kaggle.com/datasets/huseyinaydin/kdd-cup-1999-data-set-for-intrusion-detection")
    print(f"The files should be named KDDTrain+.txt and KDDTest+.txt")
    print(f"Place the files in the 'data' folder, located in the same directory as this script.")
    print(f"Run the script again after completing the download.")
    exit(1)


#Displaying a sample of the data for verification purposes.
attack_types = list(df_train['label'].unique())
attack_types.remove('normal')
print("Training Data Attack Types:")
print("\n".join(attack_types))

test_attack_types = list(df_test['label'].unique())
test_attack_types.remove('normal')
print("Testing Data Attack Types:")
print("\n".join(test_attack_types))

# Mapping the attack types to their respective categories.
label_map = {
    'smurf': 'dos',
    'neptune': 'dos',
    'teardrop': 'dos',
    'pod': 'dos',
    'back': 'dos',
    'land': 'dos',
    'processtable': 'dos',
    'worm': 'dos',
    'udpstorm': 'dos',
    'mailbomb': 'dos',
    'apache2': 'dos',
    'ipsweep': 'probe',
    'portsweep': 'probe',
    'satan': 'probe',
    'nmap': 'probe',
    'saint': 'probe',
    'mscan': 'probe',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    'guess_passwd': 'r2l',
    'ftp_write': 'r2l',
    'imap': 'r2l',
    'phf': 'r2l',
    'spy': 'r2l',
    'multihop': 'r2l',
    'xsnoop': 'r2l',
    'xlock': 'r2l',
    'named': 'r2l',
    'sendmail': 'r2l',
    'snmpgetattack': 'r2l',
    'snmpguess': 'r2l',
    'httptunnel': 'u2r',
    'sqlattack': 'u2r',
    'rootkit': 'u2r',
    'loadmodule': 'u2r',
    'perl': 'u2r',
    'buffer_overflow': 'u2r',
    'xterm': 'u2r',
    'ps': 'u2r'
}

df_train['label'] = df_train['label'].replace(label_map)

df_test['label'] = df_test['label'].replace(label_map)

print(df_train['label'].unique())
print(df_test['label'].unique())

# Separate features (X) and labels (y)
X_train = df_train.drop('label', axis=1)
y_train = df_train['label']

X_test = df_test.drop('label', axis=1)
y_test = df_test['label']

# text columns that need One-Hot Encoding
categorical_cols = ['protocol_type', 'service', 'flag']

# number columns that need Standard Scaling
numerical_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

# creating the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Model Dictionary

# models = {
#     'LogReg': LogisticRegression(max_iter=1000),
#     'RandForest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'MLP': MLPClassifier(random_state=42),
# }

# for name, model in models.items():
#     print(f"Training {name}...")
    
#     # Build a new pipeline for each model
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])

#     # Train the pipeline on training data
#     pipeline.fit(X_train, y_train)
    
#     # Make predictions on the test data
#     y_pred = pipeline.predict(X_test)
    
#     # Print the evaluation metrics
#     # print(f"Accuracy: {accuracy_score(y_test, y_pred) *100:.3f}%")
#     # print(f"Precision: {precision_score(y_test, y_pred, average='weighted') *100:.3f}%")
#     # print(f"Recall: {recall_score(y_test, y_pred, average='weighted') *100:.3f}%")
#     # print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted') *100:.3f}%")
#     # print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
#     # print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
#     # print("\n")


#     #MLP is the strongest model with "factory settings", so we will pull this machine into the "tuning bay" to see if we can improve its performance.


#Hyperparameter tuning: Using the GridSearchCV tool here to test different combinations of hyperparameters.
# We are creating a simple grid search to reduce computational cost. This grid has all the fat trimmed as is only testing 4 unique combinations.    
param_grid = {
    #classifier__ is used to specify that we are tuning the classifier part of the pipeline. These settings are send forward specifically to the tool nicknamed 'classifier'.
    'classifier__hidden_layer_sizes': [(10,), (10,10), (20,)], # Number of hidden layers and neurons in each layer. Here we are testing two 50 nueron layers back to back, and then a single 100 nueron layer.
    'classifier__activation': ['relu'], # Activation function to use. Tells the nueron when and how to fire.
    # relu is an industry standard. If a number is negative, it is set to 0. If it's positive, it passes through untouched.
    # tanh is classic and curves numbers into a range between -1 and 1. It's a bit more computationally expensive.
    'classifier__solver': ['adam'], # Solver to use.
    # adam is a popular choice for deep learning. It's an adaptive learning rate optimization algorithm. Naturally adapts to data. Much faster.
    # sgd is a classic choice for deep learning. It's a stochastic gradient descent algorithm. Slower, but reliable.
    'classifier__alpha': [0.001, 0.01, 0.1], # Regularization parameter. Sometimes nueral networks try to perfectly memorize the training data (overfitting).
    # alpha adds a penalty to the model for being too complex, forcing it to generalize better.
    'classifier__learning_rate': ['constant'], # Learning rate schedule.
    # constant: learning rate stays the same throughout training.
    # invscaling: learning rate decreases over time.
    # adaptive: learning rate decreases over time.
}

# Creating the pipeline specifically for the MLP model, instead of using the dictionary.
mlp_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(random_state=42))
    ])

# Creating the grid search object.
grid_search = GridSearchCV(
    mlp_pipeline, # The pipeline to use.
    param_grid, # The hyperparameters to tune.
    cv=5, # 5-fold cross-validation. Basically, it splits the data into 5 parts, trains on 4, tests on 1, then repeats 5 times so that each part gets tested on once.
    scoring='f1_weighted', # The scoring metric to use. We want to use F1 score because it is the harmonic mean of precision and recall. 
    # We want to be precise when we say something is an attack (precision), and we want to be sure we catch all the attacks (recall).
    verbose=1, # Print the progress. 1 means print, 0 means don't print.
    n_jobs=-1 # Use all available cores.
)

grid_search.fit(X_train, y_train) # Fitting the model to the data.

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

y_pred = grid_search.predict(X_test) # Making predictions on the test data.

f1 = f1_score(y_test, y_pred, average='weighted') *100

print("Accuracy:" + f"{accuracy_score(y_test, y_pred) *100:.3f}%\n")
print("Precision:" + f"{precision_score(y_test, y_pred, average='weighted') *100:.3f}%\n")
print("Recall:" + f"{recall_score(y_test, y_pred, average='weighted') *100:.3f}%\n")
print("F1 Score:" + f"{f1:.3f}%\n")
print("Classification Report:\n" + f"{classification_report(y_test, y_pred)}\n")
print("Confusion Matrix:\n" + f"{confusion_matrix(y_test, y_pred)}\n")



if f1 >= 99:    
    print("Model achieved F1 Score of 99% or higher, automatically deploying...")
    # Serialize the trained model so it can be used in the backend of the web application.
    trained_model = joblib.dump(grid_search.best_estimator_, 'nidd_model.pkl') #grabs the best model from the grid search using .best_estimator_ and saves it to a file called nidd_model.pkl
    print("Model saved to nidd_model.pkl")

elif f1 >= 90:
    print("Model is good enough to deploy, but not quite perfect.")
    
    while True:
        deploy = input("Do you want to deploy this model? y/n:")
        if deploy not in ['y', 'n']:
            print("Invalid input. Please enter 'y' or 'n'.")
            continue
        else:
            break
    
    if deploy == 'y':
        print("Deploying model...")
        # Serialize the trained model so it can be used in the backend of the web application.
        trained_model = joblib.dump(grid_search.best_estimator_, 'nidd_model.pkl') #grabs the best model from the grid search using .best_estimator_ and saves it to a file called nidd_model.pkl
        print("Model saved to nidd_model.pkl")

else:
    print("Model needs improvement. Try tuning the hyperparameters in the train.py file.\n")
    print("See line 204 for the hyperparameter grid.\n")





