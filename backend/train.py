## Let's load our dataset and tools. 
import pandas as pd #Pandas is a data manipulation and analysis library.
import joblib #Joblib is a library for saving and loading Python objects.

# sklearn: data splitting
from sklearn.model_selection import train_test_split, GridSearchCV #train_test_split is a function that splits the data into training and testing sets. GridSearchCV is a function that performs a grid search to find the best hyperparameters for a model.

# sklearn: preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder #StandardScaler is a function that scales the data to have a mean of 0 and a standard deviation of 1. OneHotEncoder is a function that converts categorical variables into dummy variables. LabelEncoder is a function that converts categorical variables into numerical variables.
from sklearn.compose import ColumnTransformer #ColumnTransformer is a function that applies a transformation to specific columns of the data.
from sklearn.pipeline import Pipeline #Pipeline is a function that chains multiple transformations together.
from sklearn.impute import SimpleImputer, KNNImputer #SimpleImputer is a function that imputes missing values. KNNImputer is a function that imputes missing values using the k-nearest neighbors algorithm.

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
def load_data():
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
    
    return df_train, df_test

def clean_data(df_train, df_test):
    ## Displaying a sample of the data for verification purposes.
    attack_types = list(df_train['label'].unique())
    attack_types.remove('normal')
    print("Training Data Attack Types:")
    print("\n".join(attack_types))

    test_attack_types = list(df_test['label'].unique())
    test_attack_types.remove('normal')
    print("Testing Data Attack Types:")
    print("\n".join(test_attack_types))

    # Mapping the attack types to their respective categories.
    # Using categories as keys to reduce repetition and avoid typos.
    attack_categories = {
        'dos':   ['smurf', 'neptune', 'teardrop', 'pod', 'back', 'land',
                'processtable', 'worm', 'udpstorm', 'mailbomb', 'apache2'],
        'probe': ['ipsweep', 'portsweep', 'satan', 'nmap', 'saint', 'mscan'],
        'r2l':   ['warezclient', 'warezmaster', 'guess_passwd', 'ftp_write',
                'imap', 'phf', 'spy', 'multihop', 'xsnoop', 'xlock',
                'named', 'sendmail', 'snmpgetattack', 'snmpguess'],
        'u2r':   ['httptunnel', 'sqlattack', 'rootkit', 'loadmodule',
                'perl', 'buffer_overflow', 'xterm', 'ps'],
    }

    # Creating a dictionary to map the attack types to their respective categories.
    label_map = {attack: category for category, attacks in attack_categories.items() for attack in attacks}

    df_train['label'] = df_train['label'].replace(label_map) # Map the attack types in the training dataframe to their respective categories.
    df_test['label'] = df_test['label'].replace(label_map) # Map the attack types in the testing dataframe to their respective categories.

    print(df_train['label'].unique()) # Display the unique attack types in the training dataframe.
    print(df_test['label'].unique()) # Display the unique attack types in the testing dataframe.

    # Separate features (X) and labels (y)
    X_train = df_train.drop('label', axis=1) # Drop the label column from the training dataframe. "X" just means features. This is what we feed into the model.
    y_train = df_train['label'] # Set the label column as the training labels. "y" just means labels, or the "target variable" which is what we are trying to predict.

    X_test = df_test.drop('label', axis=1) # Drop the label column from the testing dataframe.
    y_test = df_test['label'] # Set the label column as the testing labels.

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test, label_encoder

def build_preprocessor(X_train):
    # text columns that need One-Hot Encoding
    categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()

    # number columns that need Standard Scaling
    numerical_cols = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()

    # creating the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    return preprocessor
    

def train_model(X_train, y_train, X_test, y_test, preprocessor, label_encoder):
    
    # Hyperparameter tuning: Using the GridSearchCV tool here to test different combinations of hyperparameters.
    # We are creating a separate parameter grid for each model.
    lr_param_grid = {
        'classifier__C': [0.1, 1, 10], # Regularization strength. Smaller values = stronger regularization.
        'classifier__l1_ratio': [0, 0.5, 1], # Elasticnet mixing parameter: 0 = L2, 1 = L1, 0.5 = 50/50 mix.
        'classifier__solver': ['saga'], # Algorithm to use. liblinear is good for small datasets, saga can handle larger datasets and l1 penalty.
        'classifier__max_iter': [10000], # Maximum number of iterations. Reduced to 1000 to prevent infinite spinning.
    }

    rf_param_grid = {
        'classifier__n_estimators': [125, 250], # Number of trees in the forest.
        'classifier__max_depth': [10, 20, None], # Maximum depth of the trees.
        'classifier__max_features': ['sqrt', 'log2'], # Number of features to consider for each split.
        'classifier__min_samples_split': [2, 5], # Minimum number of samples required to split a node.
        'classifier__min_samples_leaf': [1, 2], # Minimum number of samples required to be at a leaf node.
        'classifier__bootstrap': [True, False], # Whether to use bootstrap sampling.
    }

    mlp_param_grid = {
        #classifier__ is used to specify that we are tuning the classifier part of the pipeline. These settings are sent forward specifically to the tool nicknamed 'classifier'.
        'classifier__hidden_layer_sizes': [(124,64), (124,64,32), (64,32), (42,24,12)], # Number of hidden layers and neurons in each layer. Testing depth and width... 2 vs 3 layers and 42 vs 128 wide.
        'classifier__activation': ['relu'], # Activation function to use. Tells the neuron when and how to fire.
        # relu is an industry standard. If a number is negative, it is set to 0. If it's positive, it passes through untouched.
        # tanh is classic and curves numbers into a range between -1 and 1. It's a bit more computationally expensive.
        'classifier__solver': ['adam'], # Solver to use.
        # adam is a popular choice for deep learning. It's an adaptive learning rate optimization algorithm. Naturally adapts to data. Much faster.
        # sgd is a classic choice for deep learning. It's a stochastic gradient descent algorithm. Slower, but reliable.
        'classifier__alpha': [0.001, 0.01, 0.1], # Regularization parameter. Sometimes neural networks try to perfectly memorize the training data (overfitting).
        # alpha adds a penalty to the model for being too complex, forcing it to generalize better.
        'classifier__max_iter': [10000], # Maximum number of iterations. Increasing the iterations allows the model to train for longer, potentially finding a better solution.
        'classifier__early_stopping': [True], # Whether to use early stopping. 
    }


# Model Dictionary - each model is paired with its matching param_grid
    model_configs = {
        'LogReg': (LogisticRegression(class_weight='balanced', random_state=42, tol=0.1), lr_param_grid),
        'RandForest': (RandomForestClassifier(class_weight='balanced', random_state=42), rf_param_grid),
        'MLP': (MLPClassifier(random_state=42), mlp_param_grid),
    }

    # Track the best model across all configs
    best_model = None
    best_f1 = 0
    best_name = ""

    for name, (model, param_grid) in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Tuning {name}...")
        print(f"{'='*60}")
    
        # Build a new pipeline for each model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Creating the grid search object.
        grid_search = GridSearchCV(
            pipeline, # The pipeline to use.
            param_grid, # The hyperparameters to tune.
            cv=5, # 5-fold cross-validation. Basically, it splits the data into 5 parts, trains on 4, tests on 1, then repeats 5 times so that each part gets tested on once.
            scoring='f1_macro', # The scoring metric to use. We want to use F1 score because it is the harmonic mean of precision and recall. 
            # We want to be precise when we say something is an attack (precision), and we want to be sure we catch all the attacks (recall).
            verbose=1, # Print the progress. 1 means print, 0 means don't print.
            n_jobs=-1 # Use all available cores.
        )

        # Train the pipeline on training data
        grid_search.fit(X_train, y_train)
    
        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
    
        # Make predictions on the test data
        y_pred = grid_search.predict(X_test)
    
        f1 = f1_score(y_test, y_pred, average='weighted') * 100
        print(f"\n{name} F1 Score: {f1:.3f}%")
        print(f"\nClassification Report:\n\n{classification_report(y_test, y_pred, target_names = label_encoder.classes_)}")
        print(f"\nConfusion Matrix:\n\n{confusion_matrix(y_test, y_pred)}")
        print("\n")

    # Track the best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = grid_search.best_estimator_
            best_name = name


    # After the loop — deploy the best model
    print(f"\n{'='*60}")
    print(f"Best model: {best_name} with F1 score of: {best_f1:.3f}%")
    print(f"{'='*60}")

    if best_f1 >= 90:    
        print(f"Model achieved F1 Score of {best_f1:.3f}%, automatically deploying...")
        # Serialize the trained model so it can be used in the backend of the web application.
        joblib.dump({'model': best_model, 'encoder': label_encoder}, 'nidd_model.pkl') # Saves the best model and encoder to a file called nidd_model.pkl
        print("Model saved to nidd_model.pkl")

    elif best_f1 >= 80:
        print(f"Model achieved an F1 score of {best_f1:.3f}% (good enough to deploy).")
    
        while True:
            deploy = input("Do you want to deploy this model? y/n:")
            if deploy not in ['y', 'n']:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            else:
                break
    
        if deploy == 'y':
            print("Deploying model...")
            joblib.dump({'model': best_model, 'encoder': label_encoder}, 'nidd_model.pkl')
            print("Model saved to nidd_model.pkl")
        else:
            print("Model not deployed.")

    else:
        print("Model needs improvement. Try tuning the hyperparameters in train.py.")



if __name__ == '__main__':
    df_train, df_test = load_data()
    X_train, y_train, X_test, y_test, label_encoder = clean_data(df_train, df_test)
    preprocessor = build_preprocessor(X_train)
    train_model(X_train, y_train, X_test, y_test, preprocessor, label_encoder) 