import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

df = pd.read_csv('loan_approval_preprocessing.csv')

# loan_status is the target
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Define categorical features
cat_features = ['occupation_status', 'product_type', 'loan_intent']

# split data into ratio -> train : valid : test = 60 : 20 : 20
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.4, random_state=210)

X_valid, X_test, y_valid, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=210)

# Create pool for catboost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment('loan_approval')

input_example = X_train[0:5]

# mlflow.autolog()

# with mlflow.start_run():
# Log parameters
max_tree = 1000
learning_rate = 0.1
eval_metrics = 'AUC'
random_seed = 210
early_stopping = 50

# mlflow.log_param("iterations", max_tree)
# mlflow.log_param("learning_rate", learning_rate)
# mlflow.log_param("eval_metric", "AUC")
# mlflow.log_param("random_seed", random_seed)
# mlflow.log_param("early_stopping_rounds", early_stopping)

# Train model
model = CatBoostClassifier(
  iterations=max_tree,
  learning_rate=learning_rate,
  eval_metric=eval_metrics,
  random_seed=random_seed,
  early_stopping_rounds=early_stopping,
  verbose=100
)

model.fit(train_pool, eval_set=valid_pool)

# # log model after fit
# mlflow.catboost.log_model(
#   cb_model=model,
#   artifact_path="model",
#   input_example=input_example
# )

# accuracy = model.score(X_test, y_test)
# mlflow.log_metric("accuracy", accuracy)

# y_prob = model.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_prob)
# mlflow.log_metric("roc_auc", roc_auc)