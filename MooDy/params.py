### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[AR] [BSAS] [JP-Merea] MooDy v2"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

PATH_1 = 'data/labeled_tweet.csv'
PATH_2 = 'data/dolar.csv'
PATH_3 = 'data/raw.csv'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'MooDy'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'
