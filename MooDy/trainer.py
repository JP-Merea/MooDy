import joblib
from termcolor import colored
import mlflow
from TaxiFareModel.data import get_data_from_gcp, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.gcp import storage_upload
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.params import MLFLOW_URI, EXPERIMENT_NAME, BUCKET_NAME, MODEL_VERSION, MODEL_VERSION
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datos import *
from index import *
from model import *
from params import *


class Trainer(object):
    def __init__(self, df_sent, df_blue):
        """
            df_sent: dataframe with sentymental analysis 
            df_blue: dataframe dolar 
        """
        self.df_sent = df_sent
        self.df_blue = df_blue
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def MooDy(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def run(self):
        self.set_pipeline()
        self.mlflow_log_param("model", "Linear")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


#if __name__ == "__main__":
    # Get and clean data
    N = 100
    df = get_data_from_gcp(nrows=N)
    df = clean_data(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model_locally()
    storage_upload()
    
if __name__ == '__main__':
    df, blue = get_data()
    df = get_clean_data(df, blue)
    target= get_labels(0, df)
    a, b, c, d, e, f, g, h = train_model_index(df, target, label='Up')
    dfX = tweet_index(df, a, b, c, d, e, f, g, h)
    df_deep= get_labels(0, dfX)