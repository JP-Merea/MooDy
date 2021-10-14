"""
In data.py you will find the upload of the dataset that we are going to use, and the cleaning of it
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_data():
    """ We will work with two inicial dataset, one that correspond to the sentymental analysis made by taking twitter and the other the value of the blue dolar 
    with their respective variation of the value between days """
    df = pd.read_csv('/content/drive/MyDrive/labeled_tweets2.csv')
    blue = pd.read_csv('/content/drive/MyDrive/dolar_oct.csv', sep=';')
    return df, blue

def get_clean_data(df, blue):
    """This function will clean the two dataset upload in the function before and then merged them, creating a new dataset """
    #process info
    blue['fecha'] = pd.to_datetime(blue['fecha'])
    blue['bid'] = pd.to_numeric(blue['bid'].str.replace(',', '.'))
    blue['fut_bid_1'] = pd.to_numeric(blue['fut_bid_1'].str.replace(',', '.'))
    blue['fut_bid_2'] = pd.to_numeric(blue['fut_bid_2'].str.replace(',', '.'))
    blue['fut_bid_3'] = pd.to_numeric(blue['fut_bid_3'].str.replace(',', '.'))
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df[df.created_at != 'created_at']
    df_blue = df.merge(blue, left_on='created_at', right_on='fecha', how='outer')
    df_blue.dropna(subset = ['created_at'], inplace=True)
    df_blue = df_blue.drop_duplicates()
    df_blue = df_blue.sort_values(by="created_at")
    df_blue[['fecha','fut_bid_1','fut_bid_2','fut_bid_3', 'bid']] = df_blue[['fecha','fut_bid_1','fut_bid_2','fut_bid_3', 'bid']].fillna(method="ffill")
    df_blue.dropna(subset = ['fecha'], inplace=True)
    #dataset columns tranformer
    df_blue['DOW'] = df_blue['fecha'].dt.weekday+1
    df_blue['Week'] = df_blue['fecha'].dt.weekofyear
    df_blue['Month'] = df_blue['fecha'].dt.month
    df_blue['Year'] = df_blue['fecha'].dt.year
    df_blue['user_verified'].loc[df_blue['user_verified'] == True] = 2
    df_blue['user_verified'].loc[df_blue['user_verified'] == False] = 1
    df_blue['user_verified'] = pd.to_numeric(df_blue['user_verified'])
    df_blue = df_blue[['fecha', 'created_at','text', 
                'favorite_count', 'retweet_count', 'user_verified','user_favourites_count',	'user_followers_count',	'user_friends_count',
                'Most_Prob',	'NEG',	'POS',	'NEU', 'fut_bid_1','fut_bid_2','fut_bid_3', 'bid']]
    scaler = MinMaxScaler()
    df_blue[["favorite_count", 'retweet_count', 'user_favourites_count',	'user_followers_count',	'user_friends_count']] = scaler.fit_transform(df_blue[['favorite_count', 'retweet_count', 'user_favourites_count',	'user_followers_count',	'user_friends_count']])
    return df_blue

if __name__ == '__main__':
    df, blue = get_data()
    df = get_clean_data(df, blue)