from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from preprocessing import preprocessing

from spectral_cluster import SpectralClustering

ss = SparkSession.builder.getOrCreate()

df = ss.read.csv('./argo_data_small.csv',header=True,inferSchema=True)

df_preprocessed = preprocessing(df)
df_preprocessed = df_preprocessed.select('profile_id',df_preprocessed['temp_pres_list'].alias('features'))
model = SpectralClustering(k=9, k_nearest=7)
predictions = model.cluster(df_preprocessed, ss)

predictions.select([col for col in predictions.columns if col != 'features']).toPandas().to_csv("clusters.csv")
