from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.linalg import Vectors,  VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from preprocessing import preprocessing
from pyspark.ml.clustering import KMeans, LDA, GaussianMixture
import sys
from spectral_cluster import SpectralClustering

import argparse

import sys
from datetime import datetime

import pyspark.sql.functions as fn
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorAssembler, PCA
from pyspark.sql import Row, SparkSession, SQLContext, Window
from pyspark.ml.linalg import Vectors,  VectorUDT
from pyspark.sql.functions import *
from pyspark.sql.types import *
from scipy import interpolate
import numpy as np
import math

from pyspark.sql.functions import udf
from pyspark.ml.clustering import KMeans
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.window import *
import pyspark.sql.types as t
import numpy as np


parser = argparse.ArgumentParser('Cluster')
parser.add_argument('path', type=str,
                   help='path to raw data')
parser.add_argument('algorithm',type=str,
                    help='clustering algorithm to run. Choose from [kmeans,lda,spectral,gmm]')
parser.add_argument('outpath',type=str,
                    help = 'output path for results as csv')
parser.add_argument("--num_nodes",type=int,default=8,
                    help='number of nodes in your cluster. Default: 8')
parser.add_argument("--k_clusters","-k", type=int,default=8, help="number of clusters. Default: 8")
parser.add_argument("--num_pca_features",'-p',type=int,default=8,help='number of pca features to use. Default 8')
args = parser.parse_args()

path = args.path
algorithm = args.algorithm
num_pca_features = args.num_pca_features
num_nodes = args.num_nodes
outpath = args.outpath
k = args.k_clusters

if algorithm not in ['kmeans','gmm','lda','spectral']: raise ValueError('Not a valid algorithm')


ss = SparkSession.builder.getOrCreate()

df = ss.read.csv(path,header=True,inferSchema=True)

df_preprocessed = preprocessing(df, num_pca=num_pca_features)



df_preprocessed.write.parquet("preprocessed",mode="Overwrite")

if algorithm == 'kmeans':
    model = KMeans(k=k).setSeed(1).fit(df_preprocessed)
    predictions = model.transform(df_preprocessed)
elif algorithm == 'spectral':
    model = SpectralClustering(k=k, k_nearest=7)
    predictions = model.cluster(df_preprocessed, ss, repartition_num=num_nodes)
elif algorithm == 'lda':
    model = LDA(k=k, maxIter=10).fit(df_preprocessed)
    predictions = model.transform(df_preprocessed)
elif algorithm == 'gmm':
    model = GaussianMixture(k=k).fit(df_preprocessed)
    predictions = model.transform(df_preprocessed)


predictions.select([col for col in predictions.columns if col != 'features'])\
           .toPandas()\
           .to_csv(sys.stdout)
