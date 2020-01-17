import sys
from datetime import datetime

import dateutil
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

def toVector(array):
    return Vectors.dense(array)
toVector_udf = udf(toVector,VectorUDT())

def interp(array):
    pres_grid = np.arange(5,1000,5)
    temp_array = [temp for temp, _ in array]
    pres_array = [pres for _, pres in array]

    interp = interpolate.interp1d(pres_array, temp_array, fill_value=(temp_array[0],np.nan),bounds_error=False)
    tempnew = interp(pres_grid)

    return [float(item) for item in tempnew]

interp_udf = udf(interp,ArrayType(DoubleType()))

def udf_null(array):
    list_ = [x for x in array if np.isnan(x)]
    if list_:
        return True  # has NAs
    else:
        return False  # does not have NAs
null_udf = udf(udf_null)

def udf_len_correct(array):
    if len(array)==199:
        return True
    else:
        return False

lenarray_udf = udf(udf_len_correct)

def udf_less_than_neg5(array):
    list_ = [x for x in array if x<-5]
    if list_:
        return True  # has NAs
    else:
        return False  # does not have NAs
neg_udf = udf(udf_less_than_neg5)

insane_sort = udf(lambda x: [item for item in sorted(
        x, key=lambda x: x[1])], ArrayType(ArrayType(DoubleType())))

def preprocessing(df,num_pca=10):
    argo_df_og = df

    # Cast temp as DoubleType()
    argo_df_og = argo_df_og.withColumn("tempTmp", argo_df_og['temp'].cast(DoubleType()))\
                       .drop("temp")\
                       .withColumnRenamed("tempTmp", "temp")\
                       .select("profile_id", "pres", "temp", "lat", "lon", "psal", "date")\
                       .persist()

    
    argo_filterby = argo_df_og.groupBy("profile_id") \
                          .agg(min("pres").alias("min_pres"), 
                               max("pres").alias("max_pres"), 
                               count("profile_id").alias("count_profile_id"))

    # Now, here are the profile_ids we want to keep, to be inner joined with original argo_df_og
    argo_keep_ids = argo_filterby.filter("count_profile_id >= 50 and min_pres <= 25 and max_pres >= 999") \
                             .select("profile_id")

    # Inner join the profile_ids to keep with original argo_df_og to filter and keep only desired IDs
    argo_df_keep = argo_keep_ids.join(argo_df_og, "profile_id", "inner").persist()

    #Final filtered df after pressure cleaning
    argo_df = argo_df_keep.select("profile_id", "pres", "temp", "lat", "lon", "psal", "date", 
                              month("date").alias("month"), year("date").alias("year")) \
                      .persist()
    
    #INTERPOLATION

    #Create vector mapping correspoding temperatures with pressures
    argo_df_listed = argo_df.select('profile_id', 'lat', 'lon', array(argo_df['temp'], argo_df['pres']).alias('temp_pres'))\
                        .groupBy('profile_id').agg(collect_list('temp_pres').alias('temp_pres_list'), 
                                                   fn.min(argo_df['lat']).alias('lat'),
                                                   fn.min(argo_df['lon']).alias('lon'))

    # Ordering by pressure
    argo_df_listed = argo_df_listed.select('profile_id', 'lat', 'lon',
                                       insane_sort(argo_df_listed['temp_pres_list']).alias('temp_pres_list'))
    
    # Interpolating missing temps at specified grid points
    pres = argo_df_listed.select('profile_id', 'lat', 'lon', interp_udf('temp_pres_list').alias('temp_interp'))

    # Finding profiles with temps as nans
    check_pres = pres.select("profile_id", "temp_interp", 'lat', 'lon',
                         null_udf("temp_interp").alias("temp_interp_hasNA"),
                         lenarray_udf("temp_interp").alias("temp_interp_len199"))
    
    # Filtering profiles with temps as nans
    filtered_pres = check_pres.filter("temp_interp_hasNA == False").select("profile_id","temp_interp", 'lat', 'lon')
    
    # Finding profiles with temps < -5
    check_pres = pres.select("profile_id", "temp_interp", 'lat', 'lon',
                         neg_udf("temp_interp").alias("temp_interp_hasNeg5s"))
    # Filtering profiles with temps < -5
    argo_df_clean = check_pres.filter("temp_interp_hasNeg5s == False").select("profile_id","temp_interp",'lat', 'lon')

    argo_df_clean = argo_df_clean.select('profile_id',
                                         toVector_udf(argo_df_clean['temp_interp']).alias('features'),
                                         'lat', 'lon')

    pca = PCA(k=num_pca, inputCol='features', outputCol='features_pca').fit(argo_df_clean)
    argo_df_clean = pca.transform(argo_df_clean)
    argo_df_clean = argo_df_clean.select('profile_id',
                                         argo_df_clean['features_pca'].alias('features'),
                                         'lat', 'lon')
    
    return argo_df_clean
