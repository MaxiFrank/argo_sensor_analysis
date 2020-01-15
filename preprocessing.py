import sys
from datetime import datetime

import dateutil
import pyspark.sql.functions as fn
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import Bucketizer, StringIndexer, VectorAssembler
from pyspark.sql import Row, SparkSession, SQLContext, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *

BIN_RANGE = [-5, 0,  5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,
             70,  75,  80,  85,  90,  95, 100, 105, 110, 115, 120, 125, 130,
             135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195,
             200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260,
             265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325,
             330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390,
             395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455,
             460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520,
             525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585,
             590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650,
             655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715,
             720, 725, 730, 735, 740, 745, 750, 755, 760, 765, 770, 775, 780,
             785, 790, 795, 800, 805, 810, 815, 820, 825, 830, 835, 840, 845,
             850, 855, 860, 865, 870, 875, 880, 885, 890, 895, 900, 905, 910,
             915, 920, 925, 930, 935, 940, 945, 950, 955, 960, 965, 970, 975,
             980, 985, 990, 995, float("inf")]

# define function to create date range


def date_range(t1, t2, step=2):
    """Returns a list of equally spaced points between t1 and t2 with stepsize step."""
    return [t1 + step*x for x in range(int((t2-t1)/step)+1)]


# define udf
date_range_udf = fn.udf(date_range, ArrayType(LongType()))

# define interpolation function


def interpol(x, x_prev, x_next, y_prev, y_next, y):
    if x_prev == x_next:
        return y
    else:
        m = (y_next-y_prev)/(x_next-x_prev)
        y_interpol = y_prev + m * (x - x_prev)
        return y_interpol


# convert function to udf
interpol_udf = fn.udf(interpol, FloatType())


def preprocessing(df):
    # Cast temp as DoubleType()
    argo_df_og = df.withColumn("tempTmp", df['temp'].cast(DoubleType()))\
        .drop("temp")\
        .withColumnRenamed("tempTmp", "temp")\
        .select("profile_id", "pres", "temp", "lat", "lon", "psal", "date")\
        .persist()

    # Filter DataFrame by conditions to keep records
    argo_filterby = argo_df_og.groupBy("profile_id") \
        .agg(min("pres").alias("min_pres"),
             max("pres").alias("max_pres"),
             count("profile_id").alias("count_profile_id"))

    # Now, here are the profile_ids we want to keep, to be inner joined with original argo_df_og
    argo_keep_ids = argo_filterby.filter("count_profile_id >= 50 and min_pres <= 25 and max_pres >= 999") \
        .select("profile_id")

    # Inner join the profile_ids to keep with original argo_df_og to filter and keep only desired IDs
    argo_df_keep = argo_keep_ids.join(
        argo_df_og, "profile_id", "inner").persist()

    argo_df = argo_df_keep.select("profile_id", "pres", "temp", "lat", "lon", "psal", "date",
                                  month("date").alias("month"), year("date").alias("year")) \
        .persist()

    bucketizer = Bucketizer(
        splits=BIN_RANGE, inputCol="pres", outputCol="pres_buckets")
    argo_df_buck = bucketizer.setHandleInvalid("keep").transform(argo_df)

    column_name = 'temp'
    df = argo_df_buck.select('date', 'profile_id', 'pres', column_name)

    df.select(fn.unix_timestamp(fn.col('date'),
                                format='yyyy-MM-dd HH:mm:ss').alias('unix_timestamp'))

    df2 = df.withColumn('date', fn.unix_timestamp(fn.col('date'), format='yyyy-MM-dd HH:mm:ss'))\
        .withColumn("readtime_existent", col("date"))

    # group data by house, obtain min and max time by house, create time arrays and explode them
    df_base = df2.groupBy('profile_id')\
        .agg(fn.min('date').cast('integer').alias('readtime_min'), fn.max('date').cast('integer').alias('readtime_max'))\
        .withColumn("date", fn.explode(date_range_udf("readtime_min", "readtime_max")))\
        .drop('readtime_min', 'readtime_max')

    # left outer join existing read values
    df_all_dates = df_base.join(df2, ["profile_id", "date"], "leftouter")

    window_ff = Window.partitionBy('profile_id')\
        .orderBy('date')\
        .rowsBetween(-sys.maxsize, 0)

    window_bf = Window.partitionBy('profile_id')\
        .orderBy('date')\
        .rowsBetween(0, sys.maxsize)

    # create the series containing the filled values
    read_last = fn.last(df_all_dates[column_name],
                        ignorenulls=True).over(window_ff)
    readtime_last = fn.last(
        df_all_dates['readtime_existent'], ignorenulls=True).over(window_ff)

    read_next = fn.first(
        df_all_dates[column_name], ignorenulls=True).over(window_bf)
    readtime_next = fn.first(
        df_all_dates['readtime_existent'], ignorenulls=True).over(window_bf)

    # add the columns to the dataframe
    df_filled = df_all_dates.withColumn('readvalue_ff', read_last)\
                            .withColumn('readtime_ff', readtime_last)\
                            .withColumn('readvalue_bf', read_next)\
                            .withColumn('readtime_bf', readtime_next)

    # add interpolated columns to dataframe and clean up
    df_filled = df_filled.withColumn('readvalue_interpol', interpol_udf('date', 'readtime_ff', 'readtime_bf', 'readvalue_ff', 'readvalue_bf', column_name))\
        .drop('readtime_existent', 'readtime_ff', 'readtime_bf')\
        .withColumnRenamed('reads_all', column_name)\
        .withColumn('date', fn.from_unixtime(col('date')))

    argo_df = df_filled

    insane_sort = udf(lambda x: [item[0] for item in sorted(
        x, key=lambda x: x[1])], ArrayType(DoubleType()))

    array(argo_df['temp'], array(argo_df['pres']))

    argo_df_listed = argo_df.select('profile_id', array(argo_df['temp'], argo_df['pres']).alias('temp_pres'))\
        .groupBy('profile_id').agg(collect_list('temp_pres').alias('temp_pres_list'))
    argo_df_listed = argo_df_listed.select('profile_id', insane_sort(
        argo_df_listed['temp_pres_list']).alias('temp_pres_list'))
    return argo_df_listed
