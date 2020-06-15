import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructType, StructField, TimestampType

def create_spark_session():
    #Creates a new Spark session or gets an existing one

    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.5") \
        .getOrCreate()
    return spark

def create_song_df(spark, song_files):
    """Creates dataframe from source song files
        1. Defines a schema with data types for song metadata
        2. Extracts input data from song files stored in S3 (JSON format)
        3. Returns song data as Spark dataframe
    Arguments:
        spark       : Spark session
        song_files  : path to song files containing input data
    """

    # define schema
    song_schema = StructType([
        StructField('artist_id', StringType(), False),
        StructField('artist_latitude', StringType(), True),
        StructField('artist_longitude', StringType(), True),
        StructField('artist_location', StringType(), True),
        StructField('artist_name', StringType(), False),
        StructField('song_id', StringType(), False),
        StructField('title', StringType(), False),
        StructField('duration', DoubleType(), True),
        StructField('year', IntegerType(), True)
    ])

    # read song data file
    song_df = spark.read.json(song_files, schema=song_schema)

    return song_df

def create_log_df(spark, log_files):
    """Creates dataframe from source log files
        1. Defines a schema with data types for log data
        2. Extracts input data from log files stored in S3 (JSON format)
        3. Returns log data as Spark dataframe
    Arguments:
        spark      : Spark session
        log_files  : path to log files containing input data
    """

    # define schema
    log_schema = StructType([
        StructField('artist', StringType(), False),
        StructField('auth', StringType(), True),
        StructField('firstName', StringType(), True),
        StructField('gender', StringType(), True),
        StructField('itemInSession', LongType(), True),
        StructField('lastName', StringType(), True),
        StructField('length', DoubleType(), True),
        StructField('level', StringType(), True),
        StructField('location', StringType(), True),
        StructField('method', StringType(), True),
        StructField('page', StringType(), False),
        StructField('registration', DoubleType(), True),
        StructField('sessionId', LongType(), True),
        StructField('song', StringType(), False),
        StructField('status', LongType(), True),
        StructField('ts', LongType(), False),
        StructField('userAgent', StringType(), True),
        StructField('userId', StringType(), False),
    ])

    # read log data file
    log_df = spark.read.json(log_files, schema=log_schema)

    return log_df

def process_song_data(spark, song_df, output_data_dir):
    """ ETL process for songs data
    1. Creates song and artist tables
    2. Writes new tables to S3
    Arguments:
        spark            : Spark session
        song_df          : Spark dataframe containing song source data
        output_dir       : output path for parquet tables 
    """

    # extract columns to create songs table
    song_fields = ['song_id','title','artist_id','year','duration']
    songs_table = song_df.select(song_fields).dropDuplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_out_path = str(output_dir + 'songs/' + 'songs.parquet')
    songs_table.write.parquet(songs_out_path, mode='overwrite', partitionBy=['year', 'artist_id'])

    # extract columns to create artists table
    artist_fields = ['artist_id','artist_name','artist_location','artist_latitude','artist_longitude']
    artists_table = song_df.select(artist_fields) \
                    .withColumnRenamed('artist_name','artist') \
                    .withColumnRenamed('artist_location','location') \
                    .withColumnRenamed('artist_latitude','latitude') \
                    .withColumnRenamed('artist_longitude','longitude') \
                    .dropDuplicates(['artist_id'])

    # write artists table to parquet files
    artists_out_path = str(output_dir + 'artists/' + 'artists.parquet')
    artists_table.write.parquet(artists_out_path, mode='overwrite')

def process_log_data(spark, song_df, log_df, output_data_dir):
    """ ETL process for Sparkify log data
    1. Creates users, time, and songplays tables
    2. Writes new tables to S3
    Arguments:
        spark            : Spark session
        song_df          : Spark dataframe containing song source data
        log_df           : Spark dataframe containing log source data
        output_data_dir  : output path for newly created tables (parquet format)
    """

    # filter by actions for song plays
    df = log_df.filter(log_df.page == 'NextSong')

    # extract columns for users table
    user_fields = ['userId', 'firstName', 'lastName', 'gender', 'level']
    users_table = df.select(user_fields).dropDuplicates(['userId'])

    # write users table to parquet files
    users_out_path = str(output_dir + 'users/' + 'users.parquet')
    users_table.write.parquet(users_out_path, mode='overwrite')

    # create timestamp column from original ts column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000), TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))

    # extract columns to create time table
    time_table = df.select(
        col('timestamp').alias('start_time'),
        hour(col('timestamp')).alias('hour'),
        dayofmonth(col('timestamp')).alias('day'),
        weekofyear(col('timestamp')).alias('week'),
        month(col('timestamp')).alias('month'),
        year(col('timestamp')).alias('year')
    ).dropDuplicates(['start_time'])

    # write time table to parquet files partitioned by year and month
    time_out_path = str(output_dir + 'time/' + 'time.parquet')
    time_table.write.parquet(time_out_path, mode='overwrite', partitionBy=['year', 'month'])

    # join song and log datasets
    df = df.join(song_df, (song_df.title==df.song)&(song_df.artist_name==df.artist)) \
        .withColumn('songplay_id', monotonically_increasing_id())

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.select(
        col('timestamp').alias('start_time'),
        col('userId'),
        col('level'),
        col('song_id'),
        col('artist_id'),
        col('sessionId'),
        col('location'),
        col('userAgent'),
        year('timestamp').alias('year'),
        month('timestamp').alias('month')
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_out_path = str(output_dir + 'songplays/' + 'songplays.parquet')
    songplays_table.write.parquet(songplays_out_path, mode='overwrite', partitionBy=['year', 'month'])

def main():
    '''Executes entire ETL process'''

    # define data paths
    song_files = 's3a://udacity-dend/song_data/*/*/*/*.json'
    log_files = 's3a://udacity-dend/log_data/*/*/*.json'
    output_dir = 's3a://labros.spark.etl.bucket/Parquet_files'

    # parse config file and set AWS keys
    '''Parses configuration file and sets AWS keys'''

    # parse config file
    config = configparser.ConfigParser()
    config.read('dl.cfg')

    # set environment variables
    os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['KEY']
    os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['SECRET']

    # initiate Spark session
    spark = create_spark_session()

    # create data frames
    song_df = create_song_df(spark, song_files)
    log_df = create_log_df(spark, log_files)

    # run ETL process
    process_song_data(spark, song_df, output_dir)
    process_log_data(spark, song_df, log_df, output_dir)


if __name__ == "__main__":
    main()