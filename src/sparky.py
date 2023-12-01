#!/usr/bin/env python3

import pandas, os
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark.sql.functions import col, lower, regexp_replace, split, from_json
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
# Skipping part where can set further config options and send job to YARN scheduler, but very possible to tune

schema = (
    StructType()
    .add("ListingID", LongType(), True)
    .add("SellerCity", StringType(), True)  # lowercase and mash to tokenize
    .add("SellerIsPriv", BooleanType(), True)
    .add("SellerListSrc", StringType(), True)  # lowercase and mash to tokenize
    .add("SellerName", StringType(), True)  # lowercase and mash to tokenize
    .add("SellerRating", DoubleType(), True)
    .add("SellerRevCnt", LongType(), True)
    .add("SellerState", VarcharType(4), True)
    .add("SellerZip", LongType(), True)  # unsure why they said this is float
    .add("VehBodystyle", StringType(), True)
    .add("VehCertified", BooleanType(), True)
    .add("VehColorExt", StringType(), True)
    .add("VehColorInt", StringType(), True)
    .add("VehDriveTrain", VarcharType(4), True)
    .add("VehEngine", ArrayType(StringType()), True)
    .add("VehFeats", ArrayType(StringType()), True)
    .add("VehFuel", VarcharType(15), True)
    .add("VehHistory", IntegerType(), True)
    .add("VehListdays", IntegerType(), True)
    .add("VehMake", StringType(), True)
    .add("VehMileage", BooleanType(), True)
    .add("VehModel", StringType(), True)
    .add("VehPriceLabel", IntegerType(), True)
    .add("VehSellerNotes", IntegerType(), True)
    .add("VehType", StringType(), True)
    .add("VehTransmission", BooleanType(), True)
    .add("VehYear", StringType(), True)
)
# Convert file names to lowercase and rename the files

folderPath = "./data"
fileNames = os.listdir(folderPath)
for f in fileNames:
    A = os.path.join(folderPath, f)
    a = os.path.join(folderPath, f.lower())

    # Rename the file
    os.rename(A, a)
print("File names converted to lowercase.")

testRaw = spark.read.option("header", True).csv("./data/test_dataset.csv")
trainRaw = spark.read.option("header", True).csv("./data/training_dataset.csv")

# Cast column to standard boolean types
testRaw, trainRaw = [
    df.withColumn("SellerIsPriv", df["SellerIsPriv"].cast(BooleanType())).withColumn(
        "VehCertified", df["VehCertified"].cast(BooleanType())
    )
    for df in [testRaw, trainRaw]
]


# Simplify city/seller names and  listing source identifier by lowercase and remove special chars
# Save other string types later to not jump the gun on transformations for other string types
# that may miss critical (rare) nuances
testRaw, trainRaw = [
    df.withColumn(
        "SellerCity", lower(regexp_replace(col("SellerCity"), "[^a-zA-Z0-9]", ""))
    )
    .drop("SellerCity")
    .withColumn(
        "SellerListSrc",
        lower(regexp_replace(col("SellerListSrc"), "[^a-zA-Z0-9]", "")),
    )
    .drop("SellerListSrc")
    .withColumn(
        "SellerName",
        lower(regexp_replace(col("SellerName"), "[^a-zA-Z0-9]", "")),
    )
    .drop("SellerListSrc")
    .withColumn(
        "VehPriceLabel",
        lower(regexp_replace(col("SellerName"), "[^a-zA-Z0-9]", "")),
    )
    .drop("VehPriceLabel")
    .withColumn(
        "VehModel",
        lower(regexp_replace(col("SellerName"), "[^a-zA-Z0-9]", "")),
    )
    .drop("VehModel")
    for df in [testRaw, trainRaw]
]

testRaw, trainRaw = [
    df.withColumn(
        "VehColorExt", split(col("VehColorExt"), " ").cast(ArrayType(StringType()))
    )
    for df in [testRaw, trainRaw]
]

testRaw, trainRaw = [
    df.withColumn(
        "VehColorInt", split(col("VehColorInt"), " ").cast(ArrayType(StringType()))
    )
    for df in [testRaw, trainRaw]
]

testRaw, trainRaw = [
    df.withColumn(
        "VehEngine", split(col("VehEngine"), " ").cast(ArrayType(StringType()))
    )
    for df in [testRaw, trainRaw]
]

testRaw, trainRaw = [
    df.withColumn(
        "VehHistory", split(col("VehHistory"), " ").cast(ArrayType(StringType()))
    )
    for df in [testRaw, trainRaw]
]

testRaw, trainRaw = [
    df.withColumn('VehFeats', from_json(col('VehFeats'), ArrayType(StringType())))
    for df in [testRaw, trainRaw]
]


testRaw.show(2)
