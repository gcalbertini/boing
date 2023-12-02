#!/usr/bin/env python3

import pandas, os
import logging
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import NumericType, StringType, ArrayType, IntegerType
from pyspark.sql.functions import col, lower, regexp_replace, split, from_json, expr
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
# Skipping part where can set further config options and send job to YARN scheduler, but very possible to tune

# Convert file names to lowercase and rename the files

folderPath = "./data"
fileNames = os.listdir(folderPath)
for f in fileNames:
    A = os.path.join(folderPath, f)
    a = os.path.join(folderPath, f.lower())

    # Rename the file
    os.rename(A, a)
print("File names converted to lowercase.")

testRaw = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("./data/test_dataset.csv")
)
trainRaw = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv("./data/training_dataset.csv")
)
trainRaw.printSchema()

# Print the number of columns (note additional 2 on train for y)
print(f"The Train DataFrame is {trainRaw.count()} by {len(trainRaw.columns)}.")
print(f"The Test DataFrame is  {testRaw.count()} by {len(testRaw.columns)}.")

# Get the unique entries for non-numeric columns and lengthy notes column
nonNumCol = [
    column
    for column in trainRaw.columns
    if not isinstance(trainRaw.schema[column].dataType, NumericType)
    and column != "VehSellerNotes"
    and column != "VehFeats"
]

# Generate condensed data structure log file for nonnumeric col
with open("nonNumCol.log", "w") as f:
    # Redirect stdout to the file just to extract the structure before preprocessing
    os.sys.stdout = f
    uniqEntryCounts = []
    nullEntryCounts = []
    for column in nonNumCol:
        uniqEntryCount = trainRaw.select(column).distinct().count()
        nullEntryCount = trainRaw.filter(col(column).isNull()).count()
        uniqEntryCounts.append(uniqEntryCount)
        nullEntryCounts.append(nullEntryCount)
        uniqEntry = trainRaw.select(column).distinct().collect()
        print(
            f"Unique entries for column '{column}' ({uniqEntryCount} with {nullEntryCount} null entries):"
        )
        for entry in uniqEntry:
            print(entry[0])
        print("====" * 20)
    print("NonNumCol/Uniques/Nulls(***):")
    zipped = list(zip(nonNumCol, uniqEntryCounts, nullEntryCounts))
    for z in zipped:
        if z[-1] > 0:
            print(f"{z}***")
        else:
            print(z)
os.sys.stdout = os.sys.__stdout__

# We can see from the log file generated that VehBodystyle only has SUV and
# VehType only has Used. Neither have nulls in the column --> drop redundancy
colRedundancy = ["VehBodystyle", "VehType"]
trainRaw = trainRaw.drop(*colRedundancy)
testRaw = testRaw.drop(*colRedundancy)

# Simplify short and simple descriptors by lowercase and remove special chars
# Save other more complex string types for later to not jump the gun on transformations for other string types
# that may miss critical (rare) nuances
testRaw, trainRaw = [
    df.withColumn(
        "SellerCity", lower(regexp_replace(col("SellerCity"), "[^a-zA-Z0-9]", ""))
    )
    .withColumn(
        "SellerListSrc",
        lower(regexp_replace(col("SellerListSrc"), "[^a-zA-Z0-9]", "")),
    )
    .withColumn(
        "SellerName",
        lower(regexp_replace(col("SellerName"), "[^a-zA-Z0-9]", "")),
    )
    .withColumn(
        "VehPriceLabel",
        lower(regexp_replace(col("VehPriceLabel"), "[^a-zA-Z0-9]", "")),
    )
    .withColumn(
        "VehModel",
        lower(regexp_replace(col("VehModel"), "[^a-zA-Z0-9]", "")),
    )
    .withColumn(
        "VehColorInt",
        lower(regexp_replace(col("VehColorInt"), "[^a-zA-Z0-9]", "")),
    )
    for df in [testRaw, trainRaw]
]

# Vectorize array 'string' types that may contain richer info and cast for new schema
testRaw, trainRaw = [
    df.withColumn(
        "VehColorExt", split(col("VehColorExt"), " ").cast(ArrayType(StringType()))
    )
    .withColumn("VehEngine", split(col("VehEngine"), " ").cast(ArrayType(StringType())))
    .withColumn(
        "VehHistory", split(col("VehHistory"), ",").cast(ArrayType(StringType()))
    )
    .withColumn("VehFeats", from_json(col("VehFeats"), ArrayType(StringType())))
    for df in [testRaw, trainRaw]
]

# We note that VehHistory [# Owner, ...] that can be featurized as NumOwners
# separate from the rest of the data. Later found explicit casting for simple
# type here was not needed

trainRaw, testRaw = [
    df.withColumn("NumOwners", split(col("VehHistory")[0], " ")[0]).withColumn(
        "NumOwners", col("NumOwners").cast(IntegerType())
    )
    for df in [testRaw, trainRaw]
]

# Now remove owner entry from original col entries. Note VehHistory has
# at most 5 elements per entry (see log file), go up to 6 to be safe?
trainRaw, testRaw = [
    df.withColumn("VehHistory", expr("slice(VehHistory, 2, 6)"))
    for df in [testRaw, trainRaw]
]

# More testing can be done to see impacts of splitting up vectorized features
# or see sentiment attached with certain elements in the entries

trainRaw.printSchema()

# Print the number of columns (note additional 2 on train for y)
print(f"The Train DataFrame is {trainRaw.count()} by {len(trainRaw.columns)}.")
print(f"The Test DataFrame is  {testRaw.count()} by {len(testRaw.columns)}.")

