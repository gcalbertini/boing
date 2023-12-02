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


def process_non_numeric_columns(df, output_file):
    nonNumCols = [
        column
        for column in df.columns
        if not isinstance(df.schema[column].dataType, NumericType)
    ]

    # Generate condensed data structure log file for non-numeric columns
    with open(output_file, "w") as f:
        # Redirect stdout to the file just to extract the structure before preprocessing
        os.sys.stdout = f
        uniqs = []
        nulls = []
        for column in nonNumCols:
            uniq = df.select(column).distinct().count()
            nullCounts = df.filter(col(column).isNull()).count()
            uniqs.append(uniq)
            nulls.append(nullCounts)
            uniqEntry = df.select(column).distinct().collect()
            print(
                f"Unique entries for column '{column}' ({uniq} with {nullCounts} null entries):"
            )
            for entry in uniqEntry:
                print(entry[0])
            print("====" * 20)
        print("NonNumCol/Uniques/Nulls(***):")
        zipped = list(zip(nonNumCols, uniqs, nulls))
        for z in zipped:
            if z[-1] > 0:
                print(f"{z}***")
            else:
                print(z)


# Call the function for trainRaw
process_non_numeric_columns(trainRaw, "nonNumCol_train.log")

# Call the function for testRaw
process_non_numeric_columns(testRaw, "nonNumCol_test.log")


def process_numeric_columns(df, output_file):
    numCols = [
        column
        for column in df.columns
        if isinstance(df.schema[column].dataType, NumericType)
    ]

    # Generate condensed data structure log file for numeric columns
    with open(output_file, "w") as f:
        # Redirect stdout to the file just to extract the structure before preprocessing
        os.sys.stdout = f
        uniqs = []
        nulls = []
        for column in numCols:
            uniq = df.select(column).distinct().count()
            nullCounts = df.filter(col(column).isNull()).count()
            uniqs.append(uniq)
            nulls.append(nullCounts)
            uniqEntry = df.select(column).distinct().collect()
            print(
                f"Unique entries for column '{column}' ({uniq} with {nullCounts} null entries):"
            )
            for entry in uniqEntry:
                print(entry[0])
            print("====" * 20)
        print("NumCol/Uniques/Nulls(***):")
        zipped = list(zip(numCols, uniqs, nulls))
        for z in zipped:
            if z[-1] > 0:
                print(f"{z}***")
            else:
                print(z)


# Call the function for trainRaw
process_numeric_columns(trainRaw, "numCol_train.log")

# Call the function for testRaw
process_numeric_columns(testRaw, "numCol_test.log")
os.sys.stdout = os.sys.__stdout__

# We can see from the log files generated that VehBodystyle only has SUV and
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

# We should find 201 null entries here to match the nulls from original column
nullEntryCount = trainRaw.filter(col("NumOwners").isNull()).count()
print(f"New column for owners contains {nullEntryCount}/201 null entries in train.")

# More testing can be done to see impacts of splitting up vectorized features
# or see sentiment attached with certain elements in the entries

trainRaw.printSchema()

# Print the number of columns (note additional 2 on train for y)
print(f"The Train DataFrame is {trainRaw.count()} by {len(trainRaw.columns)}.")
print(f"The Test DataFrame is  {testRaw.count()} by {len(testRaw.columns)}.")

# On the question of missing data. We have to study MAR, MCAR, or MNAR.
# https://bookdown.org/rwnahhas/RMPH/mi-mechanisms.html
# https://stats.stackexchange.com/questions/462507/how-to-decide-whether-missing-values-are-mar-mcar-or-mnar
# Neat study: https://pubmed.ncbi.nlm.nih.gov/35266565/
# Ref: https://www.math.wsu.edu/faculty/xchen/stat115/lectureNotes3/Rubin%20Inference%20and%20Missing%20Data.pdf

"""
    MCAR (Missing Completely At Random):
        The missingness is unrelated to the observed or unobserved data.
        No systematic pattern in the missing values.
        Probability of a data point being missing is the same for all observations.

    MAR (Missing At Random):
        The probability of missing values depends only on the observed data and not on the unobserved data.
        There is a systematic pattern in the missing values, but it can be explained by the observed data.
        Once you account for the observed data, the missingness is random.

    MNAR (Missing Not At Random):
        The missingness is related to the unobserved data (!!!).
        The probability of missing values depends on the information that was not observed.
        Missing data mechanism is related to the unobserved variable itself!
"""

# For the sake of time will not implement scripts from scratch at this point
# and use MI methods found in other libraries on a jupyter nb

# The last thing we shall do is drop rows < 10 missing values for impacted columns
colDropTrain = ["SellerZip", "VehListdays", "VehMileage", "SellerListSrc", "VehFuel"]
colDropTest = ["VehColorExt", "VehMileage"]
train = trainRaw.na.drop(subset=[*colDropTrain])
test = testRaw.na.drop(subset=[*colDropTest])

# Print the number of columns (note additional 2 on train for y)
print(f"The outgoing Train DataFrame is {train.count()} by {len(train.columns)}.")
print(f"The outgoing Test DataFrame is  {test.count()} by {len(test.columns)}.")
