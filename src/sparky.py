#!/usr/bin/env python3

import os
import numpy
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    NumericType,
    StringType,
    ArrayType,
    IntegerType,
)
from pyspark.sql.functions import (
    col,
    lower,
    regexp_replace,
    split,
    expr,
    length,
    when,
    from_json,
    sum,
    sqrt,
)
from pyspark.ml.feature import RegexTokenizer, Word2Vec, StopWordsRemover

# no truly massive arrays, iterative procedures, simple SQL ops --> Spark on local should be ok
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

trainRaw.show(2)
testRaw.show(2)


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
# 8 speed auto or an ambiguous auto for transmissions seems standard
colRedundancy = ["VehBodystyle", "VehType", "VehTransmission"]
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
    .withColumn(
        "VehColorExt",
        lower(regexp_replace(col("VehColorExt"), "[^a-zA-Z0-9]", "")),
    )
    .withColumn(
        "VehDriveTrain",
        lower(
            regexp_replace(col("VehDriveTrain"), "[^a-zA-Z0-9/]", "")
        ),  # keep '/' for 4x4/AWD etc
    )
    .withColumn(
        "VehHistory",
        lower(col("VehHistory")),
    )
    .withColumn(
        "VehFeats",
        lower(col("VehFeats")),
    )
    for df in [testRaw, trainRaw]
]

# Vectorize array 'string' types that may contain richer info and cast for new schema
testRaw, trainRaw = [
    df.withColumn(
        "VehEngine", split(col("VehEngine"), " ").cast(ArrayType(StringType()))
    )
    .withColumn(
        "VehHistory", split(col("VehHistory"), ",").cast(ArrayType(StringType()))
    )
    .withColumn("VehFeats", from_json(col("VehFeats"), ArrayType(StringType())))
    for df in [testRaw, trainRaw]
]

# We note that VehHistory [# Owner, ...] that can be featurized as NumOwners
# separate from the rest of the data. Later found explicit casting for simple
# type here was not needed

testRaw, trainRaw = [
    df.withColumn("NumOwners", split(col("VehHistory")[0], " ")[0]).withColumn(
        "NumOwners", col("NumOwners").cast(IntegerType())
    )
    for df in [testRaw, trainRaw]
]

# Now remove owner entry from original col entries. Note VehHistory has
# at most 5 elements per entry (see log file)
testRaw, trainRaw = [
    df.withColumn(
        "VehHistory", expr("slice(VehHistory, 2, 5)").cast(ArrayType(StringType()))
    )
    for df in [testRaw, trainRaw]
]
# remove annoying intial whitespace in each entry
testRaw, trainRaw = [
    df.withColumn(
        "VehHistory",
        expr("transform(VehHistory, element -> trim(element))").cast(
            ArrayType(StringType())
        ),
    )
    for df in [testRaw, trainRaw]
]


# We should find 201 null entries here to match the nulls from original column
nullEntryCount = trainRaw.filter(col("NumOwners").isNull()).count()
print(f"New column for owners contains {nullEntryCount}/201 null entries in train.")

# More testing can be done to see impacts of splitting up vectorized features
# or see sentiment attached with certain elements in the entries

# As for zipcode, lets explode to make a more natural vector, so 12345 becomes [1,2,3,4,5]

# There are only so many VehDriveTrain options. Replace drive with 'D', all with 'A' etc

testRaw, trainRaw = [
    df.withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "front", "f"))
    .withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "wheel", "w"))
    .withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "drive", "d"))
    .withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "all", "a"))
    .withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "four", "4"))
    .withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "two", "2"))
    .withColumn("VehDriveTrain", regexp_replace(col("VehDriveTrain"), "or", "/"))
    for df in [testRaw, trainRaw]
]

# We note from logs that 'ALL-WHEEL DRIVE WITH LOCKING AND LIMITED-SLIP DIFFERENTIAL' would read as awdwithlockingandlimitedslipdifferential as such:
funky = (
    trainRaw.select("VehDriveTrain", "ListingID")
    .filter(length(col("VehDriveTrain")) > 7)
    .limit(3)
)
funky.show()

trainRaw = trainRaw.withColumn(
    "VehDriveTrain",
    when(col("ListingID") == 425217, "awd").otherwise(col("VehDriveTrain")),
)

funky = (
    trainRaw.select("VehDriveTrain", "ListingID")
    .filter(length(col("VehDriveTrain")) < 3)
    .limit(3)
)
funky.show()

trainRaw = trainRaw.withColumn(
    "VehDriveTrain",
    when(col("ListingID") == 4199245, "awd").otherwise(col("VehDriveTrain")),
)


# NOTE that 4wd and 4x4 are the same thing; opt to 4wd
testRaw, trainRaw = [
    df.withColumn(
        "VehDriveTrain", regexp_replace("VehDriveTrain", "4x4", "4wd")
    ).withColumn("VehDriveTrain", regexp_replace("VehDriveTrain", "4wd/4wd", "4wd"))
    for df in [testRaw, trainRaw]
]

# NOTE awd/4wd and 4wd/awd mean the same thing! Choose one so later encoding doesn't suggest they are unique

trainRaw = trainRaw.withColumn(
    "VehDriveTrain",
    when(col("VehDriveTrain") == "awd/4wd", "4wd/awd").otherwise(col("VehDriveTrain")),
)
trainRaw.show(2)


trainRaw.printSchema()

# Print the number of columns (note additional 2 on train for y)
print(f"The Train DataFrame is {trainRaw.count()} by {len(trainRaw.columns)}.")
print(f"The Test DataFrame is  {testRaw.count()} by {len(testRaw.columns)}.")

# We now want to tokenize the VehSellerNotes entries
# https://spark.apache.org/docs/latest/ml-features.html
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2281281647728580/33322862051354/347222873603598/latest.html
# https://spark.apache.org/docs/latest/ml-linalg-guide.html << should be installed for faster numerical processing
# https://neptune.ai/blog/word-embeddings-guide#word2vec

# First nulls should be filled with `None.` as it's likley there is intentionality to notes being left blank and feature has significant heterogeneity even when tokenized
trainRaw = trainRaw.na.fill("None.", subset=["VehSellerNotes"])
testRaw = testRaw.na.fill("None.", subset=["VehSellerNotes"])


def process_text_column(input_col, trainX, testX, vector_size=20, min_count=3):
    # Tokenize the input column of both sets of data
    regexTokenizer = RegexTokenizer(
        gaps=False,
        pattern="\w+",
        inputCol=input_col,
        outputCol=f"Word2Vec1_{input_col}",
    )
    trainX = regexTokenizer.transform(trainX)
    testX = regexTokenizer.transform(testX)

    # Remove stop words from both sets of data
    swr = StopWordsRemover(
        inputCol=f"Word2Vec1_{input_col}", outputCol=f"Word2Vec2_{input_col}"
    )
    trainX = swr.transform(trainX)
    testX = swr.transform(testX)

    # Apply Word2Vec model; recall to fit on only on training data and apply transform to both sets to avoid leakage
    word2vec = Word2Vec(
        vectorSize=vector_size,
        minCount=min_count,
        inputCol=f"Word2Vec2_{input_col}",
        outputCol=f"Word2Vec_{input_col}",
        seed=42,
    )
    model = word2vec.fit(trainX)
    trainX = model.transform(trainX)
    testX = model.transform(testX)

    # Drop intermmediaries and original raw form
    col2Drop = [input_col, f"Word2Vec1_{input_col}", f"Word2Vec2_{input_col}"]
    for c in col2Drop:
        trainX = trainX.drop(c)
        testX = testX.drop(c)

    return trainX, testX


# Assume that note entries (~4000 uniques/6k) have words must show up at least 10 times in corpus (notes column) to be considered for training and reduce some noise;
# avoid overfitting on this small dataset and keep vec size near 100 --> use another project to fine tune all of this
# Larger vector sizes might be more suitable when dealing with a ``vast`` vocabulary here or when words have multiple meanings,
# as the model has more dimensions to differentiate between these nuances. Keep in mind this is a miniature text corpus and that about a third seemed
# to be duplicate (perhaps standard or saved) note entries from certain dealers
# trainRaw, testRaw = process_text_column(
#    "VehSellerNotes", trainRaw, testRaw, vector_size=100, min_count=10
# )


# No nulls in the data for seller cities (1300+ uniques). There is somewhat considerable verbal diversity here and it may be beneficial
# to capture similarity between cities and understand the relationships between them when predicting the trim-price label
# (e.g., a 2020 Honda Vroomvroom is $X in these coastal cities and $Y in this other region of cities and has Z% higher price for 2021 model).
# Let's make sure cities where most cars are sold have greater semantic weight so min count set to 3 in corpus
# A smaller vector size, such as 30, would result in more compact representations but might capture simpler relationships between names
# trainRaw, testRaw = process_text_column(
#    "SellerCity", trainRaw, testRaw, vector_size=25, min_count=3
# )

# Since seller name does not have any nulls and 700+ types lets do a last min tokenization (see logs)
# making sure all names are considered for training. Alt could use something like a dummy variable but introduces
# a lot of sparsity in the design matrix. See similar notes above.
# trainRaw, testRaw = process_text_column(
#    "SellerName", trainRaw, testRaw, vector_size=25, min_count=3
# )


trainRaw.show(3)

# ListingID is not quite an index and can spill into nonsensical feature contributions; can be indexed in jupyter nb; keep for test as will be a needed col
trainRaw = trainRaw.drop("ListingID")

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
train = trainRaw.na.drop(subset=[*colDropTrain])


# Print the number of columns (note additional 2 on train for y)
print(f"The outgoing Train DataFrame is {train.count()} by {len(train.columns)}.")
print(f"The outgoing Test DataFrame is  {testRaw.count()} by {len(testRaw.columns)}.")

# Save the Pandas DataFrame to a CSV file
train.toPandas().to_csv(folderPath + "/train_sparked.csv", index=False)
testRaw.toPandas().to_csv(folderPath + "/test_sparked.csv", index=False)

train.printSchema()
testRaw.printSchema()
