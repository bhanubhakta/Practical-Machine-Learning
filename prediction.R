library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(e1071)

# The training and test data url.
trainingDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testDataUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Create input_data folder.
if(!file.exists("./input_data")) {
  dir.create("./input_data")
}

# Downloading and storing data locally.
trainingDataFile <- "./input_data/training_data.csv"
testDataFile <- "./input_data/test_data.csv"

if(!file.exists(trainingDataFile)) {
  download.file(trainingDataUrl, destfile = trainingDataFile, method = "curl")
}

if(!file.exists(testDataFile)) {
  download.file(testDataUrl, destfile = testDataFile, method = "curl")
}

# Read the data.

trainingDataRaw <- read.csv(trainingDataFile)
testDataRaw <- read.csv(testDataFile)

trainingDataRawDim <- dim(trainingDataRaw)
testDataRawDim <- dim(testDataRaw)

print(paste("Training Data Dimensions: ", trainingDataRawDim[1], "*", trainingDataRawDim[2]))
print(paste("Test Data Dimensions: ", testDataRawDim[1], "*", testDataRawDim[2]))

# Cleaning the data.
sum(complete.cases(trainingDataRaw))

# Removing the columns containing the missing values.
trainingDataRaw <- trainingDataRaw[ ,colSums(is.na(trainingDataRaw)) == 0]
testDataRaw <- testDataRaw[ ,colSums(is.na(testDataRaw)) == 0]

print(colnames(trainingDataRaw))

# Since we are interested with the data from accelerometers on the belt, forearm, arm, and dumbell, 
# user_name, window and timestamp variables won't contribute for the outcome. Lets get rid of these columns.

trainingToRemove <- grepl("^X|timestamp|window|user_name", names(trainingDataRaw))
testToRemove <- grepl("^X|timestamp|window|user_name", names(testDataRaw))

trainingDataRaw <- trainingDataRaw[ , !trainingToRemove]
testDataRaw <- testDataRaw[ , !testToRemove]

# Get rid of the data that are not numeric.

classe <- trainingDataRaw$classe
trainingDataCleaned <- trainingDataRaw[, sapply(trainingDataRaw, is.numeric)]
trainingDataCleaned$classe <- classe

testDataCleaned <- testDataRaw[, sapply(testDataRaw, is.numeric)]

# Partition the training data into test and validation set.
# Then, we can split the cleaned training set into a pure training data set (75%) and 
# a validation data set (25%). 
# We will use the validation data set to conduct cross validation in future steps.

set.seed(113456)

inTrain <- createDataPartition(trainingDataCleaned$classe, p=0.75, list=F)
trainingData <- trainingDataCleaned[inTrain, ]
validationData <- trainingDataCleaned[-inTrain, ]

# Using algorithm for prediction.

# I prefer using Random Forest algorithm to fit a predictive model for activity recognition
# as Random Forests are often the winner for lots of problems in classification.
# Also Random Forests automatically selects important variables and is robust to correlated covariates
# & outliers in general. We will use 5-fold cross validation when applying the algorithm.

controlRandomForest <- trainControl(method="cv", 5)
modelRandomForest <- train(classe ~ ., data=trainingData, method="rf", trControl=controlRandomForest, ntree=200)
modelRandomForest

# Estimating the performance of the model in the validatin sets.

predictRandomForest <- predict(modelRandomForest, validationData)
confusionMatrix(validationData$classe, predictRandomForest)


accuracy <- postResample(predictRandomForest, validationData$classe)
accuracy

out_of_sample_error <-  1 - as.numeric(confusionMatrix(validationData$classe, predictRandomForest)$overall[1])
out_of_sample_error

result <- predict(modelRandomForest, testDataCleaned[, -ncol(testDataCleaned)])
result

# Decision tree visualization
treeModel <- rpart(classe ~ ., data=trainingData, method="class")
prp(treeModel)
