# Load or download required packages
if(!require(ellipse)) install.packages("ellipse")
if(!require(data.table)) install.packages("data.table")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# Download dataset file
dl <- tempfile()
download.file("https://www.dropbox.com/s/e2hc5rinwkectsp/falldeteciton.csv?dl=1", dl)
# load the CSV file from temp file
dataset <- read.csv(dl, header=FALSE)
# set the column names in the dataset
colnames(dataset) <- c("ACTIVITY","TIME","SL","EEG","BP","HR","CIRCLUATION")

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$ACTIVITY, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# Get dimensions of dataset
dim(dataset)

# List types for each attribute
sapply(dataset, class)

# Take a peek at the first few rows of the data
head(dataset)

# List the levels for the class
levels(dataset$ACTIVITY)

# Summarize the class distribution
percentage <- prop.table(table(dataset$ACTIVITY)) * 100
cbind(freq=table(dataset$ACTIVITY), percentage=percentage)

# Summarize attribute distributions
summary(dataset)

# Split input and output
x <- dataset[,3:7]
y <- dataset[,1]

# Barplot for class breakdown
plot(y)

# Boxplot for each attribute on one image
par(mfrow=c(1,5))
for(i in 1:5) {
  boxplot(x[,i], main=names(x)[i])
}

# Scatterplot matrix
featurePlot(x=x, y=y, plot="pairs")

# Density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(ACTIVITY~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(ACTIVITY~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(ACTIVITY~., data=dataset, method="knn", metric=metric, trControl=control)
# Random Forest
# This may take few minutes to complete
set.seed(7)
fit.rf <- train(ACTIVITY~., data=dataset, method="rf", metric=metric, trControl=control)

# Summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, rf=fit.rf))
summary(results)

# Compare accuracy of models
dotplot(results)

# Summarize Best Model (RF)
print(fit.rf)

# Estimate skill of RF on the validation dataset
predictions <- predict(fit.rf, validation)
confusionMatrix(predictions, validation$ACTIVITY)