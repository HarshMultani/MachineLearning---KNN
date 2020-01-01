# KNN algorithm

install.packages('dummies')
library('dummies')
install.packages('caTools')
library('caTools')
install.packages('tidyverse')
library('tidyverse')
install.packages('class')
library('class')


# Import the dataset
dataset <- read.csv('Social_Network_Ads.csv')


# One-Hot Encoding in R
dataset <- dummy.data.frame(dataset, names=c("Gender"), sep="_")


# Splitting the dataset into independent and dependent variables
X = dataset %>% select(2,3,4,5)
Y = dataset %>% select(6)


# Split the dataset into train and test sets
set.seed(123)
sample = sample.split(dataset, SplitRatio = 0.75)
X_train = subset(X, sample == TRUE)
X_test = subset(X, sample == FALSE)
Y_train = subset(Y, sample == TRUE)
Y_test = subset(Y, sample == FALSE)


# Feature Scaling the data
X_train <- scale(X_train)
X_test <- scale(X_test)


# Fitting KNN model to our data and predicting the test set results
Y_Pred <- knn(train = X_train, test = X_test, cl = Y_train[,1], k = 5)


# Making the confusion matrix
cm <- table(Y_test[,1], Y_Pred)