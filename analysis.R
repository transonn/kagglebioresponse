#--------------------------------------------------
#Training set analysis
#Kaggle - Predicting a Biological Response
#
#Transon Nguyen
#--------------------------------------------------

library(ggplot2) #pretty plotting
library(nnet) #neural networks
library(caret) #helpful stats functions

#FUNCTION checkerror: calculate log loss error.
checkerror <- function(y, yhat) {
	N <- length(y)
	eps <- 1e-15	
	yhat <- pmin(pmax(yhat, eps), 1-eps) #cap predictions, prevents log(0)
	error <- -1/N * sum(y*log(yhat) + (1-y)*log(1-yhat), na.rm=T)
	return(error)
}

data <- read.csv("train.csv") #read training data
#attach(data)

#split training data into training and test sets, 70% training, 30% testing
set.seed(1)
trainindex <- createDataPartition(1:nrow(data), times=1, p=.7, list=T)
trainindex <- trainindex$Resample1
trainset <- data[trainindex,]
testset <- data[-trainindex,]

y <- trainset[,1] #Activity column
x <- trainset[,-1] #all other columns
correlations <- cor(x, y) #correlation of all features relative to Activity
correlations[is.na(correlations)] <- 0 #replace NAs with 0 (no variance)

#fit gaussian to correlations, set bounds for top 5% (lol is this a good idea?)
cormean <- mean(correlations)
corsd <- sd(correlations)
corupper <- qnorm(.999, mean=cormean, sd=corsd)
corlower <- qnorm(.05, mean=cormean, sd=corsd)

#find features that correlate the most with Activity
#significants <- which(correlations > actupper | correlations < actlower)
significants <- which(correlations > corupper)

#create matrix of relevant features
features <- x[,significants]

#fit neural network based off relevant features
a <- nnet(features, y, linout=T, size=1, maxit=2000, MaxNWts=1500)
yhattrain <- a$fitted.values #prediction results
yhattrain <- pmin(pmax(yhattrain, .05), .95) #cap predictions
checkerror(y, yhattrain)

#use previously created neural network on test set features
xtest <- testset[,-1]
ytest <- testset[,1]
yhattest <- predict(a, xtest[,significants])
yhattest <- pmin(pmax(yhattest, .05), .95) #cap predictions
checkerror(ytest, yhattest)

#visualize prediction vs results
ggplot() + geom_point(aes(1:40,yhattrain[1:40]), colour="blue") + geom_point(aes(1:40,y[1:40]), colour="red")

ggplot() + geom_point(aes(1:40,yhattest[1:40]), colour="blue") + geom_point(aes(1:40,ytest[1:40]), colour="red")


#-----------------REAL MODELING FOR REAL-----------------
testdata <- read.csv("test.csv") #read test data

testfeatures <- testdata[,significants]
#use ANN on test set
yhat <- predict(a, testfeatures)

write(yhat,"20120407-2.csv", ncolumns=1)
