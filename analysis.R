#--------------------------------------------------
#Training set analysis
#Kaggle - Predicting a Biological Response
#
#Transon Nguyen
#--------------------------------------------------

library(ggplot2) #pretty plotting
library(nnet) #neural networks
library(caret) #helpful stats functions

data <- read.csv("train.csv") #read training data
attach(data)

#split training data into training and test sets, 70% training, 30% testing
set.seed(1)
trainindex <- createDataPartition(1:nrow(data), times=1, p=.7, list=T)
trainindex <- trainindex$Resample1
trainset <- data[trainindex,]
testset <- data[-trainindex,]

y <- trainset[,1] #Activity column
x <- trainset[,-1] #all other columns

y <- data[,1] #Activity column
x <- data[,-1] #all other columns
correlations <- cor(x,y) #correlation of all features relative to Activity

#fit gaussian to correlations, set bounds for top 5%
cormean <- mean(correlations)
corsd <- sd(correlations)
corupper <- qnorm(.95, mean=cormean, sd=corsd)
corlower <- qnorm(.05, mean=cormean, sd=corsd)

#find features that correlate the most with Activity
#significants <- which(correlations > actupper | correlations < actlower, arr.ind=T)
significants <- which(correlations > corupper)
#significantsorted <- sort(correlations[significants], decreasing=TRUE)

#create matrix of relevant features
#features <- get(names(significantsorted[1]))
#for (i in 2:length(significantsorted)) {
#	features <- cbind(features, get(names(significantsorted[i])))
#}

#neural network based off previously found features
a <- nnet(features, Activity, entropy=T, size=4, maxit=2000, MaxNWts=1500)
prediction <- a$fitted.values #prediction results

#visualize prediction vs results
ggplot() + geom_point(aes(1:40,prediction[1:40]), colour="blue") + geom_point(aes(1:40,Activity[1:40]), colour="red")

#check error
N <- length(Activity)
error <- -1/N * sum(Activity*log(prediction) + (1-Activity)*log(1-prediction), na.rm=T)
print(error)

#-----------------REAL MODELING FOR REAL-----------------
testdata <- read.csv("test.csv") #read testing data
detach(data)
attach(testdata)

#create matrix of relevant features from testing set
features <- get(names(significantsorted[1]))
for (i in 2:length(significantsorted)) {
	features <- cbind(features, get(names(significantsorted[i])))
}

#use previously created neural network on testing set features
yhat <- predict(a, features)

write(yhat,"20120407-1.csv", ncolumns=1)
