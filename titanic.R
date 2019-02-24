#Import necessary libraries
library(tidyverse)
library(ggplot2)
library(mice)
library(caTools)
library(scales)
library(caret)
# #Import datasets
# training_set = read.csv("train.csv")
# test_set = read.csv("test.csv")
# 
# #Pre-Processing
# #Data Cleaning and manipulation
# training_set = training_set %>%
#   select(-c(Name,Ticket,Cabin)) %>%
#   mutate(Embarked = ifelse(Embarked=="","S",as.character(Embarked))) %>%
#   mutate_at(vars(Survived, Pclass,Embarked), factor)
# 
# test_set = test_set %>%
#   select(-c(Name,Ticket,Cabin)) %>%
#   mutate_at(vars(Pclass), factor) %>%
#   mutate(Fare = ifelse(is.na(Fare), median(Fare,na.rm = T), Fare))
# 
# #IMPUTE MISSING VALUES USING MICE LIBRARY
# #Identify the missing values#md.pattern(training_set)
# imputeTraining = mice(data = training_set, method = "pmm", m = 5, maxit = 50, seed = 500)
# imputeTest = mice(data = test_set, method = "pmm", m = 5, maxit = 50, seed = 500)
# #Check the imputed values #training_imputed$imp$Age
# training_set = complete(imputeTraining,1)
# test_set = complete(imputeTest,1)
# 
# #SPLIT TRAINING INTO TRAINING AND VALIDATION
# split = sample.split(training_set$Survived, SplitRatio = 0.75)
# trainingData = subset(training_set, split == TRUE)
# validationData = subset(training_set, split == FALSE)
# write.csv(trainingData,"trainingData.csv",row.names = F)
# write.csv(validationData,"validationData.csv",row.names = F)

########################################################################################
#Import Training and Validation data
trainingData = read.csv("trainingData.csv")
validationData = read.csv("validationData.csv")
trainingData = trainingData %>% mutate_at(vars(Survived,Pclass), factor)
validationData = validationData %>% mutate_at(vars(Pclass), factor)

########################################################################################
#Exploratory Data Analysis
#Passenger Class
trainingData %>% ggplot(aes(x=Survived)) + 
  geom_bar(aes(fill=Pclass),position = "fill",width=0.5) +
  ggtitle("Survived ~ Passenger Class") +
  xlab("Survived") +
  ylab("% of Passengers") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("steelblue3","turquoise3","royalblue3"),
                      name ="Passenger\nClass", 
                      breaks = c(1,2,3),
                      labels = c("Upper","Middle","Lower")) +
  theme_classic()

#Gender of the Passengers
trainingData %>% ggplot(aes(x=Survived)) + 
  geom_bar(aes(fill=Sex),position = "fill",width=0.5) +
  ggtitle("Survived ~ Passenger's Gender") +
  xlab("Survived") +
  ylab("% of Passengers") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("turquoise3","royalblue3"),
                    name ="Gender", 
                    breaks = c("male","female"),
                    labels = c("Male","Female")) +
  theme_classic()

#Embarked Location
trainingData %>% ggplot(aes(x=Survived)) + 
  geom_bar(aes(fill=Embarked),position = "fill",width=0.5) +
  ggtitle("Survived ~ Embarked Location") +
  xlab("Survived") +
  ylab("% of Passengers") +
  scale_y_continuous(labels = percent_format()) +
  scale_fill_manual(values = c("steelblue3","turquoise3","royalblue3"),
                    name ="Embarked City", 
                    breaks = c("C","Q","S"),
                    labels = c("Cherbourg","Queenstown","Southampton")) +
  theme_classic()

#Ticket Fare
trainingData %>% 
  ggplot(aes(x=Survived,y=Fare)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Ticker Fare") +
  xlab("Survived") +
  ylab("Ticket Fare") +
  theme_classic()

#Passenger Age
trainingData %>%
  ggplot(aes(x=Survived,y=Age)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Passenger Age") +
  xlab("Survived") +
  ylab("Passenger Age") +
  theme_classic()

#Siblings/Spouses on board
trainingData %>%
  ggplot(aes(x=Survived,y=SibSp)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Passenger Age") +
  xlab("Survived") +
  ylab("Siblings/Spouses") +
  theme_classic()

#Parents/Children on board
trainingData %>%
  ggplot(aes(x=Survived,y=Parch)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Passenger Age") +
  xlab("Survived") +
  ylab("Parents/Children") +
  theme_classic()

########################################################################################
#Feature Selection
#Learning Vector Quantization - Feature Selection
control = trainControl(method="repeatedcv", number=10, repeats=3)
model.lvq = train(Survived~.-PassengerId, data=trainingData, method="lvq", 
               preProcess="scale", trControl=control)
importance = varImp(model.lvq, scale = FALSE)
print(importance)
plot(importance)
#Top 5 - Sex, Fare, Pclass, Embarked, Parch

#Recursive Feature Elimination - Feature Selection
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
model.rfe <- rfe(trainingData[,3:9], trainingData[,2], 
               sizes=c(1:7), rfeControl=control)
print(model.rfe)
predictors(model.rfe)
plot(model.rfe, type=c("g","o"))
#Filtered 4 - Sex, Pclass, Age, Fare

#Backward Elimination
model.full = glm(Survived~.-PassengerId, data = trainingData, family = binomial)
model.null = glm(Survived~1, data = trainingData, family = binomial)
step(model.full, scope = list(lower = model.null), direction = "backward")
model.final = glm(Survived~Pclass + Sex + Age + SibSp + Embarked, 
                  family = binomial, data = trainingData)
summary(model.final)

#Forward Elimination
model.full = glm(Survived~.-PassengerId, data = trainingData, family = binomial)
model.null = glm(Survived~1, data = trainingData, family = binomial)
step(model.null, scope = list(upper = model.full), direction = "forward")
model.final = glm(Survived~Pclass + Sex + Age + SibSp + Embarked, 
                  family = binomial, data = trainingData)
summary(model.final)
#Filtered 5 - Pclass, Sex, Age, SibSp, Embarked

#Building Models
#Logistic Regression - Backward Elimination Variables
#Accuracy high with Pclass, Sex, Age, Sibsp
control = trainControl(method = "cv", number = 10)
fit.glm = train(Survived~Pclass + Sex + Age + SibSp, 
                data = trainingData, method = "glm",
                metric = "Accuracy", trControl = control)
summary(fit.glm)
predict.glm = predict(fit.glm, validationData)
cm = table(predict.glm, validationData$Survived)
Accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
#Accuracy = 79.37%

#K Nearest Neighbors 
#Accuracy high with Pclass, Sex, Age, Sibsp
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"
fit.knn = train(Survived~Pclass + Sex + Age + SibSp, 
                data = trainingData, method = "knn",
                metric = metric, trControl = control)
predict.knn = predict(fit.knn, validationData)
cm = table(predict.knn, validationData$Survived)
Accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
#Accuracy 75.34%

#Support Vector Machines
#Accuracy higher with all predictors
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"
fit.svm = train(Survived~.-PassengerId,
                data = trainingData, method = "svmRadial",
                metric = metric, trControl = control)
predict.svm = predict(fit.svm, validationData)
cm = table(predict.svm, validationData$Survived)
Accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
#Accuracy 82.95%

#Random Forest
#Accuracy higher with all predictors
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"
fit.rf = train(Survived~.-PassengerId,
                data = trainingData, method = "rf",
                metric = metric, trControl = control)
predict.rf = predict(fit.rf, validationData)
cm = table(predict.rf, validationData$Survived)
Accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
#Accuracy 84.75%

#Linear Discriminant Analysis
#Accuracy higher with all predictors
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"
fit.lda = train(Survived~.-PassengerId,
               data = trainingData, method = "lda",
               metric = metric, trControl = control)
predict.lda = predict(fit.lda, validationData)
cm = table(predict.lda, validationData$Survived)
Accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[1,2]+cm[2,1])
#Accuracy 79.82%