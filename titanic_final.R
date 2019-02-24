#IMPORT LIBRARIES
library(tidyverse)
library(ggplot2)
library(mice)
library(caTools)
library(scales)
library(caret)

#IMPORT DATASETS
training_set = read.csv("train.csv")
test_set = read.csv("test.csv")
PassengerId = test_set$PassengerId

#PRE-PROCESSING
#Data Cleaning and manipulation
training_set = training_set %>%
  mutate(Embarked = ifelse(Embarked=="","S",as.character(Embarked)),
         Family = SibSp + Parch) %>%
  mutate_at(vars(Survived, Pclass,Embarked), factor) %>%
  select(-c(Name,Ticket,Cabin,PassengerId,SibSp,Parch))

test_set = test_set %>%
  mutate_at(vars(Pclass), factor) %>%
  mutate(Fare = ifelse(is.na(Fare), median(Fare,na.rm = T), Fare),
         Family = SibSp + Parch) %>%
  select(-c(Name,Ticket,Cabin,PassengerId, SibSp, Parch))

#############################################################################
#MISSING VALUE TREATMENT
#Impute missing values using MICE library
#Identify the missing values#md.pattern(training_set)
set.seed(100)
imputeTraining = mice(data = training_set, method = "pmm", m = 5, maxit = 50, seed = 500)
set.seed(100)
imputeTest = mice(data = test_set, method = "pmm", m = 5, maxit = 50, seed = 500)
#Check the imputed values #training_imputed$imp$Age
training_set = complete(imputeTraining,1)
test_set = complete(imputeTest,1)

########################################################################################
#EXPLORATORY DATA ANALYSIS
#Passenger Class
training_set %>% ggplot(aes(x=Survived)) + 
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
training_set %>% ggplot(aes(x=Survived)) + 
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
training_set %>% ggplot(aes(x=Survived)) + 
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
training_set %>% 
  ggplot(aes(x=Survived,y=Fare)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Ticker Fare") +
  xlab("Survived") +
  ylab("Ticket Fare") +
  theme_classic()

#Passenger Age
training_set %>%
  ggplot(aes(x=Survived,y=Age)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Passenger Age") +
  xlab("Survived") +
  ylab("Passenger Age") +
  theme_classic()

#Family Size
training_set %>%
  ggplot(aes(x=Survived,y=Family)) + 
  geom_boxplot(aes(fill = Survived), width=0.5) +
  ggtitle("Survived ~ Family Size") +
  xlab("Survived") +
  ylab("Family Size") +
  theme_classic()

#########################################################################
#SCALING DATA
# training_set = training_set %>% mutate_at(vars(Pclass,Sex,Embarked),as.numeric)
# test_set = test_set %>% mutate_at(vars(Pclass,Sex,Embarked),as.numeric)
# training_set[-1] = scale(training_set[-1])
# test_set = scale(test_set)

##########################################################################
#FEATURE SELECTION
#Backward Elimination
model.full = glm(Survived~., data = training_set, family = binomial)
model.null = glm(Survived~1, data = training_set, family = binomial)
model.final = step(model.full, scope = list(lower = model.null), 
                   direction = "backward")
summary(model.final)

#Forward Elimination
model.full = glm(Survived~., data = training_set, family = binomial)
model.null = glm(Survived~1, data = training_set, family = binomial)
model.final = step(model.null, scope = list(upper = model.full), 
                   direction = "forward")
summary(model.final)

#Recursive Feature Elimination - Feature Selection
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
model.rfe <- rfe(training_set[,c(3,2,4,7)], training_set[,1], 
                 sizes=c(1:6), rfeControl=control)
print(model.rfe)
predictors(model.rfe)
plot(model.rfe, type=c("g","o"))
#Final Features - PClass, Sex, Age, Family
###########################################################################
#BUILDING MODELS
control = trainControl(method = "cv", number = 10)
metric = "Accuracy"
#Random Forest
set.seed(100)
fit.rf = train(Survived~Pclass + Sex + Age + Family,
               data = training_set, method = "rf",
               metric = metric, trControl = control)
predict.rf = predict(fit.rf, test_set)
predictdfrf = data.frame(PassengerId = PassengerId,Survived=predict.rf)
write.csv(predictdfrf, "predictrf.csv",row.names = F)
#SCALING DATA
training_set = training_set %>% mutate_at(vars(Pclass,Sex,Embarked),as.numeric)
test_set = test_set %>% mutate_at(vars(Pclass,Sex,Embarked),as.numeric)
training_set[-1] = scale(training_set[-1])
test_set = scale(test_set)
#Support Vector Machines
set.seed(100)
fit.svm = train(Survived~Pclass + Sex + Age + Family,
               data = training_set, method = "svmRadial",
               metric = metric, trControl = control)
predict.svm = predict(fit.svm, test_set)
predictdfsvm = data.frame(PassengerId = PassengerId,Survived=predict.svm)
write.csv(predictdfsvm, "predictsvm.csv",row.names = F)
#KNN
set.seed(100)
fit.knn = train(Survived~Pclass + Sex + Age + Family,
                data = training_set, method = "knn",
                metric = metric, trControl = control)
predict.knn = predict(fit.knn, test_set)
predictdfknn = data.frame(PassengerId = PassengerId,Survived=predict.knn)
write.csv(predictdfknn, "predictknn.csv",row.names = F)
#Logistic Regression
set.seed(100)
fit.glm = train(Survived~Pclass + Sex + Age + Family,
                data = training_set, method = "glm",
                metric = metric, trControl = control)
predict.glm = predict(fit.glm, test_set)
predictdfglm = data.frame(PassengerId = PassengerId,Survived=predict.glm)
write.csv(predictdfglm, "predictglm.csv",row.names = F)
#LDA
set.seed(100)
fit.lda = train(Survived~Pclass + Sex + Age + Family,
                data = training_set, method = "lda",
                metric = metric, trControl = control)
predict.lda = predict(fit.lda, test_set)
predictdflda = data.frame(PassengerId = PassengerId,Survived=predict.lda)
write.csv(predictdflda, "predictlda.csv",row.names = F)
#All models together
predictdf = data.frame(PassengerId = PassengerID, 
                       lda=predict.lda,
                       rf=predict.rf,
                       svm=predict.svm,
                       glm=predict.glm,
                       knn=predict.knn)

write.csv(predictdf, "predictall.csv",row.names = F)
#XGBoost
#Build an XGBoost Model
xgboost.model = xgb.cv(data = as.matrix(training_set[-1]),
                       label = as.numeric(training_set$Survived)-1,
                       objective = "binary:logistic",
                       eval_metric = "error",
                       nrounds = 1000,
                       eta = 0.1, 
                       max_depth = 5,
                       min_child_weight = 1,
                       gamma = 0,
                       subsample = 0.9,
                       nthread = 4,
                       #colsample_bytree = 0.9,
                       early_stopping_rounds = 50,
                       nfold = 5,
                       seed = 100)


#Build an Artificial Neural Network
#Fitting ANN
# library(h2o)
# h2o.init()
# install.packages("h2o")
# classifier = h2o.deeplearning(y = "Survived",
#                               training_frame = as.h2o(training_set),
#                               activation = "Rectifier",
#                               hidden = c(6,6),
#                               epochs = 100,
#                               train_samples_per_iteration = -2)
# prob_pred = h2o.predict(classifier, as.h2o(test_set))
# pred = prob_pred > 0.5
# pred = as.vector(pred)