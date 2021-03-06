#Pregunta 1 Carga los datos. Realiza una inspecci�n por variables de la distribuci�n de aprobaci�n de cr�dito en funci�n de cada atributo visualmente. Realiza las observaciones pertinentes. � Qu� variables son mejores para separar los datos?

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
crx <- read.csv(url,na.strings = "?", header = FALSE)
write.csv(crx, file='crx.csv')
View(crx)
dim(crx)
head(crx)
str(crx)
summary(crx)

#Pregunta 2 Prepara el dataset convenientemente e imputa los valores faltantes usando la librer�a missForest

install.packages("missForest", dependencies = TRUE)
library(missForest)
data("crx")
summary(crx)
crx.imp <- missForest(crx)
crx.imp$ximp
crx.imp$OOBerror #OOB indicar el error, NRMSE es el error cuadratico medio normalizado (entre mas cercano a 0 este valor mejor el rendimiento) y PFC es la proporcion de entradas faltantes clasificadas
crx2 <- crx.imp$ximp 
View(crx2)
summary(crx2)
dim(crx2)
colnames(crx2)

#Pregunta 3 Divide el dataset tomando las primeras 590 instancias como train y las �ltimas 100 como test.

crx2$V16 <- as.numeric(crx2$V16) -1  
crx2$V16

crx2_train <- crx2[1:590, 1:16]
crx2_test <- crx2[591:690, 1:16]
X <- crx2[,1:15]
dim(X)
y <- crx2$V16 
unique(y)
dim(y)

head(y,20)
head(crx$V16, 20)

X_train <- X[1:590,]
y_train <- y[1:590]
X_test <- X[591:690,]
y_test <- y[591:690]

mod <- lm(crx2$V16~.,data=crx2) #no va es solo otra forma
mod2 <- lm(crx2$V16~V16,data=crx2) #no va es solo otra forma
summary(mod) #no va es solo otra forma
confint(mod) #no va es solo otra forma
AIC(mod) #no va es solo otra forma
AIC(mod2) #no va es solo otra forma
step <- stepAIC(lm(crx2$V16~1,data=crx2),scope = crx2$V16~V16, direction="forward") #no va es solo otra forma
  
  
help("glm")
fit1 <- glm(V16~., data=crx2, family=binomial)
fit1
summary(fit1)
fit0 <- glm(V16~1, data=crx2, family=binomial)
fit0
summary(fit0)
library(MASS)

step <-stepAIC(fit0,direction="forward",scope=list(upper=fit1,lower=fit0)) 
summary(step)

predict(step, X_test, type="response") #tambien hay otros type como raw, prob, etc verificar porque hay que usar response
predict(step, X_test, type="response")>.5
y_pred <- as.numeric(predict(step, X_test, type="response")>.5)
y_pred

install.packages(c("e1071", "caret", "e1071"))
install.packages(c("auc","class","gmodels"))
library(caret)
library(ggplot2)
library(lattice)
library(e1071)
library(auc)
library(class)
library(gmodels)

help("confusionMatrix")
View(y_pred)
View(y_test)
View(table(y_test,y_pred))
auc(y_test, y_pred)
confusionMatrix(table(y_test,y_pred), mode="everything") #La matriz de confusi�n y las m�tricas de erorr se pueden calcular. Para obtener todas las que se han dado incluir la opci�n mode="everything"
#intervalo de confianza para el accuracy

#Pregunta 4 Entrena un modelo de regresi�n log�stica con regularizaci�n Ridge y Lasso en train seleccionando el que mejor AUC tenga. Da las m�tricas en test.

X_train2 <- data.matrix(X_train) #Regresi�n log�stica con regularizaci�n Ridge
X_train2
X_test2 <- data.matrix(X_test)
X_test2
install.packages("glmnet")
library(glmnet)
set.seed(999)
cv.ridge <- cv.glmnet(X_train2, y_train, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE, type.measure='auc')
plot(cv.ridge) # Resultados
cv.ridge$lambda.min #este es el mejor valor de lambda
max(cv.ridge$cvm) #este es el valor del error que se estima para ese valor lambda m�nimo dado en MSE
coef(cv.ridge, s=cv.ridge$lambda.min) #coeficientes
#m�tricas en el test   
y_pred2 <- as.numeric(predict.glmnet(cv.ridge$glmnet.fit, newx=X_test2, s=cv.ridge$lambda.min)>.5)
y_pred2
View(table(y_test,y_pred2))
auc(y_test, y_pred2)
confusionMatrix(table(y_test,y_pred2), mode="everything")

install.packages("glmnet") #Regresi�n log�stica con regularizaci�n Lasso
library(glmnet)
set.seed(999)
cv.lasso <- cv.glmnet(X_train2, y_train, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
plot(cv.lasso)# Resultados
cv.lasso$lambda.min #este es el mejor valor de lambda
max(cv.lasso$cvm)#este es el valor del error que se estima para ese valor lambda m�nimo dado en MSE
coef(cv.lasso, s=cv.lasso$lambda.min) #coeficientes
y_pred3 <- as.numeric(predict.glmnet(cv.lasso$glmnet.fit, newx=X_test2, s=cv.lasso$lambda.min)>.5)
y_pred3 
View(table(y_test,y_pred3))
auc(y_test, y_pred3)
confusionMatrix(table(y_test,y_pred3), mode="everything")

#Pregunta 5 Aporta los log odds de las variables predictoras sobre la variable objetivo.

help("coef")
mod <- glm(y~., data=crx2, family=binomial)
mod <- glm(y ~ V9 + V11 + V15 + V13 + V5 + V6 + V14 + V10, data=SAheart, family=binomial)# No va
mod$coefficients
summary(mod)
coef(mod)
exp(coef(mod)) #Este el el log odds
exp(confint(mod))

#Pregunta 6 Si por cada verdadero positivo ganamos 100e y por cada falso positivo perdemos 20e. � Qu� rentabilidad aporta aplicar este modelo?

