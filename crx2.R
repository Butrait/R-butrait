#Pregunta 1 Carga los datos. Realiza una inspecci�n por variables de la distribuci�n de aprobaci�n de cr�dito en funci�n de cada atributo visualmente. Realiza las observaciones pertinentes. � Qu� variables son mejores para separar los datos?

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
crx <- read.csv(url,na.strings = "?", header = FALSE)
write.csv(crx, file='crx.csv')
dim(crx)
str(crx)
summary(crx)
#La mejor variable para separar los datos es el factor V16, porque separa los datos en .....

#Pregunta 2 Prepara el dataset convenientemente e imputa los valores faltantes usando la librer�a missForest

install.packages("missForest", dependencies = TRUE)
library(missForest)
crx.imp <- missForest(crx) # Cargo el dataset crx en missForest
crx.imp$ximp
crx.imp$OOBerror #OOB indicar el error, NRMSE es el error cuadratico medio normalizado (entre mas cercano a 0 este valor mejor el rendimiento) y PFC es la proporcion de entradas faltantes clasificadas
crx2 <- crx.imp$ximp # crx2 es el nuevo dataset sin valores nulos.
summary(crx2)
dim(crx2)

#Pregunta 3 Divide el dataset tomando las primeras 590 instancias como train y las �ltimas 100 como test.

crx2$V16 <- as.numeric(crx2$V16) -1  
crx2_train <- crx2[1:590, 1:16]
crx2_test <- crx2[591:690, 1:16]
X <- crx2[,1:15]
y <- crx2$V16 
unique(y)
X_train <- X[1:590,]
y_train <- y[1:590]
X_test <- X[591:690,]
y_test <- y[591:690]
#Se dividio el dataset crx2 en primeras 590 instancias como X_train y y_train, las �ltimas 100 como X_test y y_test.

#Pregunta 4 Entrena un modelo de regresi�n log�stica con regularizaci�n Ridge y Lasso en train seleccionando el que mejor AUC tenga. Da las m�tricas en test.

#Primero realizare una regresion logistica mediante AIC.

#help("glm")
fit1 <- glm(V16~., data=crx2, family=binomial)
summary(fit1)
fit0 <- glm(V16~1, data=crx2, family=binomial)
summary(fit0)
library(MASS)
step <-stepAIC(fit0,direction="forward",scope=list(upper=fit1,lower=fit0)) 
summary(step)

#El mejor modelo de regresi�n log�stica con AIC es V16 ~ V9 + V11 + V15 + V13 + V5 + V6 + V14 + V10

y_pred <- as.numeric(predict(step, X_test)>.5)
install.packages(c("e1071", "caret", "ggplot2","lattice","class","gmodels"))
library(caret)
library(ggplot2)
library(lattice)
library(e1071)
library(auc)
library(class)
library(gmodels)
#help("confusionMatrix") #La matriz de confusi�n y las m�tricas de erorr se pueden calcular. Para obtener todas las que se han dado incluir la opci�n mode="everything"
confusionMatrix(table(y_test,y_pred), mode="everything") 

# efectuaremos las regresiones log�sticas con regularizaci�n Ridge y Lasso.

#Regresi�n log�stica con regularizaci�n Ridge
X_train2 <- data.matrix(X_train)
X_test2 <- data.matrix(X_test)
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
confusionMatrix(table(y_test,y_pred2), mode="everything")

#Regresi�n log�stica con regularizaci�n Lasso
set.seed(999)
cv.lasso <- cv.glmnet(X_train2, y_train, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')
plot(cv.lasso)# Resultados
cv.lasso$lambda.min #este es el mejor valor de lambda
max(cv.lasso$cvm)#este es el valor del error que se estima para ese valor lambda m�nimo dado en MSE
coef(cv.lasso, s=cv.lasso$lambda.min) #coeficientes
y_pred3 <- as.numeric(predict.glmnet(cv.lasso$glmnet.fit, newx=X_test2, s=cv.lasso$lambda.min)>.5)
confusionMatrix(table(y_test,y_pred3), mode="everything")

#El modelo de Ridge tiene mayor precision y mejor AUC.

#Pregunta 5 Aporta los log odds de las variables predictoras sobre la variable objetivo.

help("coef")
logistic_ridge<- glm(V16~., data=crx2, family=binomial)
logistic_AIC<- glm(y ~ V9 + V11 + V15 + V13 + V5 + V6 + V14 + V10, data=crx2, family=binomial)# Este es usando el modelo por AIC
summary(logistic_ridge)
summary(logistic_AIC)

# Para el modelo de ridge como el valor de p-values es superior a 0.05 en todas las variables el log odds no aporta en las variables, en cambio se usaramos el modelo de regresi�n log�stica con AIC si hay variables que aportan los log odds de las variables predictoras sobre la variable objetivo debido que hay el p-values es superior a 0.05 en algunas varaibles.

#Pregunta 6 Si por cada verdadero positivo ganamos 100e y por cada falso positivo perdemos 20e. � Qu� rentabilidad aporta aplicar este modelo?

matrix_test_ridge <- confusionMatrix(table(y_test,y_pred2), mode="everything")
matrix_test_ridge

matrix_test_AIC <- confusionMatrix(table(y_test,y_pred), mode="everything") 
matrix_test_AIC

#La precision del modelo de ridge es 100% por lo tanto hay solo ganancias por no haber falsos positivos.
#En cambio para el modelo por AIC la presicion es de 98,84% por lo tanto...............

