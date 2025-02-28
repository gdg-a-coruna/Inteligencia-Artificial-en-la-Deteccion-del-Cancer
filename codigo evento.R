# PASO 1: Importar y preparar el conjunto de datos
# Cargar el conjunto de datos desde el repositorio UCI
url_datos <- paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/", 
                    "breast-cancer-wisconsin/breast-cancer-wisconsin.data")
cancer <- read.csv(url_datos, header = FALSE, stringsAsFactors = FALSE)

# Asignar nombres a las columnas
names(cancer) <- c("ID", "grosor", "tamaño_celula", "forma_celula", "adhesion",
                   "tamaño_epitelial", "nucleos_desnudos", "cromatina_blanda", "nucleolos_normales", "mitosis", "clase")

# Preprocesamiento de datos
cancer$nucleos_desnudos <- replace(cancer$nucleos_desnudos, cancer$nucleos_desnudos == "?", NA)
cancer <- na.omit(cancer)
cancer$clase <- (cancer$clase / 2) - 1

# Dividir el conjunto de datos en entrenamiento y prueba
set.seed(80817)
indice <- 1:nrow(cancer)
indice_prueba <- sample(indice, trunc(length(indice) / 3))
conjunto_prueba <- cancer[indice_prueba, ]
conjunto_entrenamiento <- cancer[-indice_prueba, ]

# Preparar matrices de datos
x_entrenamiento <- data.matrix(conjunto_entrenamiento[, 2:10])
y_entrenamiento <- as.numeric(conjunto_entrenamiento[, 11])
x_prueba <- data.matrix(conjunto_prueba[, 2:10])
y_prueba <- as.numeric(conjunto_prueba[, 11])

# PASO 2: Entrenar los algoritmos de aprendizaje automático
# Cargar el paquete glmnet y entrenar el modelo LASSO
require(glmnet)
modelo_glm <- cv.glmnet(x_entrenamiento, y_entrenamiento, alpha = 1, nfolds = 10)
lambda_min <- modelo_glm$lambda.min
coef_glm <- round(coef(modelo_glm, s = lambda_min), 2)

# Crear las gráficas de LASSO
plot(modelo_glm)
plot(glmnet(x_entrenamiento, y_entrenamiento, family = "gaussian", alpha = 1), "lambda", label = TRUE)
abline(v = log(lambda_min), lty = 3)

# Cargar el paquete e1071 y entrenar el modelo SVM
require(e1071)
modelo_svm <- svm(x_entrenamiento, y_entrenamiento, cost = 1, gamma = c(1 / (ncol(x_entrenamiento) - 1)), kernel = "radial", cross = 10)

# Cargar el paquete nnet y entrenar el modelo de red neuronal
require(nnet)
modelo_nnet <- nnet(x_entrenamiento, y_entrenamiento, size = 5)

# PASO 3: Testear los modelos de ML
# Hacer predicciones
prediccion_glm <- round(predict(modelo_glm, x_prueba, type = "response"), 0)
prediccion_svm <- round(predict(modelo_svm, x_prueba, type = "response"), 0)
prediccion_nnet <- round(predict(modelo_nnet, x_prueba, type = "raw"), 0)

# Modelo de conjunto (ensemble)
predicciones <- data.frame(prediccion_glm, prediccion_svm, prediccion_nnet)
names(predicciones) <- c("glm", "svm", "nnet")
predicciones$votos_ensemble <- round(rowSums(predicciones) / 3)

# PASO 4: Evaluar la sensibilidad, especificidad y precisión de los modelos
# Evaluación de la matriz de confusión
require(caret)
confusionMatrix(as.factor(prediccion_glm), as.factor(y_prueba))
confusionMatrix(as.factor(prediccion_svm), as.factor(y_prueba))
confusionMatrix(as.factor(prediccion_nnet), as.factor(y_prueba))
confusionMatrix(as.factor(predicciones$votos_ensemble), as.factor(y_prueba))

# Análisis de la curva ROC
require(pROC)
roc_glm <- roc(as.vector(y_prueba), as.vector(prediccion_glm))
roc_svm <- roc(as.vector(y_prueba), as.vector(prediccion_svm))
roc_nnet <- roc(as.vector(y_prueba), as.vector(prediccion_nnet))

# Creación de las curvas ROC
plot.roc(roc_glm, ylim = c(0, 1), xlim = c(1, 0))
lines(roc_glm, col = "blue")
lines(roc_svm, col = "red")
lines(roc_nnet, col = "green")
legend("bottomright", legend = c("GLM", "SVM", "NNet"), col = c("blue", "red", "green"), lwd = 2)

# Cálculo los valores de AUC
auc_glm <- auc(roc_glm)
auc_svm <- auc(roc_svm)
auc_nnet <- auc(roc_nnet)

# PASO 5: Aplicar nuevos datos a los modelos entrenados
# Hacer nuevas predicciones con los modelos entrenados
nuevos_datos <- c(8, 7, 8, 5, 5, 7, 9, 8, 10)
nueva_prediccion_glm <- predict(modelo_glm, data.matrix(t(nuevos_datos)), type = "response")
nueva_prediccion_svm <- predict(modelo_svm, data.matrix(t(nuevos_datos)), type = "response")
nueva_prediccion_nnet <- predict(modelo_nnet, data.matrix(t(nuevos_datos)), type = "raw")

print(nueva_prediccion_glm)
print(nueva_prediccion_svm)
print(nueva_prediccion_nnet)