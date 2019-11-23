# Copyright 2019 Observational Health Data Sciences and Informatics
#
# This file is part of MvConfounder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# memory.limit(size=10000)

# install.packages("qlcMatrix")
# install.packages("d3heatmap")

# load data
inputFolder <- "dat/20191116Complete"
outputFolder <- "res/20191118Complete"
drug <- Matrix::readMM(file=file.path(inputFolder, "drugSparseMat.txt"))
drug <- drug*1
meas <- Matrix::readMM(file=file.path(inputFolder, "measChangeSparseMat.txt"))
meas <- meas*1
measIdx <- Matrix::readMM(file=file.path(inputFolder, "measChangeIndexMat.txt"))
drugName <- as.character(read.csv(file=file.path(inputFolder, "drugName.csv"), row.names = 1)$V1)
measName <- as.character(read.csv(file=file.path(inputFolder, "measName.csv"), row.names = 1)$V1)
measName <- gsub(" in .*", "", measName)
measName <- gsub("\\s*\\([^\\)]+\\)","", measName)
measName <- gsub("\\s*\\[[^\\)]+\\]","", measName)
# find highly correlated drug pairs
corMat <-  qlcMatrix::corSparse(drug)
corMat[!lower.tri(corMat)] <- 0
condition <- which(abs(corMat)>=0.8, arr.ind = T)
drugToRemove<-unique(condition[,2])
corDrug <- data.frame(drugName[condition[,1],], drugName[condition[,2],])
colnames(corDrug)<- c("DrugName1", "DrugName2")
# remove highly correlated drugs
drug <- drug[, -c(drugToRemove)]
drugName <- drugName[-c(drugToRemove)]

# normalize measurements
meas@x <- meas@x / rep.int(Matrix::colSums(meas), diff(meas@p))

numDrug <- ncol(drug)
numMeas <- ncol(meas)
lambdas <- 10^seq(2, -2, by = -.1)
coefMat <- matrix(data = NA, nrow = numDrug, ncol = numMeas)
rownames(coefMat) <- drugName
colnames(coefMat) <- measName
res <- matrix(data = NA, nrow = 2, ncol = numMeas)
for (outcome in seq(numMeas)){
  print(paste0("Running outcome ", outcome))
  rowIdx <- which(meas[,outcome]!=0)
  y <- meas[rowIdx, outcome]
  x <- drug[rowIdx,]
  res[1,outcome]<-length(y)
  if (length(y)>5*numDrug){
    cv_fit <- glmnet::cv.glmnet(x, y,
                                alpha = 0, lambda = lambdas,
                                nfolds=5)
    res[2,outcome] <- cv_fit$lambda.min
    coefMat[,outcome] <- coef(cv_fit, s = "lambda.min")[1:numDrug+1]
    write.csv(coefMat, file=file.path(outputFolder, "coefMat.csv"))
  }
}
coefMat <- as.matrix(read.csv(file=file.path(outputFolder, "coefMat.csv"), row.names = 1))
coefMat<-coefMat[,!colSums(!is.finite(coefMat))]
drugName <- rownames(coefMat)
measName <- colnames(coefMat)
measCorMat <- cor(coefMat)
drugCorMat <- cor(t(coefMat))

d3heatmap::d3heatmap(measCorMat, symm = TRUE)
d3heatmap::d3heatmap(drugCorMat, symm = TRUE)
