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


# load data
drug <- Matrix::readMM(file="dat/drugSparseMat.txt")
drug <- drug*1
meas <- Matrix::readMM(file="dat/measChangeSparseMat.txt")
meas <- meas*1
measIdx <- Matrix::readMM(file="dat/measChangeIndexMat.txt")
drugName <- read.csv("dat/drugName.csv", row.names = 1)
measName <- read.csv("dat/measName.csv", row.names = 1)
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
lambdas <- 10^seq(3, -2, by = -.1)
coefMat <- matrix(data = NA, nrow = numDrug, ncol = numMeas)
res <- matrix(data = NA, nrow = 2, ncol = numMeas)
for (outcome in seq(numMeas)){
  y <- meas[measIdx[,outcome], outcome]
  x <- drug[measIdx[,outcome],]
  res[1,outcome]<-length(y)
  if (length(y)>5*numDrug){
    cv_fit <- glmnet::cv.glmnet(x, y,
                                alpha = 0, lambda = lambdas,
                                nfolds=5)
    res[2,outcome] <- cv_fit$lambda.min
    coefMat[,outcome] <- coef(cv_fit, s = "lambda.min")[1:numDrug+1]
  }
}

coefMat<-coefMat[,!colSums(!is.finite(coefMat))]
heatmap(coefMat, xlab = "Measurement", ylab = "Drug")
measCorMat <- cor(coefMat)
heatmap(measCorMat)

