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

drugSparseData <- Matrix::readMM("dat/drugSparseMat.txt")
drugSparseData <- drugSparseData*1
drugName <- read.csv("dat/drugName.csv", row.names = 1)

corMat <-  qlcMatrix::corSparse(drugSparseData)
corMat[!lower.tri(corMat)] <- NA
condition <- which(abs(corMat)>=0.6, arr.ind = T)
corDrug <- data.frame(drugName[condition[,1],], drugName[condition[,2],])

drugSparseData <- drugSparseData[,!apply(corMat,2,function(x) any(abs(x) >= 0.6))] ##TODO: LEFT HERE

#plot correlation matrix
melted_cormat <- reshape2::melt(corMat)
ggplot2::ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Pearson\nCorrelation") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                   size = 12, hjust = 1))+
  coord_fixed()
