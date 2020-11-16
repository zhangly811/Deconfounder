MvDeconfounder
======================

MvDeconfounder is an R package for building and validating the (multivariate) medical deconfounder using data in the OHDSI OMOP Common Data Model format. 

Introduction
============

MvDeconfounder is a causal inference model for estimating the treatment effects of medications with eletronic health records. We aim to assess the treatment effects of multiple medications on measurements using the MvDeconfounder. MvDeconfounder identifies the causal medications that have either direct effect or adverse effect on each clinical measurement. The inputs to MvDeconfounder are medication records and pre-treatment and post-treatment measurement values. The MvDeconfounder fits a probabilistic factor model (e.g., poisson factoriztion or deep exponential family) to the medication records to construct substitute confounders and adjusts for substitute confounders in the outcome model for assessing the causal effects of medications. 

Technology
==========
MvDeconfounder is an R package, with some functions implemented in python.

System Requirements
===================
Requires R (version 3.4.0 or higher), Python (version 3.7 or higher) and Java.


Dependencies
============
OHDSI R packages:

- [DatabaseConnector](https://github.com/OHDSI/DatabaseConnector)

- [SqlRender](https://github.com/OHDSI/SqlRender)
 
Python packages:

- torch==1.6.0

- numpy

- pandas

- scipy

- sklearn
 
Installation
============
Install from github using devtools package.
```r
install.packages("devtools")
devtools::install_github("zhangly811/MvDeconfounder")
```
 
User Documentation
============
Vignette: [Deconfounder on single outcome](https://github.com/zhangly811/MvDeconfounder/blob/master/inst/doc/DeconfounderSingleOutcome.pdf)



License
=======
MvDeconfounder is licensed under Apache License 2.0

Development
===========
MvDeconfounder is being developed in R Studio.

References
===========
Zhang L, Wang Y, Ostropolets A, Mulgrave JJ, Blei DM, Hripcsak G. [The Medical Deconfounder: Assessing Treatment Effects with Electronic Health Records.](https://arxiv.org/abs/1904.02098) Accepted at Machine Learning for Health Care (MLHC) Conference 2019.

Wang Y, Blei DM. [The Blessing of Multiple Causes.](https://arxiv.org/abs/1805.06826)
