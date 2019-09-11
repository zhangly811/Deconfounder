MvDeconfounder
======================

MvDeconfounder is an R package for building and validating the (multivariate) medical deconfounder using data in the OHDSI OMOP Common Data Model format. 

Introduction
============

MvDeconfounder is a causal inference model for efficiently and unbiasedly estimating the treatment effects of medications with eletronic health records. We aim to assess the treatment effects of multiple medications on multiple measurements from one cohort using the MvDeconfounder. MvDeconfounder identifies the causal medications that have either direct effect or adverse effect on each clinical measurement. The inputs to MvDeconfounder are medication records and pre-treatment and post-treatment measurement values. The MvDeconfounder fits a deep exponential family model to the medication records to construct substitute confounders and adjusts for substitute confounders in the outcome model for assessing the causal effects of medications. 

Technology
==========
MvDeconfounder is an R package, with some functions implemented in python.

System Requirements
===================
Requires R (version 3.4.0 or higher). Libraries used in MvDeconfounder require Java and Python.

The python installation is required for some of the machine learning algorithms. We advise to
install Python 3.6. 

Dependencies
============
 * DatabaseConnector
 * SqlRender
 * FeatureExtraction
 * PatientLevelPrediction

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
