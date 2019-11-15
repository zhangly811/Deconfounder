# Copyright 2019 Observational Health Data Sciences and Informatics
#
# This file is part of MvDeconfounder
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

# devtools::install_github("ohdsi/SqlRender")
# devtools::install_github("ohdsi/DatabaseConnector")
# devtools::install_github("ohdsi/FeatureExtraction")
# devtools::install_github("ohdsi/PatientLevelPrediction")''

options(fftempdir = "tmp/")
memory.limit(size=1024*12)

connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "postgresql",
                                                               server = "localhost/postgres",
                                                               port="5555",
                                                               user="lz2629",
                                                               password="ZHcar1020.")
connection = DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_pending.results"
targetCohortTable = "MVDECONFOUNDER_COHORT"
targetCohortId = 1

outputFolder <- "res/MvConfounderV1T1"
