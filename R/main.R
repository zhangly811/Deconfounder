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


library(rJava) #if not working: try running "sudo R CMD javareconf" in terminal
library(SqlRender)
library(DatabaseConnector)
?createConnectionDetails
connectionDetails <- createConnectionDetails(dbms="sql server", 
                                             server="omop.dbmi.columbia.edu",
                                             user="lz2629",
                                             password="ZHcar0811.")

DatabaseConnector::connect(connectionDetails)

listingIngredients()
createCohorts()