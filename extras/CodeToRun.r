# devtools::install_github("ohdsi/SqlRender")
# install.packages('SqlRender')

connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                             server = "omop.dbmi.columbia.edu")
connection <- DatabaseConnector::connect(connectionDetails)

vocabulary_database_schema <- "ohdsi_cumc_deid_pending.dbo"
cdm_database_schema <- "ohdsi_cumc_deid_pending.dbo"
target_database_schema <- "ohdsi_cumc_deid_pending.results"


cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"

myList<-MvConfounder::listingIngredients(connection,
                           cdmDatabaseSchema,
                           vocabularyDatabaseSchema = cdmDatabaseSchema,
                           minimumProportion = 0.2,
                           targetDrugTable = 'DRUG_ERA')
