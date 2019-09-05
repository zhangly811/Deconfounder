# devtools::install_github("ohdsi/SqlRender")
# install.packages('SqlRender')

connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                             server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"

myList<-MvConfounder::listingIngredients(connection,
                           cdmDatabaseSchema,
                           vocabularyDatabaseSchema = cdmDatabaseSchema,
                           minimumProportion = 0.1,
                           targetDrugTable = 'DRUG_ERA')
