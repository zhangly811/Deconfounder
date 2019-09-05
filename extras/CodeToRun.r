# devtools::install_github("ohdsi/SqlRender")
# install.packages('SqlRender')

connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                             server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_pending.results"
targetCohortTable = "COHORT"

ingredientList<-MvDeconfounder::listingIngredients(connection,
                           cdmDatabaseSchema,
                           vocabularyDatabaseSchema = cdmDatabaseSchema,
                           minimumProportion = 0.07,
                           targetDrugTable = 'DRUG_ERA')
ingredientList = myList
ingredientConceptIds = c(ingredientList$drugConceptIds['conceptId'])
ingredientConceptIds = c(967823, 1125315)
measurementConceptIds = c(3023103, 3016723)

myCohort = MvDeconfounder::createCohorts(connection,
                        cdmDatabaseSchema,
                        vocabularyDatabaseSchema = cdmDatabaseSchema,
                        cohortDatabaseSchema,
                        targetCohortTable,
                        # oracleTempSchema,
                        ingredientConceptIds = ingredientConceptIds,
                        measurementConceptIds = measurementConceptIds,
                        drugWindow = 35,
                        labWindow = 35,
                        targetCohortId = 999)
