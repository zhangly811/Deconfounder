# restricts to pop and saves/creates mapping
MapCovariates <- function(covariates, covariateRef, population, map){

  # restrict to population for speed
  ParallelLogger::logTrace('restricting to population for speed...')
  idx <- ffbase::ffmatch(x = covariates$rowId, table = ff::as.ff(population$rowId))
  idx <- ffbase::ffwhich(idx, !is.na(idx))
  covariates <- covariates[idx, ]

  ParallelLogger::logTrace('Now converting covariateId...')
  oldIds <- as.double(ff::as.ram(covariateRef$covariateId))
  newIds <- 1:nrow(covariateRef)

  if(!is.null(map)){
    ParallelLogger::logTrace('restricting to model variables...')
    ParallelLogger::logTrace(paste0('oldIds: ',length(map[,'oldIds'])))
    ParallelLogger::logTrace(paste0('newIds:', max(as.double(map[,'newIds']))))
    ind <- ffbase::ffmatch(x=covariateRef$covariateId, table=ff::as.ff(as.double(map[,'oldIds'])))
    ind <- ffbase::ffwhich(ind, !is.na(ind))
    covariateRef <- covariateRef[ind,]

    ind <- ffbase::ffmatch(x=covariates$covariateId, table=ff::as.ff(as.double(map[,'oldIds'])))
    ind <- ffbase::ffwhich(ind, !is.na(ind))
    covariates <- covariates[ind,]
  }
  if(is.null(map))
    map <- data.frame(oldIds=oldIds, newIds=newIds)

  return(list(covariates=covariates,
              covariateRef=covariateRef,
              map=map))
}

toSparseM <- function(plpData,
                      map,
                      timeId=NULL){
  cohorts = plpData$cohorts
  cov <- plpData$covariates #ff::clone(plpData$covariates)
  matrixDim <- c(max(cohorts$rowId), length(unique((cov$covariateId))))
  if(!is.null(timeId)){
    cov<-cov[cov$timeId==timeId,]
  }
  ParallelLogger::logDebug(paste0('covariateRef nrow: ', nrow(plpData$covariateRef)))

  covref <- plpData$covariateRef#ff::clone(plpData$covariateRef)

  cov<-ff::as.ram(cov)
  cov<-merge(cov,map, by.x="covariateId", by.y = "oldIds", all =FALSE)

  data <- Matrix::sparseMatrix(i=cov$rowId,
                               j=cov$newIds,
                               x=cov$covariateValue,
                               dims=matrixDim) # edit this to max(map$newIds)

  indexMat <- Matrix::sparseMatrix(i=cov$rowId,
                               j=cov$newIds,
                               x=T,
                               dims=matrixDim) # edit this to max(map$newIds)


  result <- list(data=data,
                 index=indexMat,
                 covariateRef=covref,
                 map=map)
  return(result)
}

