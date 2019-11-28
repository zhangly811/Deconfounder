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

# Format and check code ---------------------------------------------------
OhdsiRTools::formatRFolder()
OhdsiRTools::checkUsagePackage("MvDeconfounder")
OhdsiRTools::updateCopyrightYearFolder()

# Create manual -----------------------------------------------------------
shell("rm extras/MvDeconfounder.pdf")
shell("R CMD Rd2pdf ./ --output=extras/MvDeconfounder.pdf")

# Create vignettes ---------------------------------------------------------
# rmarkdown::render("vignettes/UsingMvDeconfounders.Rmd",
#                   output_file = "../inst/doc/UsingSkeletonPackage.pdf",
#                   rmarkdown::pdf_document(latex_engine = "pdflatex",
#                                           toc = TRUE,
#                                           number_sections = TRUE))

# Store environment in which the study was executed -----------------------
OhdsiRTools::insertEnvironmentSnapshotInPackage("MvDeconfounder")

# Format and check code ---------------------------------------------------
OhdsiRTools::formatRFolder()
OhdsiRTools::checkUsagePackage("MvDeconfounder")
OhdsiRTools::updateCopyrightYearFolder()

#usethis::create_package(getwd())
roxygen2::roxygenize()
