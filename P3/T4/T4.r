library("NoiseFiltersR")

MyData <- read.csv(file="c:/../nepal_earthquake_tra.csv", header=TRUE, sep=",")

data(iris)
# We fix a seed since there exists a random partition for the ensemble
set.seed(1)
out <- CVCF(Species~.-Sepal.Width, data = iris)
print(out)
identical(out$cleanData, iris[setdiff(1:nrow(iris),out$remIdx),])

