\name{varImpGroup}
\alias{varImpGroup}
\title{
    A grouped variable importance with Random Forests
}
\description{
    A permutation variable importance for groups of variables with Random Forests.
}
\usage{
varImpGroup(object, xdata, ngroups = length(nvarGroup), nvarGroup,
            idxGroup, groupsNames = names(nvarGroup), 
            normalize = (length(unique(nvarGroup)) != 1))
}
\arguments{
  \item{object}{
    A randomForest object.
}
  \item{xdata}{
    The input data.
}
  \item{ngroups}{
    The number of groups.
}
  \item{nvarGroup}{
    The vector of the number of variables in each group.
}
  \item{idxGroup}{
    A list of size \sQuote{ngroups} containing the indexes of each group starting from 0.
}
  \item{groupsNames}{
    The group names.
}
  \item{normalize}{
    Should the normalized grouped importance measure be computed.
}
}
\value{
    An object of class \sQuote{importance} which is a vector of the importance for each group.
}
\references{
    Gregorutti, B., Michel, B. and Saint Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis, Computational Statistics and Data Analysis 90, 15-35.
}
\author{
    Baptiste Gregorutti
}
\seealso{
  \code{\link{selectGroup}},\code{\link{selectLevel}},\code{\link{selectFunctional}},\code{\link{plot.importance}}
}
\examples{
  data(toyClassif)
  attach(toyClassif)

  rf <- randomForest(x=X,y=Y,keep.forest=TRUE, keep.inbag=TRUE, ntree=500)
  ngroups <- 3
  nvarGroup <- c(4,3,6)
  idxGroup <- list(c(0,1,2,5), c(2,4,5), c(0,1,5,6,7,8))
  grImp <- varImpGroup(rf, X, ngroups, nvarGroup, idxGroup, NULL, normalize=FALSE )
  cat("Group importance\n", grImp, "\n")

  detach(toyClassif)
}
