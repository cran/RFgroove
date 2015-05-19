\name{selectGroup}
\alias{selectGroup}
\title{
    Grouped variable selection procedure
}
\description{
    A grouped backward variable selection procedure.
}
\usage{
selectGroup(design, ydata, varNames, nvarGroup, 
            typeRF = ifelse(is.factor(ydata), "classif", "reg"), 
            verbose = TRUE, ntree = 500, ...)
}
\arguments{
  \item{design}{
    The design matrix.
}
  \item{ydata}{
    The outcome data. Must be a factor for classification.
}
  \item{varNames}{
    The vector of the group names.
}
  \item{nvarGroup}{
    The vector of the number of variables in each group.
}
  \item{typeRF}{
    The type of forest we want to construct, \sQuote{classif} for classification or \sQuote{reg} for regression.
}
  \item{verbose}{
    Should the details be printed.
}
  \item{ntree}{
    The number of trees in the forests (default: 500).
}
  \item{\dots}{
    optional parameters to be passed to the \sQuote{varImpGroup} function.
}
}

\value{
    An object of class fRFE which is a list with the following components:
    \item{nselected}{The number of selected groups ;}
    \item{selection}{The selected groups ;}
    \item{selectionIndexes}{The indexes of selected groups in the input matrix \sQuote{design} ;}
    \item{error}{The prediction error computed in each iteration of the backward procedure ;}
    \item{typeRF}{The type of the forests, classification or regression ;}
    \item{ranking}{The final ranking of the groups ;}
    \item{rankingIndexes}{The final ranking indexes of the groups.}
}

\references{
    Gregorutti, B., Michel, B. and Saint Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis, Computational Statistics and Data Analysis 90, 15-35.
}
\author{
    Baptiste Gregorutti
}



\seealso{
  \code{\link{selectLevel}},\code{\link{selectFunctional}},\code{\link{varImpGroup}}
}

\examples{
  cat("\n\nClassification\n")
  data(toyClassif)
  attach(toyClassif)
  
  cat("Case 1\n")
  nvarGroup <- c(2,1,6); names(nvarGroup) <- paste("G", 1:length(nvarGroup), sep="")
  summary(varSel <- selectGroup(design=X, ydata=Y, nvarGroup=nvarGroup, 
                                verbose=TRUE, ntree=500, normalize=FALSE))
  plot(varSel)

  cat("Case 2\n")
  nvarGroup <- rep(1,9); names(nvarGroup) <- paste("G", 1:length(nvarGroup), sep="")
  summary(varSel <- selectGroup(design=X, ydata=Y, nvarGroup=nvarGroup, 
                                verbose=TRUE, ntree=500, normalize=FALSE))
  plot(varSel)

  detach(toyClassif)

}
