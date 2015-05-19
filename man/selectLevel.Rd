\name{selectLevel}
\alias{selectLevel}
\title{
    Wavelet levels selection procedure
}
\description{
  A grouped backward variable selection procedure for selecting the most significant wavelet levels of a functional variable. The groups are the wavelet coefficients belonging to the same frequency level.
}
\usage{
selectLevel(design, ydata, typeRF = ifelse(is.factor(ydata), "classif", "reg"), 
            verbose = TRUE, ntree = 500, ...)
}
\arguments{
  \item{design}{
    The design matrix of a functional variable.
}
  \item{ydata}{
    The outcome data. Must be a factor for classification.
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
    \item{nselected}{The number of selected wavelet levels.}
    \item{selection}{The selected wavelet levels.}
    \item{selectionIndexes}{The indexes of selected wavelet levels in the input matrix \sQuote{design}.}
    \item{error}{The prediction error computed in each iteration of the backward procedure.}
    \item{typeRF}{The type of the forests, classification or regression.}
    \item{ranking}{The final ranking of the wavelet levels.}
    \item{rankingIndexes}{The final ranking indexes of the wavelet levels.}
}
\references{
    Gregorutti, B., Michel, B. and Saint Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis, Computational Statistics and Data Analysis 90, 15-35.
}
\author{
    Baptiste Gregorutti
}
\seealso{
\code{\link{selectGroup}},\code{\link{selectFunctional}},\code{\link{varImpGroup}}
}

\examples{
  data(toyRegFD)
  x <- toyRegFD$FDlist[[1]]
  y <- toyRegFD$Y

  design <- projectWavelet(xdata=x)
  summary(levSel <- selectLevel(design, y, ntree=100, verbose=TRUE))
  plot(levSel)

}

