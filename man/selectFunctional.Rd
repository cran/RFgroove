\name{selectFunctional}
\alias{selectFunctional}
\title{
    Grouped variable selection procedure for functional data
}
\description{
    Backward variable selection procedure for multivariate functional data which is a set of \eqn{p} functional covariates and \eqn{n} observations of dimension \eqn{N}.
}
\usage{
selectFunctional( FDlist, ydata, normalize=TRUE, 
                  dimensionReductionMethod=c("fpca", "wave"), 
                  nbasisInit, verbose=TRUE, ...)

}
\arguments{
  \item{FDlist}{
    A p-dimensional list containing the set of functional variables which are matrices of size \eqn{n \times N}.
  }
  \item{ydata}{
    The outcome data. Must be a factor for classification.
  }
  \item{normalize}{
    Should the functions be normalized ?
  }
  \item{dimensionReductionMethod}{
    The dimension reduction method, \sQuote{fpca} for Functional Principal Component Analysis or \sQuote{wave} for the multiple wavelet thresholding.
  }
  \item{nbasisInit}{
    The number of initial spline coefficients.
  }
  \item{verbose}{
    Should the details be printed.
  }
  \item{\dots}{
    further arguments passed to or from other methods.
  }
}
\value{
    An object of class fRFE which is a list with the following components:
    \item{nselected}{The number of selected functional variables ;}
    \item{selection}{The selected functional variables ;}
    \item{selectionIndexes}{The indexes of selected functional variables in the input data \sQuote{FDlist} ;}
    \item{error}{The prediction error computed in each iteration of the backward procedure ;}
    \item{typeRF}{The type of the forests, classification or regression ;}
    \item{ranking}{The final ranking of the functional variables ;}
    \item{rankingIndexes}{The final ranking indexes of the functional variables.}
}
\references{
    Gregorutti, B., Michel, B. and Saint Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis, Computational Statistics and Data Analysis 90, 15-35.
}
\author{
    Baptiste Gregorutti
}
\seealso{
  \code{\link{selectGroup}},\code{\link{varImpGroup}}
}
\examples{
  data(toyRegFD)
  varSel <- selectFunctional( toyRegFD$FDlist, toyRegFD$Y, normalize=FALSE, 
                              dimensionReductionMethod="fpca", nbasisInit=16, 
                              verbose=FALSE, ntree=10)
  summary(varSel)
  plot(varSel)
}
