\name{fpca}
\alias{fpca}

\title{
Functional Principal Component Analysis
}
\description{
Compute the functional PCA from a set of curves.
}
\usage{
fpca( x, nbasisInit, propVar = 0.9, reconstruct = FALSE, 
      varName = NULL, verbose = FALSE)
}

\arguments{
  \item{x}{
    The set of curves.
}
  \item{nbasisInit}{
    The number of initial spline coefficients.
}
  \item{propVar}{
    The proportion of explained variance.
}
  \item{reconstruct}{
    Should the data be reconstruct after dimension reduction ?
}
  \item{varName}{
    The name of the current functional variable.
}
  \item{verbose}{
    Should the details be printed.
}
}

\details{
  The Functional PCA is performed in two steps. First we express each discretized curves as a linear combination of \sQuote{nbasisInit} spline basis functions. Then a multivariate PCA is computed on the spline coefficients. The final number of principal components is chosen such that the proportion of explained variance is larger than \sQuote{propVar}.
}

\value{
    A list with two components:
    \item{design}{The matrix of the principal components ;}
    \item{smoothData}{The reconstructed data if \sQuote{reconstruct == TRUE}.}
}

\references{
    Ramsay, J. O., and Silverman, B. W. (2006), Functional Data Analysis, 2nd ed., Springer, New York.
}

\author{
    Baptiste Gregorutti
}

\seealso{
 \code{\link{hardThresholding}}
}

\examples{
  data(toyRegFD)
  x <- toyRegFD$FDlist[[1]]
  PCs <- fpca(x=x, nbasisInit=32, propVar=.9, reconstruct=TRUE)
  plot(x[1,])
  lines(PCs$smoothData[1,], col=2)
}
