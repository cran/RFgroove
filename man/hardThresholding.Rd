\name{hardThresholding}
\alias{hardThresholding}
\title{
    Wavelets hard-thresholding rule for independents processes
}
\description{
This function projects \eqn{n} indepedent processes on a common wavelet basis and shrinks to zero the \eqn{n} coefficients whose \eqn{\ell_2}-norm is lower than a threshold.
}
\usage{
hardThresholding(xdata, delta, verbose = FALSE, varName = NULL, wavFilter="s8")
}

\arguments{

  \item{xdata}{
    The matrix of \eqn{n} independent curves of dimension \eqn{N=2^J}, where \eqn{J} is the number of maximum wavelet level.
  }
  \item{delta}{
    The desired threshold. If missing, an automatic threshold is computed.
  }
  \item{verbose}{
    Should the details be printed.
  }
  \item{varName}{
    The name of the current functional variable.
  }
  \item{wavFilter}{
     A character string denoting the filter type. Supported types include:

          EXTREMAL PHASE (daublet): \sQuote{haar}, \sQuote{d2}, \sQuote{d4}, \sQuote{d6}, \sQuote{d8}, \sQuote{d10}, \sQuote{d12}, \sQuote{d14}, \sQuote{d16}, \sQuote{d18}, \sQuote{d20}

          LEAST ASYMMETRIC (symmlet): \sQuote{s2}, \sQuote{s4}, \sQuote{s6}, \sQuote{s8}, \sQuote{s10}, \sQuote{s12}, \sQuote{s14}, \sQuote{s16}, \sQuote{s18}, \sQuote{s20}

          BEST LOCALIZED: \sQuote{l2}, \sQuote{l4}, \sQuote{l6}, \sQuote{l14}, \sQuote{l18}, \sQuote{l20}

          COIFLET: \sQuote{c6}, \sQuote{c12}, \sQuote{c18}, \sQuote{c24}, \sQuote{c30}

          Default: \sQuote{s8}.
  }
}

\value{
    A list with two components
    \item{mht.names}{The names of the common wavelet basis after thresholding the coefficients.}
    \item{estimatedDesign}{The new design matrix after thresholding.}
}
\references{
    Gregorutti, B., Michel, B. and Saint Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis, Computational Statistics and Data Analysis 90, 15-35.
}
\author{
    Baptiste Gregorutti
}

\seealso{
    \code{\link{fpca}}
}

\examples{
  data(toyRegFD)
  x <- toyRegFD$FDlist[[1]]
  newDesignMatrix <- hardThresholding(xdata=x, verbose=TRUE)
}
