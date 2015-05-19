\name{projectWavelet}
\alias{projectWavelet}
\title{
    Projection in a wavelet basis
}

\description{
  This function projects a set of curves on a wavelet basis.
}

\usage{
  projectWavelet(xdata, wavFilter="s8")
}

\arguments{
  \item{xdata}{
    A set of \eqn{n} curves of dimension \eqn{N}.
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
  A matrix of size \eqn{n \times N} containing the wavelet coefficients.
}
\references{
    Percival, D. B. and Walden, A. T., Wavelet Methods for Time Series Analysis (2000), Cambridge University Press
}
\author{
    Baptiste Gregorutti
}
\seealso{
\code{\link{hardThresholding}},\code{\link{fpca}}
}

\examples{
  data(toyRegFD)
  x <- toyRegFD$FDlist[[1]]

  design <- projectWavelet(xdata=x)
  print(colnames(design))
}
