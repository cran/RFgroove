\name{toyRegFD}
\alias{toyRegFD}
\docType{data}
\title{
	A functional dataset for regression
}
\description{
	A set of 20 functional variables for regression. Only the 8 first variables are predictive.
}
\usage{data(toyRegFD)}
\format{
	A list of two components:
	\itemize{
		\item FDlist: a list containing the \eqn{p} functional variables, i.e. \eqn{p} matrices of size \eqn{n \times N} with \eqn{n} the number of \eqn{N}-dimensional curves ;
		\item Y: the outcome drawn from the Gaussian distribution.
	}
}

\references{
    Gregorutti, B., Michel, B. and Saint Pierre, P. (2014). Grouped variable importance with random forests and application to multivariate functional data analysis. arXiv:1411.4170
}
