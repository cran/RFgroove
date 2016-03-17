\name{plot.importance}
\alias{plot.importance}
\title{
    Plot function for object of class importance
}
\description{
    Plot the values of the grouped variable importance which is the result of the function \code{\link{varImpGroup}}.
}
\usage{
\method{plot}{importance}(x, type, horiz, ...)
}
\arguments{
  \item{x}{
    An object of class importance.
}
  \item{type}{
    A string denoting the plot type. Supported types include \sQuote{dotchart} or \sQuote{bar}.
}
  \item{horiz}{
    For barplot only. Should the bars be displayed horizontally (default: TRUE).
}
  \item{\dots}{
    optional graphical parameters.
}
}



\author{
    Baptiste Gregorutti
}
