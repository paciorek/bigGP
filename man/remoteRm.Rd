\name{remoteRm}
\alias{remoteRm}
\title{
Remote Remove Objects
}
\description{
\code{remoteRm} is used to remove objects from the global environment on the
slave processes.
}
\usage{
remoteRm(..., list = character())
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{\dots}{
    the objects to be removed, as names (unquoted) or character
    strings (quoted). 
}
  \item{list}{
    a character vector naming objects to be removed
}
}
\details{
  This is a distributed version of \code{rm}. It removes the named
  objects from all of the slave processes. Unlike \code{rm},
  \code{remoteRm} is currently not enabled to remove objects from other
  than the global environment. Note that unless \code{options(warn = 2)}
  is set on the slave processes, no warning is reported if one tries to
  remove objects that do not exist.
}
\seealso{
  \code{remoteLs}
}
\examples{
\dontrun{
bigGP.init(3)
a <- 3
b <- 7
push(a); push(b)
remoteLs()
remoteRm(a)
remoteLs()
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{rm}
\keyword{remove}
