\name{push}
\alias{push}
\title{
Copy Object from Master to Slave Processes
}
\description{
  Copies an objects from the master process to all slave processes. Objects can be copied to environments and
  ReferenceClass objects as well as the global environment on the slaves.
}
\usage{
push(.tmp, objName = deparse(substitute(.tmp)), objPos = ".GlobalEnv")
}
\arguments{
  \item{.tmp}{
    object on master process to be copied, given either as the name of
    an object or as a character.
}
\item{objName}{
  the name to use for the object on the slave processes.
}
  \item{objPos}{
  where to do the assignment, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
}
}
\section{Warning}{
  Vectors that are part of the distributed linear algebra
  computations are broken up in very specific ways on the slave processes
  and often include padded elements. In general one should not use
  \code{push} to distribute such objects as \code{push} would distribute
  the entire vector to each slave process. Rather, use \code{distributeVector}.
}
\seealso{
  \code{pull}
  \code{distributeVector}
}
\examples{
\dontrun{
bigGP.init(3)
a <- 3
push(a)
remoteLs()
}
}
\keyword{assign}