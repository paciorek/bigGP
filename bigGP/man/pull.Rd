\name{pull}
\alias{pull}
\title{
  Copy Object from Slave Processes to Master
}
\description{
  Copies all objects with a given name from the slave processes to the
  master process, returning a list with one element per slave
  process. Objects can be copied from lists, environments, and
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
pull(objName, objPos = ".GlobalEnv", tag = 1)
}
\arguments{
  \item{objName}{
    a variable name, given as a character string, giving the name of the
    object on the slave processes.
}
\item{objPos}{
  where to look for the object, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
  \item{tag}{
  non-negative integer, as in \code{mpi.send} and \code{mpi.recv}. Use \code{mpi.any.tag} for any tag flag.
}
}
\value{
  \code{pull} returns a list, with one element per slave process.
  }
\section{Warning}{
  Vectors and matrices that are part of the distributed linear algebra
  computations are broken up in very specific ways on the slave processes
  and often include padded elements. In general one should not use
  \code{pull} for retrieving such objects from the slave
  processes. Rather, use \code{collectVector}, \code{CollectTriangularMatrix}, etc.
}

\seealso{
  \code{push}
  \code{collectVector}
  \code{collectTriangularMatrix}
  \code{collectRectangularMatrix}
  \code{collectDiagonal}
}
\examples{
\dontrun{
bigGP.init(3)
a <- 3
push(a)
remoteLs()
pull('a')
}
}
\keyword{get}
