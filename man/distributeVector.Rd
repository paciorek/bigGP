\name{distributeVector}
\alias{distributeVector}
\title{
  Distribute a Vector to the Slave Processes
}
\description{
  \code{distributeVector} distributes a vector to the slave processes,
  breaking into the appropriate pieces, in some cases with padded elements.
  Objects can be distributed to environments and
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
distributeVector(obj, objName = deparse(substitute(obj)), objPos = '.GlobalEnv', n, h = 1)
}
\arguments{
  \item{obj}{
    object on master process to be copied, given either as the name of
    an object or as a character.
}
  \item{objName}{
    an object name, given as a character string, giving the name to be
    used for the object on the slave processes. If not provided, will be
    the same as the name of \code{obj} in the calling environment.
}
\item{objPos}{
  where to do the assignment, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
}
  \item{n}{
    a positive integer, the length of the vector.
  }
  \item{h}{
    a positive integer, the block replication factor, \eqn{h}, to be used when distributing the vector.
}
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
n <- 3000
x <- rnorm(n)
distributeVector(x, 'tmp', n = n)
y <- collectVector('tmp', n = n)
identical(x, y)
}
}
