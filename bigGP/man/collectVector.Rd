\name{collectVector}
\alias{collectVector}
\title{
  Return a Distributed Vector to the Master Process
}
\description{
  \code{collectVector} retrieves a distributed vector from the slave
  processes, reconstructing in the correct order on the master process. 
  Objects can be copied from environments, lists, and
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
collectVector(objName, objPos = '.GlobalEnv', n, h = 1)
}
\arguments{
  \item{objName}{
    an object name, given as a character string, giving the name of the
    object on the slave processes.
}
\item{objPos}{
  where to look for the object, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
  \item{n}{
    a positive integer, the length of the vector.
  }
  \item{h}{
    a positive integer, the block replication factor, \eqn{h}, relevant for the vector.
}
}
\value{
  \code{collectVector} returns a vector of length, \eqn{n}.
  }

\seealso{
  \code{pull}
  \code{collectTriangularMatrix}
  \code{collectRectangularMatrix}
  \code{collectDiagonal}
  \code{distributeVector}
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

