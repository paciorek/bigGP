\name{localAssign}
\alias{localAssign}
\title{
  Assign a New Name to an Object on Slave Process
}
\description{
  \code{localAssign} is an internal auxiliary function used
  to assign a new name to an object in an environment on a slave
  process. The function needs to be executed on the slave processes.
}
\usage{
localAssign(nameToAssign, currentName, objPos = ".GlobalEnv")
}
\arguments{
  \item{nameToAssign}{
    a variable name, given as a character string, giving the new name
    for the object.
  }
  \item{currentName}{
    a variable name, given as a character string, giving the current
    name for the object.
  }
  \item{objPos}{
    where to do the assignment, given as a character string (unlike
    \code{assign}). This can indicate an environment or a
    ReferenceClass object.
  }
}
\details{
  This function is primarily for internal use, but might be useful for
  developers extending the package for use cases other than the
  kriging use case contained in \code{krigeProblem} ReferenceClass.
}
\examples{
\dontrun{
bigGP.init(3)
mpi.bcast.cmd(e <- new.env())
mpi.bcast.cmd(a <- 7)
mpi.remote.exec(localAssign, "x", "a", objPos = "e")
mpi.remote.exec(e$x, ret = TRUE)
}
}
