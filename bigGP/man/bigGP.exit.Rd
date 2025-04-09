\name{bigGP.exit}
\alias{bigGP.exit}
\alias{bigGP.quit}
\title{
  Exit bigGP Environment
}
\description{
\code{bigGP.exit} terminates the package's execution environment and
detaches the package. After that, you can still work in R.

\code{bigGP.quit} terminates the package's execution environment and quits R.
}
\usage{
bigGP.exit()
bigGP.quit(save = "no")
}
\arguments{
  \item{save}{
    the same argument as \code{quit}, but defaulting to "no".}
}
\details{
  These functions should be used to safely leave the \pkg{"bigGP"}
  execution context, specifically MPI, when R is started via MPI such as
  by calling mpirun or analogous executables. They close the slave
  processes and then invoke either \code{mpi.exit} or \code{mpi.quit}.
  
   If leaving R altogether, one simply uses \code{bigGP.quit}. 
}
\seealso{
  \code{\link[Rmpi]{mpi.exit}}
  \code{\link[Rmpi]{mpi.quit}}
}
\keyword{utilities}
