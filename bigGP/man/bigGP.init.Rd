\name{bigGP.init}
\alias{bigGP.init}
\title{
  Initialize bigGP package
}
\description{
  \code{bigGP.init} initializes the bigGP and must be called before
  using any bigGP functionality. It starts slave processes, if not
  already started, and sets up the necessary objects containing
  information for distributing calculations correctly. It also
  initializes the RNG on the slave processes.
}
\usage{
bigGP.init(P = NULL, parallelRNGpkg = "rlecuyer", seed = 0)
}
\arguments{
  \item{P}{
    Number of slave processes. Should be equal to D(D+1)/2 for some
    integer D. If \code{NULL}, will be taken to be
    \code{mpi.comm.size()-1}, where the additional process is the
    master. 
}
  \item{parallelRNGpkg}{
    Package to be used for random number generation (RNG). At the moment
    this should be one of \pkg{relecuyer} or \pkg{rsprng}, and
    these packages must be installed.
}
  \item{seed}{
    Seed to be used for initializing the parallel RNG.
  }
  }
\details{
  The initialization includes starting the slave processes, calculating
  the partition factor, \eqn{D}, and providing the slave processes with
  unique identifying information. This information is stored in the
  \code{.bigGP} object on each slave process.

  Note that in general, the number of processes (number of slave
  processes, \code{P}, plus one for the master) should not exceed the number of
  physical cores on the machine(s) available.

  \code{bigGP.init} also sets up random number generation on the slaves,
  using \code{parallelRNGpkg} when specified, and setting appropriate
  seeds on each slave process. 
}
\examples{
\dontrun{
bigGP.init(3, seed = 1)
}
}