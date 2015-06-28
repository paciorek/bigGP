\name{collectDiagonal}
\alias{collectDiagonal}
\title{
  Return the Diagonal of a Distributed Square Matrix to the Master Process
}
\description{
  \code{collectDiagonal} retrieves the diagonal elements of a
  distributed square matrix from the slave
  processes in the proper order. Values can be copied from objects in environments, lists, and
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
collectDiagonal(objName, objPos = '.GlobalEnv', n, h = 1)
}
\arguments{
  \item{objName}{
    an object name, given as a character string, giving the name of the
    matrix on the slave processes.
}
\item{objPos}{
  where to look for the matrix, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
  \item{n}{
    a positive integer, the number of rows (and columns) of the matrix.
  }
  \item{h}{
    a positive integer, the block replication factor, \eqn{h}, relevant for the matrix.
}
}
\value{
  \code{collectDiagonal} returns a vector of length \eqn{n}.
  }

\seealso{
  \code{pull}
  \code{collectVector}
  \code{collectTriangularMatrix}
  \code{collectRectangularMatrix}
  \code{distributeVector}
}
\examples{
\dontrun{
if(require(fields)) {
nProc <- 3
n <- nrow(SN2011fe_subset)
inputs <- c(as.list(SN2011fe_subset), as.list(SN2011fe_newdata_subset),
  nu =2)
# initialize the problem
prob <- krigeProblem$new("prob", h_n = 1, numProcesses = nProc, n = n,
  meanFunction = SN2011fe_meanfunc, covFunction = SN2011fe_covfunc,
  inputs = inputs, params = SN2011fe_mle$par,
  data = SN2011fe_subset$flux, packages = c("fields"))
# calculate log density, primarily so Cholesky gets calculated
prob$calcLogDens()
diagC <- collectDiagonal('C', "prob", n = n, h = 1)
diagL <- collectDiagonal('L', "prob", n = n, h = 1)
diagC[1:5]
diagL[1:5]
}
}
}

