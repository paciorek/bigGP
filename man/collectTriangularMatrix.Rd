\name{collectTriangularMatrix}
\alias{collectTriangularMatrix}
\title{
  Return a Distributed Symmetric or Triangular Matrix to the Master Process
}
\description{
  \code{collectTriangularMatrix} retrieves a distributed symmetric or
  triangular matrix from the slave
  processes, reconstructing the blocks correctly on the master process. 
  Objects can be copied from environments, lists, and
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
collectTriangularMatrix(objName, objPos = '.GlobalEnv', n, h = 1)
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
    a positive integer, the number of rows (and columns) of the matrix.
  }
  \item{h}{
    a positive integer, the block replication factor, \eqn{h}, relevant for the matrix.
}
}
\value{
  \code{collectTriangularMatrix} returns a matrix of dimension, \eqn{n
    \times n}. Note that for lower triangular matrices, the upper
  triangle is non-zero and is filled with the transpose of the lower
  triangle, and vice versa for upper triangular matrices.
  }

\seealso{
  \code{pull}
  \code{collectVector}
  \code{collectRectangularMatrix}
  \code{collectDiagonal}
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
  meanFunction = SN2011fe_meanfunc, covFunction = SN2011fe_covfunc, inputs = inputs,
  params = SN2011fe_mle$par, data = SN2011fe_subset$flux, packages =
  c("fields"))
# calculate log density, primarily so Cholesky gets calculated
prob$calcLogDens()
C <- collectTriangularMatrix('C', "prob", n = n, h = 1)
L <- collectTriangularMatrix('L', "prob", n = n, h = 1)
C[1:5, 1:5]
L[1:5, 1:5]
}
}
}
