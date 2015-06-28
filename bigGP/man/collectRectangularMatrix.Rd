\name{collectRectangularMatrix}
\alias{collectRectangularMatrix}
\title{
  Return a Distributed Rectangular Matrix to the Master Process
}
\description{
  \code{collectRectangularMatrix} retrieves a distributed rectangular matrix from the slave
  processes, reconstructing the blocks correctly on the master process. 
  Objects can be copied from environments, lists, and
  ReferenceClass objects as well as the global environment on the slave
  processes. WARNING: do not use with a distributed symmetric square matrix; instead
  use \code{collectTriangularMatrix}.
}
\usage{
collectRectangularMatrix(objName, objPos = '.GlobalEnv', n1, n2, h1 = 1, h2 = 1)
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
  \item{n1}{
    a positive integer, the number of rows of the matrix.
  }
  \item{n2}{
    a positive integer, the number of columns of the matrix.
  }
  \item{h1}{
    a positive integer, the block replication factor relevant for the rows of
    the matrix.
}
  \item{h2}{
    a positive integer, the block replication factor relevant for the columns of
    the matrix.
}
}
\value{
  \code{collectRectangularMatrix} returns a matrix of dimension, \eqn{n1
    \times n2}.
  }

\seealso{
  \code{pull}
  \code{collectVector}
  \code{collectTriangularMatrix}
  \code{collectDiagonal}
  \code{distributeVector}
}
\examples{
\dontrun{
if(require(fields)) {
nProc <- 3
n <- nrow(SN2011fe_subset)
m <- nrow(SN2011fe_newdata_subset)
inputs <- c(as.list(SN2011fe_subset), as.list(SN2011fe_newdata_subset),
  nu =2)
# initialize the problem
prob <- krigeProblem$new("prob", h_n = 1, h_m = 1, numProcesses =
  nProc, n = n, m = m,
  meanFunction = SN2011fe_meanfunc, predMeanFunction = SN2011fe_predmeanfunc, 
  covFunction = SN2011fe_covfunc, crossCovFunction = SN2011fe_crosscovfunc,
predCovFunction = SN2011fe_predcovfunc, params = SN2011fe_mle$par,
  inputs = inputs, data = SN2011fe_subset$flux, packages = c("fields"))
# do predictions, primarily so cross-covariance gets calculated
pred <- prob$predict(ret = TRUE, verbose = TRUE)

crossC <- collectRectangularMatrix('crossC', "prob", n1 = n, n2 = m,
h1 = 1, h2 = 1)
crossC[1:5, 1:5]
}
}
}
