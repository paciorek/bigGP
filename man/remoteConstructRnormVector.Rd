\name{remoteConstructRnormVector}
\alias{remoteConstructRnormVector}
\alias{remoteConstructRnormMatrix}
\title{
  Create Distributed Vector or Matrix of Random Normals
}
\description{
  \code{remoteConstructRnormVector} constructs a distributed vector of
  standard normal random variables, while
  \code{remoteConstructRnormMatrix} constructs a distributed matrix. The output object can both be contained within environments or
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
remoteConstructRnormVector(objName, objPos = ".GlobalEnv", n, h = 1)
 remoteConstructRnormMatrix(objName, objPos = ".GlobalEnv", n1, n2, h1 = 1, h2 = 1)
}
\arguments{
  \item{objName}{
      the name to use for the vector or matrix, on the slave processes.
}
\item{objPos}{
  where to do the assignment of the output matrix or vector, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
}
 \item{n}{
    a positive integer, the length of the vector
  }
  \item{h}{
    a positive integer, the block replication factor, \eqn{h}, relevant
    for the vector
  }
 
 \item{n1}{
    a positive integer, the number of rows of the matrix.
  }
  \item{n2}{
    a positive integer, the number of columns of the matrix.
  }
  \item{h1}{
    a positive integer, the block replication factor, \eqn{h}, relevant
    for the rows of the matrix.
  }
  \item{h2}{
    a positive integer, the block replication factor, \eqn{h}, relevant
    for the columns of the matrix.
  }
}
\section{Warning}{
  Note that a vector and a one-column matrix are stored differently,
    with padded columns included for the matrix. For
    other distributed computation functions, providing the argument \code{n2 = NULL}
    indicates the input is a vector, while \code{n2 = 1} indicates a
    one-column matrix. 
}

\seealso{
  \code{\link{bigGP}}
}
\examples{
\dontrun{
if(require(fields)) {
  SN2011fe <- SN2011fe_subset
  SN2011fe_newdata <- SN2011fe_newdata_subset
  SN2011fe_mle <- SN2011fe_mle_subset
  nProc <- 3
n <- nrow(SN2011fe)
m <- nrow(SN2011fe_newdata)
nu <- 2
inputs <- c(as.list(SN2011fe), as.list(SN2011fe_newdata), nu = nu)
prob <- krigeProblem$new("prob", numProcesses = nProc, n = n, m = m,
predMeanFunction = SN2011fe_predmeanfunc, crossCovFunction = SN2011fe_crosscovfunc,  
predCovFunction = SN2011fe_predcovfunc, meanFunction = SN2011fe_meanfunc, 
covFunction = SN2011fe_covfunc,  inputs = inputs, params = SN2011fe_mle$par, 
data = SN2011fe$flux, packages = c("fields"))
remoteCalcChol(matName = 'C', cholName = 'L', matPos = 'prob',
  cholPos = 'prob', n = n, h = prob$h_n)
remoteConstructRnormVector('z', n = n, h = prob$h_n)
remoteMultChol(cholName = 'L', inputName = 'z', outputName = 'result',
cholPos = 'prob', n1 = n, h1 = prob$h_n)
realiz <- collectVector('result', n = n, h = prob$h_n)

r = 10
remoteConstructRnormMatrix('z2', n1 = n, n2 = r, h1 = prob$h_n, h2 = 1)
remoteMultChol(cholName = 'L', inputName = 'z2', outputName = 'result2',
cholPos = 'prob', n1 = n, n2 = r, h1 = prob$h_n, h2 = 1)
realiz2 <- collectRectangularMatrix('result2', n1 = prob$n, n2 = r, h1
= prob$h_n, h2 = 1)
}
}
}
