\name{remoteCalcChol}
\alias{remoteCalcChol}
\title{
  Calculate Distributed Cholesky Decomposition
}
\description{
  \code{remoteCalcChol} calculates a distributed Cholesky decomposition
  from a distributed positive definite matrix. The Cholesky factor and
  the original matrix can both be contained within environments and 
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
remoteCalcChol(matName, cholName, matPos = '.GlobalEnv', cholPos = '.GlobalEnv', n, h = 1)
}
\arguments{
  \item{matName}{
    name of the input (positive definite) matrix, given as a character string, giving the name of the
    object on the slave processes.
}
\item{cholName}{
    an name, given as a character string, giving the name to be
    used for the Cholesky factor matrix on the slave processes.
  }
  \item{matPos}{
    where to look for the input matrix, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
\item{cholPos}{
    where to do the assignment of the Cholesky factor matrix, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
}
  \item{n}{
    a positive integer, the number of rows and columns of the input matrix.
  }
  \item{h}{
    a positive integer, the block replication factor, \eqn{h}, relevant
    for the input matrix and used for the Cholesky factor as well.
}
}
\details{
  Computes the distributed Cholesky decomposition using a blocked
  algorithm similar to that in ScaLapack. When \eqn{h} is 1, the number
  of blocks, representing the
  lower triangle of the original matrix and of the Cholesky factor, is
  equal to the number of processes. For larger values of \eqn{h}, there
  are multiple blocks assigned to each process. 
}
\references{

  Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and
  R.C.  Thomas. 2015. Parallelizing Gaussian Process Calculations in
  R. Journal of Statistical Software, 63(10), 1-23. \doi{10.18637/jss.v063.i10}.

  Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and R.C.  Thomas. 2013. Parallelizing Gaussian Process Calculations in R. arXiv:1305.4886. \url{https://arxiv.org/abs/1305.4886}.
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
L <- collectTriangularMatrix('L', objPos = 'prob', n = n, h = prob$h_n)
}
}
}