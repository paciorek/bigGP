\name{remoteMultChol}
\alias{remoteMultChol}
\title{
  Distributed Multiplication of Lower Triangular Matrix and a Vector or Matrix
}
\description{
  \code{remoteMultChol} multiplies a distributed lower triangular matrix
  by a distributed vector or matrix. The objects can both be contained within environments or
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
remoteMultChol(cholName, inputName, outputName, cholPos = '.GlobalEnv',
inputPos = '.GlobalEnv', outputPos = '.GlobalEnv', n1, n2 = NULL, h1 = 1,
h2 = NULL)
}
\arguments{
  \item{cholName}{
    name of the lower triangular matrix, given as a character string, giving the name of the
    object on the slave processes.
}
\item{inputName}{
    name of the vector
    or matrix being multiplied by, given as a character string, giving
    the name of the object on the slave processes.
  }
\item{outputName}{
  the name to use for resulting vector or matrix product, on the slave processes.
}
\item{cholPos}{
    where to look for the lower triangular  matrix, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
\item{inputPos}{
    where to look for the input matrix or vector, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
\item{outputPos}{
  where to do the assignment of the output matrix or vector, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
}
 \item{n1}{
    a positive integer, the number of rows and columns of the lower triangular matrix.
  }
  \item{n2}{
    a positive integer, the number of columns of the vector or matrix
    being multiplied by. When equal to one, indicates multiplication by a vector.
  }
  \item{h1}{
    a positive integer, the block replication factor, \eqn{h}, relevant
    for the input matrix and used for the solution (either for a vector,
    or the rows of the solution for a matrix).
  }
  \item{h2}{
    a positive integer, the block replication factor, \eqn{h}, relevant
    for the columns of the input and output matrices when the lower
    triangular matrix is multiplied by a matrix.
  }
}
\details{
  Computes the distributed product using a blocked algorithm. 
  Note that the function works for any
  distributed lower triangular matrix, but \code{bigGP} currently only
  provides functionality for computing distributed Cholesky factors,
  hence the argument names \code{cholName} and \code{cholPos}.

  When multiplying by a vector that is stored as a vector, such as
  created by \code{distributeVector} or
  \code{remoteConstructRnormVector}, use \code{n2 = NULL}. When
  multiplying by a one-column matrix, use \code{n2 = 1}.

}
\references{
  Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and
  R.C.  Thomas. 2015. Parallelizing Gaussian Process Calculations in
  R. Journal of Statistical Software, 63(10),
  1-23. \doi{10.18637/jss.v063.i10}.
  
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
remoteConstructRnormVector('z', n = n, h = prob$h_n)
remoteMultChol(cholName = 'L', inputName = 'z', outputName = 'result',
cholPos = 'prob', n1 = n, h1 = prob$h_n)
realiz <- collectVector('result', n = n, h = prob$h_n)
}
}
}