\name{remoteCrossProdMatVec}
\alias{remoteCrossProdMatVec}
\title{
  Distributed Crossproduct of a Rectangular Matrix and a Vector
}
\description{
  \code{remoteCrossProdMatVec} multiplies the transpose of a distributed rectangular matrix
  by a distributed vector or matrix. The objects can both be contained within environments or
  ReferenceClass objects as well as the global environment on the slave processes.
}
\usage{
remoteCrossProdMatVec(matName, inputName, outputName, matPos = '.GlobalEnv',
inputPos = '.GlobalEnv', outputPos = '.GlobalEnv', n1, n2, h1 = 1, h2 = 1)
}
\arguments{
  \item{matName}{
    name of the rectangular matrix, given as a character string, giving the name of the
    object on the slave processes.
}
\item{inputName}{
    name of the vector being multiplied by, given as a character string, giving
    the name of the object on the slave processes.
  }
\item{outputName}{
  the name to use for resulting vector, on the slave processes.
}
\item{matPos}{
    where to look for the rectangular matrix, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
\item{inputPos}{
    where to look for the input vector, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
\item{outputPos}{
  where to do the assignment of the output vector, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
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
\details{
  Computes the distributed product using a blocked algorithm, resulting
  in a distributed vector.

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
predMeanFunction = SN2011fe_predmeanfunc, crossCovFunction =
SN2011fe_crosscovfunc,  predCovFunction = SN2011fe_predcovfunc,
meanFunction = SN2011fe_meanfunc, covFunction = SN2011fe_covfunc,
inputs = inputs, params = SN2011fe_mle$par, data = SN2011fe$flux,
packages = c("fields"))

remoteCalcChol(matName = "C", cholName = "L", matPos = "prob",
  cholPos = "prob", n = n, h = prob$h_n)
remoteCalc("data", "mean", `-`, "tmp1", input1Pos = "prob", input2Pos = "prob")
remoteForwardsolve(cholName = "L", inputName = "tmp1", outputName = "tmp2", 
cholPos = "prob", n1 = n, h1 = prob$h_n)
remoteBacksolve(cholName = "L", inputName = "tmp2", outputName = "tmp3",
 cholPos = "prob", n1 = n, h1 = prob$h_n)
prob$remoteConstructCov(obs = FALSE, pred = FALSE, cross = TRUE, verbose = TRUE)
# we now have a rectangular cross-covariance matrix named 'crossC'
remoteCrossProdMatVec(matName = "crossC", inputName = "tmp3", outputName = "result", 
matPos = "prob", n1 = n, n2 = m, h1 = prob$h_n, h2 = prob$h_m)
 
result <- collectVector("result", n = n, h = prob$h_n)
}
}
}