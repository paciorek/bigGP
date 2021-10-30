\name{bigGP}
\alias{bigGP}
\alias{bigGP-package}
\title{
  Package for Calculations with Big Gaussian Processes
}
\description{
  \code{bigGP} is a collection of functions for doing distributed
  calculations in the context of various kinds of Gaussian process
  models, combined with a ReferenceClass, \code{krigeProblem},
  for doing kriging calculations
  based on maximum likelihood estimation. 
}
\details{
  Full details on doing distributed kriging can be found in the help
  page for \code{\link{krigeProblem}}. For general calculations with
  distributed vectors and matrices, including extending the package for
  additional use cases beyond standard kriging, one first needs to
  create the needed vectors and matrices in a distributed fashion on the
  slave processes.  To do this, the indices associated with relevant vectors and matrices  need to be found for each slave process; see
  \code{\link{localGetVectorIndices}}. Then these indices need to be
  used by user-created functions to create the pieces of the vectors and
  matrices residing on each slave process; see
  \code{\link{localKrigeProblemConstructMean}} and
  \code{\link{localKrigeProblemConstructMean}} for examples. Following
  this, one can use the various functions for distributed linear
  algebra.

  The functions provided for distributed linear algebra are:
  \describe{
  \item{\code{remoteCalcChol}: }{calculates the Cholesky decomposition of
  a (numerically) positive definite matrix, \eqn{C = L L^{\top}}. Matrices that are not numerically positive.}
  definite will cause failure as pivoting is not implemented.
  \item{\code{remoteForwardsolve}: }{does a forwardsolve using an
  already-calculated Cholesky factor into a vector or matrix, \eqn{L^{-1}Z}.}
  \item{\code{remoteBacksolve}: }{does a backsolve using an
  already-calculated Cholesky factor into a vector or matrix, \eqn{L^{-\top}Z}.}
  \item{\code{remoteMultChol}: }{multiplies and an
  already-calculated Cholesky factor by a vector or matrix, \eqn{LZ}.}
  \item{\code{remoteCrossProdMatVec}: }{multiplies the transpose of a
  matrix by a vector, \eqn{X^{\top}z}.}
  \item{\code{remoteCrossProdMatSelf}: }{does the crossproduct of a matrix, \eqn{X^{\top}X}.}
  \item{\code{remoteCrossProdMatSelfDiag}: }{finds the diagonal of the crossproduct of a matrix, \eqn{\mbox{diag}(X^{\top}X)}.}
  \item{\code{remoteConstructRnormVector}: }{generates a vector of
  random standard normal variables.}
  \item{\code{remoteConstructRnormMatrix}: }{generates a matrix of
  random standard normal variables.}
   \item{\code{remoteCalc}: }{does arbitrary calculations on one or two
     inputs.}
   }
 }
 \section{Warnings}{
    Note that the block replication factor, \eqn{h}, needs to be
    consistent in any given calculation. So if one is doing a
    forwardsolve, the replication factor used in distributing the
    original matrix (and therefore its Cholesky factor) should be the
    same as that used in distributing the vector being solved into (or the rows
    of the matrix being solved into).

    Also note that when carrying out time-intensive calculations on the
    slave processes, the slaves will not be responsive to additional
    interaction, so commands such as \code{remoteLs} may appear to
    hang. This may occur because the slave process needs to finish a
    previous calculation before responding.

  Note that distributed vectors and distributed one-column
  matrices are stored differently, with matrices stored with padded
  columns. When using \code{remoteForwardSolve}, \code{remoteBacksolve},
  \code{remoteMultChol}, you should use \code{n2 = NULL} when the
  second argument is a vector and \code{n2 = 1} when the second column
  is a one-column matrix.

  Note that triangular and symmetric matrices are stored as vectors, column-major
  order, of the lower triangular elements. To collect a distributed
  symmetric matrix on the master process, one uses
  \code{collectTriangularMatrix}. \code{collectTriangularMatrix} always
  fills the upper triangle as the transpose of the lower triangle. 
  }
    \author{
  Christopher Paciorek and Benjamin Lipshitz, in collaboration with Tina
  Zhuo, Cari Kaufman, Rollin Thomas, and Prabhat.

  Maintainer: Christopher Paciorek <paciorek@alumni.cmu.edu>
  }
  \references{
      Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and
  R.C.  Thomas. 2015. Parallelizing Gaussian Process Calculations in
  R. Journal of Statistical Software, 63(10), 1-23. \doi{10.18637/jss.v063.i10}.
  
      Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and R.C.  Thomas. 2013. Parallelizing Gaussian Process Calculations in R. arXiv:1305.4886. \url{https://arxiv.org/abs/1305.4886}.
  }
\seealso{
See \code{\link{bigGP.init}} for the necessary initialization steps, and
\code{\link{krigeProblem}} for doing kriging based on maximum likelihood
estimation. 
}
\examples{
# this is an example of using the API to do distributed linear algebra
#    for Gaussian process calculations; we'll demonstrate generating from
#    a Gaussian process with exponential covariance; note that this can
#    be done more easily through the krigeProblem ReferenceClass
\dontrun{
bigGP.init(3)
params <- c(sigma2 = 1, rho = 0.25)
# for this example, we'll use a modest size problem, but to demo on a
#    cluster, increase m to a larger value
m <- 80 
gd <- seq(0, 1, length = m)
locns = expand.grid(x = gd, y = gd)
# indices will be a two column matrix with the index of the first set of
#  locations in the first column and of the second set in the second column
covfunc <- function(params, locns, indices) {
dd <- sqrt( (locns$x[indices[,1]] - locns$x[indices[,2]])^2 +
(locns$y[indices[,1]] - locns$y[indices[,2]])^2 )
return(params["sigma2"] * exp(-dd / params["rho"]))
}
mpi.bcast.Robj2slave(params)
mpi.bcast.Robj2slave(covfunc)
mpi.bcast.Robj2slave(locns)
mpi.bcast.cmd(indices <- localGetTriangularMatrixIndices(nrow(locns)))
mpi.bcast.cmd(C <- covfunc(params, locns, indices))
remoteLs() # this may pause before reporting, as slaves are busy doing
#    computations above
remoteCalcChol('C', 'L', n = m^2)
remoteConstructRnormVector('z', n = m^2)
remoteMultChol('L', 'z', 'x', n1 = m^2)
x <- collectVector('x', n = m^2)
image(gd, gd, matrix(x, m))
}
}

