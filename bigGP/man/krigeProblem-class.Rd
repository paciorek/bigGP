\name{krigeProblem-class}
\docType{class}
\alias{krigeProblem-class}
\alias{krigeProblem}
\alias{calcH, krigeProblem-method}
\alias{show, krigeProblem-method}
\alias{initializeSlaveProblems, krigeProblem-method}
\alias{setParams, krigeProblem-method}
\alias{remoteConstructMean, krigeProblem-method}
\alias{remoteConstructCov, krigeProblem-method}
\alias{calcLogDens, krigeProblem-method}
\alias{optimizeLogDens, krigeProblem-method}
\alias{predict, krigeProblem-method}
\alias{calcPostCov, krigeProblem-method}
\alias{simulateRealizations, krigeProblem-method}
\title{Class \code{"krigeProblem"}}
\description{
  The \code{krigeProblem} class provides functionality for kriging using
  distributed calculations, based on maximum likelihood estimation. The class includes methods for standard kriging calculations and metadata necessary for carrying out the methods in a distributed fashion.

  To carry out kriging calculations, one must first initialize an object
  of the \code{krigeProblem} class. This is done using
  \code{krigeProblem$new} and help on initialization can be obtained via
  \code{krigeProblem$help('initialize')} (but noting that the call is
  \code{krigeProblem$new} not \code{krigeProblem$initialize}).

  Note that in what follows I refer to observation and prediction
  'locations'. This is natural for spatial problems, but for non-spatial
  problems, 'locations' is meant to refer to the points within the
  relevant domain at which observations are available and predictions
  wish to be made.
  
  The user must provide functions that create the subsets of the mean
  vector(s) and the covariance matrix/matrices. Functions for the mean
  vector and covariance matrix for observation locations are required,
  while those
  for the mean vector for prediction locations, the cross-covariance
  matrix (where the first column is the index of the observation
  locations and the second of the prediction locations), and the
  prediction covariance matrix for prediction locations are required
  when doing prediction and posterior simulation. These functions should
  follow the form of \code{SN2011fe_meanfunc},
  \code{SN2011fe_predmeanfunc}, \code{SN2011fe_covfunc},
  \code{SN2011fe_predcovfunc}, and \code{SN2011fe_crosscovfunc}. Namely,
  they should take three arguments, the first a vector of all the
  parameters for the Gaussian process (both mean and covariance),
  the second an arbitrary list of inputs (in general this would include
  the observation and prediction locations), and the third being
  indices, which will be provided by the package and will differ between
  slave processes. For the
  mean functions, the indices will be a vector, indicating which of the
  vector elements are stored on a given process. For the covariance
  functions, the indices will be a two column matrix, with each row
  a pair of indices (row, column), indicating the elements of the matrix
  stored on a given process. Thus, the user-provided functions should use the second
  and third arguments to construct the elements of the vectors/matrices
  belonging on the slave process. Note that the elements of the
  matrices are stored as vectors (vectorizing matrices column-wise, as
  natural for column-major matrices). Users can simply have their
  functions operate on the rows of the index matrix without worrying
  about ordering. An optional fourth argument contains cached values that need
  not be computed at every call to the user-provided function. If the
  user wants to make use of caching of values to avoid expensive
  recomputation, the user function should mimic
  \code{SN2011fe_covfunc}. That is, when the user wishes to change the cached
  values (including on first use of the function), the function should return 
  a two-element list, with the first element being the covariance matrix
  elements and the second containing whatever object is to be
  cached. This cached object will be provided to the function on
  subsequent calls as the fourth argument. 
  
  Note that one should have all necessary packages required for
  calculation of the mean vector(s) and covariance matrix/matrices installed
  on all machines used and the names of these packages should be passed
  as the \code{packages} argument to the \code{krigeProblem} initialization.

  Help for the various methods of the class can be obtained with
  \code{krigeProblem$help('methodName')} and a list of fields and
  methods in the class with \code{krigeProblem$help()}.

  In general, \code{n} (or \code{n1} and \code{n2}) refer to the length
  or number of rows/columns of vectors and matrices and \code{h} (or
  \code{h1} and \code{h2}) to the block replication factor for these
  vectors and matrices. More details on block replication factors can be
  found in the references in \sQuote{references}; these are set at
  reasonable values automatically, and for simplicity, one can set them
  at one, in which case the number of blocks into which the primary
  covariance matrix is split is \eqn{P}, the number of slave
  processes. Cross-covariance matrices returned to the user will have
  number of rows equal
  to the number of observation locations and number of columns to the
  number of prediction locations. Matrices of realizations will have
  each realized field as a single column. 
  
}
\section{Fields}{
  \describe{
    \item{\code{localProblemName}:}{Object of class \code{"character"}
containing the name to be used for the object on the slave processes.}
    \item{\code{n}:}{Object of class \code{"numeric"}
containing the number of observation locations.}
    \item{\code{h_n}:}{Object of class \code{"numeric"}
containing the block replication factor for the observation locations,
will be set to a reasonable value by default upon initialization of an
object in the class.}
    \item{\code{h_m}:}{Object of class \code{"numeric"}
containing the block replication factor for the prediction locations,
will be set to a reasonable value by default upon initialization of an
object in the class.}
    \item{\code{meanFunction}:}{Object of class \code{"function"}
containing the function used to calculate values of the mean function at
the observation locations. See above for detailed information on how
this function should be written.}
   \item{\code{predMeanFunction}:}{Object of class \code{"function"}
containing the function used to calculate values of the mean function at
the prediction locations. See above for detailed information on how
this function should be written.}
   \item{\code{covFunction}:}{Object of class \code{"function"}
containing the function used to calculate values of the covariance
function for pairs of observation locations. See above for detailed
information on how this function should be written.}
   \item{\code{crossCovFunction}:}{Object of class \code{"function"}
containing the function used to calculate values of the covariance
function for pairs of observation and prediction locations. See above
for detailed information on how this function should be written.}
   \item{\code{predCovFunction}:}{Object of class \code{"function"}
containing the function used to calculate values of the covariance
function for pairs of prediction locations. See above for detailed
information on how this function should be written.}
    \item{\code{data}:}{Object of class \code{"ANY"}
containing the vector of data values at the observation locations. This
will be numeric, but is specified as of class \code{"ANY"} so that can
default to \code{NULL}.}
    \item{\code{params}:}{Object of class \code{"ANY"}
containing the vector of parameter values. This
will be numeric, but is specified as of class \code{"ANY"} so that can
default to \code{NULL}. This vector is what will be passed to the mean
and covariance functions.}
    \item{\code{meanCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed mean vector (for the
observation locations) on the slaves is
current (i.e., whether it is based on the current value of
\code{params}).}
    \item{\code{predMeanCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed mean vector (for the
prediction locations) on the slaves is
current (i.e., whether it is based on the current value of
\code{params}).}
    \item{\code{postMeanCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed posterior mean vector (for
the prediction locations) on the slaves is
current (i.e., whether it is based on the current value of
\code{params}).}
    \item{\code{covCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed covariance matrix (for the
observation locations) on the slaves is
current (i.e., whether it is based on the current value of
\code{params}).}
   \item{\code{crossCovCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed cross-covariance matrix
(between observation and prediction locations) on the slaves is
current (i.e., whether it is based on the current value of
\code{params}).}
   \item{\code{predCovCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed prediction covariance matrix
on the slaves is current (i.e., whether it is based on the current value of
\code{params}).}
   \item{\code{postCovCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed posterior covariance matrix
on the slaves is current (i.e., whether it is based on the current value of
\code{params}).}
   \item{\code{cholCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed Cholesky factor of the
covariance matrix (for observation locations) on the slaves is current (i.e., whether it is based on the current value of
\code{params}).}
   \item{\code{predCholCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed Cholesky factor of the
covariance matrix (the prior covariance matrix for prediction locations) on the slaves is current (i.e., whether it is based on the current value of
\code{params}). Note this is likely only relevant when generating
realizations for prediction locations not conditional on the
observations.}
   \item{\code{postCholCurrent}:}{Object of class \code{"logical"}
indicating whether the current distributed Cholesky factor of the
posterior covariance matrix on the slaves is current (i.e., whether it is based on the current value of
\code{params}).}
}
}
\section{Methods}{
  \describe{
    \item{\code{new(localProblemName = NULL, numProcesses = NULL, h_n =
NULL, h_m = NULL,  n = length(data), m = NULL, meanFunction =
function(){}, predMeanFunction = function(){}, covFunction =
function(){}, crossCovFunction = function(){}, predCovFunction =
function(){}, inputs = NULL, params = NULL, data = NULL, packages =
NULL, parallelRNGpkg = "rlecuyer", seed = 0, ...)}:}{ Initializes new
krigeProblem object, which is necessary for distributed kriging calculations.}
    \item{\code{calcH(n)}:}{ Internal method that calculates a good
choice of the block replication factor given \code{n}.}
    \item{\code{show(verbose = TRUE)}:}{ Show (i.e., print) method.}
    \item{\code{initializeSlaveProblems(packages)}:}{ Internal method
that sets up the slave
processes to carry out the \code{krigeProblem} distributed
calculations.}
    \item{\code{setParams(params, verbose = TRUE)}:}{ Sets (or changes) the value of
the parameters.}
    \item{\code{remoteConstructMean(obs = TRUE, pred = !obs, verbose =
FALSE)}:}{ Meant for internal use; calculates the value of the specified mean vector (for
observation and/or prediction locations) on the slave processes, using the
appropriate user-provided function.}
    \item{\code{remoteConstructCov(obs = TRUE, pred = FALSE, cross =
FALSE, verbose = FALSE)}:}{ Meant for internal use; calculates the value of the specified
covariance matrices on the slave processes, using the
appropriate user-provided function.}
   \item{\code{calcLogDeterminant()}:}{ Calculates the log-determinant
of the covariance matrix for the observation locations.}
    \item{\code{calcLogDens(newParams = NULL, newData = NULL, negative =
FALSE, verbose = TRUE)}:}{ Calculates the log-density of the data given
the parameters.}
   \item{\code{optimizeLogDens(newParams = NULL, newData = NULL, method
= "Nelder-Mead", verbose = FALSE, gr = NULL, lower = -Inf, upper = Inf,
control = list(), hessian = FALSE, ...)}:}{ Finds the maximum likelihood
estimate of the parameters given the data, using \code{optim}.}
   \item{\code{predict(ret = FALSE, verbose = FALSE)}:}{ Calculates
kriging predictions (i.e., the posterior mean for the prediction locations).}
   \item{\code{calcPostCov(returnDiag = TRUE, verbose = FALSE)}:}{
Calculates the prediction covariance (i.e., the posterior covariance
matrix for the prediction locations), returning the diagonal (the
variances) if requested.} 
   \item{\code{simulateRealizations(r = 1, h_r = NULL, obs = FALSE, pred
= FALSE, post = TRUE, verbose = FALSE)}:}{ Simulates realizations, which
would generally be from the posterior distribution (i.e., conditional on
the data), but could also be from the prior distribution (i.e., not
conditional on the data) at either observation or predition locations.}
}
}
\section{Extends}{
All reference classes extend and inherit methods from \code{"\linkS4class{envRefClass}"}.
}
\references{
      Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and
  R.C.  Thomas. 2015. Parallelizing Gaussian Process Calculations in
  R. Journal of Statistical Software, 63(10), 1-23. \doi{10.18637/jss.v063.i10}.
 
  Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and R.C.  Thomas. 2013. Parallelizing Gaussian Process Calculations in R. arXiv:1305.4886. \url{https://arxiv.org/abs/1305.4886}.
}
\author{
  Christopher Paciorek and Benjamin Lipshitz, in collaboration with Tina
  Zhuo, Cari Kaufman, Rollin Thomas, and Prabhat.

  Maintainer: Christopher Paciorek <paciorek@alumni.cmu.edu>
}
\seealso{
  See \code{\link{bigGP}} for general information on the package and
  \code{\link{bigGP.init}} for the necessary initialization steps
  required before using the package, including the \code{krigeProblem} class.
}
\examples{
\dontrun{
doSmallExample <- TRUE

if(require(fields)) {

if(doSmallExample){
  SN2011fe <- SN2011fe_subset
  SN2011fe_newdata <- SN2011fe_newdata_subset
  SN2011fe_mle <- SN2011fe_mle_subset
  nProc <- 3
} else {
# users should select number of processors based on their system and the
# size of the full example
nProc <- 210 
}


n <- nrow(SN2011fe)
m <- nrow(SN2011fe_newdata)
nu <- 2
inputs <- c(as.list(SN2011fe), as.list(SN2011fe_newdata), nu = nu)

prob <- krigeProblem$new("prob", numProcesses = nProc, n = n, m = m,
 predMeanFunction = SN2011fe_predmeanfunc, crossCovFunction = SN2011fe_crosscovfunc, 
predCovFunction = SN2011fe_predcovfunc, meanFunction = SN2011fe_meanfunc, 
covFunction = SN2011fe_covfunc,  inputs = inputs, params = SN2011fe_mle$par, 
data = SN2011fe$flux, packages = c("fields"))

prob$calcLogDens()

prob$optimizeLogDens(method = "L-BFGS-B", verbose = TRUE,
lower = rep(.Machine$double.eps, length(SN2011fe_initialParams)),
control = list(parscale = SN2011fe_initialParams, maxit = 2))
# the full optimization can take some time; only two iterations are done 
# are specified here; even this is not run as it takes 10s of seconds

prob$setParams(SN2011fe_mle$par)

pred <- prob$predict(ret = TRUE, se.fit = TRUE, verbose = TRUE)
realiz <- prob$simulateRealizations(r = 10, post = TRUE, verbose = TRUE)

show(prob)
}
}
}
\keyword{classes}
