\name{SN2011fe}
\alias{SN2011fe}
\alias{SN2011fe_subset}
\alias{SN2011fe_newdata}
\alias{SN2011fe_newdata_subset}
\alias{SN2011fe_mle}
\alias{SN2011fe_mle_subset}
\alias{SN2011fe_initialParams}
\alias{SN2011fe_meanfunc}
\alias{SN2011fe_predmeanfunc}
\alias{SN2011fe_covfunc}
\alias{SN2011fe_crosscovfunc}
\alias{SN2011fe_predcovfunc}
\docType{data}
\title{
  SN2011fe Supernova Dataset
}
\description{
  \code{SN2011fe} is a dataset of flux values and estimated standard errors, as a function of phase
   and wavelength, from the SN 2011fe supernova event. Data were collected over multiple nights (phases)
  and multiple wavelengths. 
}
\format{
The \code{SN2011fe} object is a data frame containing the following
columns:
\describe{
  \item{\code{phase}: }{time of measurement in days.}
  \item{\code{wavelength}: }{wavelength of measurement in \eqn{\mbox{\AA}}.}
  \item{\code{flux}: }{flux measurement in \eqn{\mbox{erg}\ \mbox{s}^{-1}\ \mbox{cm}^{-2}\ \mbox{\AA}^{-1}}.}
  \item{\code{fluxerror}: }{estimated standard deviation of the
    error in measurement of the flux.}
  \item{\code{phaseindex}: }{1-based index value of the time of
    measurement [check this]}
  \item{\code{logwavelength}: }{log of wavelength.}
}

The \code{SN2011fe_newdata} object is a data frame of prediction points
on a fine grid of phases and wavelengths. The columns correspond to
the phase and wavelength columns in \code{SN2011fe} but the initial 'p' stands for 'prediction'.

The \code{SN2011fe_mle} object is the output from maximum likelihood
fitting of the parameters of a statistical model for the dataset, with
the \code{par} element containing the MLEs.

The objects labeled '_subset' are analogous objects for a small subset
of the dataset feasible to be fit without parallel processing.

The \code{SN2011fe_initialParams} object is a set of starting values for
the maximum likelihood fitting.

The functions \code{SN2011fe_meanfunc}, \code{SN2011fe_predmeanfunc},
\code{SN2011fe_covfunc}, \code{SN2011fe_crosscovfunc}, and
\code{SN2011fe_predcovfunc} are functions for calculating the various
mean vectors and covariance matrices used in the statistical analysis of
the dataset. Users will need to create analogous functions for their own
kriging problems, so these are provided in part as templates. 
}
\section{Warning}{
Note that the \code{SN2011fe_newdata} set of prediction points was
chosen to ensure that the points were not so close together as to result
in numerically non-positive definite covariance matrices when simulating
posterior realizations. 
}
\source{
\url{https://snfactory.lbl.gov/snf/data/SNfactory_Pereira_etal_2013_SN2011fe.tar.gz}
}
\references{
  For more details on the dataset, see:
  R. Pereira, et al., 2013, "Spectrophotometric time series of SN 2011fe from
  the Nearby Supernova Factory," Astronomy and Astrophysics, accepted
  (arXiv:1302.1292v1), DOI: \doi{10.1051/0004-6361/201221008}.

  For more details on the statistical model used to fit the data, see:

  Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and
  R.C.  Thomas. 2015. Parallelizing Gaussian Process Calculations in
  R. Journal of Statistical Software, 63(10), 1-23. \doi{10.18637/jss.v063.i10}.
  
  or
  
  Paciorek, C.J., B. Lipshitz, W. Zhuo, Prabhat, C.G. Kaufman, and R.C.  Thomas. 2013. Parallelizing Gaussian Process Calculations in R. arXiv:1305.4886. \url{https://arxiv.org/abs/1305.4886}.
}
\seealso{\code{\link{krigeProblem-class}}}
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
predCovFunction = SN2011fe_predcovfunc, meanFunction =
SN2011fe_meanfunc, covFunction = SN2011fe_covfunc,  inputs = inputs,
params = SN2011fe_mle$par, data = SN2011fe$flux, packages = c("fields"))

prob$calcLogDens()
}
}
}