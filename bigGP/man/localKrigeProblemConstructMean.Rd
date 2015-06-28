\name{localKrigeProblemConstructMean}
\alias{localKrigeProblemConstructMean}
\alias{localKrigeProblemConstructCov}
\title{
  Calculate Mean Vector or Covariance Matrix on Slave Process
}
\description{
  \code{localKrigeProblemConstructMean} and
  \code{localKrigeProblemConstructCov} are internal wrapper functions
  for calculating a mean vector or covariance matrix on the slave
  processes. They are called by member functions of the \code{krigeProblem} ReferenceClass.
}
\usage{
localKrigeProblemConstructMean(problemName, obs, pred)
localKrigeProblemConstructCov(problemName, obs, pred, cross)
}
\arguments{
  \item{problemName}{name of the problem as a character string.}
  \item{obs}{logical, whether to compute the mean or covariance for the
    observation locations.}
  \item{pred}{logical, whether to compute the mean or covariance for the
    prediction locations.}
  \item{cross}{logical, whether to compute the cross-covariance.}
  }
\seealso{
  \code{krigeProblem}
  }