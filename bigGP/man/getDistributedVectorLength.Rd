\name{getDistributedVectorLength}
\alias{getDistributedVectorLength}
\alias{getDistributedTriangularMatrixLength}
\alias{getDistributedRectangularMatrixLength}
\title{
  Find Length of Subset of Vector or Matrix Stored on Slave Process
}
\description{
  \code{getDistributedVectorLength},
  \code{getDistributedTriangularMatrixLength}, and
  \code{getDistributedRectangularLength} are internal auxiliary
  functions that find the length of the vector needed to store the subset
  of a vector or matrix contained on a given slave process.
}
\usage{
getDistributedVectorLength(n, h = 1)
getDistributedTriangularMatrixLength(n, h = 1)
getDistributedRectangularMatrixLength(n1, n2, h1 = 1, h2 = 1)
}
\arguments{
  \item{n}{length of vector.}
  \item{h}{replication factor.}
  \item{n1}{number of rows.}
  \item{n2}{number of columns.}
  \item{h1}{replication factor for the rows.}
  \item{h2}{replication factor for the columns.}
}