\name{localCollectVector}
\alias{localCollectVector}
\alias{localCollectVectorTest}
\alias{localDistributeVector}
\alias{localDistributeVectorTest}
\alias{localPull}
\alias{localPullTest}
\alias{localCollectDiagonal}
\alias{localCollectDiagonalTest}
\alias{localCollectTriangularMatrix}
\alias{localCollectTriangularMatrixTest}
\alias{localCollectRectangularMatrix}
\alias{localCollectRectangularMatrixTest}

\title{
  Local Distribution and Collection Functions
}
\description{
  These internal functions carry out the tasks of their respective
  primary functions, e.g., \code{collectVector}, on the slave
  process. The functions need to be executed on the slave processes.
}
\usage{
localCollectVector(objName, objPos, n, h)
localCollectVectorTest(objName, objPos, n, h)
localDistributeVector(objName, objPos, n, h)
localDistributeVectorTest(objName, objPos, n, h)
localPull(objName, objPos, tag = 1)
localCollectDiagonal(objName, objPos, n, h)
localCollectDiagonalTest(objName, objPos, n, h)
localCollectTriangularMatrix(objName, objPos, n, h)
localCollectTriangularMatrixTest(objName, objPos, n, h)
localCollectRectangularMatrix(objName, objPos, n1, n2, h1, h2)
localCollectRectangularMatrixTest(objName, objPos, n1, n2, h1, h2)
}
\arguments{
\item{objName}{name of object as a character string.}
\item{objPos}{ where to look for the object, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. }
\item{n}{length of vector.}
\item{h}{replication factor.}
\item{tag}{MPI tag.}
\item{n1}{number of rows.}
\item{n2}{number of columns.}
\item{h1}{replication factor for the rows.}
\item{h2}{replication factor for the columns.}
  }