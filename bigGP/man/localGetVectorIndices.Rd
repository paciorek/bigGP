\name{localGetVectorIndices}
\alias{localGetVectorIndices}
\alias{localGetTriangularMatrixIndices}
\alias{localGetRectangularMatrixIndices}
\title{
  Get Indices of Vector or Matrix Elements Stored on Slave Process
}
\description{
  \code{localGetVectorIndices}, \code{localGetTriangularMatrixIndices},
  and \code{localGetRectangularMatrixIndices} are internal auxiliary
  functions that determine the indices of the elements of a vector or
  matrix that are stored on a slave process. These are primarily meant
  for internal use, but can also be used in the process of creating
  distributed vectors and matrices on the slave processes. The functions
  need to be executed on the slave processes.
}
\usage{
localGetVectorIndices(n, h = 1)
localGetTriangularMatrixIndices(n, h = 1)
localGetRectangularMatrixIndices(n1, n2, h1 = 1, h2 = 1)
}
\arguments{
  \item{n}{
    a positive integer, giving the length of the vector or number of
    rows and columns of the triangular/square matrix.
  }
  \item{n1}{
    a positive integer, giving the  number of
    rows of the rectangular matrix.
  }
  \item{n2}{
    a positive integer, giving the  number of
    columns of the rectangular matrix.
  }
  \item{h}{
    a positive integer, giving the block replication factor for the
    vector or triangular/square matrix.
  }
  \item{h1}{
    a positive integer, giving the block replication factor for the
    rows of the rectangular matrix.
  }
  \item{h2}{
    a positive integer, giving the block replication factor for the
    columns of the rectangular matrix.
  }
}
\value{
  \code{localGetVectorIndices} returns the indices (as a one-column matrix) of the
  subset of a distributed vector that will be stored on the process on
  which the function is called. \code{localGetTriangularMatrixIndices},
  and \code{localGetRectangularMatrixIndices} return a two-column
  matrix with the indices for the subset of the distributed matrix that
  will be stored (as a vector) on the process on which the function is called. I.e.,
  the ith row of the matrix gives the (row, column) position in the full
  matrix for the ith element of the vector on the local process that
  contains a subset of that matrix.

  Warning: in some cases there is a small amount of buffering involved
  in the distributed objects so that the blocks on each process are of
  the same size. In this case, the index of the first element will
  generally be added one or more times to the end of the indices assigned to the last process.
}

  