\name{remoteGetIndices}
\alias{remoteGetIndices}
\title{
  Determine Indices of Vector or Matrix Elements Stored on all Processes
}
\description{
  \code{remoteGetIndices} determines the indices of the subset of a
  matrix or vector that are stored on each process. 
}
\usage{
remoteGetIndices(type = "vector", objName, objPos = ".GlobalEnv", n1,
 n2 = NULL, h1 = 1, h2 = 1)
}
\arguments{
  \item{type}{
    a string, one of 'vector', 'symmetric', 'triangular', or 'rectangular'
  giving the type of object for which one wants the indices. Note that
  square and symmetric matrices are both stored as lower triangles, so
  these options both return the same result. For square, non-symmetric
  matrices, use 'rectangular'.
  }
  \item{objName}{
    the name to use for the object containing the indices on the slave processes.
  }
  \item{objPos}{
  where to do the assignment of the object, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass
    object.
    }
  \item{n1}{
    a positive integer, giving the length of the vector, number of rows
  and columns of a symmetric or triangular matrix and number of rows of
  a rectangular matrix, including square, non-symmetric matrices.
  }
  \item{n2}{
    a positive integer, giving the number of columns of a 
  a rectangular matrix.
  }
  \item{h1}{
    a positive integer, giving the block replication factor for a
    vector, a symmetric or triangular matrix, or the rows of
    a rectangular matrix.
  }
  \item{h2}{
    a positive integer, giving the block replication factor for the
    columns of the rectangular matrix.
  }
}
\details{
  \code{remoteGetIndices} calculates the indices as described in
  \code{\link{localGetVectorIndices}},
  \code{\link{localGetTriangularMatrixIndices}}, and
  \code{\link{localGetRectangularMatrixIndices}}, and writes them to an
  object named \code{objName}.
}

  