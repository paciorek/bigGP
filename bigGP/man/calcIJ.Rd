\name{calcIJ}
\alias{calcIJ}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Calculate Slave Process Identifiers
}
\description{
  \code{calcIJ} is an internal auxiliary function that calculates a
  unique pair of identifiers for each slave process, corresponding to
  the row and column of the block assigned to the slave process (things
  are more complicated when the block replication factor, \eqn{h}, is
  greater than one).
}
\usage{
calcIJ(D)
}
\arguments{
  \item{D}{
    a positive integer, the partition factor.
  }
  }