\name{bigGP-meta}
\alias{bigGP-meta}
\alias{.bigGP}
\alias{.bigGP.fill}

\title{
  Information about the number and identities of the processes

}
\description{
  The \code{.bigGP} object (an environment) contains information about
  the processes involved in the distributed
  computation. \code{.bigGP.fill} is an internal auxiliary function that
  fills the 
  \code{.bigGP} object with the values of \eqn{P}, \eqn{D}, \eqn{I}, and \eqn{J}.
}
\usage{
.bigGP
.bigGP.fill(init = FALSE)
}
\arguments{
  \item{init}{logical, indicating whether to initialize values to their
    defaults for before the processes are set up}
}
