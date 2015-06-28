\name{alloc}
\alias{alloc}
\title{
  Create Object with its Own Memory
}
\description{
   \code{alloc} is an internal auxiliary function that creates an object
   of the size of the input with the goal of allocating new memory for use
   in the C functions used by the package.
}
\usage{
alloc(input, inputPos = '.GlobalEnv')
}
\arguments{
  \item{input}{an object name, given as a character string, giving the
    name of the object whose size is to be mimiced in creating the
    output, or the length of the output vector to be created.}
  \item{inputPos}{ where to look for the input, given as a character
    string (unlike \code{get}). This can indicate an environment, a
    list, or a ReferenceClassobject.}
}
\value{
  A new numeric vector of the appropriate size.
  }