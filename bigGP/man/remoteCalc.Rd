\name{remoteCalc}
\alias{remoteCalc}
\title{
  Do Arbitrary Calculations on One or Two Inputs
}
\description{
  \code{remoteCalc} applies a function to either one or two input
  objects on the slave processes. Input objects can be obtained environments, lists, and
  ReferenceClass objects as well as the global environment on the slave
  processes. The output object can be assigned into a environment or a ReferenceClass objects as well as the global environment on the slave
  processes.
}
\usage{
remoteCalc(input1Name, input2Name = NULL, FUN, outputName, input1Pos = '.GlobalEnv',
input2Pos = '.GlobalEnv', outputPos = '.GlobalEnv')
}
\arguments{
\item{input1Name}{
  an object name, given as a character string, giving the name of the
    first input on the slave processes.
  }
\item{input2Name}{
  an object name, given as a character string, giving the name of the
    first input on the slave processes. This is optional so that one can
    carry out a calculation on a single input.
  }
  \item{FUN}{
    the function to be applied, see \sQuote{details}.  In the case of operators like \code{+},  the function name
    must be backquoted.
    }
\item{outputName}{
  an object name, given as a character string, giving the name to be
  used for the result of the function call.
  }
  \item{input1Pos}{
  where to look for the first input, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
 \item{input2Pos}{
  where to look for the second input, given as a character string (unlike
    \code{get}). This can indicate an environment, a list, or a ReferenceClass object. 
}
 \item{outputPos}{
  where to do the assignment of the output, given as a character string (unlike
    \code{assign}). This can indicate an environment or a ReferenceClass object. 
}
}
\details{
  \code{FUN} is found by a call to \code{match.fun} and typically is either a
     function or a symbol (e.g., a backquoted name) or a character
     string specifying a function to be searched for from the
     environment of the call to \code{remoteCalc}.
  }
  \examples{
\dontrun{
bigGP.init(3)
mpi.bcast.cmd(x <- 0:mpi.comm.rank())
remoteCalc('x', FUN = exp, outputName = 'exp.x')
remoteLs()
pull('exp.x')
remoteCalc('x', 'exp.x', FUN = `+`, outputName = 'silly')
pull('silly')
}
}

