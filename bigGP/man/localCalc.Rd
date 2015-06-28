\name{localCalc}
\alias{localCalc}
\alias{localForwardsolve}
\alias{localBacksolve}
\alias{localCalcChol}
\alias{localMultChol}
\alias{localCrossProdMatVec}
\alias{localCrossProdMatSelf}
\alias{localCrossProdMatSelfDiag}
\alias{localConstructRnormMatrix}
\title{
  Local Calculation Functions
}
\description{
  These internal functions carry out the calculations of their respective remote
  counterpart functions, e.g., \code{remoteCalc}, on the slave
  process. The functions need to be executed on the slave processes.
}

