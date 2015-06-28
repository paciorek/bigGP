#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// collect a vector of length n to the master process.  It is distributed with block size bs=(n+P-1)/P
void collectFullVec( double *Xdist, double *X, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world );
void distributeFullVec( double *X, double *Xdist, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world );
void collectFullTri( double *XDist, double *X, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world );
void collectFullDiag( double *L, double *X, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world );
void collectFullRect( double *XDist, double *X, int h, int h2, int rank, int P, int II, int JJ, int bs, int bs2, int n, int m, MPI_Comm comm_world );
