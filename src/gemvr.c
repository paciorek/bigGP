#include "local.h"
#include "gemvr.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a transposed rectangular blocked GEMV on the fundamental unit.  Example layout on 10 processors:

      0 1 2 3
  A = 1 4 5 6
      2 5 7 8
      3 6 8 9

      0
  X = 4
      7
      9

 Two two blocks of A are stored contiguously, in column major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.

 A is rectangular, stored as A^t; each block is bs x bsc.  This computes A^t X
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of X.
void gemvr( double *Xout, double *A, double *Xin, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {

  double *bufX1 = (double*) malloc( bsc*sizeof(double) );
  
  if( I == J ) {
    // broadcast along the column
    mybcast( Xin, bs, I, comms[J] );
    localGemvr( A, Xin, bufX1, bs, bsc );
    // reduce along the row
    myreduce( bufX1, Xout, bsc, J, comms[I] );
  } else {
    double *bufX2 = (double*) malloc( bsc*sizeof(double) );
    double *bufX3 = (double*) malloc( bs*sizeof(double) );
    double *bufX4 = (double*) malloc( bs*sizeof(double) );
    mybcast( bufX3, bs, J, comms[J] );
    mybcast( bufX4, bs, I, comms[I] );
    localGemvr( A, bufX3, bufX1, bs, bsc );
    localGemvr( A+bs*bsc, bufX4, bufX2, bs, bsc );
    myreduce( bufX1, Xin, bsc, I, comms[I] );
    myreduce( bufX2, Xin, bsc, J, comms[J] );
    free(bufX2);
    free(bufX3);
    free(bufX4);
  }
  free(bufX1);
}
