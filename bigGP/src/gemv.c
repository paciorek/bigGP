#include "local.h"
#include "gemv.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked GEMV on the fundamental unit.  Example layout on 10 processors:

      0 1 2 3
  A = 1 4 5 6
      2 5 7 8
      3 6 8 9

      0
  X = 4
      7
      9

 Two two blocks of X are stored contiguously, in column major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of X.
void gemv( double *Xout, double *A, double *Xin, int bs, int I, int J, int P, MPI_Comm *comms ) {

  double *bufX1 = (double*) malloc( bs*sizeof(double) );
  
  if( I == J ) {
    // broadcast along the column
    mybcast( Xin, bs, I, comms[J] );
    localGemv( A, Xin, bufX1, bs );
    // reduce along the row
    myreduce( bufX1, Xout, bs, J, comms[I] );
  } else {
    double *bufX2 = (double*) malloc( bs*sizeof(double) );
    double *bufX3 = (double*) malloc( bs*sizeof(double) );
    double *bufX4 = (double*) malloc( bs*sizeof(double) );
    mybcast( bufX1, bs, J, comms[J] );
    mybcast( bufX2, bs, I, comms[I] );
    localGemv( A, bufX1, bufX3, bs );
    localGemv( A+bs*bs, bufX2, bufX4, bs );
    myreduce( bufX3, Xin, bs, I, comms[I] );
    myreduce( bufX4, Xin, bs, J, comms[J] );
    free(bufX2);  
    free(bufX3);
    free(bufX4);  
  }
  free(bufX1);
}
