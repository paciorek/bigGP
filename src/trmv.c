#include "local.h"
#include "trmv.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked triangular MV on the fundamental unit.  Example layout on 10 processors:

      0
  A = 1 4
      2 5 7
      3 6 8 9

      0
  X = 4
      7
      9

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of A.
void trmv( double *Xout, double *A, double *Xin, int bs, int I, int J, int P, MPI_Comm *comms ) {

  if( I == J ) {
    // broadcast along the column
    sendForward( Xin, bs, I, P, comms[J] );
    localTrmv( A, Xin, Xout, bs );
    // reduce along the row
    reduceForward( Xout, bs, J, comms[I] );
  } else {
    double *bufX1 = (double*) malloc( bs*sizeof(double) );
    double *bufX2 = (double*) malloc( bs*sizeof(double) );
    recvForward( bufX1, bs, J, comms[J] );
    localGemv( A, bufX1, bufX2, bs );
    reduceForward( bufX2, bs, I, comms[I] );
    free(bufX1);  
    free(bufX2);
  }
}
