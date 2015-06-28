#include "local.h"
#include "trs.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked TRS on the fundamental unit.  Each diagonal processor owns one block of L and one of X.  Each off-diagonal processor owns one block of L and none of X, eg.

      0
  L = 1 4 
      2 5 7
      3 6 8 9

      0
  X = 4
      7
      9

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of X.
void trs( double *X, double *L, int bs, int I, int J, int P, MPI_Comm *comms ) {

  double *bufX1 = (double*) malloc( bs*sizeof(double) );

  if( I == J ) { // diagonal processor
    // perform updates from earlier in the matrix
    for( int j = 0; j < J; j++ ) {
      myrecv( bufX1, bs, j, TAG_DIRECT, comms[I] );
      localAxpy( bufX1, X, bs );
    }
    // local solve
    localTrs( L, X, bs );
    // send X down the column
    sendForward( X, bs, J, P, comms[J] );
  } else { // off-diagonal processor
    double *bufX2 = (double*) malloc( bs*sizeof(double) );
    // receive from diagonal above
    recvForward( bufX1, bs, J, comms[J] );
    localGemv( L, bufX1, bufX2, bs );
    // send to the diagonal to the right
    mysend( bufX2, bs, I, TAG_DIRECT, comms[I] );
    free(bufX2);
  }
  
  free(bufX1);
}
