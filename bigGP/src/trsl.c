#include "local.h"
#include "trsl.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked left-sided TRS on the fundamental unit.  Each diagonal processor owns one block of L and one of X.  Each off-diagonal processor owns one block of L and none of X, eg.

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
void trsl( double *X, double *L, int bs, int n, int I, int J, int P, MPI_Comm *comms ) {

  double *bufX1 = (double*) malloc( bs*sizeof(double) );

  int n2 = n - bs*I;
  if( n2 > bs )
    n2 = bs;
  if( I == J ) { // diagonal processor
    // perform updates from later in the matrix
    for( int j = P-1; j > J; j-- ) {
      myrecv( bufX1, bs, j, TAG_DIRECT, comms[I] );
      localAxpy( bufX1, X, bs );
    }
    // local solve
    localTrsl( L, X, bs, n2 );
    // send X back on the row
    sendBackward( X, bs, J, P, comms[J] );
  } else { // off-diagonal processor
    double *bufX2 = (double*) malloc( bs*sizeof(double) );
    // receive from diagonal to the right
    recvBackward( bufX1, bs, I, comms[I] );
    localGemvl2( L, bufX1, bufX2, bs, n2 );
    // send to the diagonal above
    mysend( bufX2, bs, J, TAG_DIRECT, comms[J] );
    free(bufX2);
  }
  
  free(bufX1);
}
