#include "local.h"
#include "trsm.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked TRSM on the rectangular fundamental unit.  Each diagonal processor owns one block of L and one of X^t.  Each off-diagonal processor owns one block of L and two of X.  Eg. on 10 processors

      0
  L = 1 4 
      2 5 7
      3 6 8 9

        0 1 2 3
  X^t = 1 4 5 6
        2 5 7 8
        3 6 8 9

 The two blocks of X^t are stored contiguously, in column major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.  Each block of X is rectangular with bs rows and bsc columns.

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of X.
void trsmr( double *X, double *L, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;
  int bsrect = bs*bsc;

  double *bufL1 = (double*) malloc( bs2*sizeof(double) );
  double *bufL2 = (double*) malloc( bs2*sizeof(double) );
  double *bufX1 = (double*) malloc( bsrect*sizeof(double) );
  double *bufX2 = (double*) malloc( bsrect*sizeof(double) );

  if( I == J ) {
    
    // perform updates from earlier in the matrix
    for( int j = 0; j < J; j++ ) {
      mybcast( bufL1, bs2, j, comms[J] );
      recvForward( bufX1, bsrect, j, comms[J] );
      localDgemmr( X, bufX1, bufL1, bs, bsc );      
    }
    
    // Send L to everyone in the same column
    mybcast( L, bs2, I, comms[J] );

    // local trsm on this column
    localTrsmr( L, X, bs, bsc );

    // now update the rest of the matrix: X(:,J+1:n) -= X(:,J)*(L(J+1:n,J))^t
    sendForward( X, bsrect, J, P, comms[I] );
  } else {
    double *XL = X;
    double *XU = X+bsrect;
    for( int j = 0; j <= I; j++ ) {
      if( j < J ) {
	// two blocks to update
	// receive L_Jj
	mybcast( bufL1, bs2, j, comms[I] );
	// receive L_Ij
	mybcast( bufL2, bs2, j, comms[J] );
	// receive X_Jj
	recvForward( bufX1, bsrect, j, comms[J] );
	// receive X_Ij
	recvForward( bufX2, bsrect, j, comms[I] );

	localDgemmr( XU, bufX1, bufL1, bs, bsc );
	localDgemmr( XL, bufX2, bufL2, bs, bsc );
      } else if( j == J ) {
	// work on our block in the lower triangle
	// receive the diagonal of L
	mybcast( bufL1, bs2, J, comms[J] );
	localTrsmr( bufL1, XL, bs, bsc );
	mybcast( L, bs2, J, comms[I] );
	sendForward( XL, bsrect, J, P, comms[I] );
	// send the off-diagonal of L

	// one block to update
	recvForward( bufX1, bsrect, j, comms[J] );
	localDgemmr( XU, bufX1, L, bs, bsc );
      } else if( j < I ) {
	// one block to update
	mybcast( bufL1, bs2, j, comms[I] );
	recvForward( bufX1, bsrect, j, comms[J] );
	localDgemmr( XU, bufX1, bufL1, bs, bsc );
      } else  { //j == I
	// work on our block in the upper triangle
	// receive the diagonal of L
	mybcast( bufL1, bs2, I, comms[I] );
	localTrsmr( bufL1, XU, bs, bsc );
	sendForward( XU, bsrect, I, P, comms[J] );
      }
    }
  }
  free(bufL1);
  free(bufL2);
  free(bufX1);
  free(bufX2);  
}
