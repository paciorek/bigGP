#include "comm.h"
#include "syrkr.h"

/*
  Compute L += M^t * M where M is a rectangular matrix.  Actually M^t is what is stored.  This computation takes a rectangular fundamental unit as the input, and outputs a triangular fundamental unit.

        0
  L =   1 4
	2 5 7
	3 6 8 9

        0 1 2 3
  M^t = 1 4 5 6
        2 5 7 8
	3 6 8 9

  The two blocks of M^t are stored contiguously in column-major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.  Each block of M is rectangular with bs rows and bsc columns.
  I and J indentify the processor row and column, and P is the partition number, 4 in the above example.

  Note that L is m x m, so bsc is the block size of L.
 */

void syrkr( double *L, double *M, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bsc;

  double *bufM1 = (double*) malloc( bs2*sizeof(double) );
  double *bufM2 = (double*) malloc( bs2*sizeof(double) );

  if( I == J ) {
    for( int k = 0; k < P; k++ ) {
      double *bM;
      if( k == I ) {
        bM = M;
      } else {
        bM = bufM1;
      }
      mybcast( bM, bs2, k, comms[I] );
      localSyrkr( bM, L, bs, bsc );
    }
  } else {
    for( int k = 0; k < P; k++ ) {
      double *ML = M;
      double *MU = M+bs2;
      double *bM1, *bM2;

      if( J == k )
        bM1 = ML;
      else
        bM1 = bufM1;
      mybcast( bM1, bs2, k, comms[I] );
      
      if( I == k )
        bM2 = MU;
      else
        bM2 = bufM2;
      mybcast( bM2, bs2, k, comms[J] );

      localDgemmrc( L, bM1, bM2, bs, bsc );
    }
  }
  free(bufM1);
  free(bufM2);
}
