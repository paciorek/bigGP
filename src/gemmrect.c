#include "gemmrect.h"
#include "comm.h"

/*
  Compute C += MA^t * MB where MA, MB are rectangular matrices.  Actually M^t is what is stored.  This computation takes 2 rectangular fundamental units as the input, and outputs a square fundamental unit.

        0 1 2 3
  M^t = 1 4 5 6
        2 5 7 8
	3 6 8 9

  The two blocks of M^t are stored contiguously in column-major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.  Each block of M is rectangular with bs rows and bsc columns.

  Note that C is m x m, and bsc is the block size relevant to it.

  I and J indentify the processor row and column, and P is the partition number, 4 in the above example.
 */

void gemmrect( double *C, double *MA, double *MB, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bsc;

  double *bufMA1 = (double*) malloc( bs2*sizeof(double) );
  double *bufMB1 = (double*) malloc( bs2*sizeof(double) );
  double *bufMA2 = (double*) malloc( bs2*sizeof(double) );
  double *bufMB2 = (double*) malloc( bs2*sizeof(double) );

  if( I == J ) {
    for( int k = 0; k < P; k++ ) {
      double *bMA,*bMB;
      if( k == I ) {
        bMA = MA;
        bMB = MB;
      } else {
        bMA = bufMA1;
        bMB = bufMB1;
      }
      mybcast( bMA, bs2, k, comms[I] );
      mybcast( bMB, bs2, k, comms[I] );
      localDgemmrc( C, bMA, bMB, bs, bsc );
    }
  } else {
    for( int k = 0; k < P; k++ ) {
      double *MAL = MA;
      double *MAU = MA+bs2;
      double *MBL = MB;
      double *MBU = MB+bs2;
      double *CL = C;
      double *CU = C+bsc*bsc;
      double *bMA1, *bMA2, *bMB1, *bMB2;

      if( J == k )
        bMA1 = MAL;
      else
        bMA1 = bufMA1;
      mybcast( bMA1, bs2, k, comms[I] );
      
      if( I == k )
        bMA2 = MAU;
      else
        bMA2 = bufMA2;
      mybcast( bMA2, bs2, k, comms[J] );

      if( J == k )
        bMB1 = MBL;
      else
        bMB1 = bufMB1;
      mybcast( bMB1, bs2, k, comms[I] );
      
      if( I == k )
        bMB2 = MBU;
      else
        bMB2 = bufMB2;
      mybcast( bMB2, bs2, k, comms[J] );
      
      localDgemmrc( CL, bMA1, bMB2, bs, bsc );
      localDgemmrc( CU, bMA2, bMB1, bs, bsc );
    }
  }
  free(bufMA1);
  free(bufMA2);  
  free(bufMB1);
  free(bufMB2);  

}
