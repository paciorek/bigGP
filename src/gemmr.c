#include "local.h"
#include "gemmr.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked GEMM on the fundamental unit.  Each diagonal processor owns one block of L and one of X.  Each off-diagonal processor owns one block of L and two of X.  Eg. on 10 processors

      0 1 2 3
  X = 1 4 5 6
      2 5 7 8
      3 6 8 9

 Two two blocks of X are stored contiguously, in column major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.  Rectangular matrices have rectangular blocks.

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.

 This computes C -= A * B^t, where B is square and A,C are rectangular, or equivalently C^t -= B * A^t.
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of X.
void gemmr( double *C, double *A, double *B, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int bs2s = bs*bs;
  int bs2r = bs*bsc;

  double *bufA1 = (double*) malloc( bs2r*sizeof(double) );
  double *bufB1 = (double*) malloc( bs2s*sizeof(double) );
  double *bufA2 = (double*) malloc( bs2r*sizeof(double) );
  double *bufB2 = (double*) malloc( bs2s*sizeof(double) );

  if( I == J ) {
    for( int k = 0; k < P; k++ ) {
      double *bA,*bB;
      if( k == I ) {
	bA = A;
	bB = B;
      } else {
	bA = bufA1;
	bB = bufB1;
      }
      mybcast( bA, bs2r, k, comms[I] );
      mybcast( bB, bs2s, k, comms[I] );
      localDgemmr( C, bA, bB, bs, bsc );
    }
  } else {
    for( int k = 0; k < P; k++ ) {
      double *AL = A;
      double *AU = A+bs2r;
      double *BL = B;
      double *BU = B+bs2s;
      double *CL = C;
      double *CU = C+bs2r;
      double *bA1, *bA2, *bB1, *bB2;

      if( J == k )
	bA1 = AL;
      else
	bA1 = bufA1;
      mybcast( bA1, bs2r, k, comms[I] );
      
      if( I == k )
	bA2 = AU;
      else
	bA2 = bufA2;
      mybcast( bA2, bs2r, k, comms[J] );

      if( J == k )
	bB1 = BL;
      else
	bB1 = bufB1;
      mybcast( bB1, bs2s, k, comms[I] );
      
      if( I == k )
	bB2 = BU;
      else
	bB2 = bufB2;
      mybcast( bB2, bs2s, k, comms[J] );
      
      localDgemmr( CL, bA1, bB2, bs, bsc );
      localDgemmr( CU, bA2, bB1, bs, bsc );
    }
  }
  free(bufA1);
  free(bufA2);  
  free(bufB1);
  free(bufB2);  
}
