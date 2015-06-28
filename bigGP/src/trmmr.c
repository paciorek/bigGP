#include "local.h"
#include "trmmr.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked TRMM on the fundamental unit.  Each diagonal processor owns one block of L and one of X.  Each off-diagonal processor owns one block of L and two of X.  Eg. on 10 processors

      0 1 2 3
  X = 1 4 5 6
      2 5 7 8
      3 6 8 9

 Two two blocks of X are stored contiguously, in column major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.  Rectangular matrices have rectangular blocks.

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.

 This computes C -= B * L^t, where L is triangular and A,C are rectangular, or equivalently C^t -= L * B^t.
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of X.
void trmmr( double *C, double *L, double *B, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int bs2s = bs*bs;
  int bs2r = bs*bsc;

  double *bufL1 = (double*) malloc( bs2s*sizeof(double) );
  double *bufB1 = (double*) malloc( bs2r*sizeof(double) );
  double *bufL2 = (double*) malloc( bs2s*sizeof(double) );
  double *bufB2 = (double*) malloc( bs2r*sizeof(double) );

  if( I == J ) {
    for( int k = 0; k < P; k++ ) {
      double *bL,*bB;
      if( k == I ) {
	bL = L;
	bB = B;
      } else {
	bL = bufL1;
	bB = bufB1;
      }
      mybcast( bL, bs2s, k, comms[I] );
      mybcast( bB, bs2r, k, comms[I] );
      if( k < I )
      	localDgemmrp( C, bB, bL, bs, bsc );
      else if( k == I )
	localDtrmmr( C, bB, bL, bs, bsc );
    }
  } else {
    for( int k = 0; k < P; k++ ) {
      double *BL = B;
      double *BU = B+bs2r;
      double *CL = C;
      double *CU = C+bs2r;
      double *bL1, *bL2, *bB1, *bB2;

      if( J == k )
	bL1 = L;
      else
	bL1 = bufL1;
      mybcast( bL1, bs2s, k, comms[I] );
      
      if( k <= J ) {
	bL2 = bufL2;
	mybcast( bL2, bs2s, k, comms[J] );

	if( J == k )
	  bB1 = BL;
	else
	  bB1 = bufB1;
	mybcast( bB1, bs2r, k, comms[I] );

	if( k < J )
	  localDgemmrp( CL, bB1, bL2, bs, bsc );
	else
	  localDtrmmr( CL, bB1, bL2, bs, bsc );	  
      } else {
	bL2 = bufL2;
	mybcast( bL2, bs2s, k, comms[J] );
	bB1 = bufB1;
	mybcast( bB1, bs2r, k, comms[I] );
      }
      
      if( I == k )
	bB2 = BU;
      else
	bB2 = bufB2;
      mybcast( bB2, bs2r, k, comms[J] );


      if( k < I )
	localDgemmrp( CU, bB2, bL1, bs, bsc );
      else if( k == I )
	localDtrmmr( CU, bB2, bL1, bs, bsc );
	
    }
  }
  free(bufL1);
  free(bufL2);  
  free(bufB1);
  free(bufB2);  
}
