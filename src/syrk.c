#include "local.h"
#include "syrk.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked SYRK on the fundamental unit.  Each diagonal processor owns one block of L and one of X.  Each off-diagonal processor owns one block of L and two of X.  Eg. on 10 processors

      0
  L = 1 4 
      2 5 7
      3 6 8 9

      0 1 2 3
  X = 1 4 5 6
      2 5 7 8
      3 6 8 9

 Two two blocks of X are stored contiguously, in column major order.  That is, the first block is the one in the lower triangle, and the second is the one in the upper triangle.

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of X.
void syrk( double *X, double *L, int bs, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;

  double *bufX1 = (double*) malloc( bs2*sizeof(double) );
  double *bufX2 = (double*) malloc( bs2*sizeof(double) );


  if( I == J ) {
    for( int k = 0; k < P; k++ ) {
      double *buf;
      if( k == I )
	buf = X;
      else
	buf = bufX1;
      mybcast( buf, bs2, k, comms[I] );
      localSyrk( buf, L, bs );
    }
  } else {
    for( int k = 0; k < P; k++ ) {
      double *XL = X;
      double *XU = X+bs2;
      double *buf1, *buf2;
      if( J == k )
	buf1 = XL;
      else
	buf1 = bufX1;
      mybcast( buf1, bs2, k, comms[I] );
      
      if( I == k )
	buf2 = XU;
      else
	buf2 = bufX2;
      mybcast( buf2, bs2, k, comms[J] );
      
      localDgemm( L, buf1, buf2, bs );
    }
  }
  free(bufX1);
  free(bufX2);  
}
