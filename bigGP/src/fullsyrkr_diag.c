#include "fullsyrkr_diag.h"
#include "comm.h"
#include "local.h"
#include <stdlib.h>

/*
  Compute the sums of the squares of the columns of a rectangular matrix.  The transpose of the matrix is what is stored.  The output is a vector X (with h2 pieces of size bsc per P), stored on the diagonal processors.
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of L.
void fullsyrkr_diag( double *M, double *X, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bsc;
  if( I != J ) {
    bs2 = 2*bs2;
    X = (double*) malloc( bsc*h2*sizeof(double) );
  }

  double *X2 = (double*) malloc( bsc*h2*sizeof(double) );
  double *Xp=X, *X2p=X2;

  for( int i = 0 ; i < bsc*h2; i++ ) {
    X[i] = 0.;
    X2[i] = 0.;
  }

  for( int i = 0; i < h2; i++ ) {
    for( int j = 0; j < h; j++ ) {
      double *Mp = M+i*bs2+j*h2*bs2;
      for( int ii = 0; ii < bsc; ii++ ) {
	if( I == J )
	  X2p[ii] += ddot_( &bs, Mp+ii, &bsc, Mp+ii, &bsc );
	else
	  Xp[ii] += ddot_( &bs, Mp+ii, &bsc, Mp+ii, &bsc );	  
      }
      if( I != J ) {
	Mp += bs*bsc;
	for( int ii = 0; ii < bsc; ii++ )
	  X2p[ii] += ddot_( &bs, Mp+ii, &bsc, Mp+ii, &bsc );
      }
    }
    Xp += bsc;
    X2p += bsc;
  }
  if( I == J ) {
    if( P > 1 )
      myreduce( X2, X, bsc*h2, I, comms[I] );
    else {
      int ione = 1;
      int tm= bsc*h2;
      dcopy_( &tm, X2, &ione, X, &ione );
    }
  }
  else {
    myreduce( X2, NULL, bsc*h2, J, comms[J] );
    myreduce( X, NULL, bsc*h2, I, comms[I] );
  }

  free(X2);
  if( I != J )
    free(X);
}
