#include "fullgemvr.h"
#include "gemvr.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked GEMV on the h x h2 fundamental units
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of A.
void fullgemvr( double *Xout, double *A, double *Xin, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  if( I == J )
    for( int i = 0; i < h2*bsc; i++ )
      Xout[i] = 0.;

  double *tmp = malloc( bsc*sizeof(double) );
  int bs2 = bs*bsc;
  if( I != J )
    bs2 = 2*bs2;

  for( int j = 0; j < h; j++ ) {
    for( int i = 0; i < h2; i++ ) {
      gemvr( tmp, A, Xin, bs, bsc, I, J, P, comms );
      if( I == J )
	localAxpyp( tmp, Xout+bsc*i, bsc );
      A += bs2;
    }
    Xin += bs;
  }
  free(tmp);
}
