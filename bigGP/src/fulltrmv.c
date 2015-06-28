#include "fulltrmv.h"
#include "trmv.h"
#include "gemv.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

/*
 Compute a blocked triangular MV on the h fundamental units
 */

// comms is an array of communicators of length P.  Only the I and J entries are required to be valid.  They are the communicators along the given row/column of A.
void fulltrmv( double *Xout, double *A, double *Xin, int h, int bs, int I, int J, int P, MPI_Comm *comms ) {
  if( I == J )
    for( int i = 0; i < h*bs; i++ )
      Xout[i] = 0.;

  double *tmp = malloc( bs*sizeof(double) );
  int bs2 = bs*bs;
  int sbs2 = bs2;
  if( I != J )
    sbs2 = 2*bs2;

  for( int j = 0; j < h; j++ ) {
    for( int i = j; i < h; i++ ) {
      if( i == j ) {
	trmv( tmp, A, Xin, bs, I, J, P, comms );
	if( I == J )
	  localAxpyp( tmp, Xout+bs*i, bs );
	A += bs2;
      } else {
	gemv( tmp, A, Xin, bs, I, J, P, comms );
	if( I == J )
	  localAxpyp( tmp, Xout+bs*i, bs );
	A += sbs2;
      }
    }
    Xin += bs;
  }
  free(tmp);
}
