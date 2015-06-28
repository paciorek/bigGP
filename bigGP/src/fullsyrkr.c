#include "gemmrect.h"
#include "syrkr.h"

/*
  Compute A = M^t * M where M^t is stored as a full rectangular matrix, and A is stored as lower triangular (symmetric) matrix.  Note that M^t is m x n, and A is m x m.  Thus h2 and bsc apply to A.

  This is core task 7
 */

void fullsyrkr( double *A, double *Mt, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int rbs2 = bs*bsc;
  int sbs2 = bsc*bsc;
  int tbs2 = sbs2;
  if( I != J ) {
    rbs2 = 2*rbs2;
    sbs2 = 2*sbs2;
  }

  // zero A
  for( long long i = 0; i < sbs2*h2*(h2-1)/2+tbs2*h2; i++ )
    A[i] = 0.;
  
  for( int j = 0; j < h2 ; j++ ) {
    for( int i = j; i < h2; i++ ) {
      for( int k = 0; k < h; k++ ) {
	double *Mtik = Mt + i*rbs2 + k*h2*rbs2;
	if( i == j ) {
	  syrkr( A, Mtik, bs, bsc, I, J, P, comms );
	} else {
	  double *Mtjk = Mt + j*rbs2 + k*h2*rbs2;
	  gemmrect( A, Mtik, Mtjk, bs, bsc, I, J, P, comms );
	}
      }
      if( i == j )
	A += tbs2;
      else
	A += sbs2;
    }
  }
}
