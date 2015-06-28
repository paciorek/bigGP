#include "gemmrect.h"
/* 
   Deprecated.  This function should never be called.
 */

/*
  Compute A = M^t * M where M^t is stored as a full rectangular matrix, and A is stored as a full square matrix.

  This is core task 7
 */

void fullgemmrect( double *A, double *Mt, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int rbs2 = bs*bsc;
  int sbs2 = bs*bs;
  if( I != J ) {
    rbs2 = 2*rbs2;
    sbs2 = 2*sbs2;
  }

  // zero A
  for( long long i = 0; i < sbs2*h*h; i++ )
    A[i] = 0.;
  
  for( int i = 0; i < h; i++ ) {
    for( int j = 0; j < h; j++ ) {
      for( int k = 0; k < h2; k++ ) {
	double *Aij = A + i*sbs2 + j*h*sbs2;
	double *Mtik = Mt + i*rbs2 + k*h*rbs2;
	double *Mtjk = Mt + j*rbs2 + k*h*rbs2;
	gemmrect( Aij, Mtik, Mtjk, bs, bsc, I, J, P, comms );
      }
    }
  }
}
