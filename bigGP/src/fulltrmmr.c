#include "gemmrp.h"
#include "trmmr.h"
/*
  Compute Mout = L * Min whene M^t is stored as a full rectangular matrix, L is stored as a full triangular matrix.  This is core task 9b
 */

void fulltrmmr( double *Mtout, double *L, double *Mt, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int rbs2 = bs*bsc;
  int sbs2 = bs*bs;
  int tbs2 = sbs2;
  if( I != J ) {
    rbs2 = 2*rbs2;
    sbs2 = 2*sbs2;
  }

  // zero Mtout
  for( long long i = 0; i < rbs2*h*h2; i++ )
    Mtout[i] = 0.;

  double *Mtoutp = Mtout, *Mtp = Mt;
  for( int k = 0; k < h; k++ ) {
    Mtoutp = Mtout+k*h2*rbs2;
    for( int j = k; j < h; j++ ) {
      Mtp = Mt;
      for( int i = 0; i < h2; i++ ) {
	if( j == k )
	  trmmr( Mtoutp, L, Mtp, bs, bsc, I, J, P, comms );
	else
	  gemmrp( Mtoutp, Mtp, L, bs, bsc, I, J, P, comms );
	Mtoutp += rbs2;
	Mtp += rbs2;
      }
      if( j == k )
	L += tbs2;
      else
	L += sbs2;
    }
    Mt += h2*rbs2;
  }
}
