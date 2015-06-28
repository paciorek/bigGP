#include <mpi.h>

void extractDiag( double *X, double *L, int h, int bs, int I, int J ) {
  if( I == J ) {
    int bs2 = bs*bs;
    for( int II = 0; II < h; II++ ) {
      for( int i = 0; i < bs; i++ )
	X[i] = L[i*bs+i];
      L += (h-II)*bs2;
      X += bs;
    }
  }
}
