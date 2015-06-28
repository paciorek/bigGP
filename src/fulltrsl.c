#include "fulltrsl.h"

/*
 Compute a blocked trs of a triangle (L * X_out = X_in), stored as fundamental units (triagular or square as appropriate) in column major order.  h is the number of fundamental units in a column/row
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of L.
void fulltrsl( double *L, double *X, int h, int lh, int bs, int n, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;
  int sbs2 = bs*bs;
  if( I != J )
    sbs2 = 2*sbs2;
  
  // compute the trsl of the last block  
  trsl( X+(h-1)*bs, L+(h-1)*bs2+((lh*(lh-1)-(lh-h+1)*(lh-h))/2)*sbs2, bs, n-bs*P*(h-1), I, J, P, comms );

  if( h > 1 ) {
    // update from the last block row
    double *Xp = X;
    double *Lp = L-(lh-h+1)*sbs2;
    double *Xbuf = (double*) malloc( bs*sizeof(double) );
    for( int i = 0; i < h-1; i++ ) {
      Lp += (lh-i-1)*sbs2+bs2;
      gemvl( Xbuf, Lp, X+(h-1)*bs, bs, n-bs*P*(h-1), I, J, P, comms );
      if( I == J )
	localAxpy( Xbuf, Xp, bs );
      Xp += bs;
    }
    free( Xbuf );
    // recursive call
    fulltrsl( L, X, h-1, lh, bs, n, I, J, P, comms );
  }
}
