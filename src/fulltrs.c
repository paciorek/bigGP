#include "fulltrs.h"

/*
 Compute a blocked trs of a triangle (L * X_out = X_in), stored as fundamental units (triagular or square as appropriate) in column major order.  h is the number of fundamental units in a column/row
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of L.
void fulltrs( double *L, double *X, int h, int bs, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;
  int sbs2 = bs*bs;
  if( I != J )
    sbs2 = 2*sbs2;
  
  // compute the trs of the first block  
  trs( X, L, bs, I, J, P, comms );

  if( h > 1 ) {
    // update from the first block column
    double *Xp = X+bs;
    L += bs2;
    double *Xbuf = (double*) malloc( bs*sizeof(double) );
    for( int i = 1; i < h; i++ ) {
      gemv( Xbuf, L, X, bs, I, J, P, comms );
      if( I == J )
	localAxpy( Xbuf, Xp, bs );
      L += sbs2;
      Xp += bs;
    }
    free( Xbuf );
    // recursive call
    fulltrs( L, X+bs, h-1, bs, I, J, P, comms );
  }
}
