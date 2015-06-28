#include "fulltrsmr.h"

/*
 Compute a blocked trsm of a triangle and a rectangle (L * X_out^t = X_in^t), stored as fundamental units (triagular or rectangular as appropriate) in column major order.  h is the number of fundamental units in a column/row of the triangle and a column of the rectangle; h2 is the number of fundamental units in a row of the rectangle.

 Recall that X^t is stored, so the blocks are in row-major order, and each block is the corresponding block of X^t.
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of L.
void fulltrsmr( double *L, double *X, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms ) {
  int tbs2 = bs*bs;
  int sbs2 = bs*bs;
  int rbs2 = bs*bsc;
  if( I != J ) {
    sbs2 = 2*sbs2;
    rbs2 = 2*rbs2;
  }
  
  // compute the trsm of the first block row
  for( int j = 0; j < h2; j++ ) {
    trsmr( X+j*rbs2, L, bs, bsc, I, J, P, comms );
  }
  if( h > 1 ) {
    // update from the first block column
    L += tbs2;
    double *Lp = L;
    for( int j = 0; j < h2; j++ ) {
      Lp = L;
      double *Xp = X+rbs2*h2+j*rbs2;
      for( int i = 1; i < h; i++ ) {
	gemmr( Xp, X+j*rbs2, Lp, bs, bsc, I, J, P, comms );
	Lp += sbs2;
	Xp += rbs2*h2;
      }
    }
    // recursive call
    fulltrsmr( Lp, X+rbs2*h2, h-1, h2, bs, bsc, I, J, P, comms );
  }
}
