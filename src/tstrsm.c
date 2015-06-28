#include "trsm.h"
#include "tstrsm.h"
#include <stdlib.h>

/*
 Compute a blocked TRSM of the fundamental triangular unit on a stack of h fundamental square units.
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of X.
void tstrsm( double *X, double *L, int h, int bs, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;

  for( int i = 0; i < h; i++ ) {
    trsm( X, L, bs, I, J, P, comms );
    if( I == J ) // Diagonal processor
      X += bs2;
    else // off-diagonal processor
      X += 2*bs2;
  }
}
