#include "cholesky.h"
#include "chol.h"
#include "tstrsm.h"
#include "tssyrk.h"

/*
 Compute a blocked cholesky of a triangle, stored as fundamental units (triagular or square as appropriate) in column major order.  h is the number of fundamental units in a column/row
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of X.
int cholesky( double *L, int h, int bs, int n, int I, int J, int P, MPI_Comm *comms ) {
  int info1;
  int info2 = 0;
  int bs2 = bs*bs;
  int sbs2 = bs*bs;
  if( I != J )
    sbs2 = 2*sbs2;
  
  // compute the cholesky factorization of the first block  
  info1 = chol( L, bs, n, I, J, P, comms );

  if( h > 1 ) {
    // trsm of the first block column
    double *X = L+bs2;
    tstrsm( X, L, h-1, bs, I, J, P, comms );
    // syrk of the rest of the matrix
    double *Lnew = X+(h-1)*sbs2;
    tssyrk( X, Lnew, h-1, bs, I, J, P, comms );
    // cholesky of the rest of the matrix
    info2 = cholesky( Lnew, h-1, bs, n-P*bs, I, J, P, comms );
  }
  if( info1 )
    return info1;
  return info2;
}
