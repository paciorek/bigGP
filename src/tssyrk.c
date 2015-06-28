#include "syrk.h"
#include "gemm.h"
#include "tssyrk.h"

/*
 Compute a blocked SYRK of a stack of h fundamental square units on a triangle of the corresponding size.  The triangle is stored as fundamental units (triagular or square as appropriate) in column major order.
 */

// comms is an array of communicators of length P.  Only the I and J entries will be valid.  They are the communicators along the given row/column of X.
void tssyrk( double *X, double *L, int h, int bs, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;
  int sbs2 = bs*bs;
  if( I != J )
    sbs2 = 2*sbs2;
  double *Xpi = X;
  double *Xpj = X;
  for( int j = 0; j < h; j++ ) {
    Xpj = X+j*sbs2;
    for( int i = j; i < h; i++ ) {
      Xpi = X+i*sbs2;
      if( i == j ) {
	syrk( Xpi, L, bs, I, J, P, comms );
      } else {
	gemm( L, Xpi, Xpj, bs, I, J, P, comms ); 
      }
      // Increment L;
      if( i == j )
	L += bs2;
      else
	L += sbs2;
    }
  }
}
