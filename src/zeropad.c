#include<stdio.h>

// set all pad entries in a vector to zero
void zeroPadVector( double *X, int n, int h, int bs, int I, int J, int P ) {
  if( I == J ) {
    for( int hi = 0; hi < h; hi++ ) {
      int zerostart = n - bs*I;
      if( zerostart < 0 )
	zerostart = 0;
      for( int j = zerostart; j < bs; j++ )
	X[j] = 0.;
      n -= bs*P;
      X += bs;
    }
  }
}

void zeroPadMatrix( double *M, int n, int m, int h, int hm, int bs, int bsm, int I, int J, int P ) {
  int bs2 = bs*bsm;
  for( int hi = 0; hi < h; hi++ ) {
    for( int hmi = 0; hmi < hm; hmi++ ) {
      int nstart = hi*bs*P+J*bs;
      int mstart = hmi*bsm*P+I*bsm;
      if( (nstart + bs > n) || (mstart + bsm > m ) ) {
	for( int i = 0; i < bs; i++ )
	  for( int j = 0; j < bsm; j++ )
	    if( (nstart+i >= n) || (mstart+j >= m ) ) {
	      M[j+i*bsm] = 0.;
	    }
      }
      if( I != J ) {
	M += bs2;
	int nstart = hi*bs*P+I*bs;
	int mstart = hmi*bsm*P+J*bsm;
	if( (nstart + bs > n) || (mstart + bsm > m ) ) {
	  if( (nstart + bs > n) || (mstart + bsm > m ) ) {
	    for( int i = 0; i < bs; i++ )
	      for( int j = 0; j < bsm; j++ )
		if( (nstart+i >= n) || (mstart+j >= m ) ) {
		  M[j+i*bsm] = 0.;
		}
	  }
	}
      }
      M += bs2;
    }
  }
}
  
