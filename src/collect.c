#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "lib.h"
#include "transform.h"

#define min(a,b) (((a)<(b))?(a):(b))
// collect a vector of length n to the master process.  It is distributed with block size bs=(n+P-1)/P
void collectVec( double *XDist, double *X, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  if( rank != 0 ) {
    if( II == JJ )
      MPI_Send( XDist, bs, MPI_DOUBLE, 0, 0, comm_world );
  } else {
    double *buf = (double*) malloc( bs*sizeof(double) );
    for( int I = 0; I < P; I++ ) {
      MPI_Recv( buf, bs, MPI_DOUBLE, 1+ProcRank(I,I,P), 0, comm_world, MPI_STATUS_IGNORE );
      for( int i = 0; i < bs && I*bs+i<n; i++ )
	X[I*bs+i] = buf[i];
    }
    free(buf);
  }
}

void collectFullVec( double *Xdist, double *X, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  for( int j = 0; j < h && n > 0; j++ ) {
    collectVec( Xdist, X, rank, P, II, JJ, bs, min(n,bs*P), comm_world );
    Xdist += bs;
    X += bs*P;
    n -= bs*P;
  }
}

void distributeVec( double *X, double *XDist, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  if( rank != 0 ) {
    if( II == JJ )
      MPI_Recv( XDist, bs, MPI_DOUBLE, 0, 101, comm_world, MPI_STATUS_IGNORE );
  } else {
    double *buf = (double*) malloc( bs*sizeof(double) );
    for( int I = 0; I < P; I++ ) {
      for( int i = 0; i < bs && I*bs+i<n; i++ )
	buf[i] = X[I*bs+i];
      for( int i = n-I*bs; i < bs; i++ )
	buf[i] = 0;
      MPI_Send( buf, bs, MPI_DOUBLE, 1+ProcRank(I,I,P), 101, comm_world );
    }
    free(buf);
  }
}

void distributeFullVec( double *X, double *Xdist, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  for( int j = 0; j < h && n > 0; j++ ) {
    distributeVec( X, Xdist, rank, P, II, JJ, bs, min(n,bs*P), comm_world );
    Xdist += bs;
    X += bs*P;
    n -= bs*P;
  }
}

void collectFullDiag( double *L, double *X, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  double *Xdist = (double*) malloc( sizeof(double)*h*bs );
  if( rank != 0 )
    extractDiag( Xdist, L, h, bs, II, JJ );
  collectFullVec( Xdist, X, h, rank, P, II, JJ, bs, n, comm_world );
  free(Xdist);
}

void collectTri( double *XDist, double *X, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  if( rank != 0 ) {
    MPI_Send( XDist, bs*bs, MPI_DOUBLE, 0, 0, comm_world );
  } else {
    double *buf = (double*) malloc( bs*bs*sizeof(double) );
    int src = 0;
    for( int J = 0; J < P; J++ ) {
      for( int I = 0; I < P; I++ ) {
	if( J > I )
	  continue;
	src++;
	MPI_Recv( buf, bs*bs, MPI_DOUBLE, src, 0, comm_world, MPI_STATUS_IGNORE );
	for( int i = 0; i < bs && I*bs+i < n; i++ )
	  for( int j = 0; j < bs && J*bs+j < n; j++ )
	    X[I*bs+i+(J*bs+j)*n] = buf[i+j*bs];
      }
    }
    free(buf);
  }
}

void collectSquare( double *XDist, double *X, int rank, int P, int II, int JJ, int bs, int ni, int nj, MPI_Comm comm_world ) {
  if( rank != 0 ) {
    MPI_Send( XDist, bs*bs, MPI_DOUBLE, 0, 0, comm_world );
    if( II != JJ )
      MPI_Send( XDist+bs*bs, bs*bs, MPI_DOUBLE, 0, 0, comm_world );
  } else {
    double *buf = (double*) malloc( bs*bs*sizeof(double) );
    int src = 0;
    for( int J = 0; J < P; J++ ) {
      for( int I = 0; I < P; I++ ) {
	if( J > I )
	  continue;
	src++;
	MPI_Recv( buf, bs*bs, MPI_DOUBLE, src, 0, comm_world, MPI_STATUS_IGNORE );

	for( int i = 0; i < bs && I*bs+i < ni; i++ )
	  for( int j = 0; j < bs && J*bs+j < nj; j++ )
	    X[I*bs+i+(J*bs+j)*ni] = buf[i+j*bs];
	
	if( I != J ) {
	  MPI_Recv( buf, bs*bs, MPI_DOUBLE, src, 0, comm_world, MPI_STATUS_IGNORE );
	  for( int i = 0; i < bs && J*bs+i < ni; i++ )
	    for( int j = 0; j < bs && I*bs+j < nj; j++ )
	      X[J*bs+i+(I*bs+j)*ni] = buf[i+j*bs];	  
	}
      }
    }
    free(buf);
  }
}

void collectFullTri( double *XDist, double *X, int h, int rank, int P, int II, int JJ, int bs, int n, MPI_Comm comm_world ) {
  double *buf = (double*) malloc( bs*bs*P*P*sizeof(double) );
  for( int j = 0; j < h ; j++ ) {
    for( int i = j; i < h; i++ )
      if( i == j ) {
	int ldb = min(bs*P,n-bs*P*i);
        collectTri( XDist, buf, rank, P, II, JJ, bs, ldb, comm_world );
	if( rank == 0 ) {
	  for( int ii = 0; ii < bs*P && i*bs*P+ii < n; ii++ )
	    for( int jj = 0; jj <= ii && j*bs*P+jj < n; jj++ ) {
	      X[i*bs*P+ii+(j*bs*P+jj)*n] = buf[ii+jj*ldb];
	    }
	}
        XDist += bs*bs;
      } else {
	int ldb = min(bs*P,n-bs*P*i);
        collectSquare( XDist, buf, rank, P, II, JJ, bs, ldb, min(bs*P,n-bs*P*j), comm_world );
	if( rank == 0 ) {
	  for( int ii = 0; ii < bs*P && i*bs*P+ii < n; ii++ )
	    for( int jj = 0; jj < bs*P && j*bs*P+jj < n; jj++ ) {
	      X[i*bs*P+ii+(j*bs*P+jj)*n] = buf[ii+jj*ldb];
	    }
	}
        if( II == JJ )
          XDist += bs*bs;
        else
          XDist += bs*bs*2;
      }

  }
  free(buf);
}

void collectRect( double *XDist, double *X, int rank, int P, int II, int JJ, int bs, int bs2, int ni, int nj, MPI_Comm comm_world ) {
  if( rank != 0 ) {
    MPI_Send( XDist, bs*bs2, MPI_DOUBLE, 0, 0, comm_world );
    if( II != JJ )
      MPI_Send( XDist+bs*bs2, bs*bs2, MPI_DOUBLE, 0, 0, comm_world );
  } else {
    double *buf = (double*) malloc( bs*bs2*sizeof(double) );
    int src = 0;
    for( int J = 0; J < P; J++ ) {
      for( int I = 0; I < P; I++ ) {
	if( J > I )
          continue;
	src++;
	MPI_Recv( buf, bs*bs2, MPI_DOUBLE, src, 0, comm_world, MPI_STATUS_IGNORE );

        for( int i = 0; i < bs2 && I*bs2+i < ni; i++ )
          for( int j = 0; j < bs && J*bs+j < nj; j++ )
            X[I*bs2+i+(J*bs+j)*ni] = buf[i+j*bs2];

        if( I != J ) {
          MPI_Recv( buf, bs*bs2, MPI_DOUBLE, src, 0, comm_world, MPI_STATUS_IGNORE );
          for( int i = 0; i < bs2 && J*bs2+i < ni; i++ )
            for( int j = 0; j < bs && I*bs+j < nj; j++ )
              X[J*bs2+i+(I*bs+j)*ni] = buf[i+j*bs2];
        }
      }
    }
    free(buf);
  }
}

void collectFullRect( double *XDist, double *X, int h, int h2, int rank, int P, int II, int JJ, int bs, int bs2, int n, int m, MPI_Comm comm_world ) {
  double *buf = (double*) malloc( bs*bs2*P*P*sizeof(double) );
  for( int j = 0; j < h ; j++ ) {
    for( int i = 0; i < h2; i++ ) {
      int ldb = min(m-i*bs2*P,bs2*P);
      collectRect( XDist, buf, rank, P, II, JJ, bs, bs2, ldb, min(n-j*bs*P,bs*P), comm_world );
      if( rank == 0 ) {
        for( int ii = 0; ii < bs2*P && i*bs2*P+ii < m; ii++ )
          for( int jj = 0; jj < bs*P && j*bs*P+jj < n; jj++ ) {
            X[i*bs2*P+ii+(j*bs*P+jj)*m] = buf[ii+jj*ldb];
          }
      }
      if( II == JJ )
        XDist += bs*bs2;
      else
        XDist += bs*bs2*2;
    }

  }
  free(buf);
}
