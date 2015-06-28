#include "local.h"
#include "chol.h"
#include "comm.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*                                                                                                                 
 Compute a blocked GEMM on the fundamental unit.  Each processor owns one block of L.  Eg. on 10 processors
      0
  L = 1 4
      2 5 7
      3 6 8 9

 I and J identify the processor row and column, and P is the partition number, 4 in the above example.

 If n is less than bs*P, only the leading nxn block is expected to be PSD
*/

int chol( double* A, int bs, int n, int I, int J, int P, MPI_Comm *comms ) {
  int bs2 = bs*bs;
  int info = 0;

  if ( I == J ) {
    double *M = (double*) malloc( bs2*sizeof(double) );
    for( int k = 0; k < I; k++ ) {
      recvForward( M, bs2, k, comms[I]);
      localSyrk( M, A, bs );
    }
    free(M);
    int bss = n-I*bs;
    if( bss < bs )
      info = localChol(A,bs,n-I*bs);
    else
      info = localChol(A,bs,bs);
    sendForward( A, bs2, I, P, comms[I] );
  } else {
    double *matjk = (double*) malloc( bs2*sizeof(double) );
    double *matik = (double*) malloc( bs2*sizeof(double) );
    for( int k = 0; k < J; k++ ) {
      recvForward( matjk, bs2, k, comms[J] );
      recvForward( matik, bs2, k, comms[I] );
      localDgemm( A, matik, matjk, bs );
    }
    free(matjk);
    double *M = matik;
    recvForward( M, bs2, J, comms[J] );
    localTrsm( M, A, bs );
    free(M);
    sendForward( A, bs2, J, P, comms[I] );
  }
  return info;
}
