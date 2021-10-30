#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "cholesky.h"
#include "fulltrs.h"
#include "fulltrsmr.h"
#include "fulltrsl.h"
#include "fullgemvr.h"
#include "fullsyrkr_diag.h"
#include "fullsyrkr.h"
#include "fulltrmv.h"
#include "fulltrmmr.h"
#include "collect.h"
#include "lib.h"
#include "zeropad.h"

MPI_Comm *comms;
MPI_Comm comm_world;
int initialized = 0;
int rank;

// Coretask 3a
SEXP cholesky_wrapper( SEXP RA, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *A = REAL(RA);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  int bs = (n+h*P-1)/(h*P);
  int info = cholesky( A, h, bs, n, I, J, P, comms );

  SEXP Rinfo;
  PROTECT(Rinfo = allocVector(INTSXP, 1));
  INTEGER(Rinfo)[0] = info;

  UNPROTECT(1);
  return Rinfo;
}

// Coretask 3b
SEXP forwardsolve_wrapper( SEXP RX, SEXP RL, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *X = REAL(RX);
  double *L = REAL(RL);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];

  int bs = (n+h*P-1)/(h*P);
  fulltrs( L, X, h, bs, I, J, P, comms );
  return R_NilValue;
}

// Coretask 3c
SEXP forwardsolve_matrix_wrapper( SEXP RM, SEXP RL, SEXP Rn, SEXP Rm, SEXP Rh, SEXP Rh2, SEXP RI, SEXP RJ, SEXP RP ) {
  double *M = REAL(RM);
  double *L = REAL(RL);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  int h2 = INTEGER(Rh2)[0];
  int m = INTEGER(Rm)[0];

  int bs = (n+h*P-1)/(h*P);
  int bsc = (m+h2*P-1)/(h2*P);

  fulltrsmr( L, M, h, h2, bs, bsc, I, J, P, comms );
  return R_NilValue;
}

// Coretask 4
SEXP backsolve_wrapper( SEXP RX, SEXP RL, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *X = REAL(RX);
  double *L = REAL(RL);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];

  int bs = (n+h*P-1)/(h*P);
  fulltrsl( L, X, h, h, bs, n, I, J, P, comms );
  return R_NilValue;
}

// Coretask 5
SEXP mult_cross_wrapper( SEXP RXout, SEXP RM, SEXP RXin, SEXP Rn, SEXP Rm, SEXP Rh, SEXP Rh2, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Xout = REAL(RXout);
  double *M = REAL(RM);
  double *Xin = REAL(RXin);
  int h = INTEGER(Rh)[0];
  int h2 = INTEGER(Rh2)[0];
  int n = INTEGER(Rn)[0];
  int m = INTEGER(Rm)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);
  int bsc = (m+h2*P-1)/(h2*P);

  zeroPadVector( Xin, n, h, bs, I, J, P );
  fullgemvr( Xout, M, Xin, h, h2, bs, bsc, I, J, P, comms );
  return R_NilValue;
}

// Coretask 6
SEXP cross_prod_self_diag_wrapper( SEXP RX, SEXP RM, SEXP Rn, SEXP Rm, SEXP Rh, SEXP Rh2, SEXP RI, SEXP RJ, SEXP RP ) {
  double *M = REAL(RM);
  double *X = REAL(RX);
  int h = INTEGER(Rh)[0];
  int h2 = INTEGER(Rh2)[0];
  int n = INTEGER(Rn)[0];
  int m = INTEGER(Rm)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);
  int bsc = (m+h2*P-1)/(h2*P);

  zeroPadMatrix( M, n, m, h, h2, bs, bsc, I, J, P );
  fullsyrkr_diag( M, X, h, h2, bs, bsc, I, J, P, comms );
  return R_NilValue;
}

// Coretask 7
SEXP cross_prod_self_wrapper( SEXP RMS, SEXP RM, SEXP Rn, SEXP Rm, SEXP Rh, SEXP Rh2, SEXP RI, SEXP RJ, SEXP RP ) {
  double *MS = REAL(RMS);
  double *M = REAL(RM);
  int h = INTEGER(Rh)[0];
  int h2 = INTEGER(Rh2)[0];
  int n = INTEGER(Rn)[0];
  int m = INTEGER(Rm)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);
  int bsc = (m+h2*P-1)/(h2*P);

  zeroPadMatrix( M, n, m, h, h2, bs, bsc, I, J, P );
  fullsyrkr( MS, M, h, h2, bs, bsc, I, J, P, comms );
  return R_NilValue;
}

// Coretask 9a
SEXP mult_chol_vector_wrapper( SEXP RXout, SEXP RL, SEXP RXin, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Xout = REAL(RXout);
  double *L = REAL(RL);
  double *Xin = REAL(RXin);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);

  zeroPadVector( Xin, n, h, bs, I, J, P );
  fulltrmv( Xout, L, Xin, h, bs, I, J, P, comms );
  return R_NilValue;
}

// Coretask 9b
SEXP mult_chol_matrix_wrapper( SEXP RMout, SEXP RL, SEXP RMin, SEXP Rn, SEXP Rm, SEXP Rh, SEXP Rh2, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Mout = REAL(RMout);
  double *L = REAL(RL);
  double *Min = REAL(RMin);
  int h = INTEGER(Rh)[0];
  int h2 = INTEGER(Rh2)[0];
  int n = INTEGER(Rn)[0];
  int m = INTEGER(Rm)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);
  int bsc = (m+h2*P-1)/(h2*P);

  zeroPadMatrix( Min, n, m, h, h2, bs, bsc, I, J, P );
  fulltrmmr( Mout, L, Min, h, h2, bs, bsc, I, J, P, comms );
  return R_NilValue;
}

// Coretasks 8 and 10 are currently not included.  They can be implemented trivially in R, or in C.

// Collect vectors or matrices to the master node.
SEXP collect_vector_wrapper( SEXP RXout, SEXP RX, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Xout = REAL(RXout);
  double *X = REAL(RX);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);

  collectFullVec( X, Xout, h, rank, P, I, J, bs, n, comm_world );
  return R_NilValue;
}

SEXP distribute_vector_wrapper( SEXP RXout, SEXP RX, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Xout = REAL(RXout);
  double *X = REAL(RX);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);

  distributeFullVec( X, Xout, h, rank, P, I, J, bs, n, comm_world );
  return R_NilValue;
}

SEXP collect_triangular_matrix_wrapper( SEXP RLout, SEXP RL, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Lout = REAL(RLout);
  double *L = REAL(RL);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);

  collectFullTri( L, Lout, h, rank, P, I, J, bs, n, comm_world );
  return R_NilValue;
}

SEXP collect_diagonal_wrapper( SEXP RXout, SEXP RL, SEXP Rn, SEXP Rh, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Xout = REAL(RXout);
  double *L = REAL(RL);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);
  collectFullDiag( L, Xout, h, rank, P, I, J, bs, n, comm_world );
  return R_NilValue;  
}

SEXP collect_rectangular_matrix_wrapper( SEXP RMout, SEXP RM, SEXP Rn, SEXP Rm, SEXP Rh, SEXP Rh2, SEXP RI, SEXP RJ, SEXP RP ) {
  double *Mout = REAL(RMout);
  double *M = REAL(RM);
  int h = INTEGER(Rh)[0];
  int n = INTEGER(Rn)[0];
  int h2 = INTEGER(Rh2)[0];
  int m = INTEGER(Rm)[0];
  int I = INTEGER(RI)[0];
  int J = INTEGER(RJ)[0];
  int P = INTEGER(RP)[0];
  
  int bs = (n+h*P-1)/(h*P);
  int bsc = (m+h2*P-1)/(h2*P);

  collectFullRect( M, Mout, h, h2, rank, P, I, J, bs, bsc, n, m, comm_world );
  return R_NilValue;
}

SEXP init_comms( SEXP Rcomm ) {
  if( initialized ) {
    SEXP Rinfo;
    PROTECT(Rinfo = allocVector(INTSXP, 1));
    INTEGER(Rinfo)[0] = -2;

    UNPROTECT(1);
    return Rinfo;
  }

  int fcomm = INTEGER(Rcomm)[0];

  comm_world = MPI_Comm_f2c(fcomm);
  MPI_Group initialGroup;
  MPI_Comm_group( comm_world, &initialGroup );
  int nproc;
  MPI_Comm_rank( comm_world, &rank );
  MPI_Comm_size( comm_world, &nproc );

  int P = (int)(-1+sqrt(1.+8*(nproc-1)))/2;
  if( ( nproc != P*(P+1)/2+1 ) ) {
    SEXP Rinfo;
    PROTECT(Rinfo = allocVector(INTSXP, 1));
    INTEGER(Rinfo)[0] = -1;

    UNPROTECT(1);
    return Rinfo;
  }
  comms = (MPI_Comm*) malloc( P*sizeof(MPI_Comm) );
  MPI_Group gp;
  for( int II = 0; II < P; II++ ) {
    int ranks[P];
    int r=0, i=II,j=0;
    for( ; j<=i; j++, r++ ) {
      ranks[r] = ProcRank(i,j,P)+1;
    }
    for( j=j-1, i=i+1; i<P; i++, r++ ) {
      ranks[r] = ProcRank(i,j,P)+1;
    }
    MPI_Group_incl( initialGroup, P, ranks, &gp );
    MPI_Comm_create( comm_world, gp, comms+II );
    MPI_Group_free( &gp );
  }
  MPI_Group_free( &initialGroup );
  initialized = 1;

  SEXP Rinfo;
  PROTECT(Rinfo = allocVector(INTSXP, 1));
  INTEGER(Rinfo)[0] = P;
  UNPROTECT(1);
  return Rinfo;
}

#define FUN(name, numArgs) \
  {#name, (DL_FUNC) &name, numArgs}

R_CallMethodDef CallEntries[] = {
  FUN(chol, 7),
  FUN(cholesky, 8),
  FUN(collectFullVec, 10),
  FUN(distributeFullVec, 10),
  FUN(collectFullDiag, 10),
  FUN(collectFullTri, 10),
  FUN(collectFullRect, 13),
  FUN(sendForward, 5),
  FUN(recvForward, 4),
  FUN(sendBackward, 5),
  FUN(recvBackward, 4),
  FUN(mybcast, 4),
  FUN(myreduce,5),
  FUN(mysend, 5),
  FUN(myrecv,5),
  FUN(reduceForward, 4),
  FUN(fullgemvr, 11),
  FUN(fullsyrkr, 10),
  FUN(fullsyrkr_diag, 10),
  FUN(fulltrmmr, 11),
  FUN(fulltrmv, 9),
  FUN(fulltrs, 8),
  FUN(fulltrsl, 10),
  FUN(fulltrsmr, 10),
  FUN(gemm, 8),
  FUN(gemmr, 9),
  FUN(gemv, 8),
  FUN(gemvl, 9),
  FUN(gemvr, 9),
  FUN(read_timer, 0),
  FUN(find_option, 3),
  FUN(read_int, 4),
  FUN(read_string, 4),
  FUN(ProcRank, 3),
  FUN(localGemv, 4),
  FUN(localGemvr, 5),
  FUN(localGemvl, 4),
  FUN(localGemvl2, 5),
  FUN(localTrmv, 4),
  FUN(localAxpy, 3),
  FUN(localAxpyp, 3),
  FUN(localChol, 3),
  FUN(localTrsm, 3),
  FUN(localTrsmr, 4),
  FUN(localTrs, 3),
  FUN(localTrsl, 4),
  FUN(localSyrk, 3),
  FUN(localSyrkr, 4),
  FUN(localDgemm, 4),
  FUN(localDgemmr, 5),
  FUN(localDgemmrp, 5),
  FUN(localDgemmrt, 5),
  FUN(localDgemmrc, 5),
  FUN(localDtrmmr, 5),
  FUN(syrk, 7),
  FUN(trmv, 8),
  FUN(trs, 7),
  FUN(trsl, 8),
  FUN(trsm, 7),
  FUN(trsmr, 8),
  FUN(tssyrk, 8),
  FUN(tstrsm, 8),
  FUN(cholesky_wrapper, 6),
  FUN(forwardsolve_wrapper, 7),
  FUN(forwardsolve_matrix_wrapper, 9),
  FUN(backsolve_wrapper, 7),
  FUN(mult_cross_wrapper, 10),
  FUN(cross_prod_self_diag_wrapper, 9),
  FUN(cross_prod_self_wrapper, 9),
  FUN(mult_chol_vector_wrapper, 8),
  FUN(mult_chol_matrix_wrapper, 10),
  FUN(collect_vector_wrapper, 7),
  FUN(distribute_vector_wrapper, 7),
  FUN(collect_triangular_matrix_wrapper, 7),
  FUN(collect_diagonal_wrapper, 7),
  FUN(collect_rectangular_matrix_wrapper, 9),
  FUN(init_comms,1),
  FUN(zeroPadVector, 7),
  FUN(zeroPadMatrix, 10),
  {NULL, NULL, 0}
};

void
R_init_bigGP(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
