#include "lib.h"
#ifdef TIMERS
#include "timers.h"
#endif
#include "local.h"

void localGemv( double *A, double *Xin, double* Xout, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N';
  double one = 1.;
  double zero = 0.;
  int ione = 1;
  dgemv_( &N, &n, &n, &one, A, &n, Xin, &ione, &zero, Xout, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localGemvr( double *A, double *Xin, double* Xout, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N';
  double one = 1.;
  double zero = 0.;
  int ione = 1;
  dgemv_( &N, &n2, &n, &one, A, &n2, Xin, &ione, &zero, Xout, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localGemvl( double *A, double *Xin, double* Xout, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char T = 'T';
  double one = 1.;
  double zero = 0.;
  int ione = 1;
  dgemv_( &T, &n, &n, &one, A, &n, Xin, &ione, &zero, Xout, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// Xin has size n2, Xout has size n
void localGemvl2( double *A, double *Xin, double* Xout, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char T = 'T';
  double one = 1.;
  double zero = 0.;
  int ione = 1;
  dgemv_( &T, &n2, &n, &one, A, &n, Xin, &ione, &zero, Xout, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localTrmv( double *A, double *Xin, double *Xout, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  int ione = 1;
  char N = 'N', L = 'L';
  dcopy_( &n, Xin, &ione, Xout, &ione );
  dtrmv_( &L, &N, &N, &n, A, &n, Xout, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localAxpy( double *X, double *Y, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  double none = -1.;
  int ione = 1;
  daxpy_( &n, &none, X, &ione, Y, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localAxpyp( double *X, double *Y, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  double one = 1.;
  int ione = 1;
  daxpy_( &n, &one, X, &ione, Y, &ione );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

int localChol( double *A, int n, int ns ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char L = 'L';
  int info = 0;
  dpotrf_( &L, &ns, A, &n, &info );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
  return info;
}

void localTrsm( double *A, double *B, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char R = 'R', L = 'L', T = 'T', N = 'N';
  double one = 1.;
  dtrsm_( &R, &L, &T, &N, &n, &n, &one, A, &n, B, &n );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// trsm of an nxn triangle with an nxn2 rectangle
void localTrsmr( double *A, double *B, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char R = 'R', L = 'L', T = 'T', N = 'N';
  double one = 1.;
  dtrsm_( &R, &L, &T, &N, &n2, &n, &one, A, &n, B, &n2 );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localTrs( double *A, double *X, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char R = 'R', L = 'L', T = 'T', N = 'N';
  double one = 1.;
  int ione = 1;
  dtrsm_( &R, &L, &T, &N, &ione, &n, &one, A, &n, X, &ione );  
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// A and X are stored of size n, but all entries after n2 are garbage
void localTrsl( double *A, double *X, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char L = 'L', T = 'T', N = 'N';
  double one = 1.;
  int ione = 1;
  dtrsm_( &L, &L, &T, &N, &n2, &ione, &one, A, &n, X, &n );  
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

void localSyrk( double *A, double *B, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char L = 'L', N = 'N';
  double one = 1., none = -1.;
  dsyrk_( &L, &N, &n, &n, &none, A, &n, &one, B, &n );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// n2xn2 += n2xn x (n2xn)^t
void localSyrkr( double *A, double *B, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char L = 'L', N = 'N';
  double one = 1.;
  dsyrk_( &L, &N, &n2, &n, &one, A, &n2, &one, B, &n2 );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// nxn -= nxn x (nxn)^t
void localDgemm( double *C, double*A, double *B, int n ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N', T = 'T';
  double one = 1., none = -1.;
  dgemm_( &N, &T, &n, &n, &n, &none, A, &n, B, &n, &one, C, &n );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// n2xn -= n2xn x (nxn)^t
void localDgemmr( double *C, double*A, double *B, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N', T = 'T';
  double one = 1., none = -1.;
  dgemm_( &N, &T, &n2, &n, &n, &none, A, &n2, B, &n, &one, C, &n2 );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// n2xn += n2xn x (nxn)^t
void localDgemmrp( double *C, double*A, double *B, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N', T = 'T';
  double one = 1.;
  dgemm_( &N, &T, &n2, &n, &n, &one, A, &n2, B, &n, &one, C, &n2 );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// n2xn -= n2xn x nxn
void localDgemmrt( double *C, double*A, double *B, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N';
  double one = 1., none = -1.;
  dgemm_( &N, &N, &n2, &n, &n, &none, A, &n2, B, &n, &one, C, &n2 );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// n2xn2 += n2xn x (n2xn)^t
void localDgemmrc( double *C, double *A, double *B, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char N = 'N', T = 'T';
  double one = 1.;
  dgemm_( &N, &T, &n2, &n2, &n, &one, A, &n2, B, &n2, &one, C, &n2 );
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}

// rect^t += tr * rect^t
void localDtrmmr( double *C, double *B, double *LL, int n, int n2 ) {
#ifdef TIMERS
  startTimer(TIMER_COMP);
#endif
  char R = 'R', L = 'L', T = 'T', N = 'N';
  double one = 1.;
  int ione = 1;
  int nn2 = n*n2;
  double *buf = malloc( nn2*sizeof(double) );
  dcopy_( &nn2, B, &ione, buf, &ione );
  dtrmm_( &R, &L, &T,  &N, &n2, &n, &one, LL, &n, buf, &n2 );
  daxpy_( &nn2, &one, buf, &ione, C, &ione );
  free(buf);
#ifdef TIMERS
  stopTimer(TIMER_COMP);
#endif
}
