#include <stdio.h>

void dpotrf_( char*, int*, double*, int*, int* );
void dtrsm_( char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int* );
void dsyrk_( char*, char*, int*, int*, double*, double*, int*, double*, double*, int* );
void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );
void daxpy_( int*, double*, double*, int*, double*, int* );
void dgemv_( char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );
void dcopy_( int*, double*, int*, double*, int* );
void dtrmv_( char*, char*, char*, int*, double*, int*, double*, int* );
void dtrmm_( char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int* );
double ddot_( int*, double*, int*, double*, int* );

int localChol( double *A, int n, int ns );
void localTrsm( double *A, double *B, int n );
void localTrsmr( double *A, double *B, int n, int n2 );
void localSyrk( double *A, double *B, int n );
void localSyrkr( double *A, double *B, int n, int n2 );
void localDgemm( double *C, double*A, double *B, int n );
void localDgemmr( double *C, double*A, double *B, int n, int n2 );
void localDgemmrp( double *C, double*A, double *B, int n, int n2 );
void localDgemmrt( double *C, double*A, double *B, int n, int n2 );
void localDgemmrc( double *C, double*A, double *B, int n, int n2 );
void localGemv( double *A, double *Xin, double* Xout, int n );
void localGemvr( double *A, double *Xin, double* Xout, int n, int n2 );
void localGemvl( double *A, double *Xin, double* Xout, int n );
void localGemvl2( double *A, double *Xin, double* Xout, int n, int n2 );
void localAxpy( double *X, double *Y, int n );
void localAxpyp( double *X, double *Y, int n );
void localTrs( double *A, double *X, int n );
void localTrsl( double *A, double *X, int n, int n2 );
void localTrmv( double *A, double *X, double *Xout, int n );
void localDtrmmr( double *C, double *B, double *LL, int n, int n2 );
