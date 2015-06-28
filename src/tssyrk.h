#include "syrk.h"
#include "gemm.h"

void tssyrk( double *X, double *L, int h, int bs, int I, int J, int P, MPI_Comm *comms );
