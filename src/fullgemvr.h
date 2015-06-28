#include "gemvr.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void fullgemvr( double *Xout, double *A, double *Xin, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms );
