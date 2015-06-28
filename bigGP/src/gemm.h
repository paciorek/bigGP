#include "local.h"
#include <mpi.h>
#include <stdlib.h>

void gemm( double *C, double *A, double *B, int bs, int I, int J, int P, MPI_Comm *comms );
