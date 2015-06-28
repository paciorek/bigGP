#include "local.h"
#include <mpi.h>
#include <stdlib.h>

void gemmrect( double *C, double *A, double *B, int bs, int bsc, int I, int J, int P, MPI_Comm *comms );
