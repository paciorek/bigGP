#include "local.h"
#include <mpi.h>
#include <stdlib.h>

void trmmr( double *C, double *L, double *B, int bs, int bsc, int I, int J, int P, MPI_Comm *comms );
