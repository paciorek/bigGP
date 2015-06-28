#include "local.h"
#include <mpi.h>
#include <stdlib.h>
void syrkr( double *L, double *M, int bs, int bsc, int I, int J, int P, MPI_Comm *comms );
