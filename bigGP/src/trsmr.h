#include "local.h"
#include <mpi.h>

void trsmr( double *X, double *L, int bs, int bsc, int I, int J, int P, MPI_Comm *comms );
