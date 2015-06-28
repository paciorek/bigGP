#include "local.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void trsl( double *X, double *L, int bs, int n, int I, int J, int P, MPI_Comm *comms );
