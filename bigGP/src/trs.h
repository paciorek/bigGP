#include "local.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void trs( double *X, double *L, int bs, int I, int J, int P, MPI_Comm *comms );
