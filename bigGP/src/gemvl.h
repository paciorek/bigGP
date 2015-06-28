#include "local.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void gemvl( double *Xout, double *A, double *Xin, int bs, int n, int I, int J, int P, MPI_Comm *comms );
