#include "local.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void gemvr( double *Xout, double *A, double *Xin, int bs, int bs2, int I, int J, int P, MPI_Comm *comms );
