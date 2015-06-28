#include "local.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void gemv( double *Xout, double *A, double *Xin, int bs, int I, int J, int P, MPI_Comm *comms );
