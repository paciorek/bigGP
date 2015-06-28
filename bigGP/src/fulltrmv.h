#include "trmv.h"
#include "comm.h"
#include <mpi.h>
#include <stdlib.h>

void fulltrmv( double *Xout, double *A, double *Xin, int h, int bs, int I, int J, int P, MPI_Comm *comms );
