#include "local.h"
#include <mpi.h>

void trsm( double *X, double *L, int bs, int I, int J, int P, MPI_Comm *comms );
