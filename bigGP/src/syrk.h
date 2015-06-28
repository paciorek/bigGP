#include "local.h"
#include <mpi.h>
#include <stdlib.h>

void syrk( double *X, double *L, int bs, int I, int J, int P, MPI_Comm *comms );
