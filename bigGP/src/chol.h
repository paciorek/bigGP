#include "local.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int chol( double* A, int bs, int n, int I, int J, int P, MPI_Comm *comms );
