#include "trsl.h"
#include "gemvl.h"

void fulltrsl( double *L, double *X, int h, int lh, int bs, int n, int I, int J, int P, MPI_Comm *comms );
