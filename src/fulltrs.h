#include "trs.h"
#include "gemv.h"

void fulltrs( double *L, double *X, int h, int bs, int I, int J, int P, MPI_Comm *comms );
