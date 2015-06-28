#include "trsmr.h"
#include "gemmr.h"

void fulltrsmr( double *L, double *X, int h, int h2, int bs, int bsc, int I, int J, int P, MPI_Comm *comms );
