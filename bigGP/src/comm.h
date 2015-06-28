#define TAG_BACKWARD 100
#define TAG_FORWARD 101
#define TAG_DIRECT 102
#define TAG_RF 103

#ifndef COMM_H
#define COMM_H

#include <mpi.h>

void setBF( int bf );

void sendForward( double *msg, int size, int rank, int nproc, MPI_Comm comm );
void recvForward( double *msg, int size, int source, MPI_Comm comm );
void sendBackward( double *msg, int size, int rank, int nproc, MPI_Comm comm );
void recvBackward( double *msg, int size, int source, MPI_Comm comm );
void mybcast( double *msg, int size, int source, MPI_Comm comm );
void myreduce( double *in, double *out, int size, int source, MPI_Comm comm );
void mysend( double *msg, int size, int target, int tag, MPI_Comm comm );
void myrecv( double *msg, int size, int target, int tag, MPI_Comm comm );
void reduceForward( double *msg, int size, int source, MPI_Comm comm );

#endif
