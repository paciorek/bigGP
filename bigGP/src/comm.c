#include "lib.h"
#ifdef TIMERS
#include "timers.h"
#endif
#include "comm.h"
#include "local.h"

int branchFactor = 5;

void setBF( int bf ) {
  branchFactor = bf;
}

void sendForward( double *msg, int size, int rank, int nproc, MPI_Comm comm ) {
#ifdef TIMERS
  startTimer( TIMER_SEND );
#endif
  // simple, inefficient broadcast
  for( int t = rank+1; t < nproc; t++ ) {
    MPI_Send( msg, size, MPI_DOUBLE, t, TAG_FORWARD, comm );
  }
  /* tree-based broadcast
  for( int t = rank+1; t <= rank+branchFactor && t < nproc; t++ ) {
    MPI_Send( msg, size, MPI_DOUBLE, t, TAG_FORWARD, comm );
  }
  */
#ifdef TIMERS
  stopTimer( TIMER_SEND );
#endif
}
void recvForward( double *msg, int size, int root, MPI_Comm comm ) {
#ifdef TIMERS
  startTimer( TIMER_RECV );
#endif
  // simple, inefficient broadcast
  MPI_Recv( msg, size, MPI_DOUBLE, root, TAG_FORWARD, comm, MPI_STATUS_IGNORE );
  /* tree-based broadcast
  int nproc, rank;
  MPI_Comm_rank( comm, &rank );
  MPI_Comm_size( comm, &nproc );
  int source = (rank-root-1)/branchFactor+root;
  MPI_Recv( msg, size, MPI_DOUBLE, source, TAG_FORWARD, comm, MPI_STATUS_IGNORE );
  for( int m = 1; m <= branchFactor; m++ ) {
    int target = (rank-root)*branchFactor+root+m;
    if( target >= nproc )
      break;
    MPI_Send( msg, size, MPI_DOUBLE, target, TAG_FORWARD, comm );
  }
  */
#ifdef TIMERS
  stopTimer( TIMER_RECV );
#endif
}

void sendBackward( double *msg, int size, int rank, int nproc, MPI_Comm comm ) {
#ifdef TIMERS
  startTimer( TIMER_SEND );
#endif
  for( int t = rank-1; t >= 0; t-- ) {
    MPI_Send( msg, size, MPI_DOUBLE, t, TAG_BACKWARD, comm );
  }
#ifdef TIMERS
  stopTimer( TIMER_SEND );
#endif
}
void recvBackward( double *msg, int size, int source, MPI_Comm comm ) {
#ifdef TIMERS
  startTimer( TIMER_RECV );
#endif
  MPI_Recv( msg, size, MPI_DOUBLE, source, TAG_BACKWARD, comm, MPI_STATUS_IGNORE );
#ifdef TIMERS
  stopTimer( TIMER_RECV );
#endif
}

void mybcast( double *msg, int size, int source, MPI_Comm comm ) {
#ifdef TIMERS
  int rank;
  MPI_Comm_rank( comm, &rank );
  if( rank == source )
    startTimer( TIMER_SEND );
  else
    startTimer( TIMER_RECV );
#endif
  MPI_Bcast( msg, size, MPI_DOUBLE, source, comm );
#ifdef TIMERS
  if( rank == source )
    stopTimer( TIMER_SEND );
  else
    stopTimer( TIMER_RECV );
#endif
}

void myreduce( double *in, double *out, int size, int source, MPI_Comm comm ) {
#ifdef TIMERS
  int rank;
  MPI_Comm_rank( comm, &rank );
  if( rank == source )
    startTimer( TIMER_RECV );
  else
    startTimer( TIMER_SEND );
#endif
  MPI_Reduce( in, out, size, MPI_DOUBLE, MPI_SUM, source, comm );
#ifdef TIMERS
  if( rank == source )
    stopTimer( TIMER_RECV );
  else
    stopTimer( TIMER_SEND );
#endif
}

void mysend( double *msg, int size, int target, int tag, MPI_Comm comm ) {
#ifdef TIMERS
  startTimer( TIMER_SEND );
#endif
  MPI_Send( msg, size, MPI_DOUBLE, target, tag, comm );
#ifdef TIMERS
  stopTimer( TIMER_SEND );
#endif
}

void myrecv( double *msg, int size, int target, int tag, MPI_Comm comm ) {
#ifdef TIMERS
  startTimer( TIMER_RECV );
#endif
  MPI_Recv( msg, size, MPI_DOUBLE, target, tag, comm, MPI_STATUS_IGNORE );
#ifdef TIMERS
  stopTimer( TIMER_RECV );
#endif
}

void reduceForward( double *msg, int size, int source, MPI_Comm comm ) {
  int rank;
  MPI_Comm_rank( comm, &rank );
  if( rank == source ) {
#ifdef TIMERS
    startTimer( TIMER_RECV );
#endif
    double *buf = (double*) malloc( size*sizeof(double) );
    for( int i = 0; i < source; i++ ) {
      MPI_Recv( buf, size, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_RF, comm, MPI_STATUS_IGNORE );
      localAxpyp( buf, msg, size );
    }
    free(buf);
#ifdef TIMERS
    stopTimer( TIMER_RECV );
#endif
  } else {
#ifdef TIMERS
    startTimer( TIMER_SEND );
#endif
    MPI_Send( msg, size, MPI_DOUBLE, source, TAG_RF, comm );
#ifdef TIMERS
    stopTimer( TIMER_SEND );
#endif
  }
}
