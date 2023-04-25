//#define TIMERS
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

double read_timer(void);
int find_option( int argc, char **argv, const char *option );
int read_int( int argc, char **argv, const char *option, int default_value );
char *read_string( int argc, char **argv, const char *option, char *default_value );
int ProcRank( int i, int j, int P );

