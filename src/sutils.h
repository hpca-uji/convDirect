#include "dtypes.h"

int    print_tensor4D( char *, int, int, int, int, DTYPE *, int, int, int );
int    print_tensor4D_with_padding( char *, int, int, int, int, int, int, DTYPE *, int, int, int );
int    print_tensor5D( char *, int, int, int, int, int, DTYPE *, int, int, int, int );
int    print_tensor6D( char *, int, int, int, int, int, int, DTYPE *, int, int, int, int, int );
int    print_matrix( char *, char, int, int, DTYPE *, int );
int    generate_tensor4D( int, int, int, int, DTYPE *, int, int, int );
double dclock();
