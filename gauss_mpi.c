#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
#include <string.h>
#include <mpi.h>

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}

double **matrix,*B,*V,*C;
int *swap;

int rank, nprocs;
int NSIZE = 128;

MPI_Status status;

/* Allocate the needed arrays */
void allocate_memory() {
    double *tmp;
    int i;

    matrix = (double**)malloc(NSIZE*sizeof(double*));
    assert(matrix != NULL);
    tmp = (double*)malloc(NSIZE*NSIZE*sizeof(double));
    assert(tmp != NULL);

    for(i = 0; i < NSIZE; i++){
        matrix[i] = tmp;
        tmp = tmp + NSIZE;
    }

    B = (double*)malloc(NSIZE * sizeof(double));
    assert(B != NULL);
    V = (double*)malloc(NSIZE * sizeof(double));
    assert(V != NULL);
    C = (double*)malloc(NSIZE * sizeof(double));
    assert(C != NULL);
    swap = (int*)malloc(NSIZE * sizeof(int));
    assert(swap != NULL);
}


/* Initialize the matrix */
void initMatrix() {
    int i,j;
    for(i = 0 ; i < NSIZE; i++){
        for(j = 0; j < NSIZE; j++) {
            matrix[i][j] = ((j < i ) ? 2*(j+1) : 2*(i+1));
        }
        B[i] = (double) i;
        swap[i] = i;
    }
}

/* Get the pivot row. If the value in the current pivot position is 0,
 * try to swap with a non-zero row. If that is not possible bail
 * out. Otherwise, make sure the pivot value is 1.0, and return. */

void getPivot(int currow) {
    int i, irow;
    double big;
    double tmp;

    big = matrix[currow][currow];
    irow = currow;

    if (big == 0.0) {
        for(i = currow ; i < NSIZE; i++){
            tmp = matrix[i][currow];
            if (tmp != 0.0){
                big = tmp;
                irow = i;
                break;
            }
        }
    }

    if (big == 0.0){
        printf("The matrix is singular\n");
        exit(-1);
    }

    if (irow != currow){
        for(i = currow; i < NSIZE ; i++){
            SWAP(matrix[irow][i],matrix[currow][i]);
        }
        SWAP(B[irow],B[currow]);
        SWAPINT(swap[irow],swap[currow]);
    }

    {
        double pivotVal;
        pivotVal = matrix[currow][currow];

        if (pivotVal != 1.0){
            matrix[currow][currow] = 1.0;
            for(i = currow + 1; i < NSIZE; i++){
                matrix[currow][i] /= pivotVal;
            }
            B[currow] /= pivotVal;
        }
    }
}

/* Linear Solver */
void solveGauss() {
    int i,j;

    V[NSIZE - 1] = B[NSIZE - 1];
    for (i = NSIZE - 2; i >= 0; i --){
        V[i] = B[i];
        for (j = NSIZE - 1; j > i ; j--){
            V[i] -= matrix[i][j] * V[j];
        }
    }

    for(i = 0; i < NSIZE; i++){
        C[i] = V[i];
    }
}


/* Print Times and such */
double calc_time(struct timeval *total_start, struct timeval *total_finish) {
    long compTime;
    double the_time;

    compTime = (total_finish->tv_sec - total_start->tv_sec) * 1000000;
    compTime = compTime + (total_finish->tv_usec - total_start->tv_usec);
    the_time = (double) compTime;
    return ((double) the_time / 1000000.0);
}

extern char* optarg;
int main(int argc,char *argv[]) {
    /* Init MPI things */
    MPI_Init(&argc, &argv);

    struct timeval total_start;
    struct timeval total_finish;
    double time;

    /* Parse arguments */
    int q;
    int verify = 0;
    while((q = getopt(argc, argv, "vs:")) != -1){
        switch(q){
            case 's':
            {
                int s;
                s = atoi(optarg);
                if (s > 0){
                    NSIZE = s;
                } else {
                    fprintf(stderr,"Entered size is negative, hence using the default (%d)\n",(int)NSIZE);
                }
            }
                break;
            case 'v':
            {
                verify = 1;
            }
                break;
            default:
                assert(0);
                break;
        }
    }

    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    allocate_memory();
    /* Only root populates matrices */
    if(rank == 0) {
        initMatrix();
    }

    /* Synchronize all MPI things */
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1) {
        gettimeofday(&total_start, NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* mpi solver */
    for (int i = 0; i < NSIZE; i++) {
        // get slices for matrix
//        for(int p = 0; p < nprocs; p++){
//            int slice = ((p * NSIZE) / (nprocs)) + i + 1;
//            if(slice > (NSIZE - 1)) {
//                slice = NSIZE - 1;
//            }
//            from[p] = slice;
//        }
        // Only root does pivot
        if(rank == 0) {
            getPivot(i);
        }

        for (k = J + 1 + ind ; k < NSIZE; k += NUM_PROCS){
            pivotVal = matrix[k][J];
            matrix[k][J] = 0.0;
            for (i = J + 1 ; i < NSIZE; i++){
                matrix[k][i] -= pivotVal * matrix[J][i];
            }
            B[k] -= pivotVal * B[J];
        }



        //// Try using send/receive instead of scatter/gather
        //// this will be row by row, but might work

        int pivotVal;


    }

    /* Synchronize and get time */
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1) {
        gettimeofday(&total_finish, NULL);
        time = calc_time(&total_start, &total_finish);
        MPI_Send(&time, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
    }
    if(rank == 0) {
        MPI_Recv(&time, 1, MPI_DOUBLE, 1, 100, MPI_COMM_WORLD, &status);
        printf("MPI time: %f\n", time);
        /* Linear Solver for Verification */
        if(verify) {
            gettimeofday(&total_start, 0);
            solveGauss();
            gettimeofday(&total_finish, 0);
            long compTime;
            double the_time;

            compTime = (total_finish.tv_sec - total_start.tv_sec) * 1000000;
            compTime = compTime + (total_finish.tv_usec - total_start.tv_usec);
            the_time = (double) compTime;
            printf("Verification time: %f Secs\n", (double) the_time / 1000000.0);
        }
    }

    MPI_Finalize();
    return 0;
}


