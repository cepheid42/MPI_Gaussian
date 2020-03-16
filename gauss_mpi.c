#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
#include <mpi.h>

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}

double **matrix,*B,*V,*C;
int *swap;

int proc_id, nprocs;
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

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */



///* Driver function, spawns threads for each row in matrix */
//
//void computeGauss(int begin, int end) {
//    int i,j,k;
//    int J;
//    double pivotVal;
//
//    for(i = begin; i < end; i++){
//        getPivot(i);
//        MPI_Bcast(&J, 1, MPI_INT, )
//        pivotVal = matrix[i][i];
//        for (j = J + 1 + proc_id ; j < NSIZE; j += nprocs){
//            pivotVal = matrix[j][J];
//            matrix[j][J] = 0.0;
//            for (k = i + 1 ; k < NSIZE; k++){
//                matrix[j][k] -= pivotVal * matrix[i][k];
//            }
//            B[j] -= pivotVal * B[i];
//        }
//    }
//}


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

int main(int argc,char *argv[]) {
    /* Init MPI things */
    MPI_Init(&argc, &argv);

    int q;
    int verify = 0;

    struct timeval total_start;
    struct timeval total_finish;
    double time;
    /* Parse arguments */
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
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    printf("rank: %d out of %d\n", proc_id, nprocs);
    allocate_memory(NSIZE);
    int from[nprocs];
    /* Only root populates matrices */
    if(proc_id == 0) {
        initMatrix();
    }

    /* Synchronize all MPI things */
    MPI_Barrier(MPI_COMM_WORLD);
    if(proc_id == 1) {
        gettimeofday(&total_start, NULL);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* mpi solver */
    double pivotVal;
    int piv_row = 0;
    double m_row[NSIZE];

    if(proc_id == 0) {
        printf("proc%d\n", proc_id);
        for(int i = 0; i < NSIZE; i++){
            // Create array of slice indices
            for(int a = 0; a < nprocs; a++){
                int slice = ((a * NSIZE) / (nprocs - 1)) + i + 1;
                if(slice > (NSIZE - 1)) {
                    slice = NSIZE - 1;
                }
                from[a] = slice;
            }
            getPivot(i);
            piv_row = i;
            // Give everyone updated copy of matrix


            /* Broadcast is a send and receive
             * so it cannot be inside the if statement
             *
             * so I need to redo this whole section
            */
            MPI_Bcast(matrix, NSIZE*NSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//            // Give everyone pivot row
//            MPI_Bcast(&piv_row, 1, MPI_INT, 0, MPI_COMM_WORLD);
//            // Give everyone new array slices
//            MPI_Bcast(&from, nprocs, MPI_INT, 0, MPI_COMM_WORLD);
//            // receive updated rows from helpers
//            for(int p = 1; p < nprocs; p++) {
//                for (int r = from[p]; r < from[p+1]; r++) {
//                    MPI_Recv(&matrix[r], NSIZE, MPI_DOUBLE, p, p, MPI_COMM_WORLD, &status);
//                }
//            }
        }
    }
    else {
        printf("proc%d running\n", proc_id);
//        int start = from[proc_id];
//        int end = from[proc_id + 1];
//
//        // for each row in slice
//        for(int r = start; r < end; r++){
//            // From pivot (assume square matrix) to end of row
//            for (int j = start; j < piv_row; j++){
//                pivotVal = matrix[r][j];
//                matrix[r][j] = 0.0;
//                for(int k = j + 1; k < NSIZE; k++) {
//                    matrix[j][k] -= (pivotVal * matrix[r][k]);
//                }
//            }
//            MPI_Send(&matrix[r], NSIZE, MPI_DOUBLE, 0, proc_id, MPI_COMM_WORLD);
//        }
    }


    /* Synchronize and get time */
    MPI_Barrier(MPI_COMM_WORLD);
    if(proc_id == 1) {
        gettimeofday(&total_finish, NULL);
        time = calc_time(&total_start, &total_finish);
        MPI_Send(&time, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
    }
    if(proc_id == 0) {
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


