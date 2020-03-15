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

/* Timers for stuff */
struct timeval total_start;
struct timeval total_finish;

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

/* Initialize the matrix, but multithreaded! */
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



/* Driver function, spawns threads for each row in matrix */

void computeGauss(int begin, int end) {
    int i,j,k;
    double pivotVal;

    for(i = begin; i < end; i++){
        getPivot(i);

        pivotVal = matrix[i][i];

        for (j = i + 1 ; j < NSIZE; j++){
            pivotVal = matrix[j][i];
            matrix[j][i] = 0.0;
            for (k = i + 1 ; k < NSIZE; k++){
                matrix[j][k] -= pivotVal * matrix[i][k];
            }
            B[j] -= pivotVal * B[i];
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
void calc_time() {
    long compTime;
    double the_time;

    compTime = (total_finish.tv_sec - total_start.tv_sec) * 1000000;
    compTime = compTime + (total_finish.tv_usec - total_start.tv_usec);
    the_time = (double) compTime;
    printf("MPI time: %f Secs\n", (double) the_time / 1000000.0);
}

int main(int argc,char *argv[]) {
    int q;
    int verify = 0;
    int begin, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    /* Modified to accept number of threads as input */
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

    allocate_memory(NSIZE);

    begin = (NSIZE*proc_id)/nprocs + 1;
    end   = (NSIZE*(proc_id + 1))/nprocs;

    initMatrix();
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&total_start, NULL);

    /* mpi solver */
    computeGauss(begin, end);

    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&total_finish, NULL);
    calc_time();

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

    MPI_Finalize();
    return 0;
}


