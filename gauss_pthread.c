#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
#include <pthread.h>

#define VERIFY      0

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}
#define ABS(a)          (((a) > 0) ? (a) : -(a))

double **matrix,*B,*V,*C;
int *swap;

int NUM_PROCS = 1;
int NSIZE = 128;
int J = 0;
pthread_t *gauss_threads;
pthread_t *init_threads;

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

void *initMatrix(void *s) {
    int slice = (int) s;
    int from = (slice * NSIZE) / NUM_PROCS;
    int to = ((slice + 1) * NSIZE) / NUM_PROCS;

    int i,j;
    for(i = from ; i < to; i++){
        for(j = 0; j < NSIZE; j++) {
            matrix[i][j] = ((j < i ) ? 2*(j+1) : 2*(i+1));
        }
        B[i] = (double) i;
        swap[i] = i;
    }
    pthread_exit(NULL);
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


/* Iterates over each row doing calculations, each row has no loop dependencies,
 * but is dependent on the Jth (original) row. */
void *multiGauss(void *s) {
    int k,i;
    double pivotVal;
    int ind = (int) s;

    for (k = J + 1 + ind ; k < NSIZE; k += NUM_PROCS){
        pivotVal = matrix[k][J];
        matrix[k][J] = 0.0;
        for (i = J + 1 ; i < NSIZE; i++){
            matrix[k][i] -= pivotVal * matrix[J][i];
        }
        B[k] -= pivotVal * B[J];
    }
    pthread_exit(NULL);
}

/* Driver function, spawns threads for each row in matrix */

void computeGauss() {
    int j;
    gauss_threads = (pthread_t*) malloc(NUM_PROCS * sizeof(pthread_t));

    gettimeofday(&total_start, 0);

    for(j = 0; j < NSIZE; j++){
        getPivot(j);
        J = j;  // Set global iteration counter.

        if (NSIZE != 1) {
            // Spawn Threads to compute Gaussian on remaining rows
            for (int ind = 0; ind < NUM_PROCS; ind++){
                pthread_create(&gauss_threads[ind], NULL, multiGauss, (void*) ind);
            }

            for (int ind = 0; ind < NUM_PROCS; ind++){
                pthread_join(gauss_threads[ind], NULL);
            }
        } else {
            // Doesn't spawn threads on single processor
            multiGauss(0);
        }
    }
    gettimeofday(&total_finish, 0);
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
    printf("Pthread time: %f Secs\n", (double) the_time / 1000000.0);
}

int main(int argc,char *argv[]) {
    int i;
    int verify = 0;
    /* Modified to accept number of threads as input */
    while((i = getopt(argc,argv,"vs:n:")) != -1){
        switch(i){
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
            case 'n':
            {
                int n;
                n = atoi(optarg);
                if (n > 0) {
                    NUM_PROCS = n;
                } else {
                    fprintf(stderr,"Invalid number of processors, using default of %d\n", NUM_PROCS);
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

    /* Spawn threads for matrix initialization */
    init_threads = (pthread_t*) malloc(NUM_PROCS * sizeof(pthread_t));
    for (int ind = 0; ind < NUM_PROCS; ind++){
        pthread_create(&init_threads[ind], NULL, initMatrix, (void*) ind);
    }

    for (int ind = 0; ind < NUM_PROCS; ind++){
        pthread_join(init_threads[ind], NULL);
    }


    /* pthread solver */
    computeGauss();
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

    return 0;
}
