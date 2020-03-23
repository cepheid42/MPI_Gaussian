#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include "mpi.h"

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}

int NSIZE = 128;
double** matrix;

/* Allocation and free methods */
int malloc2dchar(double ***array, int n, int m) {
	/* allocate the n*m contiguous items */
	double *p = (double *)malloc(n*m*sizeof(double));
	if (!p) return -1;

	/* allocate the row pointers into the memory */
	(*array) = (double **)malloc(n*sizeof(double*));
	if (!(*array)) {
		free(p);
		return -1;
	}

	/* set up the pointers into the contiguous memory */
	for (int i=0; i<n; i++)
		(*array)[i] = &(p[i*m]);

	return 0;
}

int free2dchar(double ***array) {
	/* free the memory - the first element of the array is at the start */
	free(&((*array)[0][0]));

	/* free the pointers into the memory */
	free(*array);

	return 0;
}

/* Initialize the matrix */
void initMatrix(double **matrix, double* B, int* swap) {
	int i,j;
	for(i = 0 ; i < NSIZE; i++){
		for(j = 0; j < NSIZE; j++) {
			matrix[i][j] = ((j < i ) ? 2*(j+1) : 2*(i+1));
		}
		B[i] = (double) i;
		swap[i] = i;
	}
}

void getPivot(int currow, double* B, int* swap) {
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

/* Single process solver function */
void singleGauss(double *B, int* swap) {
	int i;
	double pivotVal;
	for(i = 0; i < NSIZE; i++) {
		getPivot(i, B, swap);
		for (int j = i + 1; j < NSIZE; j++) {
			pivotVal = matrix[j][i];
			matrix[j][i] = 0.0;
			for (int k = i + 1; k < NSIZE; k++) {
				matrix[j][k] -= pivotVal * matrix[i][k];
			}
			B[j] -= pivotVal * B[i];
		}
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

int main(int argc, char **argv) {
	double **global;
	double *B;
	int *swap;
	int rank, nprocs;
	int q;
	int verify = 0;
	struct timeval total_start;
	struct timeval total_finish;
	double time;

	while((q = getopt(argc, argv, "vs:")) != -1){
		switch(q){
			case 's':
			{
				int s;
				s = atoi(optarg);
				if (s > 0){
					NSIZE = s;
				} else {
					fprintf(stderr,"Entered size is negative, hence using the default (%d)\n",(int) NSIZE);
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


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		/* Allocate stuff in root */
		malloc2dchar(&global, NSIZE, NSIZE);
		B = (double *) malloc(NSIZE * sizeof(double));
		swap = (int*) malloc(NSIZE * sizeof(int));

		/* Initialize matrix with values */
		initMatrix(global, B, swap);
	}

	/* Get start time */
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0) {
		gettimeofday(&total_start, NULL);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	/* Start iterating over all rows */
	for(int i = 0; i < NSIZE; i++) {
		MPI_Barrier(MPI_COMM_WORLD);    // Everybody synchronizes here

		if (rank == 0) {
			// Allows pivot function to work without requiring massive modifications
			matrix = global;
			getPivot(i, B, swap);
		}

		/* Calculate number of rows per processor for each iteration (from remaining rows) */
		int remainder = NSIZE % nprocs;
		int local_rows[nprocs];
		for (int i = 0; i < nprocs; i++) {
			local_rows[i] = NSIZE / nprocs;
			if (remainder > 0) {
				local_rows[i] += 1;
				remainder--;
			}
		}

		/* create the local array for each process to work on */
		double **local;
		malloc2dchar(&local, local_rows[rank], NSIZE);

		/* create a datatype to describe the subarrays of the global array */
		int sizes[2] = {NSIZE, NSIZE};                          /* global size */
		int subsizes[2] = {local_rows[rank], NSIZE};            /* local size */
		int starts[2] = {0, 0};                                 /* where this one starts */

		MPI_Datatype type, subarrtype;
		MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
		MPI_Type_create_resized(type, 0, NSIZE * sizeof(double), &subarrtype);
		MPI_Type_commit(&subarrtype);


		double *globalptr = NULL;
		if (rank == 0) {
			globalptr = &(global[0][0]);
		}

		/* scatter the array to all processors */
		int sendcounts[nprocs];
		int displs[nprocs];

		/* "Calculate" sendcounts and steps each processor only gets 1 "item" */
		for (int m = 0; m < nprocs; m++) sendcounts[m] = 1;
		int disp = 0;
		/* Step to the next set of rows */
		for (int i = 0; i < nprocs; i++) {
			displs[i] = disp;
			disp += local_rows[i];
		}

		// Scatter rows of matrix to each process
		MPI_Scatterv(globalptr, sendcounts, displs, subarrtype, &(local[0][0]),
		             local_rows[rank] * NSIZE, MPI_DOUBLE,
		             0, MPI_COMM_WORLD);

		/* Process local allocations */
		double *local_B = (double *) malloc(local_rows[rank] * sizeof(double));
		double *local_i = (double *) malloc(NSIZE * sizeof(double));
		double b_i;

		/* Set root values for broadcasting */
		if (rank == 0) {
			local_i = global[i];
			b_i = B[i];
		}

		// Scatter rows of B to each process
		MPI_Scatterv(&(B[0]), local_rows, displs, MPI_DOUBLE, &(local_B[0]), local_rows[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Broadcast constants to each process
		MPI_Bcast(local_i, NSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&b_i, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		/* Do work */
		double pivotVal;
		for (int k = 0; k < local_rows[rank]; k++) {
			pivotVal = local[k][i];
			local[k][i] = 0.0;
			for (int j = i; j < NSIZE; j++) {
				local[k][j] -= pivotVal * local_i[j];
			}
			local_B[k] -= pivotVal * b_i;
		}

		/* Send local matrix parts and B back to root */
		MPI_Gatherv(&(local[0][0]), local_rows[rank] * NSIZE, MPI_DOUBLE,
		            globalptr, sendcounts, displs, subarrtype,
		            0, MPI_COMM_WORLD);

		MPI_Gatherv(&(local_B[0]), local_rows[rank], MPI_DOUBLE,
		            &(B[0]), local_rows, displs, MPI_DOUBLE,
		            0, MPI_COMM_WORLD);

		/* don't need the local data anymore */
		free2dchar(&local);

		/* or the MPI data type */
		MPI_Type_free(&subarrtype);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		/* Get end time and print */
		gettimeofday(&total_finish, NULL);
		time = calc_time(&total_start, &total_finish);
		printf("%f, \n", time);

//		/* Single Process Solver for Comparison */
//		if(verify) {
//			double **single, *b;
//			int *swp;
//
//			malloc2dchar(&single, NSIZE, NSIZE);
//			b = (double *) malloc(NSIZE * sizeof(double));
//			swp = (int*) malloc(NSIZE * sizeof(int));
//
//			initMatrix(single, b, swp);
//
//			matrix = single;
//
//			gettimeofday(&total_start, 0);
//			singleGauss(b, swp);
//			gettimeofday(&total_finish, 0);
//			long compTime;
//			double the_time;
//
//			compTime = (total_finish.tv_sec - total_start.tv_sec) * 1000000;
//			compTime = compTime + (total_finish.tv_usec - total_start.tv_usec);
//			the_time = (double) compTime;
//			printf("%f, ", (double) the_time / 1000000.0);
//			free2dchar(&single);
//		}
		/* free the global array */
		free2dchar(&global);
	}

	MPI_Finalize();
	return 0;
}