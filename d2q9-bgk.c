#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <omp.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define STEP            4
#define NUM_THREADS     8
#define MASTER          0


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  double density;       /* density per link */
  double accel;         /* density redistribution */
  double omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** global_cells_ptr, t_speed** local_cells_ptr,
               t_speed** tmp_cells_ptr, int** obstacles_ptr, double** av_vels_ptr, int size);

int calc_nrows(int ny, int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int pointer_swap(t_speed** cells, t_speed** tmp_cells);
double timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles,
                int local_ncols, int local_nrows, int rank);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int row);
double comp_func(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles,
                int local_ncols, int local_nrows, int rank);
int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
double total_density(const t_param params, t_speed* cells);

/* compute average velocity */
double av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
double calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* global_cells = NULL;    /* grid containing fluid densities */
  t_speed* local_cells = NULL;
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  double* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int ii,jj;

  // mpi values
  int rank;
  int size;
  int message_length;
  int local_nrows, local_ncols;
  double local_av_vel;
  double total_av_vel;
  t_speed* sendbuf;
  t_speed* recvbuf;
  int top, bottom;
  int tag = 0;
  MPI_Status status;

  // starting MPI
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );

  // defining mpi datatype eqivalent to t_speed, MPI_t_speed
  MPI_Datatype MPI_t_speed;
  const int blocklengths = NSPEEDS;
  MPI_Datatype typelist = MPI_FLOAT;
  MPI_Type_struct(1, &blocklengths, 0, &typelist, &MPI_t_speed);
  MPI_Type_commit(&MPI_t_speed);

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &global_cells, &local_cells, &tmp_cells,
              &obstacles, &av_vels, size);
  local_ncols = params.nx;
  local_nrows = calc_nrows(params.ny, size);
  sendbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  recvbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  top = (rank + 1) % size;
  bottom = (rank == MASTER) ? (rank + size - 1) : (rank - 1);

  // scatter global_cells into local cells
  message_length = local_ncols * local_nrows;
  MPI_Scatter(global_cells, message_length, MPI_t_speed,
              &local_cells[params.nx], message_length, MPI_t_speed,
              MASTER, MPI_COMM_WORLD);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  omp_set_num_threads(NUM_THREADS);
    for (int tt = 0; tt < params.maxIters; tt++)
    {
      total_av_vel = 0.0;
      // halo exchange
      // send to bottom recieve from top
      for(jj = 0; jj < local_ncols; jj++){
        sendbuf[jj] = local_cells[params.nx + jj];
      }
      MPI_Sendrecv(sendbuf, local_ncols, MPI_t_speed, bottom, tag,
        recvbuf, local_ncols, MPI_t_speed, top, tag,
        MPI_COMM_WORLD, &status);
      for(jj = 0; jj < local_ncols; jj++){
        local_cells[(local_nrows + 1) * params.nx + jj] = recvbuf[jj];
      }
      // send to top recieve from bottom
      for(jj = 0; jj < local_ncols; jj++){
        sendbuf[jj] = local_cells[(local_nrows + 1) * params.nx + jj];
      }
      MPI_Sendrecv(sendbuf, local_ncols, MPI_t_speed, top, tag,
        recvbuf, local_ncols, MPI_t_speed, bottom, tag,
        MPI_COMM_WORLD, &status);
      for(jj = 0; jj < local_ncols; jj++){
        local_cells[params.nx + jj] = recvbuf[jj];
      }

      // do calculation
      local_av_vel = timestep(params, local_cells, tmp_cells, obstacles,
        local_ncols, local_nrows, rank);

      MPI_Reduce(&local_av_vel, &total_av_vel, 1, MPI_DOUBLE, MPI_SUM, MASTER,
        MPI_COMM_WORLD);

      if(rank == MASTER){
        av_vels[tt] = total_av_vel/(params.nx*params.ny);
      }

      // do mpi reduce over local_av_vels
      pointer_swap(&local_cells, &tmp_cells);
    }

  MPI_Gather(&local_cells[local_ncols], message_length, MPI_t_speed,
              global_cells, message_length, MPI_t_speed,
              MASTER, MPI_COMM_WORLD);

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  if(rank == MASTER){
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, global_cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, global_cells, obstacles, av_vels);
    finalise(&params, &global_cells, &tmp_cells, &obstacles, &av_vels);
  }
  free(&sendbuf);
  free(&recvbuf);
  free(&local_cells);

  MPI_Finalize();
  return EXIT_SUCCESS;
}

int pointer_swap(t_speed** cells, t_speed** tmp_cells){
  t_speed* temp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = temp;
  return EXIT_SUCCESS;
}

double timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles,
                int local_ncols, int local_nrows, int rank)
{
  //accelerates the second row of cells
  if((local_nrows * (rank-1)) <= (params.ny - 2) && (local_nrows * rank) < (params.ny - 2)){
    int row = (params.ny - 2) - (local_nrows * (rank-1));
    accelerate_flow(params, cells, obstacles, row);
  }

  //performs the bulk of the cell calculations, writing each to tmp_cells, and returns av_velocity
  return comp_func(params, cells, tmp_cells, obstacles, local_ncols, local_nrows, rank);
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int row)
{
  /* compute weighting factors */
  double w1 = params.density * params.accel / 9.0;
  double w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = row+1;
  int jj = 0;

#pragma omp parallel for private(jj)
  for (int jj = 0; jj < params.nx; jj++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii * params.nx + jj]
        && (cells[ii * params.nx + jj].speeds[3] - w1) > 0.0
        && (cells[ii * params.nx + jj].speeds[6] - w2) > 0.0
        && (cells[ii * params.nx + jj].speeds[7] - w2) > 0.0)
    {
      /* increase 'east-side' densities */
      cells[ii * params.nx + jj].speeds[1] += w1;
      cells[ii * params.nx + jj].speeds[5] += w2;
      cells[ii * params.nx + jj].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii * params.nx + jj].speeds[3] -= w1;
      cells[ii * params.nx + jj].speeds[6] -= w2;
      cells[ii * params.nx + jj].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

double comp_func(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles,
                int local_ncols, int local_nrows, int rank){
  /* loop over _all_ cells */
  const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const double w0 = 4.0 / 9.0;  /* weighting factor */
  const double w1 = 1.0 / 9.0;  /* weighting factor */
  const double w2 = 1.0 / 36.0; /* weighting factor */
  int ii,jj = 0;

  double tot_u = 0.0;

#pragma omp parallel for reduction(+:tot_u, tot_cells) private(ii,jj) collapse(2)
  for (ii = 1; ii < (local_nrows+1); ii+=STEP)
  {
    for (jj = 0; jj < local_ncols; jj+=STEP)
    {
      for (int a = ii; a < ii+STEP && a < (local_nrows+1); a++){
        for (int b = jj; b < jj+STEP && b < local_ncols; b++){
          /* determine indices of axis-direction neighbours
          ** respecting periodic boundary conditions (wrap around) */
          int y_n = (a + 1);
          int x_e = (b + 1) % local_ncols;
          int y_s = (a - 1);
          int x_w = (b == 0) ? (b + local_ncols - 1) : (b - 1);
          /* propagate densities to neighbouring cells, following
          ** appropriate directions of travel and writing into
          ** scratch space grid */
          if (!obstacles[local_nrows * rank + a * params.nx + b]){
            tmp_cells[a * local_ncols + b].speeds[0] = cells[a * local_ncols + b].speeds[0];
            tmp_cells[a * local_ncols + b].speeds[1] = cells[a * local_ncols + x_w].speeds[1];
            tmp_cells[a * local_ncols + b].speeds[2] = cells[y_s * local_ncols + b].speeds[2];
            tmp_cells[a * local_ncols + b].speeds[3] = cells[a * local_ncols + x_e].speeds[3];
            tmp_cells[a * local_ncols + b].speeds[4] = cells[y_n * local_ncols + b].speeds[4];
            tmp_cells[a * local_ncols + b].speeds[5] = cells[y_s * local_ncols + x_w].speeds[5];
            tmp_cells[a * local_ncols + b].speeds[6] = cells[y_s * local_ncols + x_e].speeds[6];
            tmp_cells[a * local_ncols + b].speeds[7] = cells[y_n * local_ncols + x_e].speeds[7];
            tmp_cells[a * local_ncols + b].speeds[8] = cells[y_n * local_ncols + x_w].speeds[8];

            /* compute local density total */
            float local_density = 0.0;

            for (int kk = 0; kk < NSPEEDS; kk++)
            {
              local_density += tmp_cells[a * params.nx + b].speeds[kk];
            }

            /* compute x velocity component */
            double u_x = (tmp_cells[a * local_ncols + b].speeds[1]
                          + tmp_cells[a * local_ncols + b].speeds[5]
                          + tmp_cells[a * local_ncols + b].speeds[8]
                          - (tmp_cells[a * local_ncols + b].speeds[3]
                             + tmp_cells[a * local_ncols + b].speeds[6]
                             + tmp_cells[a * local_ncols + b].speeds[7]))
                         / local_density;
            /* compute y velocity component */
            double u_y = (tmp_cells[a * local_ncols + b].speeds[2]
                          + tmp_cells[a * local_ncols + b].speeds[5]
                          + tmp_cells[a * local_ncols + b].speeds[6]
                          - (tmp_cells[a * local_ncols + b].speeds[4]
                             + tmp_cells[a * local_ncols + b].speeds[7]
                             + tmp_cells[a * local_ncols + b].speeds[8]))
                         / local_density;

            /* velocity squared */
            double u_sq = u_x * u_x + u_y * u_y;

            /* directional velocity components */
            float u[NSPEEDS];
            u[1] =   u_x;        /* east */
            u[2] =         u_y;  /* north */
            u[3] = - u_x;        /* west */
            u[4] =       - u_y;  /* south */
            u[5] =   u_x + u_y;  /* north-east */
            u[6] = - u_x + u_y;  /* north-west */
            u[7] = - u_x - u_y;  /* south-west */
            u[8] =   u_x - u_y;  /* south-east */

            /* equilibrium densities */
            float d_equ[NSPEEDS];
            /* zero velocity density: weight w0 */
            d_equ[0] = w0 * local_density
                       * (1.0 - u_sq / (2.0 * c_sq));
            /* axis speeds: weight w1 */
            d_equ[1] = w1 * local_density * (1.0 + u[1] / c_sq
                                             + (u[1] * u[1]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            d_equ[2] = w1 * local_density * (1.0 + u[2] / c_sq
                                             + (u[2] * u[2]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            d_equ[3] = w1 * local_density * (1.0 + u[3] / c_sq
                                             + (u[3] * u[3]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            d_equ[4] = w1 * local_density * (1.0 + u[4] / c_sq
                                             + (u[4] * u[4]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            /* diagonal speeds: weight w2 */
            d_equ[5] = w2 * local_density * (1.0 + u[5] / c_sq
                                             + (u[5] * u[5]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            d_equ[6] = w2 * local_density * (1.0 + u[6] / c_sq
                                             + (u[6] * u[6]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            d_equ[7] = w2 * local_density * (1.0 + u[7] / c_sq
                                             + (u[7] * u[7]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));
            d_equ[8] = w2 * local_density * (1.0 + u[8] / c_sq
                                             + (u[8] * u[8]) / (2.0 * c_sq * c_sq)
                                             - u_sq / (2.0 * c_sq));

            /* relaxation step */
            for (int kk = 0; kk < NSPEEDS; kk++)
            {
              tmp_cells[a * local_ncols + b].speeds[kk] = tmp_cells[a * local_ncols + b].speeds[kk]
                                                      + params.omega
                                                      * (d_equ[kk] - tmp_cells[a * local_ncols + b].speeds[kk]);
            }

            //av_velocity steps
            tot_u += sqrt(u_sq);

          } else {
            /* called after propagate, so taking values from scratch space
            ** mirroring, and writing into main grid */
            tmp_cells[a * local_ncols + b].speeds[0] = cells[a * local_ncols + b].speeds[0];
            tmp_cells[a * local_ncols + b].speeds[3] = cells[a * local_ncols + x_w].speeds[1];
            tmp_cells[a * local_ncols + b].speeds[4] = cells[y_s * local_ncols + b].speeds[2];
            tmp_cells[a * local_ncols + b].speeds[1] = cells[a * local_ncols + x_e].speeds[3];
            tmp_cells[a * local_ncols + b].speeds[2] = cells[y_n * local_ncols + b].speeds[4];
            tmp_cells[a * local_ncols + b].speeds[7] = cells[y_s * local_ncols + x_w].speeds[5];
            tmp_cells[a * local_ncols + b].speeds[8] = cells[y_s * local_ncols + x_e].speeds[6];
            tmp_cells[a * local_ncols + b].speeds[5] = cells[y_n * local_ncols + x_e].speeds[7];
            tmp_cells[a * local_ncols + b].speeds[6] = cells[y_n * local_ncols + x_w].speeds[8];
          }
        }
      }
    }
  }

  return tot_u;
}

int calc_nrows(int ny, int size){
  return ny/size;
}

double av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  double tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;
  int ii, jj = 0;

#pragma omp parallel for reduction (+:tot_u,tot_cells) private(ii,jj)
  /* loop over all non-blocked cells */
  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii * params.nx + jj])
      {
        /* local density total */
        double local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* x-component of velocity */
        double u_x = (cells[ii * params.nx + jj].speeds[1]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[8]
                      - (cells[ii * params.nx + jj].speeds[3]
                         + cells[ii * params.nx + jj].speeds[6]
                         + cells[ii * params.nx + jj].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        double u_y = (cells[ii * params.nx + jj].speeds[2]
                      + cells[ii * params.nx + jj].speeds[5]
                      + cells[ii * params.nx + jj].speeds[6]
                      - (cells[ii * params.nx + jj].speeds[4]
                         + cells[ii * params.nx + jj].speeds[7]
                         + cells[ii * params.nx + jj].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (double)tot_cells;
}

int initialise((const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** global_cells_ptr, t_speed** local_cells_ptr,
               t_speed** tmp_cells_ptr, int** obstacles_ptr, double** av_vels_ptr, int size))
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%lf\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *global_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* local grid of cells */
  *local_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (calc_nrows(params->ny,size)+2)
                                      * params->nx);

  if (*local_cells_ptr == NULL) die("cannot allocate memory for local_cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (calc_nrows(params->ny,size)+2)
                                      * params->nx);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*global_cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*global_cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*global_cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*global_cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*global_cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*global_cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*global_cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*global_cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*global_cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (double*)malloc(sizeof(double) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, double** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


double calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const double viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

double total_density(const t_param params, t_speed* cells)
{
  double total = 0.0;  /* accumulator */

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii * params.nx + jj].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels)
{
  FILE* fp;                     /* file pointer */
  const double c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  double local_density;         /* per grid cell sum of densities */
  double pressure;              /* fluid pressure in grid cell */
  double u_x;                   /* x-component of velocity in grid cell */
  double u_y;                   /* y-component of velocity in grid cell */
  double u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii * params.nx + jj].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii * params.nx + jj].speeds[1]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[8]
               - (cells[ii * params.nx + jj].speeds[3]
                  + cells[ii * params.nx + jj].speeds[6]
                  + cells[ii * params.nx + jj].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii * params.nx + jj].speeds[2]
               + cells[ii * params.nx + jj].speeds[5]
               + cells[ii * params.nx + jj].speeds[6]
               - (cells[ii * params.nx + jj].speeds[4]
                  + cells[ii * params.nx + jj].speeds[7]
                  + cells[ii * params.nx + jj].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}