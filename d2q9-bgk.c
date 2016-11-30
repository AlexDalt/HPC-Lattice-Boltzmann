#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <omp.h>
#include <mpi.h>

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
               t_param* params, t_speed** local_cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, int** global_obstacles_ptr, double** av_vels_ptr,
               int size, int rank);

int calc_nrows(int ny, int size);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int pointer_swap(t_speed** cells, t_speed** tmp_cells);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int row);

//each comp_func works on a different part of the grid
//top row
double comp_func1(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows);
//top half (- top row)
double comp_func2(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows);
//bottom half (-bottom row)
double comp_func3(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows);
//bottom row
double comp_func4(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows);

/* finalise, including freeing up allocated memory */
int write_values(const t_param params, t_speed* cells, int* obstacles, double* av_vels);
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
  char* paramfile = NULL;       /* name of the input parameter file */
  char* obstaclefile = NULL;    /* name of a the input obstacle file */
  t_param params;               /* struct to hold parameter values */
  t_speed* global_cells = NULL; /* grid containing fluid densities */
  t_speed* local_cells = NULL;  /* grid containing local fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int* obstacles = NULL;        /* grid indicating which local cells are blocked */
  int* global_obstacles = NULL; /* grid indicating which cells are blocked */
  double* av_vels = NULL;       /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int ii,jj,kk;                 /* itterating integers */

  int rank;                         // rank
  int size;                         // size
  int required=MPI_THREAD_FUNNELED; // required type of MPI
  int provided;                     // provided type of MPI
  int local_nrows;                  // local number of rows
  int top;                          // rank above
  int bottom;                       // rank below
  int tag = 0;                      // pad MPI Sendrecv
  double local_total_vel;           // local total velocity
  double global_total_vel;          // global total velocity
  int totnobst = 0;                 // total number of non-obstacle cells
  MPI_Win top_win, bottom_win;      // RMA windows for one-sided MPI calls

  // initialise mpi
  MPI_Init_thread(&argc, &argv, required, &provided);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // define a type to send with mpi
  int block_lengths[1];
  MPI_Aint displacements[1];
  MPI_Datatype typelist[1];
  MPI_Datatype MPI_t_speed;

  block_lengths[0] = NSPEEDS;
  displacements[0] = 0;
  typelist[0] = MPI_FLOAT;

  MPI_Type_create_struct(1, block_lengths, displacements, typelist, &MPI_t_speed);
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
  initialise(paramfile, obstaclefile, &params, &local_cells,
    &tmp_cells, &obstacles, &global_obstacles, &av_vels, size, rank);

  for(ii = 0; ii < params.ny; ii++){
    for(jj = 0; jj < params.nx; jj++){
      if(!global_obstacles[ii * params.nx + jj]){
        totnobst++;
      }
    }
  }

  local_nrows = calc_nrows(params.ny, size);
  top = (rank + 1) % size;
  bottom = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  omp_set_num_threads(NUM_THREADS);

  MPI_Win_create(&(local_cells[local_nrows * params.nx]), params.nx, sizeof(t_speed), MPI_INFO_NULL,
    MPI_COMM_WORLD, &top_win);

  MPI_Win_create(&(local_cells[params.nx]), params.nx, sizeof(t_speed), MPI_INFO_NULL,
    MPI_COMM_WORLD, &bottom_win);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    // accelerates if required
    if(rank*local_nrows <= params.ny-2 && (rank+1)*local_nrows > params.ny-2){
      int row = (params.ny-2) % local_nrows;
      accelerate_flow(params, local_cells, obstacles, row);
    }

    // halo exchange and work
    MPI_Win_fence(0,top_win);
    MPI_Get(&(local_cells[0]), params.nx, MPI_t_speed, bottom, 0, params.nx, MPI_t_speed, top_win);
    MPI_Win_fence(0,top_win);

    local_total_vel += comp_func3(params, local_cells, tmp_cells, obstacles, local_nrows);

    local_total_vel += comp_func4(params, local_cells, tmp_cells, obstacles, local_nrows);


    MPI_Win_fence(0,bottom_win);
    MPI_Get(&(local_cells[(local_nrows+1) * params.nx]), params.nx, MPI_t_speed, top, 0,
      params.nx, MPI_t_speed, bottom_win);
    MPI_Win_fence(0,bottom_win);

    local_total_vel = comp_func1(params, local_cells, tmp_cells, obstacles, local_nrows);

    local_total_vel += comp_func2(params, local_cells, tmp_cells, obstacles, local_nrows);


    // reduce all totals together and divide by number of cells
    global_total_vel = 0.0;
    MPI_Reduce(&local_total_vel, &global_total_vel, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    if(rank == MASTER){
      av_vels[tt] = global_total_vel / totnobst;
    }

    // swaps pointer to local_cells and tmp_cells
    pointer_swap(&local_cells, &tmp_cells);
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // gather
  if(rank == MASTER){
    global_cells = (t_speed*)malloc(sizeof(t_speed) * params.nx * params.ny);
  }

  MPI_Gather(&(local_cells[params.nx]), params.nx * local_nrows, MPI_t_speed,
    global_cells, params.nx * local_nrows, MPI_t_speed,
    MASTER, MPI_COMM_WORLD);

  /* write final values and free memory */
  if(rank == MASTER){
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, global_cells, global_obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, global_cells, global_obstacles, av_vels);
  }
  
  free(global_cells);
  finalise(&params, &local_cells, &tmp_cells, &obstacles, &av_vels);
  free(global_obstacles);
  MPI_Win_free(&top_win);
  MPI_Win_free(&bottom_win);
  MPI_Finalize();

  return EXIT_SUCCESS;
}

int pointer_swap(t_speed** cells, t_speed** tmp_cells){
  t_speed* temp = *cells;
  *cells = *tmp_cells;
  *tmp_cells = temp;
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int row)
{
  /* compute weighting factors */
  double w1 = params.density * params.accel / 9.0;
  double w2 = params.density * params.accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = row + 1;
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

//only does the topmost row
double comp_func1(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows){
  /* loop over _all_ cells */
  const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const double w0 = 4.0 / 9.0;  /* weighting factor */
  const double w1 = 1.0 / 9.0;  /* weighting factor */
  const double w2 = 1.0 / 36.0; /* weighting factor */
  int a = nrows;
  int jj = 0;

  double tot_u = 0.0;

#pragma omp parallel for reduction(+:tot_u) private(jj)
  for (jj = 0; jj < params.nx; jj+=STEP){
    for (int b = jj; b < jj+STEP && b < params.nx; b++){
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = a + 1;
      int x_e = (b + 1) % params.nx;
      int y_s = a - 1;
      int x_w = (b == 0) ? (b + params.nx - 1) : (b - 1);
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      if (!obstacles[a * params.nx + b]){
        tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
        tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_w].speeds[1];
        tmp_cells[a * params.nx + b].speeds[2] = cells[y_s * params.nx + b].speeds[2];
        tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_e].speeds[3];
        tmp_cells[a * params.nx + b].speeds[4] = cells[y_n * params.nx + b].speeds[4];
        tmp_cells[a * params.nx + b].speeds[5] = cells[y_s * params.nx + x_w].speeds[5];
        tmp_cells[a * params.nx + b].speeds[6] = cells[y_s * params.nx + x_e].speeds[6];
        tmp_cells[a * params.nx + b].speeds[7] = cells[y_n * params.nx + x_e].speeds[7];
        tmp_cells[a * params.nx + b].speeds[8] = cells[y_n * params.nx + x_w].speeds[8];

        /* compute local density total */
        float local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[a * params.nx + b].speeds[kk];
        }

        /* compute x velocity component */
        double u_x = (tmp_cells[a * params.nx + b].speeds[1]
                      + tmp_cells[a * params.nx + b].speeds[5]
                      + tmp_cells[a * params.nx + b].speeds[8]
                      - (tmp_cells[a * params.nx + b].speeds[3]
                          + tmp_cells[a * params.nx + b].speeds[6]
                          + tmp_cells[a * params.nx + b].speeds[7]))
                      / local_density;
            /* compute y velocity component */
        double u_y = (tmp_cells[a * params.nx + b].speeds[2]
                      + tmp_cells[a * params.nx + b].speeds[5]
                      + tmp_cells[a * params.nx + b].speeds[6]
                      - (tmp_cells[a * params.nx + b].speeds[4]
                          + tmp_cells[a * params.nx + b].speeds[7]
                          + tmp_cells[a * params.nx + b].speeds[8]))
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
          tmp_cells[a * params.nx + b].speeds[kk] = tmp_cells[a * params.nx + b].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[a * params.nx + b].speeds[kk]);
        }

        //av_velocity steps
        tot_u += sqrt(u_sq);

      } else {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
        tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_w].speeds[1];
        tmp_cells[a * params.nx + b].speeds[4] = cells[y_s * params.nx + b].speeds[2];
        tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_e].speeds[3];
        tmp_cells[a * params.nx + b].speeds[2] = cells[y_n * params.nx + b].speeds[4];
        tmp_cells[a * params.nx + b].speeds[7] = cells[y_s * params.nx + x_w].speeds[5];
        tmp_cells[a * params.nx + b].speeds[8] = cells[y_s * params.nx + x_e].speeds[6];
        tmp_cells[a * params.nx + b].speeds[5] = cells[y_n * params.nx + x_e].speeds[7];
        tmp_cells[a * params.nx + b].speeds[6] = cells[y_n * params.nx + x_w].speeds[8];
      }
    }
  }

  return tot_u;
}

// works on top half (- top row)
double comp_func2(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows){
  /* loop over _all_ cells */
  const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const double w0 = 4.0 / 9.0;  /* weighting factor */
  const double w1 = 1.0 / 9.0;  /* weighting factor */
  const double w2 = 1.0 / 36.0; /* weighting factor */
  int jj = 0;
  int half = (nrows + 2)/2;
  int ii = half;

  double tot_u = 0.0;

#pragma omp parallel for reduction(+:tot_u) private(ii,jj) collapse(2)
  for (ii = half; ii < nrows ; ii+=STEP)
  {
    for (jj = 0; jj < params.nx; jj+=STEP)
    {
      for (int a = ii; a < ii+STEP && a < nrows; a++){
        for (int b = jj; b < jj+STEP && b < params.nx; b++){
          /* determine indices of axis-direction neighbours
          ** respecting periodic boundary conditions (wrap around) */
          int y_n = a + 1;
          int x_e = (b + 1) % params.nx;
          int y_s = a - 1;
          int x_w = (b == 0) ? (b + params.nx - 1) : (b - 1);
          /* propagate densities to neighbouring cells, following
          ** appropriate directions of travel and writing into
          ** scratch space grid */
          if (!obstacles[a * params.nx + b]){
            tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
            tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_w].speeds[1];
            tmp_cells[a * params.nx + b].speeds[2] = cells[y_s * params.nx + b].speeds[2];
            tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_e].speeds[3];
            tmp_cells[a * params.nx + b].speeds[4] = cells[y_n * params.nx + b].speeds[4];
            tmp_cells[a * params.nx + b].speeds[5] = cells[y_s * params.nx + x_w].speeds[5];
            tmp_cells[a * params.nx + b].speeds[6] = cells[y_s * params.nx + x_e].speeds[6];
            tmp_cells[a * params.nx + b].speeds[7] = cells[y_n * params.nx + x_e].speeds[7];
            tmp_cells[a * params.nx + b].speeds[8] = cells[y_n * params.nx + x_w].speeds[8];

            /* compute local density total */
            float local_density = 0.0;

            for (int kk = 0; kk < NSPEEDS; kk++)
            {
              local_density += tmp_cells[a * params.nx + b].speeds[kk];
            }

            /* compute x velocity component */
            double u_x = (tmp_cells[a * params.nx + b].speeds[1]
                          + tmp_cells[a * params.nx + b].speeds[5]
                          + tmp_cells[a * params.nx + b].speeds[8]
                          - (tmp_cells[a * params.nx + b].speeds[3]
                             + tmp_cells[a * params.nx + b].speeds[6]
                             + tmp_cells[a * params.nx + b].speeds[7]))
                         / local_density;
            /* compute y velocity component */
            double u_y = (tmp_cells[a * params.nx + b].speeds[2]
                          + tmp_cells[a * params.nx + b].speeds[5]
                          + tmp_cells[a * params.nx + b].speeds[6]
                          - (tmp_cells[a * params.nx + b].speeds[4]
                             + tmp_cells[a * params.nx + b].speeds[7]
                             + tmp_cells[a * params.nx + b].speeds[8]))
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
              tmp_cells[a * params.nx + b].speeds[kk] = tmp_cells[a * params.nx + b].speeds[kk]
                                                      + params.omega
                                                      * (d_equ[kk] - tmp_cells[a * params.nx + b].speeds[kk]);
            }

            //av_velocity steps
            tot_u += sqrt(u_sq);

          } else {
            /* called after propagate, so taking values from scratch space
            ** mirroring, and writing into main grid */
            tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
            tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_w].speeds[1];
            tmp_cells[a * params.nx + b].speeds[4] = cells[y_s * params.nx + b].speeds[2];
            tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_e].speeds[3];
            tmp_cells[a * params.nx + b].speeds[2] = cells[y_n * params.nx + b].speeds[4];
            tmp_cells[a * params.nx + b].speeds[7] = cells[y_s * params.nx + x_w].speeds[5];
            tmp_cells[a * params.nx + b].speeds[8] = cells[y_s * params.nx + x_e].speeds[6];
            tmp_cells[a * params.nx + b].speeds[5] = cells[y_n * params.nx + x_e].speeds[7];
            tmp_cells[a * params.nx + b].speeds[6] = cells[y_n * params.nx + x_w].speeds[8];
          }
        }
      }
    }
  }

  return tot_u;
}

// works on bottom half (- bottom row)
double comp_func3(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows){
  /* loop over _all_ cells */
  const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const double w0 = 4.0 / 9.0;  /* weighting factor */
  const double w1 = 1.0 / 9.0;  /* weighting factor */
  const double w2 = 1.0 / 36.0; /* weighting factor */
  int ii,jj = 0;
  int half = (nrows + 2)/2;

  double tot_u = 0.0;

#pragma omp parallel for reduction(+:tot_u) private(ii,jj) collapse(2)
  for (ii = 2; ii < half; ii+=STEP)
  {
    for (jj = 0; jj < params.nx; jj+=STEP)
    {
      for (int a = ii; a < ii+STEP && a < half; a++){
        for (int b = jj; b < jj+STEP && b < params.nx; b++){
          /* determine indices of axis-direction neighbours
          ** respecting periodic boundary conditions (wrap around) */
          int y_n = a + 1;
          int x_e = (b + 1) % params.nx;
          int y_s = a - 1;
          int x_w = (b == 0) ? (b + params.nx - 1) : (b - 1);
          /* propagate densities to neighbouring cells, following
          ** appropriate directions of travel and writing into
          ** scratch space grid */
          if (!obstacles[a * params.nx + b]){
            tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
            tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_w].speeds[1];
            tmp_cells[a * params.nx + b].speeds[2] = cells[y_s * params.nx + b].speeds[2];
            tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_e].speeds[3];
            tmp_cells[a * params.nx + b].speeds[4] = cells[y_n * params.nx + b].speeds[4];
            tmp_cells[a * params.nx + b].speeds[5] = cells[y_s * params.nx + x_w].speeds[5];
            tmp_cells[a * params.nx + b].speeds[6] = cells[y_s * params.nx + x_e].speeds[6];
            tmp_cells[a * params.nx + b].speeds[7] = cells[y_n * params.nx + x_e].speeds[7];
            tmp_cells[a * params.nx + b].speeds[8] = cells[y_n * params.nx + x_w].speeds[8];

            /* compute local density total */
            float local_density = 0.0;

            for (int kk = 0; kk < NSPEEDS; kk++)
            {
              local_density += tmp_cells[a * params.nx + b].speeds[kk];
            }

            /* compute x velocity component */
            double u_x = (tmp_cells[a * params.nx + b].speeds[1]
                          + tmp_cells[a * params.nx + b].speeds[5]
                          + tmp_cells[a * params.nx + b].speeds[8]
                          - (tmp_cells[a * params.nx + b].speeds[3]
                             + tmp_cells[a * params.nx + b].speeds[6]
                             + tmp_cells[a * params.nx + b].speeds[7]))
                         / local_density;
            /* compute y velocity component */
            double u_y = (tmp_cells[a * params.nx + b].speeds[2]
                          + tmp_cells[a * params.nx + b].speeds[5]
                          + tmp_cells[a * params.nx + b].speeds[6]
                          - (tmp_cells[a * params.nx + b].speeds[4]
                             + tmp_cells[a * params.nx + b].speeds[7]
                             + tmp_cells[a * params.nx + b].speeds[8]))
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
              tmp_cells[a * params.nx + b].speeds[kk] = tmp_cells[a * params.nx + b].speeds[kk]
                                                      + params.omega
                                                      * (d_equ[kk] - tmp_cells[a * params.nx + b].speeds[kk]);
            }

            //av_velocity steps
            tot_u += sqrt(u_sq);

          } else {
            /* called after propagate, so taking values from scratch space
            ** mirroring, and writing into main grid */
            tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
            tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_w].speeds[1];
            tmp_cells[a * params.nx + b].speeds[4] = cells[y_s * params.nx + b].speeds[2];
            tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_e].speeds[3];
            tmp_cells[a * params.nx + b].speeds[2] = cells[y_n * params.nx + b].speeds[4];
            tmp_cells[a * params.nx + b].speeds[7] = cells[y_s * params.nx + x_w].speeds[5];
            tmp_cells[a * params.nx + b].speeds[8] = cells[y_s * params.nx + x_e].speeds[6];
            tmp_cells[a * params.nx + b].speeds[5] = cells[y_n * params.nx + x_e].speeds[7];
            tmp_cells[a * params.nx + b].speeds[6] = cells[y_n * params.nx + x_w].speeds[8];
          }
        }
      }
    }
  }

  return tot_u;
}

//works on bottom row
double comp_func4(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int nrows){
  /* loop over _all_ cells */
  const double c_sq = 1.0 / 3.0; /* square of speed of sound */
  const double w0 = 4.0 / 9.0;  /* weighting factor */
  const double w1 = 1.0 / 9.0;  /* weighting factor */
  const double w2 = 1.0 / 36.0; /* weighting factor */
  int a = 1;
  int jj = 0;

  double tot_u = 0.0;

#pragma omp parallel for reduction(+:tot_u) private(jj)
  for (jj = 0; jj < params.nx; jj+=STEP)
  {
    for (int b = jj; b < jj+STEP && b < params.nx; b++){
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = a + 1;
      int x_e = (b + 1) % params.nx;
      int y_s = a - 1;
      int x_w = (b == 0) ? (b + params.nx - 1) : (b - 1);
      /* propagate densities to neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      if (!obstacles[a * params.nx + b]){
        tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
        tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_w].speeds[1];
        tmp_cells[a * params.nx + b].speeds[2] = cells[y_s * params.nx + b].speeds[2];
        tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_e].speeds[3];
        tmp_cells[a * params.nx + b].speeds[4] = cells[y_n * params.nx + b].speeds[4];
        tmp_cells[a * params.nx + b].speeds[5] = cells[y_s * params.nx + x_w].speeds[5];
        tmp_cells[a * params.nx + b].speeds[6] = cells[y_s * params.nx + x_e].speeds[6];
        tmp_cells[a * params.nx + b].speeds[7] = cells[y_n * params.nx + x_e].speeds[7];
        tmp_cells[a * params.nx + b].speeds[8] = cells[y_n * params.nx + x_w].speeds[8];

        /* compute local density total */
        float local_density = 0.0;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += tmp_cells[a * params.nx + b].speeds[kk];
        }

        /* compute x velocity component */
        double u_x = (tmp_cells[a * params.nx + b].speeds[1]
                      + tmp_cells[a * params.nx + b].speeds[5]
                      + tmp_cells[a * params.nx + b].speeds[8]
                      - (tmp_cells[a * params.nx + b].speeds[3]
                         + tmp_cells[a * params.nx + b].speeds[6]
                         + tmp_cells[a * params.nx + b].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        double u_y = (tmp_cells[a * params.nx + b].speeds[2]
                      + tmp_cells[a * params.nx + b].speeds[5]
                      + tmp_cells[a * params.nx + b].speeds[6]
                      - (tmp_cells[a * params.nx + b].speeds[4]
                         + tmp_cells[a * params.nx + b].speeds[7]
                         + tmp_cells[a * params.nx + b].speeds[8]))
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
          tmp_cells[a * params.nx + b].speeds[kk] = tmp_cells[a * params.nx + b].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - tmp_cells[a * params.nx + b].speeds[kk]);
        }

        //av_velocity steps
        tot_u += sqrt(u_sq);

      } else {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[a * params.nx + b].speeds[0] = cells[a * params.nx + b].speeds[0];
        tmp_cells[a * params.nx + b].speeds[3] = cells[a * params.nx + x_w].speeds[1];
        tmp_cells[a * params.nx + b].speeds[4] = cells[y_s * params.nx + b].speeds[2];
        tmp_cells[a * params.nx + b].speeds[1] = cells[a * params.nx + x_e].speeds[3];
        tmp_cells[a * params.nx + b].speeds[2] = cells[y_n * params.nx + b].speeds[4];
        tmp_cells[a * params.nx + b].speeds[7] = cells[y_s * params.nx + x_w].speeds[5];
        tmp_cells[a * params.nx + b].speeds[8] = cells[y_s * params.nx + x_e].speeds[6];
        tmp_cells[a * params.nx + b].speeds[5] = cells[y_n * params.nx + x_e].speeds[7];
        tmp_cells[a * params.nx + b].speeds[6] = cells[y_n * params.nx + x_w].speeds[8];
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

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** local_cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, int** global_obstacles_ptr, double** av_vels_ptr,
               int size, int rank)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  int    totobst = 0;
  

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

  int nrows = calc_nrows(params->ny, size);

  /* main grid */
  *local_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (nrows + 2) * params->nx);

  if (*local_cells_ptr == NULL) die("cannot allocate memory for local cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (nrows + 2) * params->nx);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (nrows + 2) * params->nx);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  *global_obstacles_ptr = malloc(sizeof(int) * params->ny * params->nx);


  /* initialise densities */
  double w0 = params->density * 4.0 / 9.0;
  double w1 = params->density      / 9.0;
  double w2 = params->density      / 36.0;

  for (int ii = 0; ii < nrows+2; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      /* centre */
      (*local_cells_ptr)[ii * params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*local_cells_ptr)[ii * params->nx + jj].speeds[1] = w1;
      (*local_cells_ptr)[ii * params->nx + jj].speeds[2] = w1;
      (*local_cells_ptr)[ii * params->nx + jj].speeds[3] = w1;
      (*local_cells_ptr)[ii * params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*local_cells_ptr)[ii * params->nx + jj].speeds[5] = w2;
      (*local_cells_ptr)[ii * params->nx + jj].speeds[6] = w2;
      (*local_cells_ptr)[ii * params->nx + jj].speeds[7] = w2;
      (*local_cells_ptr)[ii * params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < nrows+2; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
      (*global_obstacles_ptr)[ii * params->nx + jj] = 0;
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

    // assign to global obstacle array
    (*global_obstacles_ptr)[yy * params->nx + xx] = blocked;

    /* assign to local array if in scope */
    if(rank*nrows <= yy && (rank+1)*nrows > yy){
      int nyy = yy%nrows;
      (*obstacles_ptr)[(nyy+1) * params->nx + xx] = blocked;
    }
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