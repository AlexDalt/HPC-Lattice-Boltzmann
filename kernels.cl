#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float* speeds[NSPEEDS];
} SOA_speeds;

kernel void accelerate_flow(global SOA_speeds* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells->speeds[3][ii * nx + jj] - w1) > 0.0
      && (cells->speeds[6][ii * nx + jj] - w2) > 0.0
      && (cells->speeds[7][ii * nx + jj] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells->speeds[1][ii * nx + jj] += w1;
    cells->speeds[5][ii * nx + jj] += w2;
    cells->speeds[8][ii * nx + jj] += w2;
    /* decrease 'west-side' densities */
    cells->speeds[3][ii * nx + jj] -= w1;
    cells->speeds[6][ii * nx + jj] -= w2;
    cells->speeds[7][ii * nx + jj] -= w2;
  }
}

kernel void comp_func(global SOA_speeds* cells,
                      global SOA_speeds* tmp_cells,
                      global float* tot_us,
                      global int* obstacles,
                      int nx, int ny,
                      float omega)
{
  int jj = get_global_id(0);
  int ii = get_global_id(1);
  int cell = ii * nx + jj;
  const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */

  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);

  if(obstacles[cell]){
    tmp_cells[0][cell] = cells[0][ii  * nx + jj ]; /* central cell, no movement */
    tmp_cells[3][cell] = cells[1][ii  * nx + x_w]; /* east */
    tmp_cells[4][cell] = cells[2][y_s * nx + jj ]; /* north */
    tmp_cells[1][cell] = cells[3][ii  * nx + x_e]; /* west */
    tmp_cells[2][cell] = cells[4][y_n * nx + jj ]; /* south */
    tmp_cells[7][cell] = cells[5][y_s * nx + x_w]; /* north-east */
    tmp_cells[8][cell] = cells[6][y_s * nx + x_e]; /* north-west */
    tmp_cells[5][cell] = cells[7][y_n * nx + x_e]; /* south-west */
    tmp_cells[6][cell] = cells[8][y_n * nx + x_w]; /* south-east */   

    tot_us[cell] = 0;
  } else {
    tmp_cells[0][cell] = cells[0][ii  * nx + jj ]; /* central cell, no movement */
    tmp_cells[1][cell] = cells[1][ii  * nx + x_w]; /* east */
    tmp_cells[2][cell] = cells[2][y_s * nx + jj ]; /* north */
    tmp_cells[3][cell] = cells[3][ii  * nx + x_e]; /* west */
    tmp_cells[4][cell] = cells[4][y_n * nx + jj ]; /* south */
    tmp_cells[5][cell] = cells[5][y_s * nx + x_w]; /* north-east */
    tmp_cells[6][cell] = cells[6][y_s * nx + x_e]; /* north-west */
    tmp_cells[7][cell] = cells[7][y_n * nx + x_e]; /* south-west */
    tmp_cells[8][cell] = cells[8][y_n * nx + x_w]; /* south-east */ 

    float local_density = 0.0;

    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_cells->speeds[kk][cell];
    }

    /* compute x velocity component */
    float u_x = (tmp_cells[1][cell]
                  + tmp_cells[5][cell]
                  + tmp_cells[8][cell]
                  - (tmp_cells[3][cell]
                     + tmp_cells[6][cell]
                     + tmp_cells[7][cell]))
                  / local_density;
    /* compute y velocity component */
    float u_y = (tmp_cells[2][cell]
                  + tmp_cells[5][cell]
                  + tmp_cells[6][cell]
                  - (tmp_cells[4][cell]
                     + tmp_cells[7][cell]
                     + tmp_cells[8][cell]))
                  / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

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
    local_density = 0;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      tmp_cells[kk][cell] = tmp_cells[kk][cell]
                                              + omega
                                              * (d_equ[kk] - tmp_cells[kk][cell]);
    }

    tot_us[cell] = sqrt((u_x * u_x) + (u_y * u_y));
  }
}