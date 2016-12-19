#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float* s0;
  float* s1;
  float* s2;
  float* s3;
  float* s4;
  float* s5;
  float* s6;
  float* s7;
  float* s8;
} SOA_speed;

kernel void accelerate_flow(global SOA_speed* cells,
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

  float mask = (!obstacles[ii * nx + jj]
      && (cells->s3[ii * nx + jj] - w1) > 0.0
      && (cells->s6[ii * nx + jj] - w2) > 0.0
      && (cells->s7[ii * nx + jj] - w2) > 0.0) ? 1.f : 0.f;

  /* increase 'east-side' densities */
  cells->s1[ii * nx + jj] += w1 * mask;
  cells->s5[ii * nx + jj] += w2 * mask;
  cells->s8[ii * nx + jj] += w2 * mask;
  /* decrease 'west-side' densities */
  cells->s3[ii * nx + jj] -= w1 * mask;
  cells->s6[ii * nx + jj] -= w2 * mask;
  cells->s7[ii * nx + jj] -= w2 * mask;
}

kernel void comp_func(global SOA_speed* cells,
                      global SOA_speed* tmp_cells,
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

  int obst  = (obstacles[cell] ? 1 : 0);
  int nobst = (obstacles[cell] ? 0 : 1);
  float diff[NSPEEDS];
  diff[0] = 0.0;

  tmp_cells.s0[cell] = cells.s0[ii  * nx + jj ]; /* central cell, no movement */
  tmp_cells.s1[cell] = cells.s1[ii  * nx + x_w]; /* east */
  tmp_cells.s2[cell] = cells.s2[y_s * nx + jj ]; /* north */
  tmp_cells.s3[cell] = cells.s3[ii  * nx + x_e]; /* west */
  tmp_cells.s4[cell] = cells.s4[y_n * nx + jj ]; /* south */
  tmp_cells.s5[cell] = cells.s5[y_s * nx + x_w]; /* north-east */
  tmp_cells.s6[cell] = cells.s6[y_s * nx + x_e]; /* north-west */
  tmp_cells.s7[cell] = cells.s7[y_n * nx + x_e]; /* south-west */
  tmp_cells.s8[cell] = cells.s8[y_n * nx + x_w]; /* south-east */

  diff[1] = tmp_cells.s3[cell];
  diff[2] = tmp_cells.s4[cell];
  diff[3] = tmp_cells.s1[cell];
  diff[4] = tmp_cells.s2[cell];
  diff[5] = tmp_cells.s7[cell];
  diff[6] = tmp_cells.s8[cell];
  diff[7] = tmp_cells.s5[cell];
  diff[8] = tmp_cells.s6[cell];

  float local_density = 0.0;

  local_density += tmp_cells.s0[cell];
  local_density += tmp_cells.s1[cell];
  local_density += tmp_cells.s2[cell];
  local_density += tmp_cells.s3[cell];
  local_density += tmp_cells.s4[cell];
  local_density += tmp_cells.s5[cell];
  local_density += tmp_cells.s6[cell];
  local_density += tmp_cells.s7[cell];
  local_density += tmp_cells.s8[cell];

  /* compute x velocity component */
  float u_x = (tmp_cells.s1[cell]
                + tmp_cells.s5[cell]
                + tmp_cells.s8[cell]
                - (tmp_cells.s3[cell]
                   + tmp_cells.s6[cell]
                   + tmp_cells.s7[cell]))
                / local_density;
  /* compute y velocity component */
  float u_y = (tmp_cells.s2[cell]
                + tmp_cells.s5[cell]
                + tmp_cells.s6[cell]
                - (tmp_cells.s4[cell]
                   + tmp_cells.s7[cell]
                   + tmp_cells.s8[cell]))
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

  tmp_cells.s0[cell] = (nobst) * (tmp_cells.s0[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s0[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s1[cell] = (nobst) * (tmp_cells.s1[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s1[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s2[cell] = (nobst) * (tmp_cells.s2[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s2[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s3[cell] = (nobst) * (tmp_cells.s3[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s3[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s4[cell] = (nobst) * (tmp_cells.s4[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s4[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s5[cell] = (nobst) * (tmp_cells.s5[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s5[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s6[cell] = (nobst) * (tmp_cells.s6[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s6[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s7[cell] = (nobst) * (tmp_cells.s7[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s7[cell]))
                               + (obst * diff[kk]);
  tmp_cells.s8[cell] = (nobst) * (tmp_cells.s8[cell]
                                            + omega
                                            * (d_equ[kk] - tmp_cells.s8[cell]))
                               + (obst * diff[kk]);

  tot_us[cell] = (nobst) * (sqrt((u_x * u_x) + (u_y * u_y)));
}