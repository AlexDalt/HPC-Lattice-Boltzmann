#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define blksz 16

kernel void accelerate_flow(global int* obstacles,
                            int nx, int ny,
                            float density, float accel,
                            global float* s0, global float* s1,
                            global float* s2, global float* s3,
                            global float* s4, global float* s5,
                            global float* s6, global float* s7,
                            global float* s8)
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
      && (s3[ii * nx + jj] - w1) > 0.0
      && (s6[ii * nx + jj] - w2) > 0.0
      && (s7[ii * nx + jj] - w2) > 0.0) ? 1.f : 0.f;

  /* increase 'east-side' densities */
  s1[ii * nx + jj] += w1 * mask;
  s5[ii * nx + jj] += w2 * mask;
  s8[ii * nx + jj] += w2 * mask;
  /* decrease 'west-side' densities */
  s3[ii * nx + jj] -= w1 * mask;
  s6[ii * nx + jj] -= w2 * mask;
  s7[ii * nx + jj] -= w2 * mask;
}

kernel void comp_func(global float* tot_us,
                      global int* obstacles,
                      int nx, int ny,
                      float omega,
                      global float* cells_s0, global float* cells_s1,
                      global float* cells_s2, global float* cells_s3,
                      global float* cells_s4, global float* cells_s5,
                      global float* cells_s6, global float* cells_s7,
                      global float* cells_s8, global float* tmp_cells_s0,
                      global float* tmp_cells_s1, global float* tmp_cells_s2,
                      global float* tmp_cells_s3, global float* tmp_cells_s4,
                      global float* tmp_cells_s5, global float* tmp_cells_s6,
                      global float* tmp_cells_s7, global float* tmp_cells_s8)
{
  const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */
  float tmp[NSPEEDS];
  float diff[NSPEEDS];

  int g_id_jj = get_global_id(0);
  int g_id_ii = get_global_id(1);
  int max_jj = get_global_size(0);
  int max_ii = get_global_size(1);
  int max_b = ny/max_jj;
  int max_a = nx/max_ii;

  #pragma unroll
  for(int a = 0; a < max_a; a++){
    #pragma unroll
    for(int b = 0; b < max_b; b++){
      int ii = g_id_ii * max_a + a;
      int jj = g_id_jj * max_b + b;

      int cell = ii * nx + jj;

      int y_n = (ii + 1) % ny;
      int x_e = (jj + 1) % nx;
      int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
      int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);

      int obst  = (obstacles[cell] ? 1 : 0);
      int nobst = (obstacles[cell] ? 0 : 1);
      float diff[NSPEEDS];
      diff[0] = 0.0;

      tmp[0] = cells_s0[ii  * nx + jj ]; /* central cell, no movement */
      tmp[1] = cells_s1[ii  * nx + x_w]; /* east */
      tmp[2] = cells_s2[y_s * nx + jj ]; /* north */
      tmp[3] = cells_s3[ii  * nx + x_e]; /* west */
      tmp[4] = cells_s4[y_n * nx + jj ]; /* south */
      tmp[5] = cells_s5[y_s * nx + x_w]; /* north-east */
      tmp[6] = cells_s6[y_s * nx + x_e]; /* north-west */
      tmp[7] = cells_s7[y_n * nx + x_e]; /* south-west */
      tmp[8] = cells_s8[y_n * nx + x_w]; /* south-east */

      diff[1] = tmp[3];
      diff[2] = tmp[4];
      diff[3] = tmp[1];
      diff[4] = tmp[2];
      diff[5] = tmp[7];
      diff[6] = tmp[8];
      diff[7] = tmp[5];
      diff[8] = tmp[6];

      float local_density = 0.0;

      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        local_density += tmp[kk];
      }

      /* compute x velocity component */
      float u_x = (tmp[1]
                    + tmp[5]
                    + tmp[8]
                    - (tmp[3]
                       + tmp[6]
                       + tmp[7]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (tmp[2]
                    + tmp[5]
                    + tmp[6]
                    - (tmp[4]
                       + tmp[7]
                       + tmp[8]))
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

      tmp_cells_s0[cell] = (nobst) * (tmp[0] + omega * (d_equ[0] - tmp[0]))
                         + (obst) * diff[0];
      tmp_cells_s1[cell] = (nobst) * (tmp[1] + omega * (d_equ[1] - tmp[1]))
                         + (obst) * diff[1];
      tmp_cells_s2[cell] = (nobst) * (tmp[2] + omega * (d_equ[2] - tmp[2]))
                         + (obst) * diff[2];
      tmp_cells_s3[cell] = (nobst) * (tmp[3] + omega * (d_equ[3] - tmp[3]))
                         + (obst) * diff[3];
      tmp_cells_s4[cell] = (nobst) * (tmp[4] + omega * (d_equ[4] - tmp[4]))
                         + (obst) * diff[4];
      tmp_cells_s5[cell] = (nobst) * (tmp[5] + omega * (d_equ[5] - tmp[5]))
                         + (obst) * diff[5];
      tmp_cells_s6[cell] = (nobst) * (tmp[6] + omega * (d_equ[6] - tmp[6]))
                         + (obst) * diff[6];
      tmp_cells_s7[cell] = (nobst) * (tmp[7] + omega * (d_equ[7] - tmp[7]))
                         + (obst) * diff[7];
      tmp_cells_s8[cell] = (nobst) * (tmp[8] + omega * (d_equ[8] - tmp[8]))
                         + (obst) * diff[8];

      tot_us[cell] = (nobst) * (sqrt((u_x * u_x) + (u_y * u_y)));
    }
  }
}