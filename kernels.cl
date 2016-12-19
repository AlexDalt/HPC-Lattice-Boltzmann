#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
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
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0) ? 1.f : 0.f;

  /* increase 'east-side' densities */
  cells[ii * nx + jj].speeds[1] += w1 * mask;
  cells[ii * nx + jj].speeds[5] += w2 * mask;
  cells[ii * nx + jj].speeds[8] += w2 * mask;
  /* decrease 'west-side' densities */
  cells[ii * nx + jj].speeds[3] -= w1 * mask;
  cells[ii * nx + jj].speeds[6] -= w2 * mask;
  cells[ii * nx + jj].speeds[7] -= w2 * mask;
}

kernel void comp_func(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global float* tot_us,
                      global int* obstacles,
                      int nx, int ny,
                      float omega,
                      local t_speed* local_cells)
{
  int g_id_jj = get_global_id(0);
  int g_id_ii = get_global_id(1);
  int max_jj = get_global_size(0);
  int max_ii = get_global_size(1);
  int max_b = ny/(max_jj+1);
  int max_a = nx/(max_ii+1);

  const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */

  for(int a = -1; a < max_a+1; a++){
    for(int b = -1; b < max_b+1; b++){
      int ref_a = a + 1;
      int ref_b = b + 1;

      int i = max_a * g_id_ii + a;
      int j = max_b * g_id_jj + b;
      for(int kk = 0; kk < NSPEEDS; kk++){
        local_cells[ref_a * (max_b+2) + ref_b].speeds[kk] = cells[i * nx + j].speeds[kk];
      }
    }
  }

  for(int a = 1; a < max_a+1; a++){
    for(int b = 1; b < max_b+1; b++){
      int ii = g_id_ii * max_a + a - 1;
      int jj = g_id_jj * max_b + b - 1;

      int cell = ii * nx + jj;

      int y_n = b + 1;
      int x_e = a + 1;
      int y_s = b - 1;
      int x_w = a - 1;

      int obst  = (obstacles[cell] ? 1 : 0);
      int nobst = (obstacles[cell] ? 0 : 1);
      float diff[NSPEEDS];
      diff[0] = 0.0;

      tmp_cells[cell].speeds[0] = local_cells[a  * nx + b ].speeds[0]; /* central cell, no movement */
      tmp_cells[cell].speeds[1] = local_cells[a  * nx + x_w].speeds[1]; /* east */
      tmp_cells[cell].speeds[2] = local_cells[y_s * nx + b ].speeds[2]; /* north */
      tmp_cells[cell].speeds[3] = local_cells[a  * nx + x_e].speeds[3]; /* west */
      tmp_cells[cell].speeds[4] = local_cells[y_n * nx + b ].speeds[4]; /* south */
      tmp_cells[cell].speeds[5] = local_cells[y_s * nx + x_w].speeds[5]; /* north-east */
      tmp_cells[cell].speeds[6] = local_cells[y_s * nx + x_e].speeds[6]; /* north-west */
      tmp_cells[cell].speeds[7] = local_cells[y_n * nx + x_e].speeds[7]; /* south-west */
      tmp_cells[cell].speeds[8] = local_cells[y_n * nx + x_w].speeds[8]; /* south-east */

      diff[1] = tmp_cells[cell].speeds[3];
      diff[2] = tmp_cells[cell].speeds[4];
      diff[3] = tmp_cells[cell].speeds[1];
      diff[4] = tmp_cells[cell].speeds[2];
      diff[5] = tmp_cells[cell].speeds[7];
      diff[6] = tmp_cells[cell].speeds[8];
      diff[7] = tmp_cells[cell].speeds[5];
      diff[8] = tmp_cells[cell].speeds[6];

      float local_density = 0.0;

      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        local_density += tmp_cells[cell].speeds[kk];
      }

      /* compute x velocity component */
      float u_x = (tmp_cells[cell].speeds[1]
                    + tmp_cells[cell].speeds[5]
                    + tmp_cells[cell].speeds[8]
                    - (tmp_cells[cell].speeds[3]
                       + tmp_cells[cell].speeds[6]
                       + tmp_cells[cell].speeds[7]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (tmp_cells[cell].speeds[2]
                    + tmp_cells[cell].speeds[5]
                    + tmp_cells[cell].speeds[6]
                    - (tmp_cells[cell].speeds[4]
                       + tmp_cells[cell].speeds[7]
                       + tmp_cells[cell].speeds[8]))
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

      for(int kk = 0; kk < NSPEEDS; kk++){
        tmp_cells[cell].speeds[kk] = (nobst) * (tmp_cells[cell].speeds[kk]
                                                + omega
                                                * (d_equ[kk] - tmp_cells[cell].speeds[kk]))
                                   + (obst * diff[kk]);
      }

      tot_us[cell] = (nobst) * (sqrt((u_x * u_x) + (u_y * u_y)));
    }
  }
}