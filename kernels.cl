#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define blksz 1

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
                      local t_speed* cells_wrk)
{
  const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */
  int Jblk, Iblk;
  t_speed tmp;
  t_speed diff;

  // work on cell (x,y) (x == jj, y == ii)
  int x = get_global_id(0);
  int y = get_global_id(1);

  // cell (x,y) is in block (Xblk,Yblk)
  int Xblk = get_group_id(0);
  int Yblk = get_group_id(1);

  // cell (x,y) is element cell (xloc, yloc) of block (Xblk, Yblk)
  int xloc = get_local_id(0);
  int yloc = get_local_id(1);
  int Num_BLK = nx / blksz;

  // upper-left-corner and inc for A and B
  int base = Yblk * nx * blksz + Xblk * blksz;

  // tmp_cell(Xblk, Yblk) = comp(cell(Xblk, Yblk))
  //Load each cell(Xblk, Yblk)
  // Each work-item loads a single element of cells
  // which is shared with teh entire work-group

  cells_wrk[yloc * blksz + xloc] = cells[base + yloc * nx + xloc];

  barrier(CLK_LOCAL_MEM_FENCE);


  // compute
  int y_n = (yloc + 1) % blksz;
  int x_e = (xloc + 1) % blksz;
  int y_s = (yloc == 0) ? (yloc + blksz - 1) : (yloc - 1);
  int x_w = (xloc == 0) ? (xloc + blksz - 1) : (xloc - 1);

  int obst = (obstacles[base + yloc * nx + xloc]) ? 1 : 0;
  int nobst = (obstacles[base + yloc * nx + xloc]) ? 0 : 1;

  tmp.speeds[0] = cells_wrk[yloc * blksz + xloc].speeds[0]; /* central cell, no movement */
  tmp.speeds[1] = cells_wrk[yloc * blksz + x_w].speeds[1]; /* east */
  tmp.speeds[2] = cells_wrk[y_s * blksz + xloc].speeds[2]; /* north */
  tmp.speeds[3] = cells_wrk[yloc * blksz + x_e].speeds[3]; /* west */
  tmp.speeds[4] = cells_wrk[y_n * blksz + xloc].speeds[4]; /* south */
  tmp.speeds[5] = cells_wrk[y_s * blksz + x_w].speeds[5]; /* north-east */
  tmp.speeds[6] = cells_wrk[y_s * blksz + x_e].speeds[6]; /* north-west */
  tmp.speeds[7] = cells_wrk[y_n * blksz + x_e].speeds[7]; /* south-west */
  tmp.speeds[8] = cells_wrk[y_n * blksz + x_w].speeds[8]; /* south-east */

  diff.speeds[0] = 0.0;
  diff.speeds[1] = tmp.speeds[3];
  diff.speeds[2] = tmp.speeds[4];
  diff.speeds[3] = tmp.speeds[1];
  diff.speeds[4] = tmp.speeds[2];
  diff.speeds[5] = tmp.speeds[7];
  diff.speeds[6] = tmp.speeds[8];
  diff.speeds[7] = tmp.speeds[5];
  diff.speeds[8] = tmp.speeds[6];

  float local_density = 0.0;
  #pragma unroll
  for(int kk = 0; kk < NSPEEDS; kk++){
    local_density += tmp.speeds[kk];
  }

  float u_x = (tmp.speeds[1]
             + tmp.speeds[5]
             + tmp.speeds[8]
             - (tmp.speeds[3]
              + tmp.speeds[6]
              + tmp.speeds[7]))
             / local_density;

  float u_y = (tmp.speeds[2]
             + tmp.speeds[5]
             + tmp.speeds[6]
             - (tmp.speeds[4]
              + tmp.speeds[7]
              + tmp.speeds[8]))
             / local_density;

  float u_sq = u_x * u_x + u_y * u_y;

  float u[NSPEEDS];
  u[1] =   u_x;        /* east */
  u[2] =         u_y;  /* north */
  u[3] = - u_x;        /* west */
  u[4] =       - u_y;  /* south */
  u[5] =   u_x + u_y;  /* north-east */
  u[6] = - u_x + u_y;  /* north-west */
  u[7] = - u_x - u_y;  /* south-west */
  u[8] =   u_x - u_y;  /* south-east */

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

  #pragma unroll
  for(int kk = 0; kk < NSPEEDS; kk++){
    tmp.speeds[kk] = (nobst) * (tmp.speeds[kk] + omega + (d_equ[kk] - tmp.speeds[kk]))
                   + (obst) * diff.speeds[kk];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  tot_us[base + yloc * nx + xloc] = (nobst) * (sqrt((u_x * u_x) + (u_y * u_y)));
}