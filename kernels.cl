#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define blksz 16
#define arsize 18*18

kernel void accelerate_flow(global int* obstacles,
                            int nx, int ny,
                            float density, float accel,
                            float* s0, float* s1,
                            float* s2, float* s3,
                            float* s4, float* s5,
                            float* s6, float* s7,
                            float* s8)
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
                      float* cells_s0, float* cells_s1,
                      float* cells_s2, float* cells_s3,
                      float* cells_s4, float* cells_s5,
                      float* cells_s6, float* cells_s7,
                      float* cells_s8, float* tmp_cells_s0,
                      float* tmp_cells_s1, float* tmp_cells_s2,
                      float* tmp_cells_s3, float* tmp_cells_s4,
                      float* tmp_cells_s5, float* tmp_cells_s6,
                      float* tmp_cells_s7, float* tmp_cells_s8)
{
  const float c_sq = 1.0 / 3.0; /* square of speed of sound */
  const float w0 = 4.0 / 9.0;  /* weighting factor */
  const float w1 = 1.0 / 9.0;  /* weighting factor */
  const float w2 = 1.0 / 36.0; /* weighting factor */
  float tmp[NSPEEDS];
  float diff[NSPEEDS];
  local float s0[arsize];
  local float s1[arsize];
  local float s2[arsize];
  local float s3[arsize];
  local float s4[arsize];
  local float s5[arsize];
  local float s6[arsize];
  local float s7[arsize];
  local float s8[arsize];

  local float* cells_wrk[arsize];

  // work on cell (x,y) (x == jj, y == ii)
  int x = get_global_id(0);
  int y = get_global_id(1);


  // cell (x,y) is in block (Xblk,Yblk)
  int Xblk = get_group_id(0);
  int Yblk = get_group_id(1);

  // cell (x,y) is element cell (xloc, yloc) of block (Xblk, Yblk)
  int xloc = get_local_id(0);
  int yloc = get_local_id(1);
  int xwrk = xloc + 1;
  int ywrk = yloc + 1;
  int XMAX = get_local_size(0);
  int YMAX = get_local_size(1);

  // tmp_cell(x, y) = comp(cell(Xblk, Yblk))
  // Load each cell(Xblk, Yblk)
  // Each work-item loads a single element of cells
  // which is shared with the entire work-group

  int y_above = ((Yblk+1) * blksz) % ny;
  int x_east  = ((Xblk+1) * blksz) % nx;
  int y_below = (Yblk == 0) ? ny - 1 : Yblk * blksz - 1;
  int x_west  = (Xblk == 0) ? nx - 1 : Xblk * blksz - 1;

  // load in working cell
  s0[ywrk * (blksz+2) + xwrk] = cells_s0[y * nx + x];
  s1[ywrk * (blksz+2) + xwrk] = cells_s1[y * nx + x];
  s2[ywrk * (blksz+2) + xwrk] = cells_s2[y * nx + x];
  s3[ywrk * (blksz+2) + xwrk] = cells_s3[y * nx + x];
  s4[ywrk * (blksz+2) + xwrk] = cells_s4[y * nx + x];
  s5[ywrk * (blksz+2) + xwrk] = cells_s5[y * nx + x];
  s6[ywrk * (blksz+2) + xwrk] = cells_s6[y * nx + x];
  s7[ywrk * (blksz+2) + xwrk] = cells_s7[y * nx + x];
  s8[ywrk * (blksz+2) + xwrk] = cells_s8[y * nx + x];


  int y_corner = (yloc < YMAX/2) ? 0 : blksz + 1;
  int x_corner = (xloc < XMAX/2) ? 0 : blksz + 1;
  int y_global = (yloc < YMAX/2) ? y_below : y_above;
  int x_global = (xloc < XMAX/2) ? x_west : x_east;

  if(yloc == xloc || xloc == (blksz - yloc - 1)){
    //printf("x = %d, y = %d\n xloc = %d, yloc = %d\n y_above = %d, y_below = %d\n x_east = %d, x_west = %d\n x_corner = %d, y_corner = %d\n x_global = %d, y_global = %d\n",x,y,xloc,yloc,y_above,y_below,x_east,x_west,x_corner,y_corner,x_global,y_global);
    //load x is corner
    s0[ywrk * (blksz + 2) + x_corner] = cells_s0[y * nx + x_global];
    s1[ywrk * (blksz + 2) + x_corner] = cells_s1[y * nx + x_global];
    s2[ywrk * (blksz + 2) + x_corner] = cells_s2[y * nx + x_global];
    s3[ywrk * (blksz + 2) + x_corner] = cells_s3[y * nx + x_global];
    s4[ywrk * (blksz + 2) + x_corner] = cells_s4[y * nx + x_global];
    s5[ywrk * (blksz + 2) + x_corner] = cells_s5[y * nx + x_global];
    s6[ywrk * (blksz + 2) + x_corner] = cells_s6[y * nx + x_global];
    s7[ywrk * (blksz + 2) + x_corner] = cells_s7[y * nx + x_global];
    s8[ywrk * (blksz + 2) + x_corner] = cells_s8[y * nx + x_global];

    //load y is corner
    s0[y_corner * (blksz + 2) + xwrk] = cells_s0[y_global * nx + x];
    s1[y_corner * (blksz + 2) + xwrk] = cells_s1[y_global * nx + x];
    s2[y_corner * (blksz + 2) + xwrk] = cells_s2[y_global * nx + x];
    s3[y_corner * (blksz + 2) + xwrk] = cells_s3[y_global * nx + x];
    s4[y_corner * (blksz + 2) + xwrk] = cells_s4[y_global * nx + x];
    s5[y_corner * (blksz + 2) + xwrk] = cells_s5[y_global * nx + x];
    s6[y_corner * (blksz + 2) + xwrk] = cells_s6[y_global * nx + x];
    s7[y_corner * (blksz + 2) + xwrk] = cells_s7[y_global * nx + x];
    s8[y_corner * (blksz + 2) + xwrk] = cells_s8[y_global * nx + x];

    //load corner
    s0[y_corner * (blksz + 2) + x_corner] = cells_s0[y_global * nx + x_global];
    s1[y_corner * (blksz + 2) + x_corner] = cells_s1[y_global * nx + x_global];
    s2[y_corner * (blksz + 2) + x_corner] = cells_s2[y_global * nx + x_global];
    s3[y_corner * (blksz + 2) + x_corner] = cells_s3[y_global * nx + x_global];
    s4[y_corner * (blksz + 2) + x_corner] = cells_s4[y_global * nx + x_global];
    s5[y_corner * (blksz + 2) + x_corner] = cells_s5[y_global * nx + x_global];
    s6[y_corner * (blksz + 2) + x_corner] = cells_s6[y_global * nx + x_global];
    s7[y_corner * (blksz + 2) + x_corner] = cells_s7[y_global * nx + x_global];
    s8[y_corner * (blksz + 2) + x_corner] = cells_s8[y_global * nx + x_global];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // compute
  int y_n = (ywrk + 1);
  int x_e = (xwrk + 1);
  int y_s = (ywrk - 1);
  int x_w = (xwrk - 1);

  int obst = (obstacles[y * nx + x]) ? 1 : 0;
  int nobst = (obstacles[y * nx + x]) ? 0 : 1;

  tmp[0] = s0[ywrk * (blksz+2) + xwrk]; /* central cell, no movement */
  tmp[1] = s1[ywrk * (blksz+2) + x_w]; /* east */
  tmp[2] = s2[y_s * (blksz+2) + xwrk]; /* north */
  tmp[3] = s3[ywrk * (blksz+2) + x_e]; /* west */
  tmp[4] = s4[y_n * (blksz+2) + xwrk]; /* south */
  tmp[5] = s5[y_s * (blksz+2) + x_w]; /* north-east */
  tmp[6] = s6[y_s * (blksz+2) + x_e]; /* north-west */
  tmp[7] = s7[y_n * (blksz+2) + x_e]; /* south-west */
  tmp[8] = s8[y_n * (blksz+2) + x_w]; /* south-east */

  diff[0] = tmp[0];
  diff[1] = tmp[3];
  diff[2] = tmp[4];
  diff[3] = tmp[1];
  diff[4] = tmp[2];
  diff[5] = tmp[7];
  diff[6] = tmp[8];
  diff[7] = tmp[5];
  diff[8] = tmp[6];

  float local_density = 0.0;
  #pragma unroll
  for(int kk = 0; kk < NSPEEDS; kk++){
    local_density += tmp[kk];
  }

  float u_x = (tmp[1]
             + tmp[5]
             + tmp[8]
             - (tmp[3]
              + tmp[6]
              + tmp[7]))
             / local_density;

  float u_y = (tmp[2]
             + tmp[5]
             + tmp[6]
             - (tmp[4]
              + tmp[7]
              + tmp[8]))
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

  tmp_cells_s0[y * nx + x] = (nobst) * (tmp[0] + omega * (d_equ[0] - tmp[0]))
                           + (obst) * diff[0];
  tmp_cells_s1[y * nx + x] = (nobst) * (tmp[1] + omega * (d_equ[1] - tmp[1]))
                           + (obst) * diff[1];
  tmp_cells_s2[y * nx + x] = (nobst) * (tmp[2] + omega * (d_equ[2] - tmp[2]))
                           + (obst) * diff[2];
  tmp_cells_s3[y * nx + x] = (nobst) * (tmp[3] + omega * (d_equ[3] - tmp[3]))
                           + (obst) * diff[3];
  tmp_cells_s4[y * nx + x] = (nobst) * (tmp[4] + omega * (d_equ[4] - tmp[4]))
                           + (obst) * diff[4];
  tmp_cells_s5[y * nx + x] = (nobst) * (tmp[5] + omega * (d_equ[5] - tmp[5]))
                           + (obst) * diff[5];
  tmp_cells_s6[y * nx + x] = (nobst) * (tmp[6] + omega * (d_equ[6] - tmp[6]))
                           + (obst) * diff[6];
  tmp_cells_s7[y * nx + x] = (nobst) * (tmp[7] + omega * (d_equ[7] - tmp[7]))
                           + (obst) * diff[7];
  tmp_cells_s8[y * nx + x] = (nobst) * (tmp[8] + omega * (d_equ[8] - tmp[8]))
                           + (obst) * diff[8];

  tot_us[y * nx + x] = (nobst) * (sqrt((u_x * u_x) + (u_y * u_y)));
  barrier(CLK_LOCAL_MEM_FENCE);
}