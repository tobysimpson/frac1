//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

//proto
int fn_idx(const int *pos, const int *dim);

//flat index
int fn_idx(const int *pos, const int *dim)
{
     return pos[0] + pos[1]*(dim[0]) + pos[2]*(dim[0]*dim[1]);
}


//init
kernel void vtx_init(constant   float  *buf_cc,
                     global     float  *vtx_xx,
                     global     float  *vtx_uu,
                     global     float  *vtx_ff,
                     global     int    *coo_ii,
                     global     int    *coo_jj,
                     global     float  *coo_aa)
{
    int vtx_dim[3] = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int vtx_pos[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
    printf("vtx %2d\n",vtx_idx);
    
    global float *x = &vtx_xx[4*vtx_idx];
    x[0] = (float)vtx_pos[0];
    x[1] = (float)vtx_pos[1];
    x[2] = (float)vtx_pos[2];
    x[3] = (float)vtx_idx;
    
    global float *u = &vtx_uu[4*vtx_idx];
    u[0] = (float) vtx_idx + 1e-1;
    u[1] = (float) vtx_idx + 2e-1;
    u[2] = (float) vtx_idx + 3e-1;
    u[3] = (float) vtx_idx + 4e-1;
    
    global float *f = &vtx_ff[4*vtx_idx];
    f[0] = (float)4*vtx_idx+0;
    f[1] = (float)4*vtx_idx+1;
    f[2] = (float)4*vtx_idx+2;
    f[3] = (float)4*vtx_idx+3;
    
    
    int blk_row_idx = 27*16*vtx_idx;
    
    global int   *blk_row_ii = &coo_ii[blk_row_idx];
    global int   *blk_row_jj = &coo_jj[blk_row_idx];
    global float *blk_row_aa = &coo_aa[blk_row_idx];

    //blocks
    for(int k=0; k<27; k++)
    {
        int blk_col_idx = k*16;
        
        global int   *blk_ii = &blk_row_ii[blk_col_idx];
        global int   *blk_jj = &blk_row_jj[blk_col_idx];
        global float *blk_aa = &blk_row_aa[blk_col_idx];
        
        
        for(int i=0; i<4; i++)
        {
            for(int j=0; j<4; j++)
            {
                blk_ii[4*i+j] = 4*vtx_idx + i;
                blk_jj[4*i+j] = 4*vtx_idx + j;
                blk_aa[4*i+j] = 1e0;
            }
        }
    }

    return;
}
