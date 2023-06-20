//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

//prototypes
int fn_idx(const int *pos, const int *dim);
int fn_bc1(const int *pos, const int *dim);
int fn_bc2(const int *pos, const int *dim);


//flat index
int fn_idx(const int *pos, const int *dim)
{
     return pos[0] + pos[1]*(dim[0]) + pos[2]*(dim[0]*dim[1]);
}

//in-bounds
int fn_bc1(const int *pos, const int *dim)
{
     return (pos[0]>-1)*(pos[1]>-1)*(pos[2]>-1)*(pos[0]<dim[0])*(pos[1]<dim[1])*(pos[2]<dim[2]);
}

//on the boundary
int fn_bc2(const int *pos, const int *dim)
{
     return (pos[0]==0)||(pos[1]==0)||(pos[2]==0)||(pos[0]==dim[0]-1)||(pos[1]==dim[1]-1)||(pos[2]==dim[2]-1);;
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
    
//    printf("vtx %3d\n",vtx_idx);
    
    int vtx_bc1 = fn_bc1(vtx_pos, vtx_dim);
    int vtx_bc2 = fn_bc2(vtx_pos, vtx_dim);
    
    global float *x = &vtx_xx[4*vtx_idx];
    x[0] = (float)vtx_pos[0];
    x[1] = (float)vtx_pos[1];
    x[2] = (float)vtx_pos[2];
    x[3] = (float) 0;
    
    global float *u = &vtx_uu[4*vtx_idx];
    u[0] = (float) vtx_idx + 1e-1;
    u[1] = (float) vtx_idx + 2e-1;
    u[2] = (float) vtx_idx + 3e-1;
    u[3] = (float) vtx_bc1;
    
    global float *f = &vtx_ff[4*vtx_idx];
    f[0] = (float) 4*vtx_idx+0;
    f[1] = (float) 4*vtx_idx+1;
    f[2] = (float) 4*vtx_idx+2;
    f[3] = (float) vtx_bc2;
    

    int blk_row_idx = 27*16*vtx_idx;

    global int   *blk_row_ii = &coo_ii[blk_row_idx];
    global int   *blk_row_jj = &coo_jj[blk_row_idx];
    global float *blk_row_aa = &coo_aa[blk_row_idx];


    //adj
    for(int adj_k=0; adj_k<3; adj_k++)
    {
        for(int adj_j=0; adj_j<3; adj_j++)
        {
            for(int adj_i=0; adj_i<3; adj_i++)
            {
                int adj_pos[3];
                adj_pos[0] = vtx_pos[0] + adj_i - 1;
                adj_pos[1] = vtx_pos[1] + adj_j - 1;
                adj_pos[2] = vtx_pos[2] + adj_k - 1;
                
                int adj_idx = fn_idx(adj_pos, vtx_dim);

                int blk_idx = adj_i + 3*adj_j + 9*adj_k;
                int blk_col_idx = blk_idx*16;
                
                global int   *blk_ii = &blk_row_ii[blk_col_idx];
                global int   *blk_jj = &blk_row_jj[blk_col_idx];
                global float *blk_aa = &blk_row_aa[blk_col_idx];
                
                int adj_bc1 = fn_bc1(adj_pos, vtx_dim);
//                int adj_bc2 = fn_bc2(adj_pos, vtx_dim);

//                printf("adj %3d %d %d %3d [%+d,%+d,%+d] %2d %d %d\n",vtx_idx, vtx_bc1, vtx_bc2, adj_idx, adj_pos[0], adj_pos[1], adj_pos[2], blk_idx, adj_bc1, adj_bc2);


                //dims
                for(int i=0; i<4; i++)
                {
                    for(int j=0; j<4; j++)
                    {
                        blk_ii[4*i+j] = 4*vtx_idx + i;
                        blk_jj[4*i+j] = adj_bc1*(4*adj_idx + j);
                        blk_aa[4*i+j] = vtx_bc2*(vtx_idx==adj_idx)*(i==j);
                    }
                }
                
            }
        }
    }
    
    return;
}



//assemble
kernel void vtx_assm(constant   float  *buf_cc,
                     global     float  *vtx_xx,
                     global     float  *vtx_uu,
                     global     float  *vtx_ff,
                     global     int    *coo_ii,
                     global     int    *coo_jj,
                     global     float  *coo_aa)
{
    //interior only
    int vtx_dim[3] = {get_global_size(0) + 2, get_global_size(1) + 2, get_global_size(2) + 2};
    int vtx_pos[3] = {get_global_id(0)   + 1, get_global_id(1)   + 1, get_global_id(2)   + 1};
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
    printf("vtx %3d\n",vtx_idx);
    
    return;
}
