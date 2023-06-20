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

    return;
}
