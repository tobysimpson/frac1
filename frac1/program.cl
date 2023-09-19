//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//


/*
 ===================================
 params
 ===================================
 */

constant float mat_lam  = 1e0f;
constant float mat_mu   = 1e0f;
constant float mat_gc   = 1e0f;
constant float mat_ls   = 1e0f;
constant float mat_gam  = 1e0f;

/*
 ===================================
 prototypes
 ===================================
 */

int fn_idx(int3 pos, int3 dim);
int fn_bc1(int3 pos, int3 dim);
int fn_bc2(int3 pos, int3 dim);

/*
 ===================================
 constants
 ===================================
 */

constant int3 off2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
constant int3 off3[27] = {  {0,0,0},{1,0,0},{2,0,0},
                            {0,1,0},{1,1,0},{2,1,0},
                            {0,2,0},{1,2,0},{2,2,0},
                            {0,0,1},{1,0,1},{2,0,1},
                            {0,1,1},{1,1,1},{2,1,1},
                            {0,2,1},{1,2,1},{2,2,1},
                            {0,0,2},{1,0,2},{2,0,2},
                            {0,1,2},{1,1,2},{2,1,2},
                            {0,2,2},{1,2,2},{2,2,2}};

/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx(int3 pos, int3 dim)
{
    return pos.x + pos.y*dim.x + pos.z*dim.x*dim.y;
}


//in-bounds
int fn_bc1(int3 pos, int3 dim)
{
    return (pos.x>-1)*(pos.y>-1)*(pos.z>-1)*(pos.x<dim.x)*(pos.y<dim.y)*(pos.z<dim.z);
}

//on the boundary
int fn_bc2(int3 pos, int3 dim)
{
    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);
}

/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_init(global float  *vtx_xx,
                     global float  *U0u,
                     global float  *U0c,
                     global float  *U1u,
                     global float  *U1c,
                     global float  *F1u,
                     global float  *F1c,
                     global int    *Juu_ii,
                     global int    *Juu_jj,
                     global float  *Juu_vv,
                     global int    *Juc_ii,
                     global int    *Juc_jj,
                     global float  *Juc_vv,
                     global int    *Jcu_ii,
                     global int    *Jcu_jj,
                     global float  *Jcu_vv,
                     global int    *Jcc_ii,
                     global int    *Jcc_jj,
                     global float  *Jcc_vv)
{
    int3 vtx_dim = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int3 vtx_pos = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    
//    printf("pos %v3d\n", vtx_pos);
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    printf("idx %3d\n",vtx_idx);
    
//    int vtx_bc1 = fn_bc1(vtx_pos, vtx_dim);
//    int vtx_bc2 = fn_bc2(vtx_pos, vtx_dim);
    
    int blk_row = vtx_idx*27*9;
    
    //adj
    for(int blk_idx=0; blk_idx<27; blk_idx++)
    {
        int3 adj_pos = vtx_pos + off3[blk_idx] - 1;
        int  adj_idx = fn_idx(adj_pos, vtx_dim);
        int  adj_bc1 = fn_bc1(adj_pos, vtx_dim);
        
        printf("adj %3d %d\n",adj_idx, adj_bc1);
        
        int blk_col = blk_idx*9;

        //row
        for(int i=0; i<3; i++)
        {
            //col
            for(int j=0; j<3; j++)
            {
                int idx = blk_row + blk_col + 3*i + j;
                
                //write
                Juu_ii[idx] = adj_bc1*(3*vtx_idx + i);
                Juu_jj[idx] = adj_bc1*(3*adj_idx + i);
                Juu_vv[idx] = (vtx_idx==adj_idx)*(i==j);
            }
        }

    }
    
    return;
}


//assemble
kernel void vtx_assm(global     float  *vtx_xx,
                     global     float  *U0u,
                     global     float  *U0c,
                     global     float  *U1u,
                     global     float  *U1c,
                     global     float  *F1u,
                     global     float  *F1c,
                     global     int    *Juu_ii,
                     global     int    *Juu_jj,
                     global     float  *Juu_vv,
                     global     int    *Juc_ii,
                     global     int    *Juc_jj,
                     global     float  *Juc_vv,
                     global     int    *Jcu_ii,
                     global     int    *Jcu_jj,
                     global     float  *Jcu_vv,
                     global     int    *Jcc_ii,
                     global     int    *Jcc_jj,
                     global     float  *Jcc_vv)
{
    int3 vtx_dim = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int3 vtx_pos = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    
    printf("pos %v3d %v3d\n", vtx_pos, vtx_dim);

    return;
}
