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
kernel void vtx_init(constant   float3 *buf_cc,
                     global     float  *vtx_xx,
                     global     float  *vtx_u0,
                     global     float  *vtx_u1,
                     global     float  *vtx_ff,
                     global     int    *coo_ii,
                     global     int    *coo_jj,
                     global     float  *coo_aa)
{
    int3 vtx_dim = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int3 vtx_pos = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    
    int vtx_idx = fn_idx1(vtx_pos, vtx_dim);
    
//    printf("idx %3d\n",vtx_idx);
    printf("pos %v3d\n", vtx_pos);
    
    int vtx_bc1 = fn_bc1(vtx_pos, vtx_dim);
    int vtx_bc2 = fn_bc2(vtx_pos, vtx_dim);
    
    return;
}


//assemble
kernel void vtx_assm(constant   float3 *buf_cc,
                     global     float  *vtx_xx,
                     global     float  *vtx_u0,
                     global     float  *vtx_u1,
                     global     float  *vtx_ff,
                     global     int    *coo_ii,
                     global     int    *coo_jj,
                     global     float  *coo_aa)
{
    //interior only
    int3 vtx_dim = {get_global_size(0) + 2, get_global_size(1) + 2, get_global_size(2) + 2};
    int3 vtx_pos = {get_global_id(0)   + 1, get_global_id(1)   + 1, get_global_id(2)   + 1};
    
    int vtx_idx = fn_idx1(vtx_pos, vtx_dim);
    
    printf("idx %3d\n",vtx_idx);
//    printf("pos %v3d\n", vtx_pos);
    

    return;
}
