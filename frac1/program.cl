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

int fn_idx1(int3 pos, int3 dim);
int fn_idx3(int3 pos);

int fn_bnd1(int3 pos, int3 dim);
int fn_bnd2(int3 pos, int3 dim);

/*
 ===================================
 constants
 ===================================
 */

constant int3 off2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
constant int3 off3[27] = {
    {-1,-1,-1},
    { 0,-1,-1},
    {+1,-1,-1},
    {-1, 0,-1},
    { 0, 0,-1},
    {+1, 0,-1},
    {-1,+1,-1},
    { 0,+1,-1},
    {+1,+1,-1},
    
    {-1,-1, 0},
    { 0,-1, 0},
    {+1,-1, 0},
    {-1, 0, 0},
    { 0, 0, 0},
    {+1, 0, 0},
    {-1,+1, 0},
    { 0,+1, 0},
    {+1,+1, 0},
    
    {-1,-1,+1},
    { 0,-1,+1},
    {+1,-1,+1},
    {-1, 0,+1},
    { 0, 0,+1},
    {+1, 0,+1},
    {-1,+1,+1},
    { 0,+1,+1},
    {+1,+1,+1}};

                            ;

/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx1(int3 pos, int3 dim)
{
    return pos.x + pos.y*dim.x + pos.z*dim.x*dim.y;
}

//index 3x3x3
int fn_idx3(int3 pos)
{
    return pos.x + pos.y*3 + pos.z*9;
}


//in-bounds
int fn_bnd1(int3 pos, int3 dim)
{
    return (pos.x>-1)*(pos.y>-1)*(pos.z>-1)*(pos.x<dim.x)*(pos.y<dim.y)*(pos.z<dim.z);
}

//on the boundary
int fn_bnd2(int3 pos, int3 dim)
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
    int3 vtx1_pos1 = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    
    printf("vtx1_pos %v3d\n", vtx1_pos1);
    
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
//    printf("vtx %3d\n",vtx_idx);

    int idx_c = vtx1_idx1;
    
    //rhs c
    U0c[idx_c] = vtx1_idx1;
    U1c[idx_c] = vtx1_idx1;
    F1c[idx_c] = vtx1_idx1;
    
    //rhs u
    for(int dim1=0; dim1<3; dim1++)
    {
        //u
        int idx_u = 3*vtx1_idx1 + dim1;
        U0u[idx_u] = vtx1_idx1 + 1e-1f*dim1;
        U1u[idx_u] = vtx1_idx1 + 1e-1f*dim1;
        F1u[idx_u] = vtx1_idx1 + 1e-1f*dim1;
    }
    
    
    //vtx2
    for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
    {
        int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3];
        int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);
        int  vtx2_bnd1 = fn_bnd1(vtx2_pos1, vtx_dim);

        printf("vtx2 %3d %d\n", vtx2_idx1, vtx2_bnd1);

        //cc
        int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
        Jcc_ii[idx_cc] = vtx2_bnd1*vtx1_idx1;
        Jcc_jj[idx_cc] = vtx2_bnd1*vtx2_idx1;
        Jcc_vv[idx_cc] = vtx2_bnd1;
        
        //row
        for(int dim1=0; dim1<3; dim1++)
        {
            //uc
            int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
            Juc_ii[idx_uc] = vtx2_bnd1*(3*vtx1_idx1 + dim1);
            Juc_jj[idx_uc] = vtx2_bnd1*(vtx2_idx1);
            Juc_vv[idx_uc] = vtx2_bnd1*(dim1+1);
            
            //uc
            int idx_cu = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
            Jcu_ii[idx_cu] = vtx2_bnd1*(vtx1_idx1);
            Jcu_jj[idx_cu] = vtx2_bnd1*(3*vtx2_idx1  + dim1);
            Jcu_vv[idx_cu] = vtx2_bnd1*(dim1+1);
            
            
            //col
            for(int dim2=0; dim2<3; dim2++)
            {
                //cc
                int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                Juu_ii[idx_uu] = vtx2_bnd1*(3*vtx1_idx1 + dim1);
                Juu_jj[idx_uu] = vtx2_bnd1*(3*vtx2_idx1 + dim2);
                Juu_vv[idx_uu] = vtx2_bnd1*(3*dim1 + dim2 + 1);
            }
        }
    }
    
    return;
}



//init
kernel void vtx_assm(global float  *vtx_xx,
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
//    int3 vtx_dim = {get_global_size(0),get_global_size(1),get_global_size(2)};
//    int3 vtx_pos = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    
//    printf("vtx_pos %v3d\n", vtx_pos);
//
//    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
////    printf("vtx %3d\n",vtx_idx);
//
//    int blk_row = vtx_idx*27*9;
//
//    //ele
//    for(int ele1_idx2=0; ele1_idx2<8; ele1_idx2++)
//    {
//        //rel
//        uint vtx1_idx2 = (uint)(7 -  ele1_idx2);
//        printf(" vtx1_idx2 %d\n", vtx1_idx2);
//
//        int3 ele1_pos2 = off2[ele1_idx2];
//
////        printf(" ele1_pos2 %v3d\n", ele1_pos2);
//
//        //vtx2
//        for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
//        {
////            printf(" vtx2_idx2 %d\n", vtx2_idx2);
//
//            int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
////            printf("  vtx2_pos3 %v3d\n", vtx2_pos3);
//
//            int vtx2_idx3 = fn_idx3(vtx2_pos3);
//            printf(" vtx2_idx3 %d\n", vtx2_idx3);
//        }
//    }
    
    return;
}
