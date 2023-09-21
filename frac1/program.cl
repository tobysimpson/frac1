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

constant float dx       = 1e0f;

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

void bas_eval(float3 p, float ee[8]);
void bas_grad(float3 p, float3 gg[8], float dx);

void mem_r3(global float *buf, float uu3[27], int3 pos, int3 dim);
void mem_r2(float uu3[27], float uu2[8], int3 pos);

void mem_r3v3(global float *buf, float uu3[27][3], int3 pos, int3 dim);
void mem_r2v3(float uu3[27][3], float uu2[8][3], int3 pos);

/*
 ===================================
 constants
 ===================================
 */

constant int3 off2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};

constant int3 off3[27] = {
    {0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},
    {0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},
    {0,0,2},{1,0,2},{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}};

/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx1(int3 pos, int3 dim)
{
    return pos.x + dim.x*(pos.y + dim.y*pos.z);
}

//index 3x3x3
int fn_idx3(int3 pos)
{
    return pos.x + 3*pos.y + 9*pos.z;
}

//in-bounds
int fn_bnd1(int3 pos, int3 dim)
{
    return all(pos>-1)*all(pos<dim);
}

/*
 ===================================
 quadrature [0,1]
 ===================================
 */

//1-point gauss [0,1]
constant float qp1 = 5e-1f;
constant float qw1 = 1e+0f;

//2-point gauss [0,1]
constant float qp2[2] = {0.211324865405187f,0.788675134594813f};
constant float qw2[2] = {5e-1f,5e-1f};

//3-point gauss [0,1]
constant float qp3[3] = {0.112701665379258f,0.500000000000000f,0.887298334620742f};
constant float qw3[3] = {0.277777777777778f,0.444444444444444f,0.277777777777778f};

/*
 ===================================
 basis
 ===================================
 */

//eval
void bas_eval(float3 p, float ee[8])
{
    float x0 = 1e0f - p.x;
    float y0 = 1e0f - p.y;
    float z0 = 1e0f - p.z;
    
    float x1 = p.x;
    float y1 = p.y;
    float z1 = p.z;
    
    ee[0] = x0*y0*z0;
    ee[1] = x1*y0*z0;
    ee[2] = x0*y1*z0;
    ee[3] = x1*y1*z0;
    ee[4] = x0*y0*z1;
    ee[5] = x1*y0*z1;
    ee[6] = x0*y1*z1;
    ee[7] = x1*y1*z1;
    
    return;
}

//grad
void bas_grad(float3 p, float3 gg[8], float dx)
{
    float x0 = 1e0f - p.x;
    float y0 = 1e0f - p.y;
    float z0 = 1e0f - p.z;
    
    float x1 = p.x;
    float y1 = p.y;
    float z1 = p.z;
    
    gg[0] = (float3){-y0*z0, -x0*z0, -x0*y0}/dx;
    gg[1] = (float3){+y0*z0, -x1*z0, -x1*y0}/dx;
    gg[2] = (float3){-y1*z0, +x0*z0, -x0*y1}/dx;
    gg[3] = (float3){+y1*z0, +x1*z0, -x1*y1}/dx;
    gg[4] = (float3){-y0*z1, -x0*z1, +x0*y0}/dx;
    gg[5] = (float3){+y0*z1, -x1*z1, +x1*y0}/dx;
    gg[6] = (float3){-y1*z1, +x0*z1, +x0*y1}/dx;
    gg[7] = (float3){+y1*z1, +x1*z1, +x1*y1}/dx;
    
    return;
}

/*
 ===================================
 memory
 ===================================
 */

//read 3x3x3 from global
void mem_r3(global float *buf, float uu3[27], int3 pos, int3 dim)
{
    for(int i=0; i<27; i++)
    {
        int3 adj_pos1 = pos + off3[i] - 1;
        int  adj_idx1 = fn_idx1(adj_pos1, dim);

        //copy
        uu3[i] = buf[adj_idx1];
    }
    return;
}

//read 2x2x2 from 3x3x3
void mem_r2(float uu3[27], float uu2[8], int3 pos)
{
    for(int i=0; i<8; i++)
    {
        int3 adj_pos3 = pos + off2[i];
        int  adj_idx3 = fn_idx3(adj_pos3);

        //copy
        uu2[i] = uu3[adj_idx3];
    }
    return;
}


//read 3x3x3 vector from global
void mem_r3v3(global float *buf, float uu3[27][3], int3 pos, int3 dim)
{
    for(int i=0; i<27; i++)
    {
        int3 adj_pos1 = pos + off3[i] - 1;
        int  adj_idx1 = fn_idx1(adj_pos1, dim);

        //copy
        uu3[i][0] = buf[adj_idx1];
        uu3[i][1] = buf[adj_idx1+1];
        uu3[i][2] = buf[adj_idx1+2];
    }
    return;
}

//read 2x2x2 from 3x3x3
void mem_r2v3(float uu3[27][3], float uu2[8][3], int3 pos)
{
    for(int i=0; i<8; i++)
    {
        int3 adj_pos3 = pos + off2[i];
        int  adj_idx3 = fn_idx3(adj_pos3);

        //copy
        uu2[i][0] = uu3[adj_idx3][0];
        uu2[i][1] = uu3[adj_idx3][1];
        uu2[i][2] = uu3[adj_idx3][2];
    }
    return;
}



/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_init(global float3 *vtx_xx,
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
    
//    printf("vtx1_pos1 %v3d\n", vtx1_pos1);
    
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
//    printf("vtx %3d\n",vtx_idx);

    //coord
    vtx_xx[vtx1_idx1].xyz = dx*convert_float3(vtx1_pos1);
    
    //rhs c
    int idx_c = vtx1_idx1;
    U0c[idx_c] = 0e0f;
    U1c[idx_c] = 0e0f;
    F1c[idx_c] = 0e0f;
    
    //rhs u
    for(int dim1=0; dim1<3; dim1++)
    {
        //u
        int idx_u = 3*vtx1_idx1 + dim1;
        U0u[idx_u] = 0e0f;
        U1u[idx_u] = 0e0f;
        F1u[idx_u] = 0e0f;
    }
    
    //vtx2
    for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
    {
        int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
        int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);
        int  vtx2_bnd1 = fn_bnd1(vtx2_pos1, vtx_dim);

//        printf("vtx2_pos1 %+v3d %d\n", vtx2_pos1, vtx2_bnd1);

        //cc
        int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
        Jcc_ii[idx_cc] = vtx2_bnd1*vtx1_idx1;
        Jcc_jj[idx_cc] = vtx2_bnd1*vtx2_idx1;
        Jcc_vv[idx_cc] = 0e0f;
        
        //dim1
        for(int dim1=0; dim1<3; dim1++)
        {
            //they are transposes => redundancy if needed
            
            //uc
            int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
            Juc_ii[idx_uc] = vtx2_bnd1*(3*vtx1_idx1 + dim1);
            Juc_jj[idx_uc] = vtx2_bnd1*(vtx2_idx1);
            Juc_vv[idx_uc] = 0e0f;
            
            //cu
            int idx_cu = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
            Jcu_ii[idx_cu] = vtx2_bnd1*(vtx1_idx1);
            Jcu_jj[idx_cu] = vtx2_bnd1*(3*vtx2_idx1  + dim1);
            Jcu_vv[idx_cu] = 0e0f;
            
            //dim2
            for(int dim2=0; dim2<3; dim2++)
            {
                //cc
                int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                Juu_ii[idx_uu] = vtx2_bnd1*(3*vtx1_idx1 + dim1);
                Juu_jj[idx_uu] = vtx2_bnd1*(3*vtx2_idx1 + dim2);
                Juu_vv[idx_uu] = 0e0f;
            } //dim2
        } //dim1
    } //vtx2
    
    return;
}



//assemble
kernel void vtx_assm(global float3 *vtx_xx,
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
    int3 vtx_dim    = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int3 vtx1_pos1  = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    int3 ele_dim    = vtx_dim - 1;
    
//    printf("vtx1_pos1 %v3d\n", vtx1_pos1);
    
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    printf("vtx1_idx1 %3d\n", vtx1_idx1);
    
    //volume
    float vlm = dx*dx*dx;
    
    //read soln
    float U0c3[27];
    float U1c3[27];
    mem_r3(U0c, U0c3, vtx1_pos1, vtx_dim);
    mem_r3(U1c, U1c3, vtx1_pos1, vtx_dim);
    
    float U1u3[27][3];
    mem_r3v3(U1u, U1u3, vtx1_pos1, vtx_dim);
    
    
    //ele1
    for(int ele1_idx2=0; ele1_idx2<8; ele1_idx2++)
    {
        int3 ele1_pos2 = off2[ele1_idx2];
        int3 ele1_pos1 = vtx1_pos1 + ele1_pos2 - 1;
        int  ele1_bnd1 = fn_bnd1(ele1_pos1, ele_dim);
        int  vtx1_idx2 = (uint)(7 - ele1_idx2);
        
        //in-bounds
        if(ele1_bnd1)
        {
            printf("ele1_pos1 %+v3d %d %d\n", ele1_pos1, ele1_bnd1, vtx1_idx2);
            
            //read soln
            float U0c2[8];
            float U1c2[8];
            mem_r2(U0c3, U0c2, ele1_pos2);
            mem_r2(U1c3, U1c2, ele1_pos2);
            
            float U1u2[8][3];
            mem_r2v3(U1u3, U1u2, ele1_pos2);
            
            
            //qpt1 (change limit with scheme 1,8,27)
            for(int qpt1=0; qpt1<1; qpt1++)
            {
                //1pt
                float3 qp = (float3){qp1,qp1,qp1};
                float  qw = qw1*qw1*qw1*vlm;
                
//                //2pt
//                float3 qp = (float3){qp2[off2[qpt1].x], qp2[off2[qpt1].y], qp2[off2[qpt1].z]};
//                float  qw = qw2[off2[qpt1].x]*qw2[off2[qpt1].y]*qw2[off2[qpt1].z]*vlm;
                
//                printf("qpt %2d %v3f %f\n", qpt1, qp, qw);
            
                //basis
                float  bas_ee[8];
                float3 bas_gg[8];
                bas_eval(qp, bas_ee);
                bas_grad(qp, bas_gg, dx);
                
                
                
                //rhs c
                int idx_c = vtx1_idx1;
                U0c[idx_c] += 1e0f;
                U1c[idx_c] += 1e0f;
                F1c[idx_c] += 1e0f;
                
                //rhs u
                for(int dim1=0; dim1<3; dim1++)
                {
                    //u
                    int idx_u = 3*vtx1_idx1 + dim1;
                    U0u[idx_u] += 1e0f;
                    U1u[idx_u] += 1e0f;
                    F1u[idx_u] += 1e0f;
                }
                
                //vtx2
                for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
                {
                    int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
                    int  vtx2_idx3 = fn_idx3(vtx2_pos3);
                    
//                    printf("vtx2 %v3d %d\n", vtx2_pos3, vtx2_idx3);
                    
                    //cc
                    int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
                    Jcc_vv[idx_cc] += 1e0f;
                    
                    //dim1
                    for(int dim1=0; dim1<3; dim1++)
                    {
                        //uc, cu
                        int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
                        Juc_vv[idx_uc] += 1e0f;
                        Jcu_vv[idx_uc] += 1e0f;
                        
                        //dim2
                        for(int dim2=0; dim2<3; dim2++)
                        {
                            //cc
                            int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                            Juu_vv[idx_uu] += 1e0f;
                            
                        } //dim2
                        
                    } //dim1
                    
                } //vtx2
                
            } //qpt
            
        } //ele1_bnd1
        
    } //ele
    
    return;
}
