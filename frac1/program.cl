//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

//testing scalar problem and linear elasticity for bc and solve

/*
 ===================================
 params
 ===================================
 */

constant float mat_g   = 1e-2f; //mm.ms^-2
constant float mat_rho = 1e+0f; //mg.mm^-3

/*
 ===================================
 prototypes
 ===================================
 */

int     fn_idx1(int3 pos, int3 dim);
int     fn_idx3(int3 pos);

int     fn_bnd1(int3 pos, int3 dim);
int     fn_bnd2(int3 pos, int3 dim);

void    bas_eval(float3 p, float ee[8]);
void    bas_grad(float3 p, float3 gg[8], float3 dx);

float   sym_tr(float8 A);
float   sym_det(float8 A);
float8  sym_vout(float3 v);
float3  sym_vmul(float8 A, float3 v);
float8  sym_mmul(float8 A, float8 B);
float   sym_tip(float8 A, float8 B);

float8  mec_E(float3 g[3]);
float8  mec_S(float8 E, float4 mat_prm);
float   mec_p(float8 E, float4 mat_prm);

float prb_a(float3 x);
float prb_f(float3 x);

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
 test problem
 ===================================
 */

//soln
float prb_a(float3 p)
{
    return pown(p.x, 3);
//    return p.x*p.y*(1e0f-p.x)*(1e0f-p.y);
}

//rhs
float prb_f(float3 p)
{
    return -6e0f*p.x;
//    return -2e0f*(pown(p.x,2) - p.x + pown(p.y,2) - p.y);
}

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
    return all(pos>=0)*all(pos<dim);
}

//on the boundary
int fn_bnd2(int3 pos, int3 dim)
{
    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);
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
void bas_grad(float3 p, float3 gg[8], float3 dx)
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
 symmetric R^3x3
 ===================================
 */

//sym trace
float sym_tr(float8 A)
{
    return A.s0 + A.s3 + A.s5;
}

//sym determinant
float sym_det(float8 A)
{
    return A.s0*A.s3*A.s5 - (A.s0*A.s4*A.s4 + A.s2*A.s2*A.s3 + A.s1*A.s1*A.s5) + 2e0f*A.s1*A.s2*A.s4;
}

//outer product vv^T
float8 sym_vout(float3 v)
{
    return (float8){v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z, 0e0f, 0e0f};
}

//sym Av
float3 sym_vmul(float8 A, float3 v)
{
    return (float3){dot(A.s012,v), dot(A.s134,v), dot(A.s245,v)};
}


//sym AB
float8 sym_mmul(float8 A, float8 B)
{
    return (float8){A.s0*B.s0 + A.s1*B.s1 + A.s2*B.s2,
                    A.s0*B.s1 + A.s1*B.s3 + A.s2*B.s4,
                    A.s0*B.s2 + A.s1*B.s4 + A.s2*B.s5,
                    A.s1*B.s1 + A.s3*B.s3 + A.s4*B.s4,
                    A.s1*B.s2 + A.s3*B.s4 + A.s4*B.s5,
                    A.s2*B.s2 + A.s4*B.s4 + A.s5*B.s5, 0e0f, 0e0f};
}

//sym tensor inner prod
float sym_tip(float8 A, float8 B)
{
    return A.s0*B.s0 + 2e0f*A.s1*B.s1 + 2e0f*A.s2*B.s2 + A.s3*B.s3 + 2e0f*A.s4*B.s4 + A.s5*B.s5;
}

/*
 ===================================
 mechanics
 ===================================
 */

//strain (du + du^T)/2
float8 mec_E(float3 g[3])
{
    return (float8){g[0].x, 5e-1f*(g[0].y+g[1].x), 5e-1f*(g[0].z+g[2].x), g[1].y,  5e-1f*(g[1].z+g[2].y), g[2].z, 0e0f, 0e0f};
}

//stress pk2 = lam*tr(e)*I + 2*mu*e
float8 mec_S(float8 E, float4 mat_prm)
{
    float8 S = 2e0f*mat_prm.w*E;
    S.s035 += mat_prm.z*sym_tr(E);
    
    return S;
}

//energy phi = 0.5*lam*(tr(E))^2 + mu*tr(E^2)
float mec_p(float8 E, float4 mat_prm)
{
    return 5e-1f*mat_prm.z*pown(sym_tr(E),2) + mat_prm.w*sym_tr(sym_mmul(E,E));
}

/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_init(int3   vtx_dim,
                     float3 x0,
                     float3 dx,
                     global float  *vtx_xx,
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
    int3 vtx1_pos1 = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
//    printf("vtx_dim %v3d\n", vtx_dim);

    //coord
    float3 x = x0 + dx*convert_float3(vtx1_pos1);
    vtx_xx[3*vtx1_idx1  ] = x.x;
    vtx_xx[3*vtx1_idx1+1] = x.y;
    vtx_xx[3*vtx1_idx1+2] = x.z;
    
    //c
    int idx_c = vtx1_idx1;
    U0c[idx_c] = prb_a(x);  //ana
    U1c[idx_c] = 0e0f;
    F1c[idx_c] = 0e0f;
    
    //u
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
kernel void vtx_assm(int3   ele_dim,
                     int3   vtx_dim,
                     float3 dx,
                     float4 mat_prm,
                     global float3 *vtx_xx,
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
    int3 vtx1_pos1  = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
//    printf("vtx1 %+v3d\n", vtx1_pos1);

    //volume
    float vlm = dx.x*dx.y*dx.z;
    
    //ref
    int  vtx1_idx2 = 8;
    
    //ele1
    for(uint ele1_idx2=0; ele1_idx2<8; ele1_idx2++)
    {
        int3 ele1_pos2 = off2[ele1_idx2];
        int3 ele1_pos1 = vtx1_pos1 + ele1_pos2 - 1;
        int  ele1_bnd1 = fn_bnd1(ele1_pos1, ele_dim);
        
//        printf(" ele1 %+v3d %d\n", ele1_pos1, ele1_bnd1);
        
        //ref vtx (decrement to avoid wierdness)
        vtx1_idx2 -= 1;
        
        //in-bounds
        if(ele1_bnd1)
        {
//            printf(" ele1 %+v3d\n", ele1_pos1 - 1);
//            printf("ele1 %d %d\n", ele1_idx2, vtx1_idx2);
//            printf("ele1 %d %+v3d %d %d\n", ele1_idx2, ele1_pos1, ele1_bnd1, vtx1_idx2);
            
            //qpt1 (change limit with scheme 1,8,27)
            for(int qpt1=0; qpt1<8; qpt1++)
            {
//                //1pt
//                float3 qp = (float3){qp1,qp1,qp1};
//                float  qw = qw1*qw1*qw1*vlm;
                
                //2pt
                float3 qp = (float3){qp2[off2[qpt1].x], qp2[off2[qpt1].y], qp2[off2[qpt1].z]};
                float  qw = qw2[off2[qpt1].x]*qw2[off2[qpt1].y]*qw2[off2[qpt1].z]*vlm;
                
//                //3pt
//                float3 qp = (float3){qp3[off3[qpt1].x], qp3[off3[qpt1].y], qp3[off3[qpt1].z]};
//                float  qw = qw3[off3[qpt1].x]*qw3[off3[qpt1].y]*qw3[off3[qpt1].z]*vlm;
                
                //qp global
                float3 qp_glb = dx*(convert_float3(ele1_pos1) + qp);
//                printf("  qp_glb %v3f\n", qp_glb);
                
                //basis
                float  bas_ee[8];
                float3 bas_gg[8];
                bas_eval(qp, bas_ee);
                bas_grad(qp, bas_gg, dx);
                
//                printf("bas %d %+v3f\n", vtx1_idx2, bas_gg[vtx1_idx2]);
                
                //rhs c
                int idx_c = vtx1_idx1;
                F1c[idx_c] += prb_f(qp_glb)*bas_ee[vtx1_idx2]*qw;
                
                //rhs u
                for(int dim1=0; dim1<3; dim1++)
                {
                    //gravity
                    float b[3] = {0e0f, 0e0f, -mat_g*mat_rho};
                    
                    //write
                    int idx_u = 3*vtx1_idx1 + dim1;
                    F1u[idx_u] += bas_ee[vtx1_idx2]*b[dim1]*qw;
                }
                
                //vtx2
                for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
                {
                    int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
                    int  vtx2_idx3 = fn_idx3(vtx2_pos3);
                    
//                    printf("vtx2 %v3d %d\n", vtx2_pos3, vtx2_idx3);
                    
                    //dots
                    float dot_e = bas_ee[vtx1_idx2]*bas_ee[vtx2_idx2];
                    float dot_g = dot(bas_gg[vtx1_idx2], bas_gg[vtx2_idx2]);
                    
                    //cc
                    int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
                    Jcc_vv[idx_cc] += dot_g*qw;
                    
                    //dim1
                    for(int dim1=0; dim1<3; dim1++)
                    {
                        //def grad
                        float3 def1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};

                        //tensor basis
                        def1[dim1] = bas_gg[vtx1_idx2];

                        //strain
                        float8 E1 = mec_E(def1);
                        
                        //couple
                        float uc = qw;
                        
                        //uc, cu
                        int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
                        Juc_vv[idx_uc] += uc;
                        Jcu_vv[idx_uc] += uc;
                        
                        //dim2
                        for(int dim2=0; dim2<3; dim2++)
                        {
                            //def grad
                            float3 def2[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};

                            //tensor basis
                            def2[dim2] = bas_gg[vtx2_idx2];
                            
                            //strain
                            float8 E2 = mec_E(def2);
                            
                            //stress
                            float8 S2 = mec_S(E2, mat_prm);
 
                            //uu
                            int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                            Juu_vv[idx_uu] += sym_tip(E1, S2)*qw;
                            
                        } //dim2
                        
                    } //dim1
                    
                } //vtx2
                
            } //qpt
            
        } //ele1_bnd1
        
    } //ele
    
    return;
}


//boundary conditions
kernel void fac_bnd1(int3   vtx_dim,
                     float3 x0,
                     float3 dx,
                     global float  *F1u,
                     global float  *F1c,
                     global float  *Juu_vv,
                     global float  *Juc_vv,
                     global float  *Jcu_vv,
                     global float  *Jcc_vv)
{
    int3 vtx1_pos1  = {0, get_global_id(0), get_global_id(1)}; //x=0
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    

    float3 x = x0 + dx*convert_float3(vtx1_pos1);
    
    //rhs c
    int idx_c = vtx1_idx1;
    F1c[idx_c] = prb_a(x);

    //rhs u
    for(int dim1=0; dim1<3; dim1++)
    {
        //u
        int idx_u = 3*vtx1_idx1 + dim1;
        F1u[idx_u] = 0e0f;
    }

    //vtx2
    for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
    {
        int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
        int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);

//        printf("vtx2_pos1 %+v3d %d\n", vtx2_pos1, vtx2_bnd1);

        //cc
        int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
        Jcc_vv[idx_cc] = (vtx1_idx1==vtx2_idx1);

        //dim1
        for(int dim1=0; dim1<3; dim1++)
        {
            //they are transposes => redundancy if needed

            //uc
            int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
            Juc_vv[idx_uc] = 0e0f;

            //cu
            int idx_cu = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
            Jcu_vv[idx_cu] = 0e0f;

            //dim2
            for(int dim2=0; dim2<3; dim2++)
            {
                //cc
                int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                Juu_vv[idx_uu] = (vtx1_idx1==vtx2_idx1)*(dim1==dim2);

            } //dim2

        } //dim1

    } //vtx2

    return;
}



//boundary conditions
kernel void vtx_bnd1(int3   vtx_dim,
                     float3 x0,
                     float3 dx,
                     global float  *F1u,
                     global float  *F1c,
                     global float  *Juu_vv,
                     global float  *Juc_vv,
                     global float  *Jcu_vv,
                     global float  *Jcc_vv)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    
    //on boundary
    if(fn_bnd2(vtx1_pos1, vtx_dim))
    {
//        printf("vtx1_pos1 %v3d\n", vtx1_pos1);
        
        int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
        

        float3 x = x0 + dx*convert_float3(vtx1_pos1);
        
        //rhs c
        int idx_c = vtx1_idx1;
        F1c[idx_c] = prb_a(x);

        //rhs u
        for(int dim1=0; dim1<3; dim1++)
        {
            //u
            int idx_u = 3*vtx1_idx1 + dim1;
            F1u[idx_u] = 0e0f;
        }

        //vtx2
        for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
        {
            int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
            int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);

    //        printf("vtx2_pos1 %+v3d %d\n", vtx2_pos1, vtx2_bnd1);

            //cc
            int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
            Jcc_vv[idx_cc] = (vtx1_idx1==vtx2_idx1);

            //dim1
            for(int dim1=0; dim1<3; dim1++)
            {
                //they are transposes => redundancy if needed

                //uc
                int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
                Juc_vv[idx_uc] = 0e0f;

                //cu
                int idx_cu = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
                Jcu_vv[idx_cu] = 0e0f;

                //dim2
                for(int dim2=0; dim2<3; dim2++)
                {
                    //cc
                    int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                    Juu_vv[idx_uu] = (vtx1_idx1==vtx2_idx1)*(dim1==dim2);

                } //dim2

            } //dim1

        } //vtx2
        
    }//bnd2

    return;
}
