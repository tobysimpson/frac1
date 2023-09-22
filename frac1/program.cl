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

int     fn_idx1(int3 pos, int3 dim);
int     fn_idx3(int3 pos);

int     fn_bnd1(int3 pos, int3 dim);
int     fn_bnd2(int3 pos, int3 dim);

void    bas_eval(float3 p, float ee[8]);
void    bas_grad(float3 p, float3 gg[8], float dx);

float   bas_itpe(float  uu2[8], float  bas_ee[8]);
void    bas_itpg(float3 uu2[8], float3 bas_gg[8], float3 u_grad[3]);

void    mem_r3f(global float *buf, float uu3[27], int3 pos, int3 dim);
void    mem_r2f(float uu3[27], float uu2[8], int3 pos);

void    mem_r3f3(global float *buf, float3 uu3[27], int3 pos, int3 dim);
void    mem_r2f3(float3 uu3[27], float3 uu2[8], int3 pos);

float   sym_tr(float8 A);
float8  sym_vout(float3 v);
float8  sym_prod(float8 A, float8 B);
float   sym_det(float8 A);
float   sym_tip(float8 A, float8 B);

float8  mec_E(float3 g[3]);
float8  mec_S(float8 E);
float   mec_p(float8 E);

float3  eig_val(float8 A);
void    eig_vec(float8 A, float3 dd, float3 vv[3]);
void    eig_A1A2(float8 A, float8 *A1, float8 *A2);
void    eig_E1E2(float3 g, int dim, float8 *E1, float8 *E2);

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


//interp eval
float bas_itpe(float uu2[8], float bas_ee[8])
{
    float u = 0e0f;
    
    for(int i=0; i<8; i++)
    {
        u += uu2[i]*bas_ee[i];
    }
    return u;
}

//interp grad, u_grad[i] = du[i]/{dx,dy,dz}
void bas_itpg(float3 uu2[8], float3 bas_gg[8], float3 u_grad[3])
{
    for(int i=0; i<8; i++)
    {
        u_grad[0] += uu2[i].x*bas_gg[i];
        u_grad[1] += uu2[i].y*bas_gg[i];
        u_grad[2] += uu2[i].z*bas_gg[i];
    }
    return;
}

/*
 ===================================
 memory
 ===================================
 */

//read 3x3x3 from global
void mem_r3f(global float *buf, float uu3[27], int3 pos, int3 dim)
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
void mem_r2f(float uu3[27], float uu2[8], int3 pos)
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
void mem_r3f3(global float *buf, float3 uu3[27], int3 pos, int3 dim)
{
    for(int i=0; i<27; i++)
    {
        int3 adj_pos1 = pos + off3[i] - 1;
        int  adj_idx1 = fn_idx1(adj_pos1, dim);

        //copy/cast
        uu3[i] = (float3){buf[adj_idx1], buf[adj_idx1+1], buf[adj_idx1+2]};
    }
    return;
}

//read 2x2x2 from 3x3x3
void mem_r2f3(float3 uu3[27], float3 uu2[8], int3 pos)
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

//outer product vv^T
float8 sym_vout(float3 v)
{
    return (float8){v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z, 0e0f, 0e0f};
}

//sym prod
float8 sym_prod(float8 A, float8 B)
{
    return (float8){A.s0*B.s0 + A.s1*B.s1 + A.s2*B.s2,
                    A.s0*B.s1 + A.s1*B.s3 + A.s2*B.s4,
                    A.s0*B.s2 + A.s1*B.s4 + A.s2*B.s5,
                    A.s1*B.s1 + A.s3*B.s3 + A.s4*B.s4,
                    A.s1*B.s2 + A.s3*B.s4 + A.s4*B.s5,
                    A.s2*B.s2 + A.s4*B.s4 + A.s5*B.s5, 0e0f, 0e0f};
}

//sym determinant
float sym_det(float8 A)
{
    return A.s0*A.s3*A.s5 - (A.s0*A.s4*A.s4 + A.s2*A.s2*A.s3 + A.s1*A.s1*A.s5) + 2e0f*A.s1*A.s2*A.s4;
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
float8 mec_S(float8 E)
{
    float8 S = 2e0f*mat_mu*E;
    S.s035 += mat_lam*sym_tr(E);
    
    return S;
}

//energy phi = 0.5*lam*(tr(E))^2 + mu*tr(E^2)
float mec_p(float8 E)
{
    return 5e-1f*mat_lam*pown(sym_tr(E),2) + mat_mu*sym_tr(sym_prod(E,E));
}

/*
 ===================================
 eigs (sym 3x3)
 ===================================
 */

//eigenvalues - Deledalle2017
float3 eig_val(float8 A)
{
    //weird layout
    float a = A.s0;
    float b = A.s3;
    float c = A.s5;
    float d = A.s1;
    float e = A.s4;
    float f = A.s2;
    
    float x1 = a*a + b*b + c*c - a*b - a*c - b*c + 3e0f*(d*d + e*e + f*f);
    float x2 = -(2e0f*a - b - c)*(2e0f*b - a - c)*(2e0f*c - a - b) + 9e0f*(2e0f*c - a - b)*d*d + (2e0f*b - a - c)*f*f + (2e0f*a - b - c)*e*e - 5.4e1f*d*e*f;
    
    float p1 = atan(sqrt(4e0f*x1*x1*x1 - x2*x2)/x2);
    
    //logic
    float phi = 5e-1f*M_PI_F;
    phi = (x2>0e0f)?p1         :phi;       //x2>0
    phi = (x2<0e0f)?p1 + M_PI_F:phi;       //x2<0
 
    float3 dd;
    dd.x = (a + b + c - 2e0f*sqrt(x1)*cos((phi         )/3e0f))/3e0f;
    dd.y = (a + b + c + 2e0f*sqrt(x1)*cos((phi - M_PI_F)/3e0f))/3e0f;
    dd.z = (a + b + c + 2e0f*sqrt(x1)*cos((phi + M_PI_F)/3e0f))/3e0f;
    
    return dd;
}


//eigenvectors - Kopp2008
void eig_vec(float8 A, float3 dd, float3 vv[3])
{
    //cross, normalise, skip when lam=0
    vv[0] = normalize(cross((float3){A.s0-dd.x, A.s1, A.s2},(float3){A.s1, A.s3-dd.x, A.s4}))*(dd.x!=0e0f);
    vv[1] = normalize(cross((float3){A.s0-dd.y, A.s1, A.s2},(float3){A.s1, A.s3-dd.y, A.s4}))*(dd.y!=0e0f);
    vv[2] = normalize(cross((float3){A.s0-dd.z, A.s1, A.s2},(float3){A.s1, A.s3-dd.z, A.s4}))*(dd.z!=0e0f);

    return;
}


//split
void eig_A1A2(float8 A, float8 *A1, float8 *A2)
{
    //vals, vecs
    float3 dd;
    float3 vv[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
    
    //calc
    dd = eig_val(A);
    eig_vec(A, dd, vv);
    
    *A1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
    *A2 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
    
    //outer, sum
    *A1 += sym_vout(vv[0])*(dd.x>+0e0f)*dd.x;
    *A1 += sym_vout(vv[1])*(dd.y>+0e0f)*dd.y;
    *A1 += sym_vout(vv[2])*(dd.z>+0e0f)*dd.z;
    
    *A2 += sym_vout(vv[0])*(dd.x<-0e0f)*dd.x;
    *A2 += sym_vout(vv[1])*(dd.y<-0e0f)*dd.y;
    *A2 += sym_vout(vv[2])*(dd.z<-0e0f)*dd.z;

    return;
}


//split direct from basis gradient and dim
void eig_E1E2(float3 g, int dim, float8 *E1, float8 *E2)
{
    float nrm = length(g);
    
    float3 g1 = 5e-1f*(g-nrm);
    float3 g2 = 5e-1f*(g+nrm);
    
    //vals (d2 is always zero)
    float d0[3] = {g1.x, g1.y, g1.z};
    float d1[3] = {g2.x, g2.y, g2.z};
    
    //vecs
    float3 v0[3];
    v0[0] = normalize((float3){g1.x, g.y, g.z});
    v0[1] = normalize((float3){g.x, g1.y, g.z});
    v0[2] = normalize((float3){-g.x*g2.z, -g.y*g2.z, g.x*g.x + g.y*g.y});
    
    float3 v1[3];
    v1[0] = normalize((float3){g2.x, g.y, g.z});
    v1[1] = normalize((float3){g.x, g2.y, g.z});
    v1[2] = normalize((float3){-g.x*g1.z, -g.y*g1.z, g.x*g.x + g.y*g.y});
    
    //select
    *E1 = sym_vout(v0[dim])*(d0[dim]>0e0f)*d0[dim] + sym_vout(v1[dim])*(d1[dim]>0e0f)*d1[dim];
    *E2 = sym_vout(v0[dim])*(d0[dim]<0e0f)*d0[dim] + sym_vout(v1[dim])*(d1[dim]<0e0f)*d1[dim];
    
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
    printf("vtx1 %3d\n", vtx1_idx1);
    
    //volume
    float vlm = dx*dx*dx;
    
    //read
    float U0c3[27];
    float U1c3[27];
    float3 U1u3[27];
    mem_r3f(U0c, U0c3, vtx1_pos1, vtx_dim);
    mem_r3f(U1c, U1c3, vtx1_pos1, vtx_dim);
    mem_r3f3(U1u, U1u3, vtx1_pos1, vtx_dim);
    
    //reset
    int vtx1_idx2 = 8; //wierd subraction thing
    
    //ele1
    for(int ele1_idx2=0; ele1_idx2<8; ele1_idx2++)
    {
        int3 ele1_pos2 = off2[ele1_idx2];
        int3 ele1_pos1 = vtx1_pos1 + ele1_pos2 - 1;
        int  ele1_bnd1 = fn_bnd1(ele1_pos1, ele_dim);
        vtx1_idx2 -= 1;
        
        //in-bounds
        if(ele1_bnd1)
        {
//            printf("ele1 %d %d\n", ele1_idx2, vtx1_idx2);
//            printf("ele1 %d %+v3d %d %d\n", ele1_idx2, ele1_pos1, ele1_bnd1, vtx1_idx2);
            
            //read
            float U0c2[8];
            float U1c2[8];
            float3 U1u2[8];
            mem_r2f(U0c3, U0c2, ele1_pos2);
            mem_r2f(U1c3, U1c2, ele1_pos2);
            mem_r2f3(U1u3, U1u2, ele1_pos2);
            
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
                
//                printf("bas %d %+v3f\n", vtx1_idx2, bas_gg[vtx1_idx2]);
                
                //interp
                float ch0 = bas_itpe(U0c2, bas_ee);
                float ch1 = bas_itpe(U1c2, bas_ee);
                
                //grad
                float3 uh1_grad[3] = {{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f}};
                bas_itpg(U1u2, bas_gg, uh1_grad);
                
                //strain
                float8 Eh = mec_E(uh1_grad);
                
                //split
                float8 Eh1, Eh2;
                eig_A1A2(Eh, &Eh1, &Eh2);
                
                //stress
                float8 Sh1 = mec_S(Eh1);
                float8 Sh2 = mec_S(Eh2);
                
                //energy
                float ph1 = mec_p(Eh1);
                
                //crack
                float c1 = pown(1e0f - ch1, 2);
                float c2 = 2e0f*(ch1 - 1e0f);
                
                //rhs c
                int idx_c = vtx1_idx1;
                F1c[idx_c] += 0e0f;
                
                //rhs
                for(int dim1=0; dim1<3; dim1++)
                {
                    //def grad - reset
                    float3 def1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};

                    //tensor basis
                    def1[dim1] = bas_gg[vtx1_idx2];

                    //strain
                    float8 E1 = mec_E(def1);
                    
                    //write
                    int idx_u = 3*vtx1_idx1 + dim1;
                    F1u[idx_u] += sym_tip(c1*Sh1 + Sh2,E1)*qw;
                }
                
                //vtx2
                for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
                {
                    int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
                    int  vtx2_idx3 = fn_idx3(vtx2_pos3);
                    
//                    printf("vtx2 %v3d %d\n", vtx2_pos3, vtx2_idx3);
                    
                    //dots
                    float dot_e = bas_ee[vtx1_idx2]*bas_ee[vtx2_idx2];
                    float dot_g = dot(bas_gg[vtx1_idx2],bas_gg[vtx2_idx2]);
                    
                    //cc
                    int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
                    Jcc_vv[idx_cc] += ((2e0f*ph1*dot_e) + (mat_gc*(dot_e/mat_ls + dot_g*mat_ls)) + (mat_gam*(ch1<ch0)*dot_e))*qw;
                    
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
                        float uc = c2*bas_ee[vtx2_idx2]*sym_tip(Sh1, E1)*qw;
                        
                        //uc, cu
                        int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
                        Juc_vv[idx_uc] += uc;
                        Jcu_vv[idx_uc] += uc;
                        
                        //dim2
                        for(int dim2=0; dim2<3; dim2++)
                        {
                            //split
                            float8 E21 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                            float8 E22 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                            eig_E1E2(bas_gg[vtx2_idx2], dim2, &E21, &E22);
                            
                            //stress
                            float8 S21 = mec_S(E21);
                            float8 S22 = mec_S(E22);

                            //uu
                            int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
                            Juu_vv[idx_uu] += sym_tip(c1*S21 + S22, E1)*qw;
                            
                        } //dim2
                        
                    } //dim1
                    
                } //vtx2
                
            } //qpt
            
        } //ele1_bnd1
        
    } //ele
    
    return;
}
