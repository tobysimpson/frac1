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
float   bas_itpe(float uu2[8], float bas_ee[8]);
void    bas_itpg(float3 uu2[8], float3 bas_gg[8], float3 u_grad[3]);

float   prb_a(float3 x);
float   prb_f(float3 x);

float3  mtx_mv(float3 A[3], float3 v);
void    mtx_mm(float3 A[3], float3 B[3], float3 C[3]);
void    mtx_mmT(float3 A[3], float3 B[3], float3 C[3]);
void    mtx_mdmT(float3 A[3], float D[3], float3 B[3], float3 C[3]);
void    mtx_sum(float3 A[3], float3 B[3], float3 C[3]);

float   sym_tr(float8 A);
float   sym_det(float8 A);
float8  sym_vvT(float3 v);
float3  sym_mv(float8 A, float3 v);
float8  sym_mm(float8 A, float8 B);
float8  sym_mdmT(float3 A[3], float D[3]);
float8  sym_sumT(float3 A[3]);
float   sym_tip(float8 A, float8 B);

float8  mec_E(float3 g[3]);
float8  mec_S(float8 E, float4 mat_prm);
float   mec_p(float8 E, float4 mat_prm);

void    mem_gr3f(global float *buf, float uu3[27], int3 pos, int3 dim);
void    mem_gr2f(global float *buf, float uu2[8], int3 pos, int3 dim);
void    mem_lr2f(float uu3[27], float uu2[8], int3 pos);
void    mem_gr3f3(global float *buf, float3 uu3[27], int3 pos, int3 dim);
void    mem_lr2f3(float3 uu3[27], float3 uu2[8], int3 pos);

void    eig_val(float8 A, float dd[3]);
void    eig_vec(float8 A, float dd[3], float3 vv[3]);
void    eig_dcm(float8 A, float dd[3], float3 vv[3]);
void    eig_drv(float3 dA[3], float D[3], float3 V[3], float8 A1, float8 A2);


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
//    return pown(p.x,2);
    return sin(M_PI_F*p.x)*sin(M_PI_F*p.y)*sin(M_PI_F*p.z);
//    return pown(p.x, 3);
//    return p.x*p.y*(1e0f-p.x)*(1e0f-p.y);
}

//rhs -u''
float prb_f(float3 p)
{
//    return -2e0f;
    return 3e0f*pown(M_PI_F,2)*sin(M_PI_F*p.x)*sin(M_PI_F*p.y)*sin(M_PI_F*p.z);
//    return -6e0f*p.x;
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


//interp grad, u_grad[i] = du[i]/{dx,dy,dz}, rows of Jacobian
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

//global read 3x3x3 float
void mem_gr3f(global float *buf, float uu3[27], int3 pos, int3 dim)
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

//global read 2x2x2 float
void mem_gr2f(global float *buf, float uu2[8], int3 pos, int3 dim)
{
    for(int i=0; i<8; i++)
    {
        int3 adj_pos1 = pos + off2[i];
        int  adj_idx1 = fn_idx1(adj_pos1, dim);

        //copy
        uu2[i] = buf[adj_idx1];
    }
    return;
}

//local read 2x2x2 from 3x3x3
void mem_lr2f(float uu3[27], float uu2[8], int3 pos)
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

//global read 3x3x3 vector from global
void mem_gr3f3(global float *buf, float3 uu3[27], int3 pos, int3 dim)
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

//local read 2x2x2 from 3x3x3 vector
void mem_lr2f3(float3 uu3[27], float3 uu2[8], int3 pos)
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
 matrix R^3x3
 ===================================
 */

//mmult Av
float3 mtx_mv(float3 A[3], float3 v)
{
    return A[0]*v.x + A[1]*v.y + A[2]*v.z;
}

//mmult C = AB
void mtx_mm(float3 A[3], float3 B[3], float3 C[3])
{
    C[0] = mtx_mv(A,B[0]);
    C[1] = mtx_mv(A,B[1]);
    C[2] = mtx_mv(A,B[2]);

    return;
}

//mmult C = AB^T
void mtx_mmT(float3 A[3], float3 B[3], float3 C[3])
{
    C[0] = A[0]*B[0].x + A[1]*B[1].x + A[2]*B[2].x;
    C[1] = A[0]*B[0].y + A[1]*B[1].y + A[2]*B[2].y;
    C[2] = A[0]*B[0].z + A[1]*B[1].z + A[2]*B[2].z;

    return;
}

//mmult C = ADB^T, diagonal D
void mtx_mdmT(float3 A[3], float D[3], float3 B[3], float3 C[3])
{
    C[0] = D[0]*A[0]*B[0].x + D[1]*A[1]*B[1].x + D[2]*A[2]*B[2].x;
    C[1] = D[0]*A[0]*B[0].y + D[1]*A[1]*B[1].y + D[2]*A[2]*B[2].y;
    C[2] = D[0]*A[0]*B[0].z + D[1]*A[1]*B[1].z + D[2]*A[2]*B[2].z;

    return;
}

//sum
void mtx_sum(float3 A[3], float3 B[3], float3 C[3])
{
    C[0] = A[0] + B[0];
    C[1] = A[1] + B[1];
    C[2] = A[2] + B[2];
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
    return dot((float3){A.s0,A.s1,A.s2}, cross((float3){A.s1, A.s3, A.s4}, (float3){A.s2, A.s4, A.s5}));
//    return A.s0*A.s3*A.s5 - (A.s0*A.s4*A.s4 + A.s2*A.s2*A.s3 + A.s1*A.s1*A.s5) + 2e0f*A.s1*A.s2*A.s4;
}

//outer product vv^T
float8 sym_vvT(float3 v)
{
    return (float8){v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z, 0e0f, 0e0f};
}

//sym mtx-vec
float3 sym_mv(float8 A, float3 v)
{
    return (float3){dot(A.s012,v), dot(A.s134,v), dot(A.s245,v)};
}

//sym mtx-mtx
float8 sym_mm(float8 A, float8 B)
{
    return (float8){A.s0*B.s0 + A.s1*B.s1 + A.s2*B.s2,
                    A.s0*B.s1 + A.s1*B.s3 + A.s2*B.s4,
                    A.s0*B.s2 + A.s1*B.s4 + A.s2*B.s5,
                    A.s1*B.s1 + A.s3*B.s3 + A.s4*B.s4,
                    A.s1*B.s2 + A.s3*B.s4 + A.s4*B.s5,
                    A.s2*B.s2 + A.s4*B.s4 + A.s5*B.s5, 0e0f, 0e0f};
}

//mul A = VDV^T, diagonal D
float8  sym_mdmT(float3 V[3], float D[3])
{
    float8 A;
    
    A.s0 = D[0]*V[0].x*V[0].x + D[1]*V[1].x*V[1].x + D[2]*V[2].x*V[2].x;
    A.s1 = D[0]*V[0].x*V[0].y + D[1]*V[1].x*V[1].y + D[2]*V[2].x*V[2].y;
    A.s2 = D[0]*V[0].x*V[0].z + D[1]*V[1].x*V[1].z + D[2]*V[2].x*V[2].z;
    A.s3 = D[0]*V[0].y*V[0].y + D[1]*V[1].y*V[1].y + D[2]*V[2].y*V[2].y;
    A.s4 = D[0]*V[0].y*V[0].z + D[1]*V[1].y*V[1].z + D[2]*V[2].y*V[2].z;
    A.s5 = D[0]*V[0].z*V[0].z + D[1]*V[1].z*V[1].z + D[2]*V[2].z*V[2].z;
    A.s6 = 0e0f;
    A.s7 = 0e0f;
    
    return A;
}

//sum S = A+A^T
float8 sym_sumT(float3 A[3])
{
    float8 S;
    
    S.s0 = A[0].x + A[0].x;
    S.s1 = A[1].x + A[0].y;
    S.s2 = A[2].x + A[0].z;
    S.s3 = A[1].y + A[1].y;
    S.s4 = A[2].y + A[1].z;
    S.s5 = A[2].z + A[2].z;
    
    return S;
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
float8 mec_E(float3 du[3])
{
    return 5e-1f*sym_sumT(du);
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
    return 5e-1f*mat_prm.z*pown(sym_tr(E),2) + mat_prm.w*sym_tr(sym_mm(E,E));
}

/*
 ===================================
 eigs (sym 3x3)
 ===================================
 */

//eigenvalues (A real symm) - Deledalle2017
void eig_val(float8 A, float D[3])
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
 
    //write
    D[0] = (a + b + c - 2e0f*sqrt(x1)*cos((phi         )/3e0f))/3e0f;
    D[1] = (a + b + c + 2e0f*sqrt(x1)*cos((phi - M_PI_F)/3e0f))/3e0f;
    D[2] = (a + b + c + 2e0f*sqrt(x1)*cos((phi + M_PI_F)/3e0f))/3e0f;
    
    return;
}


//eigenvectors (A real symm) - Kopp2008
void eig_vec(float8 A, float D[3], float3 V[3])
{
    //cross, normalise, skip when lam=0
    V[0] = normalize(cross((float3){A.s0-D[0], A.s1, A.s2},(float3){A.s1, A.s3-D[0], A.s4}))*(D[0]!=0e0f);
    V[1] = normalize(cross((float3){A.s0-D[1], A.s1, A.s2},(float3){A.s1, A.s3-D[1], A.s4}))*(D[1]!=0e0f);
    V[2] = normalize(cross((float3){A.s0-D[2], A.s1, A.s2},(float3){A.s1, A.s3-D[2], A.s4}))*(D[2]!=0e0f);

    return;
}


//eigen decomposition
void eig_dcm(float8 A, float D[3], float3 V[3])
{
    eig_val(A, D);
    eig_vec(A, D, V);
    
    return;
}

//derivative of A in direction of dA where A = VDV^T, Jodlbauer2020, dA arrives transposed
void eig_drv(float3 dA[3], float D[3], float3 V[3], float8 A1, float8 A2)
{
    //L = (A - D[i]*I)
    
    //derivs, per eig
    float  dD[3];
    float3 dV[3];
    
    //split (1=pos, 2=neg)
    float dD1[3];
    float dD2[3];
    float D1[3];
    float D2[3];
    
    //loop eigs
    for(int i=0; i<3; i++)
    {
        //D inverse
        float  D[3];
        D[0] = (D[0]==D[i])?0e0f:1e0f/(D[0]-D[i]);
        D[1] = (D[1]==D[i])?0e0f:1e0f/(D[1]-D[i]);
        D[2] = (D[2]==D[i])?0e0f:1e0f/(D[2]-D[i]);
        
        //L inverse
        float3 L[3];
        mtx_mdmT(V,D,V,L);
        
        float3 LdA[3];
        mtx_mmT(L, dA, LdA); //because dA arrives as rows
        
        //derivs
        dV[i] = -mtx_mv(LdA, V[i]);
        dD[i] = dot(V[i], mtx_mv(dA, V[i]));
        
        //split
        dD1[i] = (D[i]>0e0f)?dD[i]:0e0f;
        dD2[i] = (D[i]<0e0f)?dD[i]:0e0f;
        
        D1[i] = (D[i]>0e0f)?D[i]:0e0f;
        D2[i] = (D[i]<0e0f)?D[i]:0e0f;
        
    }//i
    
    //A_pos = VD_posV^T
    
    float3 M1[3];               //dV*D_pos*V^T
    float3 M2[3];               //dV*D_neg*V^T
    mtx_mdmT(dV,D1,V,M1);
    mtx_mdmT(dV,D2,V,M2);
    
    //finally, A1/A2, pos/neg
    A1 = sym_sumT(M1) + sym_mdmT(V,dD1);
    A2 = sym_sumT(M2) + sym_mdmT(V,dD2);
    
    return;
}




/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_init(const  int3    vtx_dim,
                     const  float3  x0,
                     const  float3  dx,
                     global float  *vtx_xx,
                     global float  *U1u,
                     global float  *F1u,
                     global float  *U1c,
                     global float  *F1c,
                     global float  *A1c,
                     global float  *E1c,
                     global int    *Juu_ii,
                     global int    *Juu_jj,
                     global float  *Juu_vv,
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
    U1c[idx_c] = 0e0f;
    F1c[idx_c] = 0e0f;
    A1c[idx_c] = prb_a(x);
    E1c[idx_c] = 0e0f;
    
    //u
    for(int dim1=0; dim1<3; dim1++)
    {
        //u
        int idx_u = 3*vtx1_idx1 + dim1;
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
//            //uc
//            int idx_uc = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
//            Juc_ii[idx_uc] = vtx2_bnd1*(3*vtx1_idx1 + dim1);
//            Juc_jj[idx_uc] = vtx2_bnd1*(vtx2_idx1);
//            Juc_vv[idx_uc] = 0e0f;
//
//            //cu
//            int idx_cu = 27*3*vtx1_idx1 + 3*vtx2_idx3 + dim1;
//            Jcu_ii[idx_cu] = vtx2_bnd1*(vtx1_idx1);
//            Jcu_jj[idx_cu] = vtx2_bnd1*(3*vtx2_idx1  + dim1);
//            Jcu_vv[idx_cu] = 0e0f;
            
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
kernel void vtx_assm(const int3     vtx_dim,
                     const float3   x0,
                     const float3   dx,
                     const float4   mat_prm,
                     global float  *U1u,
                     global float  *F1u,
                     global float  *U1c,
                     global float  *F1c,
                     global float  *Juu_vv,
                     global float  *Jcc_vv)
{
    int3 ele_dim = vtx_dim - 1;
    int3 vtx1_pos1  = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    int  vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
//    printf("vtx1 %+v3d\n", vtx1_pos1);

    //volume
    float vlm = dx.x*dx.y*dx.z;
    
    //read
    float3 uu3[27];
    mem_gr3f3(U1u, uu3, vtx1_pos1, vtx_dim);
    
    //ref - avoids -ve int bug
    int  vtx1_idx2 = 8;
    
    //ele1
    for(uint ele1_idx2=0; ele1_idx2<8; ele1_idx2++)
    {
        int3 ele1_pos2 = off2[ele1_idx2];
        int3 ele1_pos1 = vtx1_pos1 + ele1_pos2 - 1;
        int  ele1_bnd1 = fn_bnd1(ele1_pos1, ele_dim);
        
        //ref vtx (decrement to avoid bug)
        vtx1_idx2 -= 1;
        
        //in-bounds
        if(ele1_bnd1)
        {
            //read
            float3 uu2[8];
            mem_lr2f3(uu3, uu2, ele1_pos2);
            
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
                float3 qp_glb = x0 + dx*(convert_float3(ele1_pos1) + qp);
                
                //basis
                float  bas_ee[8];
                float3 bas_gg[8];
                bas_eval(qp, bas_ee);
                bas_grad(qp, bas_gg, dx);
                
                //interp
                float3 duh[3] = {{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f},{0e0f,0e0f,0e0f}};
                bas_itpg(uu2, bas_gg, duh);
                
                //strain
                float8 Eh = mec_E(duh);
                
                //test trace tr(e(u)) > 0 for later
                float trEh = (sym_tr(Eh)>0e0f);
                
                //decompose Eh
                float  D[3];
                float3 V[3];
                eig_dcm(Eh, D, V);
                
                //eig_drv(float3 dA[3], float D[3], float3 V[3], float8 A1, float8 A2)
                

                
                //vtx2
                for(int vtx2_idx2=0; vtx2_idx2<8; vtx2_idx2++)
                {
                    int3 vtx2_pos3 = ele1_pos2 + off2[vtx2_idx2];
                    int  vtx2_idx3 = fn_idx3(vtx2_pos3);
                    

                    
                    //dim1
                    for(int dim1=0; dim1<3; dim1++)
                    {
                        //tensor basis
                        float3 du1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                        du1[dim1] = bas_gg[vtx1_idx2];

                        //strain
                        float8 E1 = mec_E(du1);
                        
                        //dim2
                        for(int dim2=0; dim2<3; dim2++)
                        {
                            //tensor basis
                            float3 du2[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                            du2[dim2] = bas_gg[vtx2_idx2];
                            
                            //strain
                            float8 E2 = mec_E(du2);
                            
                            //stress
//                            float8 S2 = mec_S(E2, mat_prm);
 
                            //uu
                            int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
//                            Juu_vv[idx_uu] += sym_tip(E1, S2)*qw;
                            
                        } //dim2
                        
                    } //dim1
                    
                } //vtx2
                
            } //qpt
            
        } //ele1_bnd1
        
    } //ele
    
    return;
}


//bc - external surface
kernel void vtx_bnd1(const int3    vtx_dim,
                     const float3  x0,
                     const float3  dx,
                     global float *U1c,
                     global float *F1c,
                     global float *Jcc_vv)
{
    int3 vtx1_pos1  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    
    //on boundary
    if(fn_bnd2(vtx1_pos1, vtx_dim))
    {
//        printf("vtx1_pos1 %v3d\n", vtx1_pos1);
        
        int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
        

        float3 x = x0 + dx*convert_float3(vtx1_pos1);
        
        //c - (soln and rhs for cg)
        int idx_c = vtx1_idx1;
        U1c[idx_c] = prb_a(x);
        F1c[idx_c] = prb_a(x);

        //rhs u
        for(int dim1=0; dim1<3; dim1++)
        {
            //u
//            int idx_u = 3*vtx1_idx1 + dim1;
//            F1u[idx_u] = 0e0f;
        }

        //vtx2
        for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
        {
            int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
            int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);

    //        printf("vtx2_pos1 %+v3d %d\n", vtx2_pos1, vtx2_bnd1);

            //cc - I
            int idx_cc = 27*vtx1_idx1 + vtx2_idx3;
            Jcc_vv[idx_cc] = (vtx1_idx1==vtx2_idx1);

            //dim1
            for(int dim1=0; dim1<3; dim1++)
            {
                //dim2
                for(int dim2=0; dim2<3; dim2++)
                {
                    //uu
//                    int idx_uu = 27*9*vtx1_idx1 + 9*vtx2_idx3 + 3*dim1 + dim2;
//                    Juu_vv[idx_uu] = (vtx1_idx1==vtx2_idx1)*(dim1==dim2);

                } //dim2

            } //dim1

        } //vtx2
        
    }//bnd2

    return;
}


//error
kernel void vtx_err1(const int3     vtx_dim,
                     global float   *U1c,
                     global float   *A1c,
                     global float   *E1c)
{
    int3 vtx_pos  = {get_global_id(0), get_global_id(1), get_global_id(2)};
    int  vtx_idx = fn_idx1(vtx_pos, vtx_dim);

    //write
    E1c[vtx_idx] = fabs(A1c[vtx_idx] - U1c[vtx_idx]);

    return;
}


//bc - zero dirichlet mech (2D)
kernel void fac_bnd1(const int3     vtx_dim,
                     global float  *U1u,
                     global float  *F1u,
                     global float  *Juu_vv)
{
    int3 vtx1_pos1  = {0, get_global_id(0), get_global_id(1)}; //x=0
    int vtx1_idx1 = fn_idx1(vtx1_pos1, vtx_dim);
    
    //rhs u
    for(int dim1=0; dim1<3; dim1++)
    {
        //u
        int idx_u = 3*vtx1_idx1 + dim1;
        U1u[idx_u] = 0e0f;
        F1u[idx_u] = 0e0f;
    }

    //vtx2
    for(int vtx2_idx3=0; vtx2_idx3<27; vtx2_idx3++)
    {
        int3 vtx2_pos1 = vtx1_pos1 + off3[vtx2_idx3] - 1;
        int  vtx2_idx1 = fn_idx1(vtx2_pos1, vtx_dim);

        //dim1
        for(int dim1=0; dim1<3; dim1++)
        {
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

