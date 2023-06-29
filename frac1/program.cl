//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//


//if you don't have opencl uncomment this:
/*

struct int3
{
    int x;
    int y;
    int z;
};
typedef struct int3 int3;

struct float3
{
    float x;
    float y;
    float z;
};
typedef struct float3 float3;

struct float8
{
    float s0;
    float s1;
    float s2;
    float s3;
    float s4;
    float s5;
    float s6;
    float s7;
};
typedef struct float8 float8;

 */

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
int fn_idx2(int3 pos);
int fn_idx3(int3 pos);

int fn_bc1(int3 pos, int3 dim);
int fn_bc2(int3 pos, int3 dim);

void bas_eval(float3 p, float ee[8]);
void bas_grad(float3 p, float3 gg[8]);
void bas_itpe(float uu2[8][4], float  bas_ee[8], float  u_eval[4]);
void bas_itpg(float uu2[8][4], float3 bas_gg[8], float3 u_grad[4]);

float  vec_dot(float3 a, float3 b);
float  vec_norm(float3 a);
float3 vec_unit(float3 a);
float3 vec_cross(float3 a, float3 b);
float8 vec_out(float3 v);

float3 vec_vaddf(float3 a, float3 b);
float3 vec_saddf(float3 a, float b);

int3   vec_vaddi(int3 a, int3 b);
int3   vec_saddi(int3 a, int b);

float3 vec_smulf(float3 a, float b);

float  sym_tr(float8 A);
float8 sym_prod(float8 A, float8 B);
float  sym_det(float8 A);
float  sym_tip(float8 A, float8 B);
float8 sym_smul(float8 A, float b);
float8 sym_add(float8 A, float8 B);

float8 mec_E(float3 g[3]);
float8 mec_S(float8 E);
float  mec_p(float8 E);

float3 eig_val(float8 A);
void   eig_vec(float8 A, float3 d, float3 v[3]);
void   eig_A1A2(float8 A, float8 *A1, float8 *A2);
void   eig_E1E2(float3 g, int dim, float8 *E1, float8 *E2);

void mem_read3(global float *buf, float uu[27][4], int3 pos, int3 dim);
void mem_read2(float uu3[27][4], float uu2[8][4], int3 ref);

/*
 ===================================
 constants
 ===================================
 */

constant int3 off2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
constant int3 off3[27] = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},{0,0,2},{1,0,2},{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}};

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
 utilities
 ===================================
 */

//flat index
int fn_idx1(int3 pos, int3 dim)
{
    return pos.x + pos.y*dim.x + pos.z*dim.x*dim.y;
}

//index 2x2x2
int fn_idx2(int3 pos)
{
    return pos.x + pos.y*2 + pos.z*4;
}

//index 3x3x3
int fn_idx3(int3 pos)
{
    return pos.x + pos.y*3 + pos.z*9;
}

//in-bounds
int fn_bc1(int3 pos, int3 dim)
{
    return (pos.x>-1)*(pos.y>-1)*(pos.z>-1)*(pos.x<dim.x)*(pos.y<dim.y)*(pos.z<dim.z);
}

//on the boundary
int fn_bc2(int3 pos, int3 dim)
{
    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);;
}

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
void bas_grad(float3 p, float3 gg[8])
{
    float x0 = 1e0f - p.x;
    float y0 = 1e0f - p.y;
    float z0 = 1e0f - p.z;
    
    float x1 = p.x;
    float y1 = p.y;
    float z1 = p.z;
    
    gg[0] = (float3){-y0*z0, -x0*z0, -x0*y0};
    gg[1] = (float3){+y0*z0, -x1*z0, -x1*y0};
    gg[2] = (float3){-y1*z0, +x0*z0, -x0*y1};
    gg[3] = (float3){+y1*z0, +x1*z0, -x1*y1};
    gg[4] = (float3){-y0*z1, -x0*z1, +x0*y0};
    gg[5] = (float3){+y0*z1, -x1*z1, +x1*y0};
    gg[6] = (float3){-y1*z1, +x0*z1, +x0*y1};
    gg[7] = (float3){+y1*z1, +x1*z1, +x1*y1};
    
    return;
}

//interp eval
void bas_itpe(float uu2[8][4], float bas_ee[8], float u_eval[4])
{
    for(int i=0; i<8; i++)
    {
        u_eval[0] += uu2[i][0]*bas_ee[i];
        u_eval[1] += uu2[i][1]*bas_ee[i];
        u_eval[2] += uu2[i][2]*bas_ee[i];
        u_eval[3] += uu2[i][3]*bas_ee[i];
    }
    return;
}

//interp grad
void bas_itpg(float uu2[8][4], float3 bas_gg[8], float3 u_grad[4])
{
    for(int i=0; i<8; i++)
    {
        u_grad[0] = vec_vaddf(u_grad[0], vec_smulf(bas_gg[i], uu2[i][0]));
        u_grad[1] = vec_vaddf(u_grad[1], vec_smulf(bas_gg[i], uu2[i][1]));
        u_grad[2] = vec_vaddf(u_grad[2], vec_smulf(bas_gg[i], uu2[i][2]));
        u_grad[3] = vec_vaddf(u_grad[3], vec_smulf(bas_gg[i], uu2[i][3]));
    }
    return;
}

/*
 ===================================
 vector R^3
 ===================================
 */

//vector inner prod
float vec_dot(float3 a, float3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

//vector 2-norm
float vec_norm(float3 a)
{
    return sqrt(vec_dot(a,a));
}

//vector normalize
float3 vec_unit(float3 a)
{
    float r = 1e0f/vec_norm(a);
    
    return (float3){a.x*r, a.y*r, a.z*r};
}

//vector cross product
float3 vec_cross(float3 a, float3 b)
{
    return (float3){a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

//outer product
float8 vec_out(float3 v)
{
    return (float8){v.x*v.x, v.x*v.y, v.x*v.z, v.y*v.y, v.y*v.z, v.z*v.z, 0e0f, 0e0f};
}

//add vector float
float3 vec_vaddf(float3 a, float3 b)
{
    return (float3){a.x + b.x, a.y + b.y, a.z + b.z};
}

//add scalar float
float3 vec_saddf(float3 a, float b)
{
    return (float3){a.x + b, a.y + b, a.z + b};
}

//add vector int
int3 vec_vaddi(int3 a, int3 b)
{
    return (int3){a.x + b.x, a.y + b.y, a.z + b.z};
}

//add scalar int
int3 vec_saddi(int3 a, int b)
{
    return (int3){a.x + b, a.y + b, a.z + b};
}

//mult scalar float
float3 vec_smulf(float3 a, float b)
{
    return (float3){a.x*b, a.y*b, a.z*b};
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

//sym scalar mult
float8 sym_smul(float8 A, float b)
{
    return (float8){A.s0*b, A.s1*b, A.s2*b, A.s3*b, A.s4*b, A.s5*b, 0e0f, 0e0f};
}

//sym add
float8 sym_add(float8 A, float8 B)
{
    return (float8){A.s0 + B.s0, A.s1 + B.s1, A.s2 + B.s2, A.s3 + B.s3, A.s4 + B.s4, A.s5 + B.s5, 0e0f, 0e0f};
}

/*
 ===================================
 mechanics
 ===================================
 */

//strain
float8 mec_E(float3 g[3])
{
    return (float8){g[0].x, 5e-1f*(g[0].y+g[1].x), 5e-1f*(g[0].z+g[2].x), g[1].y, 5e-1f*(g[1].z+g[2].y), g[2].z, 0e0f, 0e0f};
}

//stress pk2 = lam*tr(e)*I + 2*mu*e
float8 mec_S(float8 E)
{
    float a = 2e0f*mat_mu;
    float b = mat_lam*sym_tr(E);
    
    return (float8){a*E.s0 + b, a*E.s1, a*E.s2, a*E.s3 + b, a*E.s4, a*E.s5 + b, 0e0f, 0e0f};
}

//energy phi = 0.5*lam*(tr(e))^2 + mu*tr(e^2)
float mec_p(float8 E)
{
    return 5e-1f*mat_lam*pown(sym_tr(E),2) + mat_mu*sym_tr(sym_prod(E,E));
}

/*
 ===================================
 eigs (sym 3x3)
 ===================================
 */

//eigenvalues - cuppen
float3 eig_val(float8 A)
{
    float3 d;
    
    //off-diag
    float p1 = A.s1*A.s1 + A.s2*A.s2 + A.s4*A.s4;
    
    //diag
    if(p1==0e0f)
    {
        d.x = A.s0;
        d.y = A.s3;
        d.z = A.s5;
        
        return d;
    }
    
    float q  = sym_tr(A)/3e0f;
    float p2 = pown(A.s0-q,2) + pown(A.s3-q,2) + pown(A.s5-q,2) + 2e0f*p1;
    float p  = sqrt(p2/6e0f);
    
    //B = (A - qI)/p
    float8 B = (float8){(A.s0 - q)/p, A.s1/p, A.s2/p, (A.s3 - q)/p, A.s4/p, (A.s5 - q)/p, 0e0f, 0e0f};
    float r = 5e-1f*sym_det(B);
    
    float phi = acos(r)/3e0f;
    phi = (r<=-1e0f)?M_PI_F/3e0f:phi;
    phi = (r>=+1e0f)?0e0f:phi;
    
    //decreasing order
    d.z = q + 2e0f*p*cos(phi);
    d.x = q + 2e0f*p*cos(phi + (2e0f*M_PI_F/3e0f));
    d.y = 3e0f*q - (d[0] + d[2]);

    return d;
}

////eigenvectors - kopp2008
//void eig_vec(float8 A, float3 d, float3 v[3])
//{
//    //lam1
//    float3 c1 = {A.s1, A.s3-d.x, A.s4};
//    float3 c2 = {A.s2, A.s4, A.s5-d.x};
//    //lam2
//    float3 c3 = {A.s0-d.y, A.s1, A.s2};
//    float3 c4 = {A.s2, A.s4, A.s5-d.y};
//    //lam3
//    float3 c5 = {A.s0-d.z, A.s1, A.s2};
//    float3 c6 = {A.s1, A.s3-d.z, A.s4};
//
//    //cross, normalise
//    v[0] = vec_unit(vec_cross(c1, c2));
//    v[1] = vec_unit(vec_cross(c3, c4));
//    v[2] = vec_unit(vec_cross(c5, c6));
//
//    return;
//}

//eigenvectors - deledalle2017
void eig_vec(float8 A, float3 d, float3 v[3])
{
    float m0 = (A.s1*(A.s5-d.x)-A.s4*A.s2)/(A.s2*(A.s3-d.x)-A.s1*A.s4);
    float m1 = (A.s1*(A.s5-d.y)-A.s4*A.s2)/(A.s2*(A.s3-d.y)-A.s1*A.s4);
    float m2 = (A.s1*(A.s5-d.z)-A.s4*A.s2)/(A.s2*(A.s3-d.z)-A.s1*A.s4);

    //vecs
    v[0] = vec_unit((float3){(d.x - A.s5 - A.s4*m0), A.s2*m0, A.s2});
    v[1] = vec_unit((float3){(d.y - A.s5 - A.s4*m1), A.s2*m1, A.s2});
    v[2] = vec_unit((float3){(d.z - A.s5 - A.s4*m2), A.s2*m2, A.s2});

    return;
}


//split
void eig_A1A2(float8 A, float8 *A1, float8 *A2)
{
    //vals, vecs
    float3 d;
    float3 v[3];
    
    //calc
    d = eig_val(A);
    eig_vec(A, d, v);
    
//    A1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
    
    *A1 = sym_add(*A1, sym_smul(vec_out(v[0]),(d.x>0e0f)*d.x));
    *A1 = sym_add(*A1, sym_smul(vec_out(v[1]),(d.y>0e0f)*d.y));
    *A1 = sym_add(*A1, sym_smul(vec_out(v[2]),(d.z>0e0f)*d.z));
    
    *A2 = sym_add(*A2, sym_smul(vec_out(v[0]),(d.x<0e0f)*d.x));
    *A2 = sym_add(*A2, sym_smul(vec_out(v[1]),(d.y<0e0f)*d.y));
    *A2 = sym_add(*A2, sym_smul(vec_out(v[2]),(d.z<0e0f)*d.z));
        
    return;
}


//split direct from basis gradient and dim
void eig_E1E2(float3 g, int dim, float8 *E1, float8 *E2)
{
    float n = vec_norm(g);
    
    float3 g1 = vec_saddf(g, -n);
    float3 g2 = vec_saddf(g, +n);
    
    //vals
    float d0[3] = {5e-1*g1.x, 5e-1*g1.y, 5e-1*g1.z};
    float d1[3] = {5e-1*g2.x, 5e-1*g2.y, 5e-1*g2.z};
    
    //vecs
    float3 v0[3];
    v0[0] = vec_unit((float3){g1.x, g.y, g.z});
    v0[1] = vec_unit((float3){g.x, g1.y, g.z});
    v0[2] = vec_unit((float3){-g.x*g2.z, -g.y*g2.z, g.x*g.x + g.y*g.y});
    
    float3 v1[3];
    v1[0] = vec_unit((float3){g2.x, g.y, g.z});
    v1[1] = vec_unit((float3){g.x, g2.y, g.z});
    v1[2] = vec_unit((float3){-g.x*g1.z, -g.y*g1.z, g.x*g.x + g.y*g.y});
    
    //select
    *E1 = sym_smul(vec_out(v0[dim]),(d0[dim]>0e0f)*d0[dim]) + sym_smul(vec_out(v1[dim]),(d1[dim]>0e0f)*d1[dim]);
    *E2 = sym_smul(vec_out(v0[dim]),(d0[dim]<0e0f)*d0[dim]) + sym_smul(vec_out(v1[dim]),(d1[dim]<0e0f)*d1[dim]);
    
    return;
}

/*
 ===================================
 memory
 ===================================
 */

//read 3x3x3 fom buffer
void mem_read3(global float *buf, float uu3[27][4], int3 pos, int3 dim)
{
    for(int i=0; i<27; i++)
    {
        int3 adj = vec_vaddi(pos,vec_saddi(off3[i],-1));
        int idx1 = fn_idx1(adj, dim);

//        printf("%2d | %d %d %d | %3d\n",i, adj.x, adj.y, adj.z, idx1);
        
        for(int j=0; j<4; j++)
        {
            uu3[i][j] = buf[idx1+j];
        }
    }
    return;
}


//read 2x2x2 from 3x3x3
void mem_read2(float uu3[27][4], float uu2[8][4], int3 ref)
{
    for(int i=0; i<8; i++)
    {
        int3 adj = vec_vaddi(ref,off2[i]);
        int idx3 = fn_idx3(adj);

//        printf("%2d | %d %d %d | %3d\n",i, adj.x, adj.y, adj.z, idx3);
        
        for(int j=0; j<4; j++)
        {
            uu2[i][j] = uu3[idx3][j];
        }
    }
    
    return;
}

/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_init(constant   float  *buf_cc,
                     global     float  *vtx_xx,
                     global     float  *vtx_u0,
                     global     float  *vtx_u1,
                     global     float  *vtx_ff,
                     global     int    *coo_ii,
                     global     int    *coo_jj,
                     global     float  *coo_aa)
{
    int3 vtx_dim1 = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int3 vtx_pos1 = {get_global_id(0)  ,get_global_id(1)  ,get_global_id(2)};
    
    int vtx_idx1 = fn_idx1(vtx_pos1, vtx_dim1);
    
    //    printf("vtx_idx1 %3d\n",vtx_idx);
    //    printf("vtx_pos1 [%d,%d,%d]\n", vtx_pos1[0], vtx_pos1[1], vtx_pos1[2]);
    
//    int vtx_bc1 = fn_bc1(vtx_pos1, vtx_dim1);
    int vtx_bc2 = fn_bc2(vtx_pos1, vtx_dim1);
    
    int vec_row_idx  = 4*vtx_idx1;
    global float *x  = &vtx_xx[vec_row_idx];
    global float *u0 = &vtx_u0[vec_row_idx];
    global float *u1 = &vtx_u1[vec_row_idx];
    global float *f  = &vtx_ff[vec_row_idx];
    
    x[0] = (float) vtx_pos1.x;
    x[1] = (float) vtx_pos1.y;
    x[2] = (float) vtx_pos1.z;
    x[3] = (float) vtx_bc2;
    
    u0[0] = 0e0f;
    u0[1] = 0e0f;
    u0[2] = 0e0f;
    u0[3] = 0e0f;
    
    u1[0] = 0e0f;
    u1[1] = 0e0f;
    u1[2] = 0e0f;
    u1[3] = 0e0f;
    
    f[0] = 0e0f;
    f[1] = 0e0f;
    f[2] = 0e0f;
    f[3] = 0e0f;
    
    
    int blk_row = 27*16*vtx_idx1;

    //vtx
    for(int adj1=0; adj1<27; adj1++)
    {
        int3 adj_pos1 = vec_vaddi(vtx_pos1,vec_saddi(off3[adj1],-1));
        int  adj_idx1 = fn_idx1(adj_pos1, vtx_dim1);
        int  adj_bc1  = fn_bc1(adj_pos1, vtx_dim1);
        
        int blk_col = 16*adj1;
        global int   *blk_ii = &coo_ii[blk_row + blk_col];
        global int   *blk_jj = &coo_jj[blk_row + blk_col];
        global float *blk_aa = &coo_aa[blk_row + blk_col];
        
        //dims
        for(int dim1=0; dim1<4; dim1++)
        {
            for(int dim2=0; dim2<4; dim2++)
            {
                int dim_idx = 4*dim1+dim2;
                
                blk_ii[dim_idx] = adj_bc1*(4*vtx_idx1 + dim1);
                blk_jj[dim_idx] = adj_bc1*(4*adj_idx1 + dim2);
                blk_aa[dim_idx] = vtx_bc2*(vtx_idx1==adj_idx1)*(dim1==dim2);  //I
            }
        }
        
    }
    return;
}



//assemble
kernel void vtx_assm(constant   float  *buf_cc,
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
    
//    printf("vtx %2d [%d,%d,%d]\n", vtx_idx, vtx_pos.x, vtx_pos.y, vtx_pos.z);
    
    //soln 3x3x3
    float uu30[27][4];
    float uu31[27][4];
    mem_read3(vtx_u0, uu30, vtx_pos, vtx_dim);
    mem_read3(vtx_u1, uu31, vtx_pos, vtx_dim);
    
    //pointers
    int blk_row = 27*16*vtx_idx;
    int vec_row = 4*vtx_idx;
    
    //loop ele
//    for(int ele1=0; ele1<8; ele1++)
    for(int vtx1=7; vtx1>=0; vtx1--) //wierd bug
    {
//        int vtx1 = 7 - ele1;
        int ele1 = 7 - vtx1;
        int3 ele_pos = off2[ele1];
        
//        printf("ele %2d [%v3d] %d\n", ele1, ele_pos, vtx1);
        
        //soln 2x2x2
        float uu20[8][4];
        float uu21[8][4];
        mem_read2(uu30, uu20, ele_pos);
        mem_read2(uu31, uu21, ele_pos);
        
        //loop quad points (change limit with scheme 1,8,27)
        for(int qpt1=0; qpt1<1; qpt1++)
        {
            //1pt
            float3 qp = (float3){qp1,qp1,qp1};
            float  qw = qw1*qw1*qw1;
            
//            //2pt
//            float3 qp = (float3){qp2[off2[qpt1].x], qp2[off2[qpt1].y], qp2[off2[qpt1].z]};
//            float  qw = qw2[off2[qpt1].x]*qw2[off2[qpt1].y]*qw2[off2[qpt1].z];
            
//            //3pt
//            float3 qp = (float3){qp3[off3[qpt1].x], qp3[off3[qpt1].y], qp3[off3[qpt1].z]};
//            float  qw = qw3[off3[qpt1].x]*qw3[off3[qpt1].y]*qw3[off3[qpt1].z];
            
//            printf("qpt %2d [%v3e] %e\n", qpt1, qp, qw);
            
            //basis
            float  bas_ee[8];
            float3 bas_gg[8];
            
            bas_eval(qp, bas_ee);
            bas_grad(qp, bas_gg);
            
            //eval soln
            float  u0_eval[4];
            float  u1_eval[4];
            float3 u1_grad[4];
            
            bas_itpe(uu20, bas_ee, u0_eval);
            bas_itpe(uu21, bas_ee, u1_eval);
            bas_itpg(uu21, bas_gg, u1_grad);
            
            //strain
            float8 Eh = mec_E((float3*)u1_grad);
            
            //split
            float8 Eh1 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
            float8 Eh2 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
            eig_A1A2(Eh, &Eh1, &Eh2);
            
//            printf("%+e %+e %+e\n", Eh1.s0, Eh1.s1, Eh1.s2);
//            printf("%+e %+e %+e\n", Eh1.s1, Eh1.s3, Eh1.s4);
//            printf("%+e %+e %+e\n", Eh1.s2, Eh1.s4, Eh1.s5);
            
            //stress
            float8 Sh1 = mec_S(Eh1);
            float8 Sh2 = mec_S(Eh2);
            
            //energy
            float ph1 = mec_p(Eh1);
            
            //crack
            float ch0 = u0_eval[3];
            float ch1 = u1_eval[3];
            float c1 = pown(1e0f - ch1, 2);
            float c2 = 2e0f*(ch1 - 1e0f);
            
            //rhs
            for(int dim1=0; dim1<3; dim1++)
            {
                //def grad
                float3 def1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};

                //tensor basis
                def1[dim1] = bas_gg[vtx1];

                //strain
                float8 E1 = mec_E(def1);
                
                //write rhs
                vtx_ff[vec_row + dim1] += sym_tip(sym_add(sym_smul(Sh1, c1), Sh2),E1)*qw;
            }
            
            //loop adj
            for(int vtx2=0; vtx2<8; vtx2++)
            {
                int vtx_idx3 = fn_idx3(vec_vaddi(off2[ele1], off2[vtx2]));
                int blk_col = 16*vtx_idx3;
            
//                printf("vtx2 %d %d %2d\n", vtx2, vtx1, vtx_idx3);
                
                //dots
                float dot_e = bas_ee[vtx1]*bas_ee[vtx2];
                float dot_g = vec_dot(bas_gg[vtx1],bas_gg[vtx2]);
                
                printf("%+e\n", ph1);
                
                //write block cc
                coo_aa[blk_row + blk_col + 15] += ((2e0f*ph1*dot_e) + (mat_gc*(dot_e/mat_ls + dot_g*mat_ls)) + (mat_gam*(ch1<ch0)*dot_e))*qw;
                
                //loop dim1
                for(int dim1=0; dim1<3; dim1++)
                {
//                    printf("dim1 %d\n", dim1);
                    
                    //def grad
                    float3 def1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};

                    //tensor basis
                    def1[dim1] = bas_gg[vtx1];

                    //strain
                    float8 E1 = mec_E(def1);
                    
                    //couple
                    float uc = c2*bas_ee[vtx2]*sym_tip(Sh1,E1)*qw;
                    
                    //write blocks uc,cu
                    coo_aa[blk_row + blk_col + 4*dim1+3] += uc;  //uc
                    coo_aa[blk_row + blk_col + 12+dim1]  += uc;  //cu
                    
                    //loop dim2
                    for(int dim2=0; dim2<3; dim2++)
                    {
//                        printf("dim %d %d\n", dim1, dim2);

                        //split
                        float8 E21 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                        float8 E22 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                        eig_E1E2(bas_gg[vtx2], dim2, &E21, &E22);
                        
                        //stress
                        float8 S21 = mec_S(E21);
                        float8 S22 = mec_S(E22);

                        //write block uu
                        coo_aa[blk_row + blk_col + 4*dim1+dim2] += sym_tip(sym_add(sym_smul(S21, c1), S22),E1)*qw;
                        
                    }//dim2
                    
                }//dim1
                
            }//vtx2
            
        }//qpt1
        
    }//ele1
    
    return;
}
