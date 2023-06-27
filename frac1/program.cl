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

constant float mat_lam = 1e0f;
constant float mat_mu  = 1e0f;

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

float  vec_dot(float3 a, float3 b);
float  vec_norm(float3 a);
float3 vec_unit(float3 a);
float3 vec_cross(float3 a, float3 b);
float8 vec_out(float3 v);
float3 vec_add(float3 a, float3 b);
float3 vec_smul(float3 a, float b);

float  sym_tr(float8 A);
float8 sym_sq(float8 A);
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

/*
 ===================================
 constants
 ===================================
 */

constant int3 off2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
constant int3 off3[27] = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},{0,0,2},{1,0,2},{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}};

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

int fn_idx2(int3 pos)
{
    return pos.x + pos.y*2 + pos.z*4;
}

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
 quadrature [0,1]
 ===================================
 */

//1-point gauss [0,1]
constant float qpt_x = 5e-1f;
constant float qpt_w = 1e+0f;

////2-point gauss [0,1]
//constant float qpt_x[2] = {0.211324865405187f,0.788675134594813f};
//constant float qpt_w[2] = {5e-1f,5e-1f};

////3-point gauss [0,1]
//constant float qpt_x[3] = {0.112701665379258f,0.500000000000000f,0.887298334620742f};
//constant float qpt_w[3] = {0.277777777777778f,0.444444444444444f,0.277777777777778f};

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

float3 vec_add(float3 a, float3 b)
{
    return (float3){a.x + b.x, a.y + b.y, a.z + b.z};
}

float3 vec_smul(float3 a, float b)
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

//sym squared
float8 sym_sq(float8 A)
{
    return (float8){A.s0*A.s0 + A.s1*A.s1 + A.s2*A.s2,
                    A.s0*A.s1 + A.s1*A.s3 + A.s2*A.s4,
                    A.s0*A.s2 + A.s1*A.s4 + A.s2*A.s5,
                    A.s1*A.s1 + A.s3*A.s3 + A.s4*A.s4,
                    A.s1*A.s2 + A.s3*A.s4 + A.s4*A.s5,
                    A.s2*A.s2 + A.s4*A.s4 + A.s5*A.s5, 0e0f, 0e0f};
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

//strain, g[0] = [u0_x, u0_y u0_z]
float8 mec_E(float3 g[3])
{
    return (float8){g[0].x, 5e-1f*g[0].y+g[1].x, 5e-1f*g[0].z+g[2].x, g[1].y, 5e-1f*g[1].z+g[2].y, g[2].z, 0e0f, 0e0f};;
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
    return 5e-1f*mat_lam*pown(sym_tr(E),2) + mat_mu*sym_tr(sym_sq(E));
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

////eigenvectors
//void eig_vec(float a[6], float d[3], float v[3][3])
//{
//    //lam1
//    float c1[3] = {a[1], a[3]-d[0], a[4]};
//    float c2[3] = {a[2], a[4], a[5]-d[0]};
//    //lam2
//    float c3[3] = {a[0]-d[1], a[1], a[2]};
//    float c4[3] = {a[2], a[4], a[5]-d[1]};
//    //lam3
//    float c5[3] = {a[0]-d[2], a[1], a[2]};
//    float c6[3] = {a[1], a[3]-d[2], a[4]};
//
//    //vecs
//    vec_cross(c1, c2, v[0]);
//    vec_cross(c3, c4, v[1]);
//    vec_cross(c5, c6, v[2]);
//
//    //normalise
//    vec_unit(v[0]);
//    vec_unit(v[1]);
//    vec_unit(v[2]);
//
//
//    return;
//}

//eigenvectors
void eig_vec(float8 A, float3 d, float3 v[3])
{
    float m0 = (A.s1*(A.s5-d.x)-A.s4*A.s2)/(A.s2*(A.s3-d.x)-A.s1*A.s4);
    float m1 = (A.s1*(A.s5-d.y)-A.s4*A.s2)/(A.s2*(A.s3-d.y)-A.s1*A.s4);
    float m2 = (A.s1*(A.s5-d.z)-A.s4*A.s2)/(A.s2*(A.s3-d.z)-A.s1*A.s4);

    //vecs
    v[0] = vec_unit((float3){(d.x - A.s5 - A.s4*m0)/A.s2, m0, 1e0f});
    v[1] = vec_unit((float3){(d.y - A.s5 - A.s4*m1)/A.s2, m1, 1e0f});
    v[2] = vec_unit((float3){(d.z - A.s5 - A.s4*m2)/A.s2, m2, 1e0f});

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
    
    sym_add(*A1, (d.x>0e0f)*d.x*vec_out(v[0]));
    sym_add(*A1, (d.y>0e0f)*d.y*vec_out(v[1]));
    sym_add(*A1, (d.z>0e0f)*d.z*vec_out(v[2]));
    
    sym_add(*A2, (d.x<0e0f)*d.x*vec_out(v[0]));
    sym_add(*A2, (d.y<0e0f)*d.y*vec_out(v[1]));
    sym_add(*A2, (d.z<0e0f)*d.z*vec_out(v[2]));
        
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
                     global     float  *vtx_uu,
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
    
    int vtx_bc1 = fn_bc1(vtx_pos1, vtx_dim1);
    int vtx_bc2 = fn_bc2(vtx_pos1, vtx_dim1);
    
    int vec_row_idx = 4*vtx_idx1;
    global float *x = &vtx_xx[vec_row_idx];
    global float *u = &vtx_uu[vec_row_idx];
    global float *f = &vtx_ff[vec_row_idx];
    
    x[0] = (float) vtx_pos1.x;
    x[1] = (float) vtx_pos1.y;
    x[2] = (float) vtx_pos1.z;
    x[3] = (float) vtx_bc2;
    
    u[0] = (float) 1;
    u[1] = (float) 2;
    u[2] = (float) 3;
    u[3] = (float) 4;
    
    f[0] = (float) 2;
    f[1] = (float) 4;
    f[2] = (float) 6;
    f[3] = (float) 8;
    
    
    int blk_row_idx = 27*16*vtx_idx1;
    global int   *blk_row_ii = &coo_ii[blk_row_idx];
    global int   *blk_row_jj = &coo_jj[blk_row_idx];
    global float *blk_row_aa = &coo_aa[blk_row_idx];
    
    
    //vtx
    for(int adj1=0; adj1<27; adj1++)
    {
        int3 adj_pos1 = {vtx_pos1.x + off3[adj1].x - 1, vtx_pos1.y + off3[adj1].y - 1, vtx_pos1.z + off3[adj1].z - 1};
        int  adj_idx1 = fn_idx1(adj_pos1, vtx_dim1);
        int  adj_bc1  = fn_bc1(adj_pos1, vtx_dim1);
        
        int blk_col_idx = adj1*16;
        global int   *blk_ii = &blk_row_ii[blk_col_idx];
        global int   *blk_jj = &blk_row_jj[blk_col_idx];
        global float *blk_aa = &blk_row_aa[blk_col_idx];
        
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
                     global     float  *vtx_uu,
                     global     float  *vtx_ff,
                     global     int    *coo_ii,
                     global     int    *coo_jj,
                     global     float  *coo_aa)
{
    //interior only
    int3 vtx_dim1 = {get_global_size(0) + 2, get_global_size(1) + 2, get_global_size(2) + 2};
    int3 vtx_pos1 = {get_global_id(0)   + 1, get_global_id(1)   + 1, get_global_id(2)   + 1};
    
    int vtx_idx1 = fn_idx1(vtx_pos1, vtx_dim1);
    
    printf("vtx %3d\n",vtx_idx1);
    //    printf("vtx_pos [%d,%d,%d]\n", vtx_pos[0], vtx_pos[1], vtx_pos[2]);
    
    //pointers K, U, F
    int blk_row_idx = 27*16*vtx_idx1;
    global float *blk_row_aa = &coo_aa[blk_row_idx];
    
    int vec_row_idx = 4*vtx_idx1;
    global float *vec_row_uu = &vtx_uu[vec_row_idx];
    global float *vec_row_ff = &vtx_ff[vec_row_idx];
    
    //loop ele
//    for(int ele_i=0; ele_i<8; ele_i++)
    for(int vtx1=7; vtx1>=0; vtx1--)//such a wierd bug?
    {
        //vtx1 (blk row)
        int ele_i = 0x7 - vtx1;
        
        printf("ele  %d %d\n",ele_i, vtx1);
        
        //per vtx
        int3    vv_pos1[8];     //global pos
        int     vv_idx1[8];     //global idx
        int3    vv_pos3[8];     //3x3x3 pos
        int     vv_idx3[8];     //3x3x3 idx
        
        float   vv_u[8][4];     //soln val
        
        //loop vtx - eval
        for(int vtx_i=0; vtx_i<8; vtx_i++)
        {
            vv_pos3[vtx_i] = (int3){off2[ele_i].x + off2[vtx_i].x, off2[ele_i].y + off2[vtx_i].y, off2[ele_i].z + off2[vtx_i].z};
            
//            printf("%v3d\n", vv_pos3[vtx_i]);
            
            
            vv_idx3[vtx_i] = fn_idx3(vv_pos3[vtx_i]);
            
            vv_pos1[vtx_i] = (int3){vtx_pos1.x + vv_pos3[vtx_i].x - 1, vtx_pos1.y + vv_pos3[vtx_i].y - 1, vtx_pos1.z + vv_pos3[vtx_i].z - 1};
            
            vv_idx1[vtx_i] = fn_idx1(vv_pos1[vtx_i], vtx_dim1);
            
            printf("vv_idx1 %3d\n",vv_idx1[vtx_i]);
            
            //soln
            global float *u = &vtx_uu[4*vv_idx1[vtx_i]];
            vv_u[vtx_i][0] = u[0];
            vv_u[vtx_i][1] = u[1];
            vv_u[vtx_i][2] = u[2];
            vv_u[vtx_i][3] = u[3];
            
            printf("vv_u[vtx_i] %e\n",vv_u[vtx_i][2]);
            
        }//vtx
        
        
        //loop qpt
        for(int qpt1=0; qpt1<1; qpt1++)
        {
            printf("qpt %2d\n",qpt1);
            
            //1pt
            float3 qp = (float3){qpt_x,qpt_x,qpt_x};
            float  qw = qpt_w*qpt_w*qpt_w;
            
//            //2pt
//            float3 qp = (float3){qpt_x[off2[qpt1][0]],qpt_x[off2[qpt1][1]],qpt_x[off2[qpt1][2]]};
//            float qw    =        qpt_w[off2[qpt1][0]]*qpt_w[off2[qpt1][1]]*qpt_w[off2[qpt1][2]];
            
//            //3pt
//            float qp[3] = {qpt_x[off3[qpt1][0]],qpt_x[off3[qpt1][1]],qpt_x[off3[qpt1][2]]};
//            float qw    =  qpt_w[off3[qpt1][0]]*qpt_w[off3[qpt1][1]]*qpt_w[off3[qpt1][2]];
            
            //basis
            float  ee[8];
            float3 gg[8];
            
            bas_eval(qp, ee);
            bas_grad(qp, gg);
            
//            printf("ee   %e %e %e %e %e %e %e %e\n", ee[0], ee[1], ee[2], ee[3], ee[4], ee[5], ee[6], ee[7]);
//            printf("gg.x %+e %+e %+e %+e %+e %+e %+e %+e\n", gg[0].x, gg[1].x, gg[2].x, gg[3].x, gg[4].x, gg[5].x, gg[6].x, gg[7].x);
//            printf("gg.y %+e %+e %+e %+e %+e %+e %+e %+e\n", gg[0].y, gg[1].y, gg[2].y, gg[3].y, gg[4].y, gg[5].y, gg[6].y, gg[7].y);
//            printf("gg.z %+e %+e %+e %+e %+e %+e %+e %+e\n", gg[0].z, gg[1].z, gg[2].z, gg[3].z, gg[4].z, gg[5].z, gg[6].z, gg[7].z);
            
            
            //soln
            float3 u_grad[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
            float c_eval     = 0e0f;

            //eval
            for(int vtx_i=0; vtx_i<8; vtx_i++)
            {
                //u_grad
                for(int dim1=0; dim1<3; dim1++)
                {
                    u_grad[dim1] = vec_add(u_grad[dim1], vec_smul(gg[vtx_i], vv_u[vtx_i][dim1]));
                }
                //c_eval
                c_eval += vv_u[vtx_i][3]*ee[vtx_i];
            }

            // notation
            // E/S/p = strain/stress/energy
            // h/1/2 = mesh/vtx1/vtx2
            // 1/2   = pos/neg


            //strain (sym)
            float8 Eh = mec_E(u_grad);

            //split
            float8 Eh1 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
            float8 Eh2 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
            eig_A1A2(Eh, &Eh1, &Eh2);

            //stress
            float8 Sh1 = mec_S(Eh1);

            //energy
            float ph1 = mec_p(Eh1);

            //crack
            float c1 = pown(1e0f - c_eval, 2);
            float c2 = 2e0f*(c_eval - 1e0f);
            
            printf("c1 %e c2 %e\n",c1,c2);
            
        
            //loop vtx2 (blk col)
            for(int vtx2=0; vtx2<8; vtx2++)
            {
                printf("vtx %d %d\n", vtx2, vtx1);
                
                //blk
                global float *blk_aa = &blk_row_aa[16*vv_idx3[vtx2]];
                
                //dims 3x3
                for(int dim1=0; dim1<3; dim1++)
                {
                    
                    
                    //grad
                    //tensor basis
                    //strain
                    
//                    //uc
//                    blk_aa[4*dim1+3] += 1e0f;
//
//                    //cu
//                    blk_aa[12+dim1] += 1e0f;
                    

                    for(int dim2=0; dim2<3; dim2++)
                    {
//                        printf("dim %d %d\n", dim1, dim2);
                        
                        //grad
                        float3 g1[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                        float3 g2[3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
                        
                        //tensor basis
                        g1[dim1] = gg[vtx1];
                        g2[dim2] = gg[vtx2];
                        

//                        printf("g1 %+e %+e %+e\n   %+e %+e %+e\n   %+e %+e %+e\n", g1[0].x, g1[0].y, g1[0].z, g1[1].x, g1[1].y, g1[1].z, g1[2].x, g1[2].y, g1[2].z);
//                        printf("g2 %+e %+e %+e\n   %+e %+e %+e\n   %+e %+e %+e\n", g2[0].x, g2[0].y, g2[0].z, g2[1].x, g2[1].y, g2[1].z, g2[2].x, g2[2].y, g2[2].z);
                        
                        //strain
                        float8 E1 = mec_E(g1);
                        float8 E2 = mec_E(g2);
                        
//                        printf("g2 %+e %+e %+e\n   %+e %+e %+e\n   %+e %+e %+e\n", g2[0].x, g2[0].y, g2[0].z, g2[1].x, g2[1].y, g2[1].z, g2[2].x, g2[2].y, g2[2].z);
                        
                        //split
                        float8 E21 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                        float8 E22 = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                        eig_A1A2(E2, &E21, &E22);
                        
                        //stress
                        float8 S21 = mec_S(E21);
                        float8 S22 = mec_S(E22);

                        //uu
//                        blk_aa[4*dim1+dim2] += sym_tip(E2,E1)*qw;
//                        blk_aa[4*dim1+dim2] += sym_tip(sym_add(sym_smul(S21, c1), S22),E1)*qw;
                        
                        float8 T = E2;

                        blk_aa[0] = T.s0;
                        blk_aa[1] = T.s1;
                        blk_aa[2] = T.s2;


                        blk_aa[4] = T.s1;
                        blk_aa[5] = T.s3;
                        blk_aa[6] = T.s4;


                        blk_aa[8] = T.s2;
                        blk_aa[9] = T.s4;
                        blk_aa[10] = T.s5;
                        

                        
                    }
                }
                
                //cc
//                blk_aa[15] += vec_dot(gg[vtx1], gg[vtx2])*qw;
                
            } //vtx
            
        } //qpt
        
    } //ele
    
    return;
}
