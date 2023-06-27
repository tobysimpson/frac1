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

int fn_idx(int3 pos, int3 dim);
int fn_bc1(int3 pos, int3 dim);
int fn_bc2(int3 pos, int3 dim);

void bas_eval(float3 p, float ee[8]);
void bas_grad(float3 p, float3 gg[8]);

float  vec_dot(float3 a, float3 b);
float  vec_norm(float3 a);
float3 vec_unit(float3 a);
float3 vec_cross(float3 a, float3 b);
float8 vec_out(float3 v);

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
void   eig_split(float8 A, float8 A1, float8 A2);

/*
 ===================================
 constants
 ===================================
 */

constant int3 idx2[8] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
constant int3 idx3[27] = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},{0,0,2},{1,0,2},{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}};

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
    return (pos.x>-1)*(pos.y>-1)*(pos.z>-1)*(pos.x<dim.x)*(pos.y<dim.y)*(pos.x<dim.z);
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

////1-point gauss [0,1]
//constant float qpt_x[1] = {5e-1f};
//constant float qpt_w[1] = {1e+0f};

//2-point gauss [0,1]
constant float qpt_x[2] = {0.211324865405187f,0.788675134594813f};
constant float qpt_w[2] = {5e-1f,5e-1f};

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

/*
 ===================================
 symmetric R^3x3
 ===================================
 */

//sym trace
float sym_tr(float8 a)
{
    return a.s0 + a.s3 + a.s5;
}

//sym squared
float8 sym_sq(float8 a)
{
    return (float8){a.s0*a.s0 + a.s1*a.s1 + a.s2*a.s2,
                    a.s0*a.s1 + a.s1*a.s3 + a.s2*a.s4,
                    a.s0*a.s2 + a.s1*a.s4 + a.s2*a.s5,
                    a.s1*a.s1 + a.s3*a.s3 + a.s4*a.s4,
                    a.s1*a.s2 + a.s3*a.s4 + a.s4*a.s5,
                    a.s2*a.s2 + a.s4*a.s4 + a.s5*a.s5, 0e0f, 0e0f};
}

//sym determinant
float sym_det(float8 a)
{
    return a.s0*a.s3*a.s5 - (a.s0*a.s4*a.s4 + a.s2*a.s2*a.s3 + a.s1*a.s1*a.s5) + 2e0f*a.s1*a.s2*a.s4;
}

//sym tensor inner prod
float sym_tip(float8 a, float8 b)
{
    return a.s0*b.s0 + 2e0f*a.s1*b.s1 + 2e0f*a.s2*b.s2 + a.s3*b.s3 + 2e0f*a.s4*b.s4 + a.s5*b.s5;
}

//sym scalar mult
float8 sym_smul(float8 a, float b)
{
    return (float8){a.s0*b, a.s1*b, a.s2*b, a.s3*b, a.s4*b, a.s5*b, 0e0f, 0e0f};
}

//sym add
float8 sym_add(float8 a, float8 b)
{
    return (float8){a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, 0e0f, 0e0f};
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
float8 mec_S(float8 e)
{
    float a = 2e0f*mat_mu;
    float b = mat_lam*sym_tr(e);
    
    return (float8){a*e.s0 + b, a*e.s1, a*e.s2, a*e.s3 + b, a*e.s4, a*e.s5 + b, 0e0f, 0e0f};
}

//energy phi = 0.5*lam*(tr(e))^2 + mu*tr(e^2)
float mec_p(float8 e)
{
    return 5e-1f*mat_lam*pown(sym_tr(e),2) + mat_mu*sym_tr(sym_sq(e));
}

/*
 ===================================
 eigs (sym 3x3)
 ===================================
 */

//eigenvalues - cuppen
float3 eig_val(float8 A)
{
    //off-diag
    float p1 = A.s1*A.s1 + A.s2*A.s2 + A.s4*A.s4;
    
    //diag
    if(p1==0e0f)
    {
        d.x = A.s0;
        d.y = A.s3;
        d.z = A.s5;
        
        return;
    }
    
    float q  = sym_tr(A)/3e0f;
    float p2 = pown(A.s0-q,2) + pown(A.s3-q,2) + pown(A.s5-q,2) + 2e0f*p1;
    float p  = sqrt(p2/6e0f);
    
    //B = (A - qI)/p
    float8 b = (float8){(A.s0 - q)/p, A.s1/p, A.s2/p, (A.s3 - q)/p, A.s4/p, (A.s5 - q)/p, 0e0f, 0e0f};
    float r = 5e-1f*sym_det(b);
    
    float phi = acos(r)/3e0f;
    phi = (r<=-1e0f)?M_PI_F/3e0f:phi;
    phi = (r>=+1e0f)?0e0f:phi;
    
    //decreasing order
    d[2] = q + 2e0f*p*cos(phi);
    d[0] = q + 2e0f*p*cos(phi + (2e0f*M_PI_F/3e0f));
    d[1] = 3e0f*q - (d[0] + d[2]);

    return;
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
//    vec_unt(v[0]);
//    vec_unt(v[1]);
//    vec_unt(v[2]);
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
    v[0] = vec_unt((float3){(d.x - A.s5 - A.s4*m0)/A.s2, m0, 1e0f});
    v[1] = vec_unt((float3){(d.y - A.s5 - A.s4*m1)/A.s2, m1, 1e0f});
    v[2] = vec_unt((float3){(d.z - A.s5 - A.s4*m2)/A.s2, m2, 1e0f});

    return;
}


//split
void eig_A1A2(float8 A, float8 A1, float8 A2)
{
    //vals, vecs
    float3 d;
    float3 v[3];
    
    //calc
    d = eig_val(A);
    eig_vec(A, d, v);
    
//    A1 = (float8){0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
    
    sym_add(A1, (d.x>0e0f)*vec_out(v[0]));
    sym_add(A1, (d.x>0e0f)*vec_out(v[1]));
    sym_add(A1, (d.x>0e0f)*vec_out(v[2]));
    
    sym_add(A2, (d.x<0e0f)*vec_out(v[0]));
    sym_add(A2, (d.x<0e0f)*vec_out(v[1]));
    sym_add(A2, (d.x<0e0f)*vec_out(v[2]));
        
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
    int3 vtx_dim = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int3 vtx_pos = {get_global_id(0),get_global_id(1),get_global_id(2)};
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
    //    printf("vtx_idx %3d\n",vtx_idx);
    //    printf("vtx_pos [%d,%d,%d]\n", vtx_pos[0], vtx_pos[1], vtx_pos[2]);
    
    int vtx_bc1 = fn_bc1(vtx_pos, vtx_dim);
    int vtx_bc2 = fn_bc2(vtx_pos, vtx_dim);
    
    global float *x = &vtx_xx[4*vtx_idx];
    x[0] = (float) vtx_pos[0];
    x[1] = (float) vtx_pos[1];
    x[2] = (float) vtx_pos[2];
    x[3] = (float) vtx_bc2;
    
    global float *u = &vtx_uu[4*vtx_idx];
    u[0] = (float) 1e-4f;
    u[1] = (float) 2;
    u[2] = (float) 3;
    u[3] = (float) vtx_bc1;
    
    global float *f = &vtx_ff[4*vtx_idx];
    f[0] = (float) 1;
    f[1] = (float) 2;
    f[2] = (float) 3;
    f[3] = (float) vtx_bc2;
    
    
    int blk_row_idx = 27*16*vtx_idx;
    global int   *blk_row_ii = &coo_ii[blk_row_idx];
    global int   *blk_row_jj = &coo_jj[blk_row_idx];
    global float *blk_row_aa = &coo_aa[blk_row_idx];
    
    
    //vtx
    for(int adj_i=0; adj_i<27; adj_i++)
    {
        int adj_pos[3] = {vtx_pos[0] + idx3[adj_i][0] - 1, vtx_pos[1] + idx3[adj_i][1] - 1, vtx_pos[2] + idx3[adj_i][2] - 1};
        int adj_idx = fn_idx(adj_pos, vtx_dim);
        int adj_bc1 = fn_bc1(adj_pos, vtx_dim);
        
        int blk_col_idx = adj_i*16;
        global int   *blk_ii = &blk_row_ii[blk_col_idx];
        global int   *blk_jj = &blk_row_jj[blk_col_idx];
        global float *blk_aa = &blk_row_aa[blk_col_idx];
        

        //dims
        for(int dim_i=0; dim_i<4; dim_i++)
        {
            for(int dim_j=0; dim_j<4; dim_j++)
            {
                int dim_idx = 4*dim_i+dim_j;
                
                blk_ii[dim_idx] = adj_bc1*4*vtx_idx + dim_i;
                blk_jj[dim_idx] = adj_bc1*(4*adj_idx + dim_j);
                blk_aa[dim_idx] = vtx_bc2*(vtx_idx==adj_idx)*(dim_i==dim_j);
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
    int3 vtx_dim = {get_global_size(0) + 2, get_global_size(1) + 2, get_global_size(2) + 2};
    int3 vtx_pos = {get_global_id(0)   + 1, get_global_id(1)   + 1, get_global_id(2)   + 1};
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
    //    printf("vtx %3d\n",vtx_idx);
    //    printf("vtx_pos [%d,%d,%d]\n", vtx_pos[0], vtx_pos[1], vtx_pos[2]);
    
    //pointers K, U, F
    int blk_row_idx = 27*16*vtx_idx;
    global float *blk_row_aa = &coo_aa[blk_row_idx];
    
    int vec_row_idx = 4*vtx_idx;
    global float *vec_row_uu = &vtx_uu[vec_row_idx];
    global float *vec_row_ff = &vtx_ff[vec_row_idx];
    
    //loop ele
    for(int ele_i=0; ele_i<8; ele_i++)
    {
        //vtx1 (blk row)
        int vtx1_i = 7 - ele_i;
        
        //per vtx
        int     vv_loc[8][3];   //local pos
        int     vv_idx3[8];     //3x3x3 idx
        int     vv_pos[8][3];   //global pos
        int     vv_idx[8];      //global idx
        float   vv_u[8][4];     //soln
        
        //loop vtx - eval
        for(int vtx_i=0; vtx_i<8; vtx_i++)
        {
            vv_loc[vtx_i][0] = idx2[ele_i][0] + idx2[vtx_i][0];
            vv_loc[vtx_i][1] = idx2[ele_i][1] + idx2[vtx_i][1];
            vv_loc[vtx_i][2] = idx2[ele_i][2] + idx2[vtx_i][2];
            
            vv_idx3[vtx_i] = vv_loc[vtx_i][0] + 3*vv_loc[vtx_i][1] + 9*vv_loc[vtx_i][2];
            
            vv_pos[vtx_i][0] = vtx_pos[0] + vv_loc[vtx_i][0] - 1;
            vv_pos[vtx_i][1] = vtx_pos[1] + vv_loc[vtx_i][1] - 1;
            vv_pos[vtx_i][2] = vtx_pos[2] + vv_loc[vtx_i][2] - 1;
            
            vv_idx[vtx_i] = fn_idx(vv_pos[vtx_i], vtx_dim);
            
            //soln
            global float *u = &vtx_uu[4*vv_idx[vtx_i]];
            vv_u[vtx_i][0] = u[0];
            vv_u[vtx_i][1] = u[1];
            vv_u[vtx_i][2] = u[2];
            vv_u[vtx_i][3] = u[3];
            
        }//vtx
        
        
        //loop qpt
        for(int qpt_i=0; qpt_i<8; qpt_i++)
        {
//            //1pt
//            float qp[3] = {qpt_x[0],qpt_x[0],qpt_x[0]};
//            float qw    = qpt_w[0]*qpt_w[0]*qpt_w[0];
            
            //2pt
            float qp[3] = {qpt_x[idx2[qpt_i][0]],qpt_x[idx2[qpt_i][1]],qpt_x[idx2[qpt_i][2]]};
            float qw    = qpt_w[idx2[qpt_i][0]]*qpt_w[idx2[qpt_i][1]]*qpt_w[idx2[qpt_i][2]];
            
//            //3pt
//            float qp[3] = {qpt_x[idx3[qpt_i][0]],qpt_x[idx3[qpt_i][1]],qpt_x[idx3[qpt_i][2]]};
//            float qw    = qpt_w[idx3[qpt_i][0]]*qpt_w[idx3[qpt_i][1]]*qpt_w[idx3[qpt_i][2]];
            
            //basis
            float ee[8];
            float gg[8][3];
            
            bas_eval(qp, ee);
            bas_grad(qp, gg);
            
            //soln
            float u_grad[3][3] = {{0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}, {0e0f, 0e0f, 0e0f}};
            float c_eval       = 0e0f;
            
            //eval
            for(int vtx_i=0; vtx_i<8; vtx_i++)
            {
                //u_grad
                for(int dim_i=0; dim_i<3; dim_i++)
                {
                    for(int dim_j=0; dim_j<3; dim_j++)
                    {
                        u_grad[dim_i][dim_j] += vv_u[vtx_i][dim_i]*gg[vtx_i][dim_j];
                    }
                }
                //c_eval
                c_eval += vv_u[vtx_i][3]*ee[vtx_i];
            }
            
            // notation
            // e/s/p = strain/stress/energy
            // h/i/j = mesh/basis i/basis j
            // 1/2   = pos/neg
            
            
            //strain (sym)
            float eh[6];
            mec_e(u_grad, eh);
            
            //split
            float eh1[6] = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
            float eh2[6] = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
            eig_a1a2(eh, eh1, eh2);
            
            //stress
            float sh1[6] = {0e0f};
            float sh2[6] = {0e0f};
            mec_s(eh1, sh1);
            mec_s(eh2, sh2);
            
            //energy
            float ph1 = mec_p(eh1);
            
            //crack
            float c1 = pown(1e0f - c_eval, 2);
            float c2 = 2e0f*(c_eval - 1e0f);
            
        
            //loop vtx2 (blk col)
            for(int vtx2_i=0; vtx2_i<8; vtx2_i++)
            {
                //blk
                global float *blk_aa = &blk_row_aa[16*vv_idx3[vtx2_i]];
                
                //dims 3x3
                for(int dim_i=0; dim_i<3; dim_i++)
                {
                    //grad
                    float gi[3][3];
                    
                    //tensor basis
                    gi[dim_i][0] = gg[vtx1_i][0];
                    gi[dim_i][1] = gg[vtx1_i][1];
                    gi[dim_i][2] = gg[vtx1_i][2];
                    
                    //strain
                    float ei[6];
                    mec_e(gi, ei);
                    
                    
                    
//                    //uc
//                    blk_aa[4*dim_i+3] += 1e0f;
//
//                    //cu
//                    blk_aa[12+dim_i] += 1e0f;
                    
                    
                    
                    for(int dim_j=0; dim_j<3; dim_j++)
                    {
                        //grad
                        float gj[3][3];
                        
                        //tensor basis
                        gj[dim_j][0] = gg[vtx2_i][0];
                        gj[dim_j][1] = gg[vtx2_i][1];
                        gj[dim_j][2] = gg[vtx2_i][2];
                        
                        //strain
                        float ej[6];
                        mec_e(gj, ej);
                        
                        //split
                        float ej1[6] = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                        float ej2[6] = {0e0f, 0e0f, 0e0f, 0e0f, 0e0f, 0e0f};
                        eig_a1a2(ej, ej1, ej2);
                        
                        //stress
                        float sj1[6] = {0e0f};
                        float sj2[6] = {0e0f};
                        mec_s(ej1, sj1);
                        mec_s(ej2, sj2);
                        
                        
                        //sj1 = c1*sj1 + sj2
                        sym_smul(c1, sj1);
                        sym_add(sj1, sj2);
                        
                        //uu
                        int dim_idx = 4*dim_i+dim_j;
                        blk_aa[dim_idx] += sym_tip(sj1,ei)*qw;
                    }
                }
                
                //cc
//                blk_aa[15] += 1e0f; //vec_dot(gg[vtx1_i], gg[vtx2_i])*qw;
                
                
            } //vtx
            
        } //qpt
        
    } //ele
    
    return;
}
