//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

//params
constant float mat_lam = 1e0f;
constant float mat_mu  = 1e0f;

//prototypes
int fn_idx(int *pos, int *dim);
int fn_bc1(int *pos, int *dim);
int fn_bc2(int *pos, int *dim);

void bas_eval(float p[3], float ee[8]);
void bas_grad(float p[3], float gg[8][3]);
void bas_tens(int i, float g[3], float a[3][3]);

float vec_dot(float *a, float *b);

float sym_tr(float *a);
void  sym_sq(float *a, float *b);
float sym_dot(float *a, float *b);

void  mec_e(float *u, float *e);
void  mec_s(float *e, float *s);
float mec_p(float *e);

constant int idx2[8][3] = {{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}};
constant int idx3[27][3] = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},{0,0,2},{1,0,2},{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}};

/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx(int *pos, int *dim)
{
    return pos[0] + pos[1]*(dim[0]) + pos[2]*(dim[0]*dim[1]);
}

//in-bounds
int fn_bc1(int *pos, int *dim)
{
    return (pos[0]>-1)*(pos[1]>-1)*(pos[2]>-1)*(pos[0]<dim[0])*(pos[1]<dim[1])*(pos[2]<dim[2]);
}

//on the boundary
int fn_bc2(int *pos, int *dim)
{
    return (pos[0]==0)||(pos[1]==0)||(pos[2]==0)||(pos[0]==dim[0]-1)||(pos[1]==dim[1]-1)||(pos[2]==dim[2]-1);;
}

/*
 ===================================
 basis
 ===================================
 */

//eval
void bas_eval(float p[3], float ee[8])
{
    float x0 = 1e0f - p[0];
    float y0 = 1e0f - p[1];
    float z0 = 1e0f - p[2];
    
    float x1 = p[0];
    float y1 = p[1];
    float z1 = p[2];
    
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
void bas_grad(float p[3], float gg[8][3])
{
    float x0 = 1e0f - p[0];
    float y0 = 1e0f - p[1];
    float z0 = 1e0f - p[2];

    float x1 = p[0];
    float y1 = p[1];
    float z1 = p[2];

    gg[0][0] = -y0*z0;
    gg[1][0] = +y0*z0;
    gg[2][0] = -y1*z0;
    gg[3][0] = +y1*z0;
    gg[4][0] = -y0*z1;
    gg[5][0] = +y0*z1;
    gg[6][0] = -y1*z1;
    gg[7][0] = +y1*z1;

    gg[0][1] = -x0*z0;
    gg[1][1] = -x1*z0;
    gg[2][1] = +x0*z0;
    gg[3][1] = +x1*z0;
    gg[4][1] = -x0*z1;
    gg[5][1] = -x1*z1;
    gg[6][1] = +x0*z1;
    gg[7][1] = +x1*z1;

    gg[0][2] = -x0*y0;
    gg[1][2] = -x1*y0;
    gg[2][2] = -x0*y1;
    gg[3][2] = -x1*y1;
    gg[4][2] = +x0*y0;
    gg[5][2] = +x1*y0;
    gg[6][2] = +x0*y1;
    gg[7][2] = +x1*y1;

    return;
}

//basis tensor (outer prod of basis col and grad row)
void bas_tens(int i, float g[3], float a[3][3])
{
//    a[i][0] = g[0];
//    a[i][1] = g[1];
//    a[i][2] = g[2];
    
    memcpy(a[i], g, 3);
    
    return;
}



/*
 ===================================
 linear algebra R^3
 ===================================
 */

//vector inner prod
float vec_dot(float *a, float *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


//sym trace
float sym_tr(float *a)
{
    return a[0] + a[3] + a[5];
}

//sym squared
void sym_sq(float *a, float *b)
{
    b[0] = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    b[1] = a[0]*a[1] + a[1]*a[3] + a[2]*a[4];
    b[2] = a[0]*a[2] + a[1]*a[4] + a[2]*a[5];
    b[3] = a[1]*a[1] + a[3]*a[3] + a[4]*a[4];
    b[4] = a[1]*a[2] + a[3]*a[4] + a[4]*a[5];
    b[5] = a[2]*a[2] + a[4]*a[4] + a[5]*a[5];
    
    return;
}

//sym tensor inner prod
float sym_dot(float *a, float *b)
{
    return a[0]*b[0] + 2e0f*a[1]*b[1] + 2e0f*a[2]*b[2] + a[3]*b[3] + 2e0f*a[4]*b[4] + a[5]*b[5];
}

/*
 ===================================
 mechanics
 ===================================
 */

//strain = 0.5(u + u')
void mec_e(float *u, float *e)
{
    e[0] = u[0];
    e[1] = 5e-1f*(u[1]+u[3]);
    e[2] = 5e-1f*(u[2]+u[6]);
    e[3] = u[4];
    e[4] = 5e-1f*(u[5]+u[7]);
    e[5] = u[8];

    return;
}

//stress pk2 = lam*tr(e)*I + 2*mu*e
void mec_s(float *e, float *s)
{
    float a = 2e0f*mat_mu;
    float b = mat_lam*sym_tr(e);
    
    s[0] = a*e[0] + b;
    s[1] = a*e[1];
    s[2] = a*e[2];
    s[3] = a*e[3] + b;
    s[4] = a*e[4];
    s[5] = a*e[5] + b;

    return;
}

//energy phi = 0.5*lam*(tr(e))^2 + mu*tr(e^2)
float mec_p(float *e)
{
    float a = sym_tr(e);
    float *b;
    sym_sq(e, b);
    
    return 5e-1f*mat_lam*a*a + mat_mu*sym_tr(b);
}

/*
 ===================================
 quadrature [0,1]
 ===================================
 */

////1-point gauss [0,1]
//constant int   qpt_n    = 1;
//constant float qpt_x[1] = {5e-1f};
//constant float qpt_w[1] = {1e+0f};

////2-point gauss [0,1]
//constant int   qpt_n    = 2;
//constant float qpt_x[2] = {0.211324865405187f,0.788675134594813f};
//constant float qpt_w[2] = {0.500000000000000f,0.500000000000000f};

//3-point gauss [0,1]
constant int   qpt_n    = 3;
constant float qpt_x[3] = {0.112701665379258f,0.500000000000000f,0.887298334620742f};
constant float qpt_w[3] = {0.277777777777778f,0.444444444444444f,0.277777777777778f};

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
    int vtx_dim[3] = {get_global_size(0),get_global_size(1),get_global_size(2)};
    int vtx_pos[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    
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
    u[0] = (float) 1e-4;
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
    
    //adj
    for(int adj_k=0; adj_k<3; adj_k++)
    {
        for(int adj_j=0; adj_j<3; adj_j++)
        {
            for(int adj_i=0; adj_i<3; adj_i++)
            {
                int adj_pos[3];
                adj_pos[0] = vtx_pos[0] + adj_i - 1;
                adj_pos[1] = vtx_pos[1] + adj_j - 1;
                adj_pos[2] = vtx_pos[2] + adj_k - 1;
                
                int adj_idx = fn_idx(adj_pos, vtx_dim);
                
                int adj_idx3 = adj_i + 3*adj_j + 9*adj_k; //3x3x3
                int blk_col_idx = adj_idx3*16;
                
                global int   *blk_ii = &blk_row_ii[blk_col_idx];
                global int   *blk_jj = &blk_row_jj[blk_col_idx];
                global float *blk_aa = &blk_row_aa[blk_col_idx];
                
                int adj_bc1 = fn_bc1(adj_pos, vtx_dim);
                //int adj_bc2 = fn_bc2(adj_pos, vtx_dim);
                
                //printf("adj %3d %d %d %3d [%+d,%+d,%+d] %2d %d %d\n",vtx_idx, vtx_bc1, vtx_bc2, adj_idx, adj_pos[0], adj_pos[1], adj_pos[2], blk_idx, adj_bc1, adj_bc2);
                
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
    int vtx_dim[3] = {get_global_size(0) + 2, get_global_size(1) + 2, get_global_size(2) + 2};
    int vtx_pos[3] = {get_global_id(0)   + 1, get_global_id(1)   + 1, get_global_id(2)   + 1};
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
//    printf("vtx %3d\n",vtx_idx);
//    printf("vtx_pos [%d,%d,%d]\n", vtx_pos[0], vtx_pos[1], vtx_pos[2]);
    
    //pointers
    int blk_row_idx = 27*16*vtx_idx;
    global float *blk_row_aa = &coo_aa[blk_row_idx];
    
//    int vec_row_idx = 4*vtx_idx;
//    global float *vec_row_uu = &vtx_uu[vec_row_idx];x
//    global float *vec_row_ff = &vtx_ff[vec_row_idx];
    
    
    int vv_pos[8][3];   //global
    int vv_idx[8];      //global
    int vv_idx3[8];     //3x3x3
    

    //ele
    for(int ele_i=0; ele_i<8; ele_i++)
    {
        //key
        int vtx1_i = 7 - ele_i;
        
        
        //vtx
        for(int vtx_i=0; vtx_i<8; vtx_i++)
        {
            vv_idx3[vtx_i] = (idx2[ele_i][0] + idx2[vtx_i][0]) + 3*(idx2[ele_i][1] + idx2[vtx_i][1]) + 9*(idx2[ele_i][2] + idx2[vtx_i][2]);
            
            vv_pos[vtx_i][0] = vtx_pos[0] + vv_idx3[vtx_i] - 1;
            vv_pos[vtx_i][1] = vtx_pos[1] + vv_idx3[vtx_i] - 1;
            vv_pos[vtx_i][2] = vtx_pos[2] + vv_idx3[vtx_i] - 1;
            
            vv_idx[vtx_i] = fn_idx(vv_pos[vtx_i], vtx_dim);
            
            
            
        }//vtx
        
        //qpt
        for(int qpt_i=0; qpt_i<27; qpt_i++)
        {
            float qp[3] = {qpt_x[idx3[qpt_i][0]],qpt_x[idx3[qpt_i][1]],qpt_x[idx3[qpt_i][2]]};
            float qw    = qpt_w[idx3[qpt_i][0]]*qpt_w[idx3[qpt_i][1]]*qpt_w[idx3[qpt_i][2]];
            
            //basis
            float ee[8];
            float gg[8][3];

            bas_eval(qp, ee);
            bas_grad(qp, gg);
            
            
            //adj
            for(int vtx2_i=0; vtx2_i<8; vtx2_i++)
            {
                //blk
                global float *blk_aa = &blk_row_aa[16*vv_idx3[vtx2_i]];

                //write
                blk_aa[15] += vec_dot(gg[vtx1_i], gg[vtx2_i])*qw;
                
                
                
                
//                //dims 3x3
//                for(int dim_i=0; dim_i<3; dim_i++)
//                {
//                    for(int dim_j=0; dim_j<3; dim_j++)
//                    {
//                        int dim_idx = 4*dim_i+dim_j;
//                        
//                        blk_aa[dim_idx] = dim_idx;
//                    }
//                }
                
                
                
            }//vtx
            
            
            
            
        }//qpt
        
        
    }//ele
    

    
//
//
//    //ele
//    for(int ele_k=0; ele_k<2; ele_k++)
//    {
//        for(int ele_j=0; ele_j<2; ele_j++)
//        {
//            for(int ele_i=0; ele_i<2; ele_i++)
//            {
////                //ref vtx
////                int ele_ref[3];
////                ele_ref[0] = vtx_pos[0] + ele_i - 1;
////                ele_ref[1] = vtx_pos[1] + ele_j - 1;
////                ele_ref[2] = vtx_pos[2] + ele_k - 1;
//
//                int ele_idx2 = ele_i + 2*ele_j + 4*ele_k;
//                int vtx_idx2 = 7 - ele_idx2;
//
////                printf("ele [%d,%d,%d] %d %d\n", ele_ref[0], ele_ref[1], ele_ref[2], ele_idx2, vtx_idx2);
//
//                //qpt
//                for(int qpt_k=0; qpt_k<qpt_n; qpt_k++)
//                {
//                    for(int qpt_j=0; qpt_j<qpt_n; qpt_j++)
//                    {
//                        for(int qpt_i=0; qpt_i<qpt_n; qpt_i++)
//                        {
//                            float qp[3] = {qpt_x[qpt_i],qpt_x[qpt_j],qpt_x[qpt_k]};
//                            float qw    = qpt_w[qpt_i]*qpt_w[qpt_j]*qpt_w[qpt_k];
//
//
//                            //basis
//                            float ee[8];
//                            float gg[8][3];
//
//                            bas_eval(qp, ee);
//                            bas_grad(qp, gg);
//
//                            //eval
////                            float u_h[4] = {0e0f, 0e0f, 0e0f, 0e0f};
//
//                            //adj
//                            for(int adj_k=0; adj_k<2; adj_k++)
//                            {
//                                for(int adj_j=0; adj_j<2; adj_j++)
//                                {
//                                    for(int adj_i=0; adj_i<2; adj_i++)
//                                    {
////                                        int adj_pos[3];
////                                        adj_pos[0] = ele_ref[0] + adj_i;
////                                        adj_pos[1] = ele_ref[1] + adj_j;
////                                        adj_pos[2] = ele_ref[2] + adj_k;
//
////                                        int adj_idx = fn_idx(adj_pos, vtx_dim);                                     //global
//                                        int adj_idx2 = adj_i + 2*adj_j + 4*adj_k;                                   //2x2x2
//                                        int adj_idx3 = (ele_i + adj_i) + 3*(ele_j + adj_j) + 9*(ele_k + adj_k);     //3x3x3
//
//
//                                        //tensor basis
//
//
//
//                                        //blk
//                                        global float *blk_aa = &blk_row_aa[16*adj_idx3];
//
//                                        //write
//                                        blk_aa[15] += vec_dot(gg[vtx_idx2], gg[adj_idx2])*qw;
//
//
//
//
//
//
//
//                                    }
//                                }
//                            }//adj
//
//
//
//
//                        }
//                    }
//                }//qpt
//
//
//
//
//            }
//        }
//    }//ele
//
    return;
}
