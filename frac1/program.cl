//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

//prototypes
int fn_idx(const int *pos, const int *dim);
int fn_bc1(const int *pos, const int *dim);
int fn_bc2(const int *pos, const int *dim);

void bas_eval(const float p[3], float ee[8]);
void bas_grad(const float p[3], float gg[8][3]);

float fn_dot(float *a, float *b);

/*
 ===================================
 utilities
 ===================================
 */

//flat index
int fn_idx(const int *pos, const int *dim)
{
    return pos[0] + pos[1]*(dim[0]) + pos[2]*(dim[0]*dim[1]);
}

//in-bounds
int fn_bc1(const int *pos, const int *dim)
{
    return (pos[0]>-1)*(pos[1]>-1)*(pos[2]>-1)*(pos[0]<dim[0])*(pos[1]<dim[1])*(pos[2]<dim[2]);
}

//on the boundary
int fn_bc2(const int *pos, const int *dim)
{
    return (pos[0]==0)||(pos[1]==0)||(pos[2]==0)||(pos[0]==dim[0]-1)||(pos[1]==dim[1]-1)||(pos[2]==dim[2]-1);;
}

/*
 ===================================
 basis
 ===================================
 */

//eval
void bas_eval(const float p[3], float ee[8])
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
void bas_grad(const float p[3], float gg[8][3])
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

/*
 ===================================
 tensors
 ===================================
 */

//vector inner prod
float fn_dot(float *a, float *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

//tensor inner prod


/*
 ===================================
 mechanics
 ===================================
 */

//stress

//strain

//energy




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
                        
                        blk_ii[dim_idx] = 4*vtx_idx + dim_i;
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
    
    printf("vtx %3d\n",vtx_idx);
    printf("vtx_pos [%d,%d,%d]\n", vtx_pos[0], vtx_pos[1], vtx_pos[2]);
    
    
    int blk_row_idx = 27*16*vtx_idx;
    global float *blk_row_aa = &coo_aa[blk_row_idx];
    
    //ele
    for(int ele_k=0; ele_k<2; ele_k++)
    {
        for(int ele_j=0; ele_j<2; ele_j++)
        {
            for(int ele_i=0; ele_i<2; ele_i++)
            {
                //ref vtx
                int ele_ref[3];
                ele_ref[0] = vtx_pos[0] + ele_i - 1;
                ele_ref[1] = vtx_pos[1] + ele_j - 1;
                ele_ref[2] = vtx_pos[2] + ele_k - 1;
                
                int ele_idx2 = ele_i + 2*ele_j + 4*ele_k;
                int vtx_idx2 = 7 - ele_idx2;
                
                printf("ele [%d,%d,%d] %d %d\n", ele_ref[0], ele_ref[1], ele_ref[2], ele_idx2, vtx_idx2);
                
                //qpt
                for(int qpt_k=0; qpt_k<qpt_n; qpt_k++)
                {
                    for(int qpt_j=0; qpt_j<qpt_n; qpt_j++)
                    {
                        for(int qpt_i=0; qpt_i<qpt_n; qpt_i++)
                        {

                            float qp[3] = {qpt_x[qpt_i],qpt_x[qpt_j],qpt_x[qpt_k]};
                            float qw    = qpt_w[qpt_i]*qpt_w[qpt_j]*qpt_w[qpt_k];

                            printf("qpt [%6.4f,%6.4f,%6.4f] %6.4f\n",qp[0], qp[1], qp[2], qw);
                            
                            //basis
                            float ee[8];
                            float gg[8][3];

                            bas_eval(qp, ee);
                            bas_grad(qp, gg);
                            

                            
//                            printf("vtx_gg %d [%+6.4f,%+6.4f,%+6.4f]\n", vtx_idx2, gg[vtx_idx2][0], gg[vtx_idx2][1], gg[vtx_idx2][2]);

                            //adj
                            for(int adj_k=0; adj_k<2; adj_k++)
                            {
                                for(int adj_j=0; adj_j<2; adj_j++)
                                {
                                    for(int adj_i=0; adj_i<2; adj_i++)
                                    {
                                        int adj_pos[3];
                                        adj_pos[0] = ele_ref[0] + adj_i;
                                        adj_pos[1] = ele_ref[1] + adj_j;
                                        adj_pos[2] = ele_ref[2] + adj_k;

                                        int adj_idx = fn_idx(adj_pos, vtx_dim);                                     //global
                                        int adj_idx2 = adj_i + 2*adj_j + 4*adj_k;                                   //2x2x2
//                                        int adj_idx3 = (ele_i + adj_i) + 3*(ele_j + adj_j) + 9*(ele_k + adj_k);     //3x3x3

//                                        printf("   adj [%d,%d,%d] %d %d | %2d %3d\n", adj_pos[0], adj_pos[1], adj_pos[2], vtx_idx2, adj_idx2, adj_idx3, adj_idx);
                                        
                                    
//                                        float *g1 = gg[vtx_idx2];
//                                        float *g2 = gg[adj_idx2];
                                        
                                        for(int i=0; i<8; i++)
                                        {
            //                                float *g = gg[i];
                                            
                                            printf("gg %d [%+6.4f %+6.4f %+6.4f]\n", i, gg[i][0], gg[i][1], gg[i][2]);
                                            
                                        }
                                        
//                                        printf("dt %d [%+6.4f,%+6.4f,%+6.4f] %d [%+6.4f,%+6.4f,%+6.4f] %+e %+e\n", vtx_idx2, g1[0], g1[1], g1[2], adj_idx2, g2[0], g2[1], g2[2], fn_dot(g1, g2),qw);
                                        
                                        printf("vv %d  %+6.4f %+6.4f %+6.4f\n",vtx_idx2,gg[vtx_idx2][0],gg[vtx_idx2][1],gg[vtx_idx2][2]);
                                        
                                        //blk
                                        global float *blk_aa = &blk_row_aa[16*adj_idx];
                                        
                                        
                                        //calc
//                                      blk_aa[15] += fn_dot(gg[vtx_idx2], gg[adj_idx2]);
                                        
                                        
                                        
                                        //write all
                                        for(int i=0; i<16; i++)
                                        {
                                            blk_aa[i] += fn_dot(gg[vtx_idx2], gg[adj_idx2])*qw;
                                        }


                                        
                                        
                                        

                                    }
                                }
                            }//adj
                            
                            
                            
                            
                        }
                    }
                }//qpt
                
                
                
                
            }
        }
    }//ele
    
    return;
}
