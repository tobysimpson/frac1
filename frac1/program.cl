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

/*
 ===================================
 util
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
 basis [0,1]
 ===================================
 */

//eval
void bas_eval(const float p[3], float ee[8])
{
    ee[0] = (1e0f-p[0])*(1e0f-p[1])*(1e0f-p[2]);
    ee[1] = (     p[0])*(1e0f-p[1])*(1e0f-p[2]);
    ee[2] = (1e0f-p[0])*(     p[1])*(1e0f-p[2]);
    ee[3] = (     p[0])*(     p[1])*(1e0f-p[2]);
    ee[4] = (1e0f-p[0])*(1e0f-p[1])*(     p[2]);
    ee[5] = (     p[0])*(1e0f-p[1])*(     p[2]);
    ee[6] = (1e0f-p[0])*(     p[1])*(     p[2]);
    ee[7] = (     p[0])*(     p[1])*(     p[2]);
    
    return;
}

//grad
void bas_grad(const float p[3], float gg[8][3])
{
    gg[0][0] = (-1e0f)*(1e0f-p[1])*(1e0f-p[2]);
    gg[0][1] = (1e0f-p[0])*(-1e0f)*(1e0f-p[2]);
    gg[0][2] = (1e0f-p[0])*(1e0f-p[1])*(-1e0f);
    gg[1][0] = (+1e0f)*(1e0f-p[1])*(1e0f-p[2]);
    gg[1][1] = (     p[0])*(-1e0f)*(1e0f-p[2]);
    gg[1][2] = (     p[0])*(1e0f-p[1])*(-1e0f);
    gg[2][0] = (-1e0f)*(     p[1])*(1e0f-p[2]);
    gg[2][1] = (1e0f-p[0])*(+1e0f)*(1e0f-p[2]);
    gg[2][2] = (1e0f-p[0])*(     p[1])*(-1e0f);
    gg[3][0] = (+1e0f)*(     p[1])*(1e0f-p[2]);
    gg[3][1] = (     p[0])*(+1e0f)*(1e0f-p[2]);
    gg[3][2] = (     p[0])*(     p[1])*(-1e0f);
    gg[4][0] = (-1e0f)*(1e0f-p[1])*(     p[2]);
    gg[4][1] = (1e0f-p[0])*(-1e0f)*(     p[2]);
    gg[4][2] = (1e0f-p[0])*(1e0f-p[1])*(+1e0f);
    gg[5][0] = (+1e0f)*(1e0f-p[1])*(     p[2]);
    gg[5][1] = (     p[0])*(-1e0f)*(     p[2]);
    gg[5][2] = (     p[0])*(1e0f-p[1])*(+1e0f);
    gg[6][0] = (-1e0f)*(     p[1])*(     p[2]);
    gg[6][1] = (1e0f-p[0])*(+1e0f)*(     p[2]);
    gg[6][2] = (1e0f-p[0])*(     p[1])*(+1e0f);
    gg[7][0] = (+1e0f)*(     p[1])*(     p[2]);
    gg[7][1] = (     p[0])*(+1e0f)*(     p[2]);
    gg[7][2] = (     p[0])*(     p[1])*(+1e0f);

    return;
}



////grad
//void bas_grad(const float3 p, float3 gg[8], const float3 dx)
//{
//     gg[0] = (float3){(-1e0f)*(1e0f-p.y)*(1e0f-p.z), (1e0f-p.x)*(-1e0f)*(1e0f-p.z), (1e0f-p.x)*(1e0f-p.y)*(-1e0f)}/dx;
//     gg[1] = (float3){(+1e0f)*(1e0f-p.y)*(1e0f-p.z), (     p.x)*(-1e0f)*(1e0f-p.z), (     p.x)*(1e0f-p.y)*(-1e0f)}/dx;
//     gg[2] = (float3){(-1e0f)*(     p.y)*(1e0f-p.z), (1e0f-p.x)*(+1e0f)*(1e0f-p.z), (1e0f-p.x)*(     p.y)*(-1e0f)}/dx;
//     gg[3] = (float3){(+1e0f)*(     p.y)*(1e0f-p.z), (     p.x)*(+1e0f)*(1e0f-p.z), (     p.x)*(     p.y)*(-1e0f)}/dx;
//     gg[4] = (float3){(-1e0f)*(1e0f-p.y)*(     p.z), (1e0f-p.x)*(-1e0f)*(     p.z), (1e0f-p.x)*(1e0f-p.y)*(+1e0f)}/dx;
//     gg[5] = (float3){(+1e0f)*(1e0f-p.y)*(     p.z), (     p.x)*(-1e0f)*(     p.z), (     p.x)*(1e0f-p.y)*(+1e0f)}/dx;
//     gg[6] = (float3){(-1e0f)*(     p.y)*(     p.z), (1e0f-p.x)*(+1e0f)*(     p.z), (1e0f-p.x)*(     p.y)*(+1e0f)}/dx;
//     gg[7] = (float3){(+1e0f)*(     p.y)*(     p.z), (     p.x)*(+1e0f)*(     p.z), (     p.x)*(     p.y)*(+1e0f)}/dx;
//
//     return;
//}

/*
 ===================================
 quadrature [0,1]
 ===================================
 */

//1-point gauss [0,1]
constant int   qpt_n    = 1;
constant float qpt_x[1] = {5e-1f};
constant float qpt_w[1] = {1e+0f};

////2-point gauss [0,1]
//constant int   qpt_n    = 2;
//constant float qpt_x[2] = {0.211324865405187f,0.788675134594813f};
//constant float qpt_w[2] = {0.500000000000000f,0.500000000000000f};

////3-point gauss [0,1]
//constant int   qpt_n    = 3;
//constant float qpt_x[3] = {0.112701665379258f,0.500000000000000f,0.887298334620742f};
//constant float qpt_w[3] = {0.277777777777778f,0.444444444444444f,0.277777777777778f};

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
    x[0] = (float)vtx_pos[0];
    x[1] = (float)vtx_pos[1];
    x[2] = (float)vtx_pos[2];
    x[3] = (float) 0;
    
    global float *u = &vtx_uu[4*vtx_idx];
    u[0] = (float) vtx_idx + 1e-1;
    u[1] = (float) vtx_idx + 2e-1;
    u[2] = (float) vtx_idx + 3e-1;
    u[3] = (float) vtx_bc1;
    
    global float *f = &vtx_ff[4*vtx_idx];
    f[0] = (float) 4*vtx_idx+0;
    f[1] = (float) 4*vtx_idx+1;
    f[2] = (float) 4*vtx_idx+2;
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
                
                int blk_idx = adj_i + 3*adj_j + 9*adj_k;
                int blk_col_idx = blk_idx*16;
                
                global int   *blk_ii = &blk_row_ii[blk_col_idx];
                global int   *blk_jj = &blk_row_jj[blk_col_idx];
                global float *blk_aa = &blk_row_aa[blk_col_idx];
                
                int adj_bc1 = fn_bc1(adj_pos, vtx_dim);
                //int adj_bc2 = fn_bc2(adj_pos, vtx_dim);
                
                //printf("adj %3d %d %d %3d [%+d,%+d,%+d] %2d %d %d\n",vtx_idx, vtx_bc1, vtx_bc2, adj_idx, adj_pos[0], adj_pos[1], adj_pos[2], blk_idx, adj_bc1, adj_bc2);
                
                //dims
                for(int i=0; i<4; i++)
                {
                    for(int j=0; j<4; j++)
                    {
                        blk_ii[4*i+j] = 4*vtx_idx + i;
                        blk_jj[4*i+j] = adj_bc1*(4*adj_idx + j);
                        blk_aa[4*i+j] = vtx_bc2*(vtx_idx==adj_idx)*(i==j);
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
                int ele_ref[3];
                ele_ref[0] = vtx_pos[0] + ele_i - 1;
                ele_ref[1] = vtx_pos[1] + ele_j - 1;
                ele_ref[2] = vtx_pos[2] + ele_k - 1;
                
                printf("ele_ref [%d,%d,%d]\n", ele_ref[0], ele_ref[1], ele_ref[2]);
                
                //qpt
                for(int qpt_k=0; qpt_k<qpt_n; qpt_k++)
                {
                    for(int qpt_j=0; qpt_j<qpt_n; qpt_j++)
                    {
                        for(int qpt_i=0; qpt_i<qpt_n; qpt_i++)
                        {
                            
                            float qp[3] = {qpt_x[qpt_i],qpt_x[qpt_j],qpt_x[qpt_k]};
                            float qw    = qpt_w[qpt_i]*qpt_w[qpt_j]*qpt_w[qpt_k];
                            
                            float ee[8];
                            float gg[8][3];
                            
                            bas_eval(qp, ee);
                            bas_grad(qp, gg);
                            
                            
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
                                        
                                        int adj_idx = fn_idx(adj_pos, vtx_dim);
                                        
                                        int adj_loc = adj_i + 2*adj_j + 4*adj_k;
                                        
                                      
                                        printf("%d %e [%+e,%+e,%+e] %e\n",adj_loc,ee[adj_loc], gg[adj_loc][0], gg[adj_loc][1], gg[adj_loc][2],qw);
                                        
                                        
                                        //blk
                                        int blk_idx = (ele_i + adj_i) + 3*(ele_j + adj_j) + 9*(ele_k + adj_k);
                                        global float *blk_aa = &blk_row_aa[16*blk_idx];
                                        
                                        
                                        //write
                                        for(int i=0; i<16; i++)
                                        {
                                            blk_aa[i] += 1e0f;
                                        }
                                        
                                        
//                                        printf("adj_pos [%d,%d,%d] %2d %3d\n", adj_pos[0], adj_pos[1], adj_pos[2], blk_idx, adj_idx);
                                        
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
