//
//  msh.h
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#ifndef msh_h
#define msh_h


//object
struct msh_obj
{
    int         ne[3];      //ele_dim
    int         nv[3];      //vtx_dim
    
    int         ne_tot;     //totals
    int         nv_tot;
    
    //device
    cl_int3     vtx_dim;
    cl_float4   mat_prm;
};


//init
void msh_init(struct msh_obj *msh)
{
    //ele
    msh->ne[0] = 4;
    msh->ne[1] = 4;
    msh->ne[2] = 4;
    
    //vtx
    msh->nv[0] = msh->ne[0] + 1;
    msh->nv[1] = msh->ne[1] + 1;
    msh->nv[2] = msh->ne[2] + 1;
    
    //material
    msh->mat_prm.x = 1e-0f;                                                                                 //youngs E
    msh->mat_prm.y = 0.25f;                                                                                 //poisson v
    msh->mat_prm.z = (msh->mat_prm.x*msh->mat_prm.y)/((1e0f+msh->mat_prm.y)*(1e0f-2e0f*msh->mat_prm.y));    //lame lambda
    msh->mat_prm.w = msh->mat_prm.x/(2e0f*(1e0f+msh->mat_prm.y));                                           //lame mu
    
    printf("mat_prm %e %e %e %e\n", msh->mat_prm.x, msh->mat_prm.y, msh->mat_prm.z, msh->mat_prm.w);
    
    msh->vtx_dim = (cl_int3){msh->nv[0], msh->nv[1], msh->nv[2]};
    
    //totals
    msh->ne_tot = msh->ne[0]*msh->ne[1]*msh->ne[2];
    msh->nv_tot = msh->nv[0]*msh->nv[1]*msh->nv[2];
    
    printf("ne=[%d,%d %d]\n",msh->ne[0],msh->ne[1],msh->ne[2]);
    printf("nv=[%d,%d,%d]\n",msh->nv[0],msh->nv[1],msh->nv[2]);
    
    printf("ne_tot=%d\n", msh->ne_tot);
    printf("nv_tot=%d\n", msh->nv_tot);
    
    return;
}


#endif /* msh_h */
