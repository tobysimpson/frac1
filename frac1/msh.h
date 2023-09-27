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
    cl_int3     ele_dim;
    cl_int3     vtx_dim;
    
    cl_float3   x0;
    cl_float3   x1;
    cl_float3   dx;
    
    cl_float4   mat_prm;
    
    int         ne_tot;     //totals
    int         nv_tot;
};


//init
void msh_init(struct msh_obj *msh)
{
    //dim
    msh->ele_dim = (cl_int3){4,4,4};
    msh->vtx_dim = (cl_int3){msh->ele_dim.x+1, msh->ele_dim.y+1, msh->ele_dim.z+1};
    
    printf("ele_dim %d %d %d\n", msh->ele_dim.x, msh->ele_dim.y, msh->ele_dim.z);
    printf("vtx_dim %d %d %d\n", msh->vtx_dim.x, msh->vtx_dim.y, msh->vtx_dim.z);
    
    //range
    msh->x0 = (cl_float3){-1e0f,-1e0f,-1e0f};
    msh->x1 = (cl_float3){+1e0f,+1e0f,+1e0f};
    msh->dx = (cl_float3){(msh->x1.x - msh->x0.x)/(float)msh->ele_dim.x, (msh->x1.y - msh->x0.y)/(float)msh->ele_dim.y, (msh->x1.z - msh->x0.z)/(float)msh->ele_dim.z};
    
    printf("x0 %+e %+e %+e\n", msh->x0.x, msh->x0.y, msh->x0.z);
    printf("x1 %+e %+e %+e\n", msh->x1.x, msh->x1.y, msh->x1.z);
    printf("dx %+e %+e %+e\n", msh->dx.x, msh->dx.y, msh->dx.z);
    
    //material
    msh->mat_prm.x = 1e-0f;                                                                                 //youngs E
    msh->mat_prm.y = 0.25f;                                                                                 //poisson v
    msh->mat_prm.z = (msh->mat_prm.x*msh->mat_prm.y)/((1e0f+msh->mat_prm.y)*(1e0f-2e0f*msh->mat_prm.y));    //lame lambda
    msh->mat_prm.w = msh->mat_prm.x/(2e0f*(1e0f+msh->mat_prm.y));                                           //lame mu
    
    printf("mat_prm %e %e %e %e\n", msh->mat_prm.x, msh->mat_prm.y, msh->mat_prm.z, msh->mat_prm.w);
    
    //totals
    msh->ne_tot = msh->ele_dim.x*msh->ele_dim.y*msh->ele_dim.z;
    msh->nv_tot = msh->vtx_dim.x*msh->vtx_dim.y*msh->vtx_dim.z;
    
    printf("ne_tot=%d\n", msh->ne_tot);
    printf("nv_tot=%d\n", msh->nv_tot);
    
    return;
}


#endif /* msh_h */
