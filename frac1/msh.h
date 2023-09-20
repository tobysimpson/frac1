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
    size_t      ne[3];      //ele_dim
    size_t      nv[3];      //vtx_dim
    
    size_t      ne_tot;     //totals
    size_t      nv_tot;
    
    cl_float3   x0;         //x min
    cl_float3   x1;         //x max
    cl_float3   dx;
    
    size_t      ie[3];      //interior
    size_t      iv[3];
    
    cl_float3   cc[3];      //const for device
};


//init
void msh_init(struct msh_obj *msh)
{
    //ele
    msh->ne[0] = 8;
    msh->ne[1] = msh->ne[0];
    msh->ne[2] = msh->ne[0];
    
    //vtx
    msh->nv[0] = msh->ne[0] + 1;
    msh->nv[1] = msh->ne[1] + 1;
    msh->nv[2] = msh->ne[2] + 1;
    
    //totals
    msh->ne_tot = msh->ne[0]*msh->ne[1]*msh->ne[2];
    msh->nv_tot = msh->nv[0]*msh->nv[1]*msh->nv[2];
    
    printf("ne=[%zu,%zu,%zu]\n",msh->ne[0],msh->ne[1],msh->ne[2]);
    printf("nv=[%zu,%zu,%zu]\n",msh->nv[0],msh->nv[1],msh->nv[2]);
    
    printf("ne_tot=%zu\n", msh->ne_tot);
    printf("nv_tot=%zu\n", msh->nv_tot);
    
    //size
    msh->x0 = (cl_float3){0e0f, 0e0f, 0e0f};
    msh->x1 = (cl_float3){1e0f, 1e0f, 1e0f};
    
    //dx
    msh->dx.x = (msh->x1.x - msh->x0.x)/(float)msh->ne[0];
    msh->dx.y = (msh->x1.y - msh->x0.y)/(float)msh->ne[1];
    msh->dx.z = (msh->x1.z - msh->x0.z)/(float)msh->ne[2];

    printf("xmin=[%+f,%+f,%+f]\n", msh->x0.x, msh->x0.y, msh->x0.z);
    printf("xmax=[%+f,%+f,%+f]\n", msh->x1.x, msh->x1.y, msh->x1.z);
    printf("dx  =[%+f,%+f,%+f]\n", msh->dx.x, msh->dx.y, msh->dx.z);
    
    //interior dims
    msh->ie[0] = msh->ne[0] - 2;
    msh->ie[1] = msh->ne[1] - 2;
    msh->ie[2] = msh->ne[2] - 2;

    msh->iv[0] = msh->nv[0] - 2;
    msh->iv[1] = msh->nv[1] - 2;
    msh->iv[2] = msh->nv[2] - 2;
    
    //constants
    msh->cc[0] = msh->x0;
    msh->cc[1] = msh->x1;
    msh->cc[2] = msh->dx;
    
    return;
}


#endif /* msh_h */
