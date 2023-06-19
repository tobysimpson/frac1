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
    
    cl_float3   xmin;
    cl_float3   xmax;
    cl_float3   dx;

    size_t      nv[3];      //vtx_dim
    size_t      ne_tot;     //totals
    size_t      nv_tot;
    
    size_t      ie[3];      //interior
    size_t      iv[3];
    
    cl_float3   cc[3];      //constants {xmin, xmax, dx}
};


//init
void msh_init(struct msh_obj *msh)
{
    //dims
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
    
    //constants
    msh->cc[0] = msh->xmin;
    msh->cc[1] = msh->xmax;
    msh->cc[2] = (cl_float4){(msh->cc[1].x - msh->cc[0].x)/(float)msh->ne[0],(msh->cc[1].y - msh->cc[0].y)/(float)msh->ne[1], (msh->cc[1].z - msh->cc[0].z)/(float)msh->ne[2], 0e0f};  //dx
    
    printf("xmin=[%+f,%+f,%+f]\n",msh->cc[0].x,msh->cc[0].y,msh->cc[0].z);
    printf("xmax=[%+f,%+f,%+f]\n",msh->cc[1].x,msh->cc[1].y,msh->cc[1].z);
    printf("dx  =[%+f,%+f,%+f]\n",msh->cc[2].x,msh->cc[2].y,msh->cc[2].z);
    
    //interior dims
    msh->ie[0] = msh->ne[0] - 2;
    msh->ie[1] = msh->ne[1] - 2;
    msh->ie[2] = msh->ne[2] - 2;

    msh->iv[0] = msh->nv[0] - 2;
    msh->iv[1] = msh->nv[1] - 2;
    msh->iv[2] = msh->nv[2] - 2;
    
    return;
}


#endif /* msh_h */
