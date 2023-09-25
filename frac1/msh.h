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
    
    size_t      f1[2];      //face
    
    cl_ulong3   vtx_dim;
};


//init
void msh_init(struct msh_obj *msh)
{
    //ele
    msh->ne[0] = 3;
    msh->ne[1] = msh->ne[0];
    msh->ne[2] = msh->ne[0];
    
    //vtx
    msh->nv[0] = msh->ne[0] + 1;
    msh->nv[1] = msh->ne[1] + 1;
    msh->nv[2] = msh->ne[2] + 1;
    
    //face
    msh->f1[0] = msh->nv[1];    //x=y*z
    msh->f1[1] = msh->nv[2];
    
    msh->vtx_dim = (cl_ulong3){msh->nv[0], msh->nv[1], msh->nv[2]};
    
    //totals
    msh->ne_tot = msh->ne[0]*msh->ne[1]*msh->ne[2];
    msh->nv_tot = msh->nv[0]*msh->nv[1]*msh->nv[2];
    
    printf("ne=[%zu,%zu,%zu]\n",msh->ne[0],msh->ne[1],msh->ne[2]);
    printf("nv=[%zu,%zu,%zu]\n",msh->nv[0],msh->nv[1],msh->nv[2]);
    
    printf("ne_tot=%zu\n", msh->ne_tot);
    printf("nv_tot=%zu\n", msh->nv_tot);
    
    return;
}


#endif /* msh_h */
