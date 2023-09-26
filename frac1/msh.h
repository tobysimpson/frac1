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
    int     ne[3];      //ele_dim
    int     nv[3];      //vtx_dim
    
    int     ne_tot;     //totals
    int     nv_tot;
    
    cl_int3 vtx_dim;    //pass
};


//init
void msh_init(struct msh_obj *msh)
{
    //ele
    msh->ne[0] = 10;
    msh->ne[1] = msh->ne[0];
    msh->ne[2] = msh->ne[0];
    
    //vtx
    msh->nv[0] = msh->ne[0] + 1;
    msh->nv[1] = msh->ne[1] + 1;
    msh->nv[2] = msh->ne[2] + 1;
    
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
