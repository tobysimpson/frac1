//
//  program.cl
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//


//proto
int fn_idx(const int3 pos, const int3 dim);


//flat index
int fn_idx(const int3 pos, const int3 dim)
{
     return pos.x + pos.y*(dim.x) + pos.z*(dim.x*dim.y);
}


//init
kernel void vtx_init(constant   float4  *buf_cc,
                     global     float4  *vtx_xx,
                     global     float4  *vtx_uu,
                     global     int     *mtx_ii,
                     global     int     *mtx_jj,
                     global     float   *mtx_kk,
                     global     float   *mtx_ff)
{
    const int3 vtx_dim = (int3){get_global_size(0),get_global_size(1),get_global_size(2)};
    const int3 vtx_pos = (int3){get_global_id(0),get_global_id(1),get_global_id(2)};

    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
//    printf("vtx [%v3d] [%v3d]\n", vtx_pos, vtx_dim);
    
    float3 x = buf_cc[0].xyz + convert_float3(vtx_pos)*buf_cc[2].xyz;
    
    vtx_xx[vtx_idx].xyz = x;
    vtx_xx[vtx_idx].w   = 0;
    
    vtx_uu[vtx_idx].xyz = convert_float3(vtx_pos);
    vtx_uu[vtx_idx].w   = 1e0f;
    
    //zero coo
    for(int i=0; i<27; i++)
    {
        int coo_idx = 27*vtx_idx + i;
        
        mtx_ii[coo_idx] = vtx_idx;
        mtx_jj[coo_idx] = 0e0f;
        mtx_kk[coo_idx] = 0e0f;
    }
    
    mtx_ff[vtx_idx] = 0e0f;
    
    //set identity
    if(any(vtx_pos==0)|any(vtx_pos==(vtx_dim-1)))
    {
        int coo_idx = 27*vtx_idx + 13;

        mtx_jj[coo_idx] = vtx_idx;
        mtx_kk[coo_idx] = 1e0f;
    }
    
    return;
}
