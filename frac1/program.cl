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
                     global     float4  *vtx_ff,
                     global     int4    *coo_ii,
                     global     int4    *coo_jj,
                     global     float4  *coo_aa)
{
    const int3 vtx_dim = (int3){get_global_size(0),get_global_size(1),get_global_size(2)};
    const int3 vtx_pos = (int3){get_global_id(0),get_global_id(1),get_global_id(2)};

    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
//    printf("vtx [%v3d]\n", vtx_pos);
    
    float3 x = buf_cc[0].xyz + convert_float3(vtx_pos)*buf_cc[2].xyz;
    
    vtx_xx[vtx_idx].xyz = x;
    vtx_xx[vtx_idx].w   = 0e0f;
    
    vtx_uu[vtx_idx] = 4*vtx_idx + (float4){0e0f,1e0f,2e0f,3e0f};
    vtx_ff[vtx_idx] = 4*vtx_idx + (float4){0e0f,1e0f,2e0f,3e0f};
    
    
//    int coo_idx = 27*16*vtx_idx;
    
//    //blk
//    for(int i=0; i<27; i++)
//    {
//
//
//        //blk row
//        for(int i=0; i<4; i++)
//        {
//            //blk_col
//            for(int i=0; i<4; i++)
//            {
//
//            }
//        }
//    }
  
    
    return;
}


kernel void vtx_assm(constant   float4  *buf_cc,
                     global     float4  *vtx_xx,
                     global     float4  *vtx_uu,
                     global     float4  *vtx_ff,
                     global     int4    *coo_ii,
                     global     int4    *coo_jj,
                     global     float4  *coo_aa)
{
    const int3 vtx_dim = (int3){get_global_size(0),get_global_size(1),get_global_size(2)};
    const int3 vtx_pos = (int3){get_global_id(0),get_global_id(1),get_global_id(2)} + 1;    //int
    
    int vtx_idx = fn_idx(vtx_pos, vtx_dim);
    
    printf("vtx [%v3d]\n", vtx_pos);
    
    //ele
    for(int ek=0; ek<2; ek++)
    {
        for(int ej=0; ej<2; ej++)
        {
            for(int ei=0; ei<2; ei++)
            {
                int3 ele_pos = vtx_pos + (int3){ei,ej,ek} - (int3)1;

                printf("ele [%v3d]\n", ele_pos);

                //vtx
                for(int vk=0; vk<2; vk++)
                {
                    for(int vj=0; vj<2; vj++)
                    {
                        for(int vi=0; vi<2; vi++)
                        {
                            int3 adj_pos = ele_pos + (int3){vi,vj,vk};

                            printf("adj [%v3d]\n", adj_pos);
                        }
                    }
                }

            }
        }
    }

    return;
}
