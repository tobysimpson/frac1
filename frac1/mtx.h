//
//  mtx.h
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#ifndef mtx_h
#define mtx_h

#define ROOT_WRITE  "/Users/toby/Downloads/"

//write
void mtx_coo(struct msh_obj *msh, struct ocl_obj *ocl)
{
    //ptr
    void *ptr1;
    void *ptr2;
    void *ptr3;
    void *ptr4;
    
    //file
    FILE* file1;
    FILE* file2;
    FILE* file3;
    FILE* file4;
    
    //name
    char file1_name[250];
    char file2_name[250];
    char file3_name[250];
    char file4_name[250];


    sprintf(file1_name, "%s%s.raw", ROOT_WRITE, "coo_ii");
    sprintf(file2_name, "%s%s.raw", ROOT_WRITE, "coo_jj");
    sprintf(file3_name, "%s%s.raw", ROOT_WRITE, "coo_kk");
    sprintf(file4_name, "%s%s.raw", ROOT_WRITE, "coo_ff");

    //open
    file1 = fopen(file1_name,"wb");
    file2 = fopen(file2_name,"wb");
    file3 = fopen(file3_name,"wb");
    file4 = fopen(file4_name,"wb");
    
    //map
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, ocl->mtx_ii, CL_TRUE, CL_MAP_READ, 0, 27*msh->nv_tot*sizeof(int)  , 0, NULL, NULL, &ocl->err);
    ptr2 = clEnqueueMapBuffer(ocl->command_queue, ocl->mtx_jj, CL_TRUE, CL_MAP_READ, 0, 27*msh->nv_tot*sizeof(int)  , 0, NULL, NULL, &ocl->err);
    ptr3 = clEnqueueMapBuffer(ocl->command_queue, ocl->mtx_vv, CL_TRUE, CL_MAP_READ, 0, 27*msh->nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    ptr4 = clEnqueueMapBuffer(ocl->command_queue, ocl->mtx_ff, CL_TRUE, CL_MAP_READ, 0,    msh->nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    
//    for(int i=0; i<msh->nv_tot; i++)
//    {
//        printf("%e\n", ptr4[i]);
//    }
 
    //write
    fwrite(ptr1, sizeof(int),   27*msh->nv_tot, file1);
    fwrite(ptr2, sizeof(int),   27*msh->nv_tot, file2);
    fwrite(ptr3, sizeof(float), 27*msh->nv_tot, file3);
    fwrite(ptr4, sizeof(float),    msh->nv_tot, file4);
    
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->mtx_ii, ptr1, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->mtx_jj, ptr2, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->mtx_vv, ptr3, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->mtx_ff, ptr4, 0, NULL, NULL);
    
    //close
    fclose(file1);
    fclose(file2);
    fclose(file3);
    fclose(file4);

    return;
}


//write
void mtx_vtk(struct msh_obj *msh, struct ocl_obj *ocl)
{
    //host
    cl_float4 *hst_ptr;
    
    FILE* file1;
    char file_name1[250];
    
    //file name
    sprintf(file_name1, "%s%s.%03d.vtk", ROOT_WRITE, "grid1", 0);
    
    //open
    file1 = fopen(file_name1,"w");
    
    //write
    fprintf(file1,"# vtk DataFile Version 3.0\n");
    fprintf(file1,"grid1\n");
    fprintf(file1,"ASCII\n");
    fprintf(file1,"DATASET STRUCTURED_GRID\n");
    fprintf(file1,"DIMENSIONS %zu %zu %zu\n", msh->nv[0], msh->nv[1], msh->nv[2]);
    
    /*
     ===================
     coords
     ===================
     */
    
    //map read
    hst_ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_xx, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
    fprintf(file1,"POINTS %zu float\n", msh->nv_tot);

    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%+e %+e %+e\n", hst_ptr[i].x, hst_ptr[i].y, hst_ptr[i].z);
    }

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_xx, hst_ptr, 0, NULL, NULL);
    
    /*
     ===================
     vertex data
     ===================
     */
    
    //map read
    hst_ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_uu, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
    fprintf(file1,"\nPOINT_DATA %zu\n", msh->nv_tot);
    fprintf(file1,"VECTORS pv1 float\n");
    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e\n", hst_ptr[i].x, hst_ptr[i].y, hst_ptr[i].z);
    }
    
    fprintf(file1,"FIELD FieldData1 1\n");
    fprintf(file1,"pf1 4 %zu float\n", msh->nv_tot);
    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e %e\n", hst_ptr[i].x, hst_ptr[i].y, hst_ptr[i].z, hst_ptr[i].w);
    }

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_uu, hst_ptr, 0, NULL, NULL);

    fclose(file1);

    return;
}


#endif /* mtx_h */

