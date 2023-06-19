//
//  coo.h
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#ifndef coo_h
#define coo_h

#define ROOT_WRITE  "/Users/toby/Downloads/"

//write
void wrt_coo(struct msh_obj *msh, struct ocl_obj *ocl)
{
    //ptr
    void *ptr1;
    void *ptr2;
    void *ptr3;
    void *ptr4;
    void *ptr5;
    
    //file
    FILE* file1;
    FILE* file2;
    FILE* file3;
    FILE* file4;
    FILE* file5;
    
    //name
    char file1_name[250];
    char file2_name[250];
    char file3_name[250];
    char file4_name[250];
    char file5_name[250];

    sprintf(file1_name, "%s%s.raw", ROOT_WRITE, "vtx_uu");
    sprintf(file2_name, "%s%s.raw", ROOT_WRITE, "vtx_ff");
    sprintf(file3_name, "%s%s.raw", ROOT_WRITE, "coo_ii");
    sprintf(file4_name, "%s%s.raw", ROOT_WRITE, "coo_jj");
    sprintf(file5_name, "%s%s.raw", ROOT_WRITE, "coo_aa");
    
    //open
    file1 = fopen(file1_name,"wb");
    file2 = fopen(file2_name,"wb");
    file3 = fopen(file3_name,"wb");
    file4 = fopen(file4_name,"wb");
    file5 = fopen(file5_name,"wb");
  
    //map
    ptr1 = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_uu, CL_TRUE, CL_MAP_READ, 0,    msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    ptr2 = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_ff, CL_TRUE, CL_MAP_READ, 0,    msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    ptr3 = clEnqueueMapBuffer(ocl->command_queue, ocl->coo_ii, CL_TRUE, CL_MAP_READ, 0, 27*msh->nv_tot*sizeof(cl_int4)  , 0, NULL, NULL, &ocl->err);
    ptr4 = clEnqueueMapBuffer(ocl->command_queue, ocl->coo_jj, CL_TRUE, CL_MAP_READ, 0, 27*msh->nv_tot*sizeof(cl_int4)  , 0, NULL, NULL, &ocl->err);
    ptr5 = clEnqueueMapBuffer(ocl->command_queue, ocl->coo_aa, CL_TRUE, CL_MAP_READ, 0, 27*msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
     
    //write
    fwrite(ptr1, sizeof(cl_float4),    msh->nv_tot, file1);
    fwrite(ptr2, sizeof(cl_float4),    msh->nv_tot, file2);
    fwrite(ptr3, sizeof(cl_int4),   27*msh->nv_tot, file3);
    fwrite(ptr4, sizeof(cl_int4),   27*msh->nv_tot, file4);
    fwrite(ptr5, sizeof(cl_float4), 27*msh->nv_tot, file5);
    
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_uu, ptr1, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_ff, ptr2, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->coo_ii, ptr3, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->coo_jj, ptr4, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->coo_aa, ptr5, 0, NULL, NULL);
    
    //close
    fclose(file1);
    fclose(file2);
    fclose(file3);
    fclose(file4);
    fclose(file5);

    return;
}


//write
void wrt_vtk(struct msh_obj *msh, struct ocl_obj *ocl)
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


#endif /* coo_h */

