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
    FILE* file1 = fopen(file1_name,"wb");
    FILE* file2 = fopen(file2_name,"wb");
    FILE* file3 = fopen(file3_name,"wb");
    FILE* file4 = fopen(file4_name,"wb");
    FILE* file5 = fopen(file5_name,"wb");
  
    //map
    void *ptr1 = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_uu, CL_TRUE, CL_MAP_READ, 0,     4*msh->nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    void *ptr2 = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_ff, CL_TRUE, CL_MAP_READ, 0,     4*msh->nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    void *ptr3 = clEnqueueMapBuffer(ocl->command_queue, ocl->coo_ii, CL_TRUE, CL_MAP_READ, 0, 27*16*msh->nv_tot*sizeof(int)  , 0, NULL, NULL, &ocl->err);
    void *ptr4 = clEnqueueMapBuffer(ocl->command_queue, ocl->coo_jj, CL_TRUE, CL_MAP_READ, 0, 27*16*msh->nv_tot*sizeof(int)  , 0, NULL, NULL, &ocl->err);
    void *ptr5 = clEnqueueMapBuffer(ocl->command_queue, ocl->coo_aa, CL_TRUE, CL_MAP_READ, 0, 27*16*msh->nv_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
     
    //write
    fwrite(ptr1, sizeof(float),     4*msh->nv_tot, file1);
    fwrite(ptr2, sizeof(float),     4*msh->nv_tot, file2);
    fwrite(ptr3, sizeof(int),   27*16*msh->nv_tot, file3);
    fwrite(ptr4, sizeof(int),   27*16*msh->nv_tot, file4);
    fwrite(ptr5, sizeof(float), 27*16*msh->nv_tot, file5);
    
    //close
    fclose(file1);
    fclose(file2);
    fclose(file3);
    fclose(file4);
    fclose(file5);
    
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_uu, ptr1, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_ff, ptr2, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->coo_ii, ptr3, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->coo_jj, ptr4, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->coo_aa, ptr5, 0, NULL, NULL);

    return;
}


//write
void wrt_vtk(struct msh_obj *msh, struct ocl_obj *ocl)
{

    FILE* file1;
    char file1_name[250];
    
    //file name
    sprintf(file1_name, "%s%s.%03d.vtk", ROOT_WRITE, "grid1", 0);
    
    //open
    file1 = fopen(file1_name,"w");
    
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
    cl_float4 *ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_xx, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
    fprintf(file1,"POINTS %zu float\n", msh->nv_tot);

    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e\n", ptr[i].x, ptr[i].y, ptr[i].z);
    }

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_xx, ptr, 0, NULL, NULL);
    
    /*
     ===================
     soln
     ===================
     */
    
    //map read
    ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_uu, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
    fprintf(file1,"\nPOINT_DATA %zu\n", msh->nv_tot);
    fprintf(file1,"VECTORS pv1 float\n");
    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e\n", ptr[i].x, ptr[i].y, ptr[i].z);
    }
    
    fprintf(file1,"FIELD FieldData1 1\n");
    fprintf(file1,"pf1 4 %zu float\n", msh->nv_tot);
    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e %e\n", ptr[i].x, ptr[i].y, ptr[i].z, ptr[i].w);
    }

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_uu, ptr, 0, NULL, NULL);
    
    /*
     ===================
     rhs
     ===================
     */
    
    //map read
    ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->vtx_ff, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
    fprintf(file1,"FIELD FieldData2 1\n");
    fprintf(file1,"pf2 4 %zu float\n", msh->nv_tot);
    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e %e\n", ptr[i].x, ptr[i].y, ptr[i].z, ptr[i].w);
    }

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->vtx_ff, ptr, 0, NULL, NULL);

    fclose(file1);

    return;
}


#endif /* coo_h */

