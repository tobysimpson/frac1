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
void wrt_raw(struct ocl_obj *ocl, cl_mem buf, size_t n, size_t bytes, char *file_name)
{
//    printf("%s\n",file_name);
    
    //name
    char file1_path[250];
    sprintf(file1_path, "%s%s.raw", ROOT_WRITE, file_name);

    //open
    FILE* file1 = fopen(file1_path,"wb");
  
    //map
    void *ptr1 = clEnqueueMapBuffer(ocl->command_queue, buf, CL_TRUE, CL_MAP_READ, 0, n*bytes, 0, NULL, NULL, &ocl->err);
     
    //write
    fwrite(ptr1, bytes, n, file1);
    
    //unmap
    clEnqueueUnmapMemObject(ocl->command_queue, buf, ptr1, 0, NULL, NULL);

    //close
    fclose(file1);
    
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
    ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->U1u, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
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
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->U1u, ptr, 0, NULL, NULL);
    
    /*
     ===================
     rhs
     ===================
     */
    
    //map read
    ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->F1u, CL_TRUE, CL_MAP_READ, 0, msh->nv_tot*sizeof(cl_float4), 0, NULL, NULL, &ocl->err);
    
    fprintf(file1,"FIELD FieldData2 1\n");
    fprintf(file1,"pf2 4 %zu float\n", msh->nv_tot);
    
    for(int i=0; i<msh->nv_tot; i++)
    {
        fprintf(file1, "%e %e %e %e\n", ptr[i].x, ptr[i].y, ptr[i].z, ptr[i].w);
    }

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->F1u, ptr, 0, NULL, NULL);

    fclose(file1);

    return;
}


#endif /* coo_h */

