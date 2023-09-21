//
//  ocl.h
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#ifndef ocl_h
#define ocl_h


#define ROOT_PRG    "/Users/toby/Documents/USI/postdoc/fracture/xcode/frac1/frac1"


//object
struct ocl_obj
{
    //environment
    cl_int              err;
    cl_platform_id      platform_id;
    cl_device_id        device_id;
    cl_uint             num_devices;
    cl_uint             num_platforms;
    cl_context          context;
    cl_command_queue    command_queue;
    cl_program          program;
    char                device_str[100];
        
    //device memory
    cl_mem              vtx_xx;
    
    cl_mem              U0u;    //prior
    cl_mem              U0c;
    
    cl_mem              U1u;    //current
    cl_mem              U1c;
    
    cl_mem              F1u;    //rhs
    cl_mem              F1c;
    
    cl_mem              Juu_ii; //coo
    cl_mem              Juu_jj;
    cl_mem              Juu_vv;
    
    cl_mem              Juc_ii;
    cl_mem              Juc_jj;
    cl_mem              Juc_vv;
    
    cl_mem              Jcu_ii;
    cl_mem              Jcu_jj;
    cl_mem              Jcu_vv;

    cl_mem              Jcc_ii;
    cl_mem              Jcc_jj;
    cl_mem              Jcc_vv;
    
    //kernels
    cl_kernel           vtx_init;
    cl_kernel           vtx_assm;
};


//init
void ocl_init(struct msh_obj *msh, struct ocl_obj *ocl)
{
    /*
     =============================
     environment
     =============================
     */
    
    ocl->err            = clGetPlatformIDs(1, &ocl->platform_id, &ocl->num_platforms);                                              //platform
    ocl->err            = clGetDeviceIDs(ocl->platform_id, CL_DEVICE_TYPE_GPU, 1, &ocl->device_id, &ocl->num_devices);              //devices
    ocl->context        = clCreateContext(NULL, ocl->num_devices, &ocl->device_id, NULL, NULL, &ocl->err);                          //context
    ocl->command_queue  = clCreateCommandQueue(ocl->context, ocl->device_id, 0, &ocl->err);                                         //command queue
    ocl->err            = clGetDeviceInfo(ocl->device_id, CL_DEVICE_NAME, sizeof(ocl->device_str), &ocl->device_str, NULL);         //device info
    
    printf("%s\n", ocl->device_str);
    
    /*
     =============================
     program
     =============================
     */
    
    //name
    char prg_name[200];
    sprintf(prg_name,"%s/%s", ROOT_PRG, "program.cl");

    printf("%s\n",prg_name);

    //file
    FILE* src_file = fopen(prg_name, "r");
    if(!src_file)
    {
        fprintf(stderr, "Failed to load kernel. check ROOT_PRG\n");
        exit(1);
    }

    //length
    fseek(src_file, 0, SEEK_END);
    size_t  prg_len =  ftell(src_file);
    rewind(src_file);

//    printf("%lu\n",prg_len);

    //source
    char *prg_src = (char*)malloc(prg_len);
    fread(prg_src, sizeof(char), prg_len, src_file);
    fclose(src_file);

//    printf("%s\n",prg_src);

    //create
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&prg_src, (const size_t*)&prg_len, &ocl->err);
    printf("prg %d\n",ocl->err);

    //build
    ocl->err = clBuildProgram(ocl->program, 1, &ocl->device_id, NULL, NULL, NULL);
    printf("bld %d\n",ocl->err);

    //log
    size_t log_size = 0;
    
    //log size
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    //allocate
    char *log = (char*)malloc(log_size);

    //log text
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    //print
    printf("%s\n", log);

    //clear
    free(log);

    //clean
    free(prg_src);

    //unload compiler
    ocl->err = clUnloadPlatformCompiler(ocl->platform_id);
    
    /*
     =============================
     kernels
     =============================
     */

    ocl->vtx_init = clCreateKernel(ocl->program, "vtx_init", &ocl->err);
    ocl->vtx_assm = clCreateKernel(ocl->program, "vtx_assm", &ocl->err);
    
    /*
     =============================
     memory
     =============================
     */
    
    //CL_MEM_HOST_READ_ONLY/CL_MEM_HOST_NO_ACCESS

    //memory
    ocl->vtx_xx = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->U0u    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->U0c    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    1*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->U1u    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->U1c    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    1*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->F1u    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->F1c    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    1*msh->nv_tot*sizeof(float), NULL, &ocl->err);

    ocl->Juu_ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*9*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Juu_jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*9*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Juu_vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*9*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->Juc_ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*3*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Juc_jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*3*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Juc_vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->Jcu_ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*3*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Jcu_jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*3*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Jcu_vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->Jcc_ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*1*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Jcc_jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*1*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->Jcc_vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*1*msh->nv_tot*sizeof(float), NULL, &ocl->err);

    /*
     =============================
     arguments
     =============================
     */

    ocl->err = clSetKernelArg(ocl->vtx_init,  0, sizeof(cl_mem), (void*)&ocl->vtx_xx);
    ocl->err = clSetKernelArg(ocl->vtx_init,  1, sizeof(cl_mem), (void*)&ocl->U0u);
    ocl->err = clSetKernelArg(ocl->vtx_init,  2, sizeof(cl_mem), (void*)&ocl->U0c);
    ocl->err = clSetKernelArg(ocl->vtx_init,  3, sizeof(cl_mem), (void*)&ocl->U1u);
    ocl->err = clSetKernelArg(ocl->vtx_init,  4, sizeof(cl_mem), (void*)&ocl->U1c);
    ocl->err = clSetKernelArg(ocl->vtx_init,  5, sizeof(cl_mem), (void*)&ocl->F1u);
    ocl->err = clSetKernelArg(ocl->vtx_init,  6, sizeof(cl_mem), (void*)&ocl->F1c);
    ocl->err = clSetKernelArg(ocl->vtx_init,  7, sizeof(cl_mem), (void*)&ocl->Juu_ii);
    ocl->err = clSetKernelArg(ocl->vtx_init,  8, sizeof(cl_mem), (void*)&ocl->Juu_jj);
    ocl->err = clSetKernelArg(ocl->vtx_init,  9, sizeof(cl_mem), (void*)&ocl->Juu_vv);
    ocl->err = clSetKernelArg(ocl->vtx_init, 10, sizeof(cl_mem), (void*)&ocl->Juc_ii);
    ocl->err = clSetKernelArg(ocl->vtx_init, 11, sizeof(cl_mem), (void*)&ocl->Juc_jj);
    ocl->err = clSetKernelArg(ocl->vtx_init, 12, sizeof(cl_mem), (void*)&ocl->Juc_vv);
    ocl->err = clSetKernelArg(ocl->vtx_init, 13, sizeof(cl_mem), (void*)&ocl->Jcu_ii);
    ocl->err = clSetKernelArg(ocl->vtx_init, 14, sizeof(cl_mem), (void*)&ocl->Jcu_jj);
    ocl->err = clSetKernelArg(ocl->vtx_init, 15, sizeof(cl_mem), (void*)&ocl->Jcu_vv);
    ocl->err = clSetKernelArg(ocl->vtx_init, 16, sizeof(cl_mem), (void*)&ocl->Jcc_ii);
    ocl->err = clSetKernelArg(ocl->vtx_init, 17, sizeof(cl_mem), (void*)&ocl->Jcc_jj);
    ocl->err = clSetKernelArg(ocl->vtx_init, 18, sizeof(cl_mem), (void*)&ocl->Jcc_vv);
    
    ocl->err = clSetKernelArg(ocl->vtx_assm,  0, sizeof(cl_mem), (void*)&ocl->vtx_xx);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  1, sizeof(cl_mem), (void*)&ocl->U0u);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  2, sizeof(cl_mem), (void*)&ocl->U0c);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  3, sizeof(cl_mem), (void*)&ocl->U1u);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  4, sizeof(cl_mem), (void*)&ocl->U1c);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  5, sizeof(cl_mem), (void*)&ocl->F1u);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  6, sizeof(cl_mem), (void*)&ocl->F1c);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  7, sizeof(cl_mem), (void*)&ocl->Juu_ii);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  8, sizeof(cl_mem), (void*)&ocl->Juu_jj);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  9, sizeof(cl_mem), (void*)&ocl->Juu_vv);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 10, sizeof(cl_mem), (void*)&ocl->Juc_ii);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 11, sizeof(cl_mem), (void*)&ocl->Juc_jj);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 12, sizeof(cl_mem), (void*)&ocl->Juc_vv);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 13, sizeof(cl_mem), (void*)&ocl->Jcu_ii);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 14, sizeof(cl_mem), (void*)&ocl->Jcu_jj);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 15, sizeof(cl_mem), (void*)&ocl->Jcu_vv);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 16, sizeof(cl_mem), (void*)&ocl->Jcc_ii);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 17, sizeof(cl_mem), (void*)&ocl->Jcc_jj);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 18, sizeof(cl_mem), (void*)&ocl->Jcc_vv);
    
}


//final
void ocl_final(struct ocl_obj *ocl)
{
    ocl->err = clFlush(ocl->command_queue);
    ocl->err = clFinish(ocl->command_queue);

    //kernels
    ocl->err = clReleaseKernel(ocl->vtx_init);
    ocl->err = clReleaseKernel(ocl->vtx_assm);

    //memory
    ocl->err = clReleaseMemObject(ocl->vtx_xx);
    ocl->err = clReleaseMemObject(ocl->U0u);
    ocl->err = clReleaseMemObject(ocl->U0c);
    ocl->err = clReleaseMemObject(ocl->U1u);
    ocl->err = clReleaseMemObject(ocl->U1c);
    ocl->err = clReleaseMemObject(ocl->F1u);
    ocl->err = clReleaseMemObject(ocl->F1c);
    ocl->err = clReleaseMemObject(ocl->Juu_ii);
    ocl->err = clReleaseMemObject(ocl->Juu_jj);
    ocl->err = clReleaseMemObject(ocl->Juu_vv);
    ocl->err = clReleaseMemObject(ocl->Juc_ii);
    ocl->err = clReleaseMemObject(ocl->Juc_jj);
    ocl->err = clReleaseMemObject(ocl->Juc_vv);
    ocl->err = clReleaseMemObject(ocl->Jcu_ii);
    ocl->err = clReleaseMemObject(ocl->Jcu_jj);
    ocl->err = clReleaseMemObject(ocl->Jcu_vv);
    ocl->err = clReleaseMemObject(ocl->Jcc_ii);
    ocl->err = clReleaseMemObject(ocl->Jcc_jj);
    ocl->err = clReleaseMemObject(ocl->Jcc_vv);

    ocl->err = clReleaseProgram(ocl->program);
    ocl->err = clReleaseCommandQueue(ocl->command_queue);
    ocl->err = clReleaseContext(ocl->context);
    
    return;
}



#endif /* ocl_h */
