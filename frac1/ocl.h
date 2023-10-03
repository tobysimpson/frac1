//
//  ocl.h
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#ifndef ocl_h
#define ocl_h


#define ROOT_PRG    "/Users/toby/Documents/USI/postdoc/fracture/xcode/frac1/frac1"


struct coo_dev
{
    cl_mem  ii;
    cl_mem  jj;
    cl_mem  vv;
};

struct coo_hst
{
    int     *ii;
    int     *jj;
    float   *vv;
};

struct mem_hst
{
    float*  vtx_xx;
    
    float*  U1u;    //sln
    float*  F1u;    //rhs
    
    float*  U1c;
    float*  F1c;
    float*  A1c;    //ana
    float*  E1c;    //err
    
    struct coo_hst Juu;
    struct coo_hst Jcc;
};

struct mem_dev
{
    cl_mem vtx_xx;

    cl_mem U1u;
    cl_mem F1u;
    
    cl_mem U1c;
    cl_mem F1c;
    cl_mem A1c;
    cl_mem E1c;
    
    struct coo_dev Juu;
    struct coo_dev Jcc;
};

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
        
    //memory
    struct mem_hst hst;
    struct mem_dev dev;
    
    //kernels
    cl_kernel           vtx_init;
    cl_kernel           vtx_assm;
    cl_kernel           vtx_bnd1;
    cl_kernel           vtx_err1;
    cl_kernel           fac_bnd1;
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
    ocl->vtx_bnd1 = clCreateKernel(ocl->program, "vtx_bnd1", &ocl->err);
    ocl->vtx_err1 = clCreateKernel(ocl->program, "vtx_err1", &ocl->err);
    ocl->fac_bnd1 = clCreateKernel(ocl->program, "fac_bnd1", &ocl->err);
    
    /*
     =============================
     memory
     =============================
     */
    
    //host
    ocl->hst.vtx_xx = malloc(3*msh->nv_tot*sizeof(float));
    
    ocl->hst.U1u = malloc(3*msh->nv_tot*sizeof(float));
    ocl->hst.F1u = malloc(3*msh->nv_tot*sizeof(float));
    
    ocl->hst.U1c = malloc(msh->nv_tot*sizeof(float));
    ocl->hst.F1c = malloc(msh->nv_tot*sizeof(float));
    ocl->hst.A1c = malloc(msh->nv_tot*sizeof(float));
    ocl->hst.E1c = malloc(msh->nv_tot*sizeof(float));
    
    ocl->hst.Juu.ii = malloc(27*9*msh->nv_tot*sizeof(int));
    ocl->hst.Juu.jj = malloc(27*9*msh->nv_tot*sizeof(int));
    ocl->hst.Juu.vv = malloc(27*9*msh->nv_tot*sizeof(float));
    
    ocl->hst.Jcc.ii = malloc(27*1*msh->nv_tot*sizeof(int));
    ocl->hst.Jcc.jj = malloc(27*1*msh->nv_tot*sizeof(int));
    ocl->hst.Jcc.vv = malloc(27*1*msh->nv_tot*sizeof(float));
    
    //CL_MEM_READ_WRITE/CL_MEM_HOST_READ_ONLY/CL_MEM_HOST_NO_ACCESS
    
    //device
    ocl->dev.vtx_xx = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->dev.U1u    = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,     3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->dev.F1u    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 3*msh->nv_tot*sizeof(float), NULL, &ocl->err);
    
    ocl->dev.U1c    = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,     msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->dev.F1c    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->dev.A1c    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, msh->nv_tot*sizeof(float), NULL, &ocl->err);
    ocl->dev.E1c    = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, msh->nv_tot*sizeof(float), NULL, &ocl->err);

    ocl->dev.Juu.ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*9*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->dev.Juu.jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*9*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->dev.Juu.vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*9*msh->nv_tot*sizeof(float), NULL, &ocl->err);

    ocl->dev.Jcc.ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->dev.Jcc.jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(int),   NULL, &ocl->err);
    ocl->dev.Jcc.vv = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(float), NULL, &ocl->err);

    /*
     =============================
     arguments
     =============================
     */

    ocl->err = clSetKernelArg(ocl->vtx_init,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_init,  1, sizeof(cl_float3), (void*)&msh->x0);
    ocl->err = clSetKernelArg(ocl->vtx_init,  2, sizeof(cl_float3), (void*)&msh->dx);
    ocl->err = clSetKernelArg(ocl->vtx_init,  3, sizeof(cl_mem),    (void*)&ocl->dev.vtx_xx);
    ocl->err = clSetKernelArg(ocl->vtx_init,  4, sizeof(cl_mem),    (void*)&ocl->dev.U1u);
    ocl->err = clSetKernelArg(ocl->vtx_init,  5, sizeof(cl_mem),    (void*)&ocl->dev.F1u);
    ocl->err = clSetKernelArg(ocl->vtx_init,  6, sizeof(cl_mem),    (void*)&ocl->dev.U1c);
    ocl->err = clSetKernelArg(ocl->vtx_init,  7, sizeof(cl_mem),    (void*)&ocl->dev.F1c);
    ocl->err = clSetKernelArg(ocl->vtx_init,  8, sizeof(cl_mem),    (void*)&ocl->dev.A1c);
    ocl->err = clSetKernelArg(ocl->vtx_init,  9, sizeof(cl_mem),    (void*)&ocl->dev.E1c);
    ocl->err = clSetKernelArg(ocl->vtx_init, 10, sizeof(cl_mem),    (void*)&ocl->dev.Juu.ii);
    ocl->err = clSetKernelArg(ocl->vtx_init, 11, sizeof(cl_mem),    (void*)&ocl->dev.Juu.jj);
    ocl->err = clSetKernelArg(ocl->vtx_init, 12, sizeof(cl_mem),    (void*)&ocl->dev.Juu.vv);
    ocl->err = clSetKernelArg(ocl->vtx_init, 13, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.ii);
    ocl->err = clSetKernelArg(ocl->vtx_init, 14, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.jj);
    ocl->err = clSetKernelArg(ocl->vtx_init, 15, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.vv);
    
    ocl->err = clSetKernelArg(ocl->vtx_assm,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  1, sizeof(cl_float3), (void*)&msh->x0);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  2, sizeof(cl_float3), (void*)&msh->dx);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  3, sizeof(cl_float4), (void*)&msh->mat_prm);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  4, sizeof(cl_mem),    (void*)&ocl->dev.U1u);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  5, sizeof(cl_mem),    (void*)&ocl->dev.F1u);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  6, sizeof(cl_mem),    (void*)&ocl->dev.U1c);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  7, sizeof(cl_mem),    (void*)&ocl->dev.F1c);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  8, sizeof(cl_mem),    (void*)&ocl->dev.Juu.ii);
    ocl->err = clSetKernelArg(ocl->vtx_assm,  9, sizeof(cl_mem),    (void*)&ocl->dev.Juu.jj);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 10, sizeof(cl_mem),    (void*)&ocl->dev.Juu.vv);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 11, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.ii);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 12, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.jj);
    ocl->err = clSetKernelArg(ocl->vtx_assm, 13, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.vv);
    
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  1, sizeof(cl_float3), (void*)&msh->x0);
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  2, sizeof(cl_float3), (void*)&msh->dx);
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  3, sizeof(cl_mem),    (void*)&ocl->dev.U1c);
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  4, sizeof(cl_mem),    (void*)&ocl->dev.F1c);
    ocl->err = clSetKernelArg(ocl->vtx_bnd1,  5, sizeof(cl_mem),    (void*)&ocl->dev.Jcc.vv);
    
    ocl->err = clSetKernelArg(ocl->vtx_err1,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->vtx_err1,  1, sizeof(cl_mem),    (void*)&ocl->dev.U1c);
    ocl->err = clSetKernelArg(ocl->vtx_err1,  2, sizeof(cl_mem),    (void*)&ocl->dev.A1c);
    ocl->err = clSetKernelArg(ocl->vtx_err1,  3, sizeof(cl_mem),    (void*)&ocl->dev.E1c);
    
    ocl->err = clSetKernelArg(ocl->fac_bnd1,  0, sizeof(cl_int3),   (void*)&msh->vtx_dim);
    ocl->err = clSetKernelArg(ocl->fac_bnd1,  1, sizeof(cl_mem),    (void*)&ocl->dev.F1u);
    ocl->err = clSetKernelArg(ocl->fac_bnd1,  2, sizeof(cl_mem),    (void*)&ocl->dev.Juu.vv);
}


//final
void ocl_final(struct ocl_obj *ocl)
{
    ocl->err = clFlush(ocl->command_queue);
    ocl->err = clFinish(ocl->command_queue);

    //kernels
    ocl->err = clReleaseKernel(ocl->vtx_init);
    ocl->err = clReleaseKernel(ocl->vtx_assm);
    ocl->err = clReleaseKernel(ocl->vtx_bnd1);
    ocl->err = clReleaseKernel(ocl->vtx_err1);
    ocl->err = clReleaseKernel(ocl->fac_bnd1);
    

    //device
    ocl->err = clReleaseMemObject(ocl->dev.vtx_xx);
    
    ocl->err = clReleaseMemObject(ocl->dev.U1u);
    ocl->err = clReleaseMemObject(ocl->dev.F1u);
    
    ocl->err = clReleaseMemObject(ocl->dev.U1c);
    ocl->err = clReleaseMemObject(ocl->dev.F1c);
    ocl->err = clReleaseMemObject(ocl->dev.A1c);
    ocl->err = clReleaseMemObject(ocl->dev.E1c);
    
    ocl->err = clReleaseMemObject(ocl->dev.Juu.ii);
    ocl->err = clReleaseMemObject(ocl->dev.Juu.jj);
    ocl->err = clReleaseMemObject(ocl->dev.Juu.vv);
    
    ocl->err = clReleaseMemObject(ocl->dev.Jcc.ii);
    ocl->err = clReleaseMemObject(ocl->dev.Jcc.jj);
    ocl->err = clReleaseMemObject(ocl->dev.Jcc.vv);
    
    ocl->err = clReleaseProgram(ocl->program);
    ocl->err = clReleaseCommandQueue(ocl->command_queue);
    ocl->err = clReleaseContext(ocl->context);
    
    //host
    free(ocl->hst.vtx_xx);
    
    free(ocl->hst.U1u);
    free(ocl->hst.F1u);
    
    free(ocl->hst.U1c);
    free(ocl->hst.F1c);
    free(ocl->hst.A1c);
    free(ocl->hst.E1c);
    
    free(ocl->hst.Juu.ii);
    free(ocl->hst.Juu.jj);
    free(ocl->hst.Juu.vv);

    free(ocl->hst.Jcc.ii);
    free(ocl->hst.Jcc.jj);
    free(ocl->hst.Jcc.vv);

    return;
}



#endif /* ocl_h */
