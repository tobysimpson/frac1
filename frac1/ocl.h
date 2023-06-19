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
    cl_mem              buf_cc;
    cl_mem              vtx_xx;
    cl_mem              vtx_uu; //sln
    cl_mem              vtx_ff; //rhs
    
    //coo
    cl_mem              coo_ii;
    cl_mem              coo_jj;
    cl_mem              coo_aa;
    
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
        fprintf(stderr, "Failed to load kernel.\n");
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
    
    //log
    size_t log_size = 0;;
    char *log = NULL;
    
//    printf("%s\n",prg_src);
    
    //create
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&prg_src, (const size_t*)&prg_len, &ocl->err);
    printf("prg %d\n",ocl->err);
            
    //build
    ocl->err = clBuildProgram(ocl->program, 1, &ocl->device_id, NULL, NULL, NULL);
    printf("bld %d\n",ocl->err);
    
    //log size
    clGetProgramBuildInfo(ocl->program, ocl->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    
    //allocate
    log = (char*)malloc(log_size);
    
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
     memory
     =============================
     */
    
    //CL_MEM_HOST_READ_ONLY/CL_MEM_HOST_NO_ACCESS

    //constants
    ocl->buf_cc = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, 3*sizeof(cl_float4), (void*)&msh->cc, &ocl->err);

    //memory
    ocl->vtx_xx = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    ocl->vtx_uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    ocl->vtx_ff = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY,    msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    
    ocl->coo_ii = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(cl_int4),   NULL, &ocl->err);
    ocl->coo_jj = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(cl_int4),   NULL, &ocl->err);
    ocl->coo_aa = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, 27*msh->nv_tot*sizeof(cl_float4), NULL, &ocl->err);
    

    /*
     =============================
     kernels
     =============================
     */
    
    ocl->vtx_init = clCreateKernel(ocl->program, "vtx_init", &ocl->err);
    ocl->vtx_assm = clCreateKernel(ocl->program, "vtx_assm", &ocl->err);
  
    /*
     =============================
     arguments
     =============================
     */
    
    ocl->err = clSetKernelArg(ocl->vtx_init, 0, sizeof(cl_mem), (void*)&ocl->buf_cc);
    ocl->err = clSetKernelArg(ocl->vtx_init, 1, sizeof(cl_mem), (void*)&ocl->vtx_xx);
    ocl->err = clSetKernelArg(ocl->vtx_init, 2, sizeof(cl_mem), (void*)&ocl->vtx_uu);
    ocl->err = clSetKernelArg(ocl->vtx_init, 3, sizeof(cl_mem), (void*)&ocl->vtx_ff);
    ocl->err = clSetKernelArg(ocl->vtx_init, 4, sizeof(cl_mem), (void*)&ocl->coo_ii);
    ocl->err = clSetKernelArg(ocl->vtx_init, 5, sizeof(cl_mem), (void*)&ocl->coo_jj);
    ocl->err = clSetKernelArg(ocl->vtx_init, 6, sizeof(cl_mem), (void*)&ocl->coo_aa);
    
    
}


//final
void ocl_final(struct ocl_obj *ocl)
{
    ocl->err = clFlush(ocl->command_queue);
    ocl->err = clFinish(ocl->command_queue);
    
    //kernels
    ocl->err = clReleaseKernel(ocl->vtx_init);

    //memory
    ocl->err = clReleaseMemObject(ocl->buf_cc);
    ocl->err = clReleaseMemObject(ocl->vtx_xx);
    ocl->err = clReleaseMemObject(ocl->vtx_uu);
    ocl->err = clReleaseMemObject(ocl->vtx_ff);
    
    ocl->err = clReleaseMemObject(ocl->coo_ii);
    ocl->err = clReleaseMemObject(ocl->coo_jj);
    ocl->err = clReleaseMemObject(ocl->coo_aa);
    


    ocl->err = clReleaseProgram(ocl->program);
    ocl->err = clReleaseCommandQueue(ocl->command_queue);
    ocl->err = clReleaseContext(ocl->context);
    
    return;
}



#endif /* ocl_h */
