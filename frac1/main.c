//
//  main.c
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#include <stdio.h>
#include <OpenCL/opencl.h>
#include <Accelerate/Accelerate.h>

#include "msh.h"
#include "ocl.h"
#include "slv.h"
#include "io.h"

//here
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //objects
    struct msh_obj msh;
    struct ocl_obj ocl;
    
    //init obj
    msh_init(&msh);
    ocl_init(&msh, &ocl);
    
    //cast dims
    size_t nv[3] = {msh.nv[0],msh.nv[1],msh.nv[2]};
    size_t f1[3] = {msh.nv[1],msh.nv[2],0}; //x=y*z
    
    //kernels
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.fac_bnd1, 2, NULL, f1, NULL, 0, NULL, NULL);
    
    //solve
    slv_test1(&msh, &ocl);
    
    //debug mtx
    wrt_raw(&ocl, ocl.Juu_ii, 27*9*msh.nv_tot, sizeof(int),   "Juu_ii");
    wrt_raw(&ocl, ocl.Juu_jj, 27*9*msh.nv_tot, sizeof(int),   "Juu_jj");
    wrt_raw(&ocl, ocl.Juu_vv, 27*9*msh.nv_tot, sizeof(float), "Juu_vv");
    
    wrt_raw(&ocl, ocl.Juc_ii, 27*3*msh.nv_tot, sizeof(int),   "Juc_ii");
    wrt_raw(&ocl, ocl.Juc_jj, 27*3*msh.nv_tot, sizeof(int),   "Juc_jj");
    wrt_raw(&ocl, ocl.Juc_vv, 27*3*msh.nv_tot, sizeof(float), "Juc_vv");
    
    wrt_raw(&ocl, ocl.Jcu_ii, 27*3*msh.nv_tot, sizeof(int),   "Jcu_ii");
    wrt_raw(&ocl, ocl.Jcu_jj, 27*3*msh.nv_tot, sizeof(int),   "Jcu_jj");
    wrt_raw(&ocl, ocl.Jcu_vv, 27*3*msh.nv_tot, sizeof(float), "Jcu_vv");
    
    wrt_raw(&ocl, ocl.Jcc_ii, 27*msh.nv_tot, sizeof(int),   "Jcc_ii");
    wrt_raw(&ocl, ocl.Jcc_jj, 27*msh.nv_tot, sizeof(int),   "Jcc_jj");
    wrt_raw(&ocl, ocl.Jcc_vv, 27*msh.nv_tot, sizeof(float), "Jcc_vv");
    

    wrt_vtk(&msh, &ocl);
    

    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}
