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
    
    //init
    msh_init(&msh);
    ocl_init(&msh, &ocl);
    
    //assemble
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, msh.nv, NULL, 0, NULL, NULL);
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, msh.iv, NULL, 0, NULL, NULL);
    
    //solve
//    slv_test1(1);
    
    //write
    wrt_raw(&ocl, ocl.Juu_ii, 27*9*msh.nv_tot, sizeof(int),   "Juu_ii");
    wrt_raw(&ocl, ocl.Juu_jj, 27*9*msh.nv_tot, sizeof(int),   "Juu_jj");
    wrt_raw(&ocl, ocl.Juu_vv, 27*9*msh.nv_tot, sizeof(float), "Juu_vv");
    
    wrt_raw(&ocl, ocl.U0u, 3*msh.nv_tot, sizeof(float), "U0u");
    wrt_raw(&ocl, ocl.U0c, 1*msh.nv_tot, sizeof(float), "U0c");
    wrt_raw(&ocl, ocl.U1u, 3*msh.nv_tot, sizeof(float), "U1u");
    wrt_raw(&ocl, ocl.U1c, 1*msh.nv_tot, sizeof(float), "U1c");
    wrt_raw(&ocl, ocl.F1u, 3*msh.nv_tot, sizeof(float), "F1u");
    wrt_raw(&ocl, ocl.F1c, 1*msh.nv_tot, sizeof(float), "F1c");

    
//    wrt_vtk(&msh, &ocl);
    
    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}
