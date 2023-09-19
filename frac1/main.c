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
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, msh.iv, NULL, 0, NULL, NULL);
    
    //solve
    slv_test1(1);
    
    //write
    wrt_coo(&msh, &ocl);
    wrt_vtk(&msh, &ocl);
    
    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}
