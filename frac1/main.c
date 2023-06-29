//
//  main.c
//  frac1
//
//  Created by Toby Simpson on 19.06.23.
//

#include <stdio.h>
#include <OpenCL/opencl.h>

#include "msh.h"
#include "ocl.h"
#include "wrt.h"

//here
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //params
    size_t  ne = 4;
    float   x0 = 0e0f;
    float   x1 = ne;
    
    //objects
    struct msh_obj msh = {{ne,ne,ne}, {x0,x0,x0}, {x1,x1,x1}};   //ne,x0,x1
    struct ocl_obj ocl;
    
    //init
    msh_init(&msh);
    ocl_init(&msh, &ocl);
    
    //calc
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, msh.nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, msh.iv, NULL, 0, NULL, NULL);
    
    //write
    wrt_coo(&msh, &ocl);
    wrt_vtk(&msh, &ocl);
    
    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}
