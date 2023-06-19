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
#include "mtx.h"


//here
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    /*
     ===============
     params
     ===============
     */
    
    size_t  ne   = 4;
    float   xmin = -1e0f;
    float   xmax = +1e0f;
    
    /*
     ===============
     init
     ===============
     */
    
    struct msh_obj msh = {{ne,ne,ne}, {xmin,xmin,xmin}, {xmax,xmax,xmax}};   //ne,xmin,xmax
    struct ocl_obj ocl;
    
    msh_init(&msh);
    ocl_init(&msh, &ocl);
    
    /*
     ===============
     calc
     ===============
     */
    
    //init
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, msh.nv, NULL, 0, NULL, NULL);
    

    
    //write
    mtx_coo(&msh, &ocl);
    mtx_vtk(&msh, &ocl);
    
    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}
