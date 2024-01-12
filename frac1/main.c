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

//for later
//clSetKernelArg(myKernel, 0, sizeof(cl_int), &myVariable).

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
    size_t nv[3] = {msh.vtx_dim.x, msh.vtx_dim.y, msh.vtx_dim.z};
//    size_t f1[2] = {msh.vtx_dim.y, msh.vtx_dim.z};
    
    //kernels
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, nv, NULL, 0, NULL, &ocl.event);
    
    //for profiling
    clWaitForEvents(1, &ocl.event);
    
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd1, 3, NULL, nv, NULL, 0, NULL, NULL); //c
//    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.fac_bnd1, 2, NULL, f1, NULL, 0, NULL, NULL); //u
    

    
    //read from device
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.vtx_xx, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.vtx_xx, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.U1u, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.F1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.F1u, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U0c, CL_TRUE, 0, 1*msh.nv_tot*sizeof(float), ocl.hst.U0c, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.U1c, CL_TRUE, 0, 1*msh.nv_tot*sizeof(float), ocl.hst.U1c, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.F1c, CL_TRUE, 0, 1*msh.nv_tot*sizeof(float), ocl.hst.F1c, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Juu.ii, CL_TRUE, 0, 27*9*msh.nv_tot*sizeof(int),   ocl.hst.Juu.ii, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Juu.jj, CL_TRUE, 0, 27*9*msh.nv_tot*sizeof(int),   ocl.hst.Juu.jj, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Juu.vv, CL_TRUE, 0, 27*9*msh.nv_tot*sizeof(float), ocl.hst.Juu.vv, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Juc.ii, CL_TRUE, 0, 27*3*msh.nv_tot*sizeof(int),   ocl.hst.Juc.ii, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Juc.jj, CL_TRUE, 0, 27*3*msh.nv_tot*sizeof(int),   ocl.hst.Juc.jj, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Juc.vv, CL_TRUE, 0, 27*3*msh.nv_tot*sizeof(float), ocl.hst.Juc.vv, 0, NULL, NULL);
    
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Jcc.ii, CL_TRUE, 0, 27*1*msh.nv_tot*sizeof(int),   ocl.hst.Jcc.ii, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Jcc.jj, CL_TRUE, 0, 27*1*msh.nv_tot*sizeof(int),   ocl.hst.Jcc.jj, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.dev.Jcc.vv, CL_TRUE, 0, 27*1*msh.nv_tot*sizeof(float), ocl.hst.Jcc.vv, 0, NULL, NULL);
    
    //reset
//    memset(ocl.hst.U1u, 0, 3*msh.nv_tot*sizeof(float));
//    memset(ocl.hst.U1c, 0, 1*msh.nv_tot*sizeof(float));
    
//    //solve
//    slv_u(&msh, &ocl);
//    slv_c(&msh, &ocl);
    
    //store prior
//    ocl.err = clEnqueueCopyBuffer( ocl.command_queue, ocl.dev.U1c, ocl.dev.U0c, 0, 0, 1*msh.nv_tot*sizeof(float), 0, NULL, NULL);
    
    //write to device
//    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.dev.U1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.hst.U1u, 0, NULL, NULL);
//    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.dev.U1c, CL_TRUE, 0, 1*msh.nv_tot*sizeof(float), ocl.hst.U1c, 0, NULL, NULL);
     
    
    //write vtk
    wrt_vtk(&msh, &ocl);
    
    //write for matlab
    wrt_raw(ocl.hst.Juu.ii, 27*9*msh.nv_tot, sizeof(int),   "Juu_ii");
    wrt_raw(ocl.hst.Juu.jj, 27*9*msh.nv_tot, sizeof(int),   "Juu_jj");
    wrt_raw(ocl.hst.Juu.vv, 27*9*msh.nv_tot, sizeof(float), "Juu_vv");
    
    //write for matlab
    wrt_raw(ocl.hst.Juc.ii, 27*3*msh.nv_tot, sizeof(int),   "Juc_ii");
    wrt_raw(ocl.hst.Juc.jj, 27*3*msh.nv_tot, sizeof(int),   "Juc_jj");
    wrt_raw(ocl.hst.Juc.vv, 27*3*msh.nv_tot, sizeof(float), "Juc_vv");
    
    wrt_raw(ocl.hst.Jcc.ii, 27*1*msh.nv_tot, sizeof(int),   "Jcc_ii");
    wrt_raw(ocl.hst.Jcc.jj, 27*1*msh.nv_tot, sizeof(int),   "Jcc_jj");
    wrt_raw(ocl.hst.Jcc.vv, 27*1*msh.nv_tot, sizeof(float), "Jcc_vv");
    
    wrt_raw(ocl.hst.U1u, 3*msh.nv_tot, sizeof(float), "U1u");
    wrt_raw(ocl.hst.F1u, 3*msh.nv_tot, sizeof(float), "F1u");
    
    wrt_raw(ocl.hst.U0c, 1*msh.nv_tot, sizeof(float), "U0c");
    wrt_raw(ocl.hst.U1c, 1*msh.nv_tot, sizeof(float), "U1c");
    wrt_raw(ocl.hst.F1c, 1*msh.nv_tot, sizeof(float), "F1c");

    //clean
    ocl_final(&msh, &ocl);
    
    printf("done\n");
    
    return 0;
}
