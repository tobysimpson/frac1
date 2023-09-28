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
    size_t ne[3] = {msh.ele_dim.x, msh.ele_dim.y, msh.ele_dim.z};
    size_t nv[3] = {msh.vtx_dim.x, msh.vtx_dim.y, msh.vtx_dim.z};
    size_t f1[2] = {msh.vtx_dim.y, msh.vtx_dim.z};
    
    //kernels
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_init, 3, NULL, nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_assm, 3, NULL, nv, NULL, 0, NULL, NULL);
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_bnd1, 3, NULL, nv, NULL, 0, NULL, NULL); //c
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.fac_bnd1, 2, NULL, f1, NULL, 0, NULL, NULL); //u
    
    //read from device
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.U1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.uu, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.F1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.fu, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.U0c, CL_TRUE, 0,   msh.nv_tot*sizeof(float), ocl.ac, 0, NULL, NULL); //ana for vtk
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.U1c, CL_TRUE, 0,   msh.nv_tot*sizeof(float), ocl.uc, 0, NULL, NULL);
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.F1c, CL_TRUE, 0,   msh.nv_tot*sizeof(float), ocl.fc, 0, NULL, NULL);
    
    //reset
    memset(ocl.uu, 0e0f, 3*msh.nv_tot*sizeof(float));
    memset(ocl.uc, 0e0f,   msh.nv_tot*sizeof(float));
    
    //solve
//    slv_u(&msh, &ocl);
    slv_c(&msh, &ocl);
    
    //write to device
    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.U1c, CL_TRUE, 0,   msh.nv_tot*sizeof(float), ocl.uc, 0, NULL, NULL);
    ocl.err = clEnqueueWriteBuffer(ocl.command_queue, ocl.U1u, CL_TRUE, 0, 3*msh.nv_tot*sizeof(float), ocl.uu, 0, NULL, NULL);
    
    //store prior
    ocl.err = clEnqueueCopyBuffer( ocl.command_queue, ocl.U1c, ocl.U0c, 0, 0, msh.nv_tot*sizeof(float), 0, NULL, NULL);
    
    //calc error
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.ele_err1, 3, NULL, ne, NULL, 0, NULL, NULL);
    
    //read from device
    ocl.err = clEnqueueReadBuffer(ocl.command_queue, ocl.ele_ec, CL_TRUE, 0, msh.ne_tot*sizeof(float), ocl.ec, 0, NULL, NULL);
    
    err_nrm(&msh, &ocl);
    
    //write vtk
    wrt_vtk(&msh, &ocl);
    
    //write for matlab
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
    
    wrt_raw(&ocl, ocl.U1u, 3*msh.nv_tot, sizeof(float), "U1u");
    wrt_raw(&ocl, ocl.F1u, 3*msh.nv_tot, sizeof(float), "F1u");
    
    wrt_raw(&ocl, ocl.U0c, msh.nv_tot, sizeof(float), "U0c");
    wrt_raw(&ocl, ocl.U1c, msh.nv_tot, sizeof(float), "U1c");
    wrt_raw(&ocl, ocl.F1c, msh.nv_tot, sizeof(float), "F1c");
    
    //clean
    ocl_final(&ocl);
    
    printf("done\n");
    
    return 0;
}
