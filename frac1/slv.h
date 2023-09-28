//
//  slv.h
//  frac1
//
//  Created by Toby Simpson on 18.09.23.
//

#ifndef slv_h
#define slv_h


void dsp_vec(DenseVector_Float v)
{
    for(int i=0; i<v.count; i++)
    {
        printf("%+e ", v.data[i]);
    }
    printf("\n\n");
    
    return;
}


//solve
int slv_u(struct msh_obj *msh, struct ocl_obj *ocl)
{
    printf("slv u\n");
    
    //init mtx
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
    atts.transpose  = false;

    //size of input array
    long blk_num = 27*9*msh->nv_tot;
    int num_rows = 3*msh->nv_tot;
    int num_cols = 3*msh->nv_tot;

    //map read
    int*    ii = clEnqueueMapBuffer(ocl->command_queue, ocl->Juu_ii, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(int),   0, NULL, NULL, &ocl->err);
    int*    jj = clEnqueueMapBuffer(ocl->command_queue, ocl->Juu_jj, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(int),   0, NULL, NULL, &ocl->err);
    float*  vv = clEnqueueMapBuffer(ocl->command_queue, ocl->Juu_vv, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(float), 0, NULL, NULL, &ocl->err);

    //create
    SparseMatrix_Float A = SparseConvertFromCoordinate(num_rows, num_cols, blk_num, 1, atts, ii, jj, vv);  //duplicates sum
    
    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Juu_ii, ii, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Juu_jj, jj, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Juu_vv, vv, 0, NULL, NULL);
    
    //debug
    printf("nnz=%lu\n", A.structure.columnStarts[A.structure.columnCount]);
    
    //vecs
    DenseVector_Float u;
    DenseVector_Float f;
    
    u.count = 3*msh->nv_tot;
    f.count = 3*msh->nv_tot;
    
    u.data = ocl->uu;
    f.data = ocl->fu;

    /*
     ========================
     solve
     ========================
     */
    
    //iterate
//    SparseSolve(SparseConjugateGradient(), A, f, u);    //yes SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
//    SparseSolve(SparseGMRES(), A, f, u);              //yes
    SparseSolve(SparseLSMR(), A, f, u);               //yes
    
    //QR
//    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);       //no
//    SparseSolve(QR, f , u);
//    SparseCleanup(QR);
    
    //clean
    SparseCleanup(A);

    return 0;
}


//solve
int slv_c(struct msh_obj *msh, struct ocl_obj *ocl)
{
    printf("slv c\n");
    
    //init mtx
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
    atts.transpose  = false;

    //size of input array
    long blk_num = 27*msh->nv_tot;
    
    int num_rows = msh->nv_tot;
    int num_cols = msh->nv_tot;

    //map read
    int*    ii = clEnqueueMapBuffer(ocl->command_queue, ocl->Jcc_ii, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(int),   0, NULL, NULL, &ocl->err);
    int*    jj = clEnqueueMapBuffer(ocl->command_queue, ocl->Jcc_jj, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(int),   0, NULL, NULL, &ocl->err);
    float*  vv = clEnqueueMapBuffer(ocl->command_queue, ocl->Jcc_vv, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(float), 0, NULL, NULL, &ocl->err);

    //create
    SparseMatrix_Float A = SparseConvertFromCoordinate(num_rows, num_cols, blk_num, 1, atts, ii, jj, vv);  //duplicates sum
    
    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Jcc_ii, ii, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Jcc_jj, jj, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Jcc_vv, vv, 0, NULL, NULL);
    
    //debug
    printf("nnz=%lu\n", A.structure.columnStarts[A.structure.columnCount]);
    
    //vecs
    DenseVector_Float u;
    DenseVector_Float f;
    
    u.count = msh->nv_tot;
    f.count = msh->nv_tot;
    
    u.data = ocl->uc;
    f.data = ocl->fc;

    /*
     ========================
     solve
     ========================
     */
    
    //iterate
//    SparseSolve(SparseConjugateGradient(), A, f, u);    //SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
//    SparseSolve(SparseGMRES(), A, f, u);
//    SparseSolve(SparseLSMR(), A, f, u);
    
    //QR
    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);
    SparseSolve(QR, f , u);
    SparseCleanup(QR);
    
    //clean
    SparseCleanup(A);

    return 0;
}


void err_nrm(struct msh_obj *msh, struct ocl_obj *ocl)
{
    //map read
    float *ptr = clEnqueueMapBuffer(ocl->command_queue, ocl->ele_ee, CL_TRUE, CL_MAP_READ, 0, msh->ne_tot*sizeof(float), 0, NULL, NULL, &ocl->err);
    
    float e_sum = 0e0f;
    
    //sum
    for(int i=0; i<msh->ne_tot; i++)
    {
        e_sum += ptr[i];
        
//        printf("%03d %e %e\n", i, ptr[i], e_sum);
    }
    printf("\n");

    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->ele_ee, ptr, 0, NULL, NULL);
    
    //disp
    printf("%03d %e %e\n", msh->ele_dim.x, msh->dx.x, sqrtf(e_sum));
    
    return;
}


#endif /* slv_h */

