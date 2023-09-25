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
    printf("\n");
    
    return;
}


//solve
int slv_test1(struct msh_obj *msh, struct ocl_obj *ocl)
{
    printf("slv %d\n", msh->nv[0]);
    
    /*
     ========================
     init mtx
     ========================
     */
    
    SparseAttributes_t atts;
    atts.kind       = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
    atts.transpose  = false;
//    atts.triangle = SparseUpperTriangle;
    
//    int     A_ii[]  = { 0,   1,   2,    3,    0,   1,   0,   2};
//    int     A_jj[]  = { 0,   1,   2,    3,    1,   0,   2,   0};
//    float   A_vv[]  = { 1,   1,   1,    1,    1,   1,   1,   1};
    
    //size of input arrays
    long            blk_num = 27*9*msh->nv_tot;
    unsigned char   blk_sz  = 1;
    
    int num_rows = 3*msh->nv_tot;
    int num_cols = 3*msh->nv_tot;

    //map read
    int*    ii = clEnqueueMapBuffer(ocl->command_queue, ocl->Juu_ii, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(int),   0, NULL, NULL, &ocl->err);
    int*    jj = clEnqueueMapBuffer(ocl->command_queue, ocl->Juu_jj, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(int),   0, NULL, NULL, &ocl->err);
    float*  vv = clEnqueueMapBuffer(ocl->command_queue, ocl->Juu_vv, CL_TRUE, CL_MAP_READ, 0, blk_num*sizeof(float), 0, NULL, NULL, &ocl->err);

    //create
    SparseMatrix_Float A = SparseConvertFromCoordinate(num_rows, num_cols, blk_num, blk_sz, atts, ii, jj, vv);  //duplicates sum
    
    //unmap read
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Juu_ii, ii, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Juu_jj, jj, 0, NULL, NULL);
    clEnqueueUnmapMemObject(ocl->command_queue, ocl->Juu_vv, vv, 0, NULL, NULL);
    
    //debug
    printf("nnz=%lu\n", A.structure.columnStarts[A.structure.columnCount]);
    
    /*
     ========================
     disp mtx
     ========================
     */
    
    printf("disp\n");

    int col_idx = 0;

    for(int i=0; i<A.structure.columnStarts[A.structure.columnCount]; i++)
    {
        if(i == A.structure.columnStarts[col_idx+1])
        {
            col_idx += 1;
        }
        printf("(%d,%d) %+e\n",A.structure.rowIndices[i],col_idx,A.data[i]);
    }
    
    /*
     ========================
     vecs
     ========================
     */
    
    printf("u\n");
    
    float u_vv[4] = {1,2,3,4};

    DenseVector_Float u;
    u.count = 4;
    u.data = u_vv;
    
    dsp_vec(u);

    
    printf("b\n");
    
    float b_vv[4] = {0,0,0,0};

    DenseVector_Float b;
    b.count = 4;
    b.data = b_vv;

    dsp_vec(b);
    
    /*
     ========================
     multiply
     ========================
     */

    printf("Au\n");
    SparseMultiply(A,u,b);
    dsp_vec(b);
    
    //reset - wont solve without reset
    memset(u.data, 0e0f, 4*sizeof(float));
    printf("u\n");
    dsp_vec(u);
    
    /*
     ========================
     solve
     ========================
     */
    
    //iterate
    SparseSolve(SparseConjugateGradient(), A, b, u);    //yes SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
//    SparseSolve(SparseGMRES(), A, b, u);              //yes
//    SparseSolve(SparseLSMR(), A, b, u);               //yes
    
    //QR
//    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);       //yes
//    SparseSolve(QR, b , u);
//    SparseCleanup(A_QR);
    
    printf("u\n");
    dsp_vec(u);
    
    /*
     ========================
     clean up
     ========================
    */
    
    SparseCleanup(A);

    
    return 0;
}



////sparse from (i,j,v) multiply and solve (QR)
//int slv_test1(int a)
//{
//    printf("slv %d\n", a);
//
//    /*
//     ========================
//     init mtx
//     ========================
//     */
//
//    SparseAttributes_t atts;
//    atts.kind     = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
////    atts.triangle = SparseUpperTriangle;
//
//    int     A_ii[]  = { 0,   1,   2,    3,    0,   1,   0,   2};
//    int     A_jj[]  = { 0,   1,   2,    3,    1,   0,   2,   0};
//    float   A_vv[]  = { 1,   1,   1,    1,    1,   1,   1,   1};
//
//    //size of input arrays
//    long            blk_num = 8;
//    unsigned char   blk_sz  = 1;
//
//    int A_row_num = 4;
//    int A_col_num = 4;
//
//    SparseMatrix_Float A = SparseConvertFromCoordinate(A_row_num, A_col_num, blk_num, blk_sz, atts, A_ii, A_jj, A_vv);  //duplicates sum
//
//    printf("nnz=%lu\n", A.structure.columnStarts[A.structure.columnCount]);      //this is key nnz = the length of the data and row_idx arrays
//
//    /*
//     ========================
//     disp mtx
//     ========================
//     */
//
//    printf("csc\n");
//
//    int col_idx = 0;
//
//    for(int i=0; i<A.structure.columnStarts[A.structure.columnCount]; i++)
//    {
//        if(i == A.structure.columnStarts[col_idx+1])
//        {
//            col_idx += 1;
//        }
//        printf("(%d,%d) %+e\n",A.structure.rowIndices[i],col_idx,A.data[i]);
//    }
//
//    /*
//     ========================
//     vecs
//     ========================
//     */
//
//    printf("u\n");
//
//    float u_vv[4] = {1,2,3,4};
//
//    DenseVector_Float u;
//    u.count = 4;
//    u.data = u_vv;
//
//    dsp_vec(u);
//
//
//    printf("b\n");
//
//    float b_vv[4] = {0,0,0,0};
//
//    DenseVector_Float b;
//    b.count = 4;
//    b.data = b_vv;
//
//    dsp_vec(b);
//
//    /*
//     ========================
//     multiply
//     ========================
//     */
//
//    printf("Au\n");
//    SparseMultiply(A,u,b);
//    dsp_vec(b);
//
//    //reset - wont solve without reset
//    memset(u.data, 0e0f, 4*sizeof(float));
//    printf("u\n");
//    dsp_vec(u);
//
//    /*
//     ========================
//     solve
//     ========================
//     */
//
//    //iterate
//    SparseSolve(SparseConjugateGradient(), A, b, u);    //yes SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
////    SparseSolve(SparseGMRES(), A, b, u);              //yes
////    SparseSolve(SparseLSMR(), A, b, u);               //yes
//
//    //QR
////    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);       //yes
////    SparseSolve(QR, b , u);
////    SparseCleanup(A_QR);
//
//    printf("u\n");
//    dsp_vec(u);
//
//    /*
//     ========================
//     clean up
//     ========================
//    */
//
//    SparseCleanup(A);
//
//
//    return 0;
//}


#endif /* slv_h */
