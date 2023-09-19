//
//  slv.h
//  frac1
//
//  Created by Toby Simpson on 18.09.23.
//

#ifndef slv_h
#define slv_h



void vec_print(DenseVector_Float v)
{
    for(int i=0; i<v.count; i++)
    {
        printf(" %+e\n", v.data[i]);
    }
    
    return;
}


//sparse from (i,j,v) multiply and solve (QR)
int slv_test1(int a)
{
    printf("slv %d\n", a);
    
    /*
     ========================
     init mtx
     ========================
     */
    
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;
    
    
    int     A_ii[]  = { 0,   1,   2,    3,    0,   0,   0,   1};
    int     A_jj[]  = { 0,   1,   2,    3,    1,   2,   3,   0};
    float   A_vv[]  = { 1,   2,   1,    1,    1,   0,   0,   0};
    
    //size of input arrays
    long            blk_num = 8;
    unsigned char   blk_sz  = 1;
    
    int A_row_num = 4;
    int A_col_num = 4;
    
    SparseMatrix_Float A = SparseConvertFromCoordinate(A_row_num, A_col_num, blk_num, blk_sz, atts, A_ii, A_jj, A_vv);  //duplicates sum
    
    printf("nnz=%lu\n", A.structure.columnStarts[A.structure.columnCount]);      //this is key nnz = the length of the data and row_idx arrays
    

    
    /*
     ========================
     disp mtx
     ========================
     */
    
    printf("csc\n");
    
    int col_idx = 0;

    for(int i=0; i<A.structure.columnStarts[A.structure.columnCount]; i++)
    {
        if(i == A.structure.columnStarts[col_idx+1])
        {
            col_idx += 1;
        }
        
        printf("(%d,%d) %e\n",A.structure.rowIndices[i],col_idx,A.data[i]);
    }

//    printf("colstarts\n");
//
//    for(int j=0; j<A.structure.columnCount+1; j++)
//    {
//        printf("%ld\n",A.structure.columnStarts[j]);
//    }
    
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
    
    vec_print(u);

    
    printf("b\n");
    
    float b_vv[4] = {0,0,0,0};

    DenseVector_Float b;
    b.count = 4;
    b.data = b_vv;

    vec_print(b);
    
    /*
     ========================
     multiply
     ========================
     */

    printf("Au\n");
    SparseMultiply(A,u,b);
    vec_print(b);
    
    //reset - wont solve without reset
    memset(u.data, 0e0f, 4*sizeof(float));
    printf("u\n");
    vec_print(u);
    
    //solve
//    SparseSolve(SparseConjugateGradient(), A, b, u); //yes
//    SparseSolve(SparseLSMR(), A, b, u);     //yes
//    SparseSolve(SparseGMRES(), A, b, u);        //yes
    
    
    //solve QR
//    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);       //yes
//    SparseSolve(QR, b , u);
    
    printf("u\n");
    vec_print(u);
    


    /*
     ========================
     solve LSMR
     ========================
     */

//    SparseSolve(SparseLSMR(), A, b, u);
//
//    SparseIterativeStatus_t status = SparseSolve(SparseLSMR(), A, b, u, SparsePreconditionerDiagScaling);
//
//    if(status!=SparseIterativeConverged)
//    {
//        printf("Failed to converge. Returned with error %d\n", status);}
//    else
//    {
//        printf("u\n");
//        vec_print(u);
//    }
    
    /*
     ========================
     solve GMRES
     ========================
     */
    
//    SparseGMRESOptions options;
//
//    options.atol = 0.01;
//    options.maxIterations = 1000;
//    options.nvec = 100;
//    options.rtol = 0.01;
//    options.variant = SparseVariantGMRES; //SparseVariantGMRES SparseVariantDQGMRES SparseVariantFGMRES (needs precon)
//    options.reportError = NULL;
//    options.reportStatus = NULL;
//
////    SparseSolve(SparseGMRES(options), A, b, u, SparsePreconditionerDiagScaling); //SparsePreconditionerDiagonal SparsePreconditionerDiagScaling
//

    
    
//    SparseIterativeStatus_t status = SparseSolve(SparseConjugateGradient(), A, b, u);
//
//    if(status!=SparseIterativeConverged)
//    {
//        printf("Failed to converge. Returned with error %d\n", status);}
//    else
//    {
//        printf("u\n");
//        vec_print(u);
//    }
    

    
    /*
     ========================
     clean up
     ========================
    */
    
    SparseCleanup(A);
//    SparseCleanup(A_QR);
    
    return 0;
}




#endif /* slv_h */
