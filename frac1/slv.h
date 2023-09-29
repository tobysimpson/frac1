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
    printf("slv_u\n");
    
    //init mtx
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
    atts.transpose  = false;

    //size of input array
    long blk_num = 27*9*msh->nv_tot;
    int num_rows = 3*msh->nv_tot;
    int num_cols = 3*msh->nv_tot;

    //create
    SparseMatrix_Float A = SparseConvertFromCoordinate(num_rows, num_cols, blk_num, 1, atts, ocl->hst.Juu.ii, ocl->hst.Juu.jj, ocl->hst.Juu.vv);  //duplicates sum
    
    //vecs
    DenseVector_Float u;
    DenseVector_Float f;
    
    u.count = 3*msh->nv_tot;
    f.count = 3*msh->nv_tot;
    
    u.data = ocl->hst.U1u;
    f.data = ocl->hst.F1u;

    /*
     ========================
     solve
     ========================
     */
    
    //iterate
//    SparseSolve(SparseConjugateGradient(), A, f, u);    // SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
    SparseSolve(SparseGMRES(), A, f, u);
//    SparseSolve(SparseLSMR(), A, f, u);
    
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
    printf("slv_c\n");
    
    //init mtx
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;        // SparseOrdinary/SparseSymmetric
    atts.transpose  = false;

    //size of input array
    long blk_num = 27*msh->nv_tot;
    
    int num_rows = msh->nv_tot;
    int num_cols = msh->nv_tot;

    //create
    SparseMatrix_Float A = SparseConvertFromCoordinate(num_rows, num_cols, blk_num, 1, atts, ocl->hst.Jcc.ii, ocl->hst.Jcc.jj, ocl->hst.Jcc.vv);
    
    //debug
    printf("nnz=%lu\n", A.structure.columnStarts[A.structure.columnCount]);
    
    //vecs
    DenseVector_Float u;
    DenseVector_Float f;
    
    u.count = msh->nv_tot;
    f.count = msh->nv_tot;
    
    u.data = ocl->hst.U1c;
    f.data = ocl->hst.F1c;

    /*
     ========================
     solve
     ========================
     */
    
    //iterate
    SparseSolve(SparseConjugateGradient(), A, f, u);    //SparsePreconditionerDiagonal/SparsePreconditionerDiagScaling
//    SparseSolve(SparseGMRES(), A, f, u);
//    SparseSolve(SparseLSMR(), A, f, u);
    
    //QR
//    SparseOpaqueFactorization_Float QR = SparseFactor(SparseFactorizationQR, A);
//    SparseSolve(QR, f , u);
//    SparseCleanup(QR);

    //clean
    SparseCleanup(A);

    return 0;
}


void err_nrm(struct msh_obj *msh, struct ocl_obj *ocl)
{
    float e_max = ocl->hst.E1c[0];
    
    //sum
    for(int i=0; i<msh->ne_tot; i++)
    {
        float e = ocl->hst.E1c[i];
        
        e_max = (e>e_max)?e:e_max;
        
//        printf("%03d %e %e\n", i, e, e_max);
    }
    printf("\n");

    //disp
    printf("%03d %e %e\n", msh->ele_dim.x, msh->dx.x, e_max);
    
    return;
}


#endif /* slv_h */

