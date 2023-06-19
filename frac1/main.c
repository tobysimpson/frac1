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


//constants
#define ROOT_SRC    "/Users/toby/Documents/USI/postdoc/fracture/xcode/frac1/frac1/"
#define ROOT_WRITE  "/Users/toby/Downloads/"


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
    
    msh_init(&msh);
    
    /*
     ===============
     ass
     ===============
     */
    
    char prg_pth[1000];
    sprintf(prg_pth,"%s/%s", ROOT_SRC, "program.cl");
    
    
    //source
    FILE* prg_file = fopen(prg_pth, "r");
    if(prg_file)
    {
        printf("yes\n");
        fclose(prg_file);
    }
    else
    {
        printf("no\n");
    }
    
    /*
     ===============
     solve
     ===============
     */
    
    
    
    /*
     ===============
     write
     ===============
     */
    
    
    
    /*
     ===============
     clean
     ===============
     */
    
    printf("done\n");
    
    return 0;
}
