#include <stdio.h>
#include <stdlib.h>
// #include <time.h>
#include <math.h>
#include "string.h"
#include "R.h"

#define NODE_TERMINAL -1
#define SMALL_INT int



//
//  Prototypes
//


void zeroInt(int *x, int length);

void predictClassTree( double *x, int n, int mdim, int *treemap, int *nodestatus, double *xbestsplit, int *bestvar, int *nodeclass,
			                 int treeSize, int *cat, int nclass, int *jts, int *nodex, int maxcat);

void predictRegTree(double *x, int nsample, int mdim,
                    int *lDaughter, int *rDaughter, SMALL_INT *nodestatus,
                    double *ypred, double *split, double *nodepred,
                    int *splitVar, int treeSize, int *cat, int maxcat,
                    int *nodex);

void permuteOOBgroup(double *xdata, int n, int p, int m, int d, int indexTree, int *inbag);




//
//  Functions
//


void zeroInt(int *x, int length) {
    memset(x, 0, length * sizeof(int)); 
}

// m: index of the first variable of the group
// d: size of the group
void permuteOOBgroup(double *xdata, int n, int p, int m, int d, int indexTree, int *inbag){

  if(m+d>p){
    return;
  }

  double *tp = (double *) malloc(n*d*sizeof(double));
  double tmp;
  int i, j, last, k, nOOB = 0;

  for (i = 0; i < n; ++i) {
    /* make a copy of the OOB part of the data into tp (for permuting) */
    if (inbag[i + indexTree*n] == 0){
      for (j = m; j < m + d; ++j)
        tp[nOOB*d + j - m] = xdata[j + i*p];
      nOOB++;
    }
  }

  /* Permute tp */
  last = nOOB;
  for (i = 0; i < nOOB; ++i) {
    k = (int) last * unif_rand();
    for(j = 0; j < d; j++){
      tmp = tp[last*d - j - 1];
      tp[last*d - j - 1] = tp[k*d - j + d - 1];
      tp[k*d - j + d - 1] = tmp;
    }
    last--;
  }

  /* Copy the permuted OOB data back into xdata. */
  nOOB = 0;
  for (i = 0; i < n; ++i) {
    if (inbag[i + indexTree*n] == 0) {
      for(j = m; j < m + d; ++j)
        xdata[j + i*p] = tp[nOOB*d + j - m];
      nOOB++;
    }
  }

  free(tp);
}



void permuteOOBgroupByIndexes(double *xdata, int n, int p, int d, int *ixdGroupOOB, int nOOB){

  double *OOBpart = (double*)malloc(n*d * sizeof(double));
  int idxOOB=0, last = nOOB, i, j, k;
  double tmp;

  /* make a copy of the OOB part of the data into OOBpart (for permuting) */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < p; ++j){
      if (ixdGroupOOB[i*p+j] == 1){
        OOBpart[idxOOB] = xdata[i*p+j];
        idxOOB++;
      }
    }
  }

  /* Permute OOBpart */
  for (i = 0; i < nOOB; ++i) {
    k = (int) last * unif_rand();
    for(j = 0; j < d; j++){
      tmp = OOBpart[last*d - j - 1];
      OOBpart[last*d - j - 1] = OOBpart[k*d - j + d - 1];
      OOBpart[k*d - j + d - 1] = tmp;
    }
    last--;
  }

  /* Copy the permuted OOB data back into xdata. */
  idxOOB=0;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < p; ++j){
      if (ixdGroupOOB[i*p + j] == 1){
        xdata[i*p + j] = OOBpart[idxOOB];
        idxOOB++;
      }
    }
  }
  free(OOBpart);
}










// 
// Classification
// 

void predictClassTree(double *x, int n, int mdim, int *treemap,
                      int *nodestatus, double *xbestsplit,
                      int *bestvar, int *nodeclass,
                      int treeSize, int *cat, int nclass,
                      int *jts, int *nodex, int maxcat)
{
  
  int m, i, j, k, *cbestsplit, niter;
  unsigned int npack;

   // decode the categorical splits 
  if (maxcat > 1) {
    cbestsplit = (int *) calloc(maxcat * treeSize, sizeof(int));
    zeroInt(cbestsplit, maxcat * treeSize);
    for (i = 0; i < treeSize; ++i) {
      if (nodestatus[i] != NODE_TERMINAL) {
        if (cat[bestvar[i] - 1] > 1) {
          npack = (unsigned int) xbestsplit[i];
           // unpack `npack' into bits 
          for (j = 0; npack; npack >>= 1, ++j)
            cbestsplit[j + i*maxcat] = npack & 01;
        }
      }
    }
  }
  for (i = 0; i < n; ++i) {
    k = 0;
    niter=0;
    while (nodestatus[k] != NODE_TERMINAL) {
      
      m = bestvar[k] - 1;
      if (cat[m] == 1) {
         // Split by a numerical predictor 
        k = (x[m + i * mdim] <= xbestsplit[k]) ? treemap[k * 2] - 1 : treemap[1 + k * 2] - 1;
      } else {
         // Split by a categorical predictor
        k = cbestsplit[(int) x[m + i * mdim] - 1 + k * maxcat] ? treemap[k * 2] - 1 : treemap[1 + k * 2] - 1;
      }
      if(k==-1){
        return;
      }
      niter++;
    }
     // Terminal node: assign class label 
    jts[i] = nodeclass[k];
    nodex[i] = k + 1;
  }
  if (maxcat > 1) free(cbestsplit);
}


void R_varImpGroup(  double *permutation_measure_group, double *xdata, int *ydata, int *n, int *p, int *ntree, int *treemap, 
                      int *nodestatus, double *xbestsplit, int *bestvar, int* nodepred, int *ndbigtree, int *cat, int *maxcat, 
                      int *nclass, int *ngroups, int *nvarGroup, int *maxnvarGroup, int *idxGroup, int *inbag, int *nrnodes)
{

  if(*ngroups==1){
    return;
  }

  int i, j, k, indexTree, itx, m;
  int idxByNnode = 0;
  int correctPred, correctPredPerm, nOOB;
  
  int *nodex = (int *)malloc(*n * sizeof(int));
  int *pred = (int *)malloc(*n * sizeof(int));
  int *predPerm = (int *)malloc(*n * sizeof(int));
  int *idxCurrentGroup = (int *)malloc(*maxnvarGroup * sizeof(int));
  int *currentGroup = (int *)malloc(*p * sizeof(int));
  int *idxGroupOOB = (int *)malloc(*n * *p * sizeof(int));

  double *tx = (double *)malloc(*n * *maxnvarGroup * sizeof(double));
  int sizeTx = *n * *maxnvarGroup;
  double impTree, errorBefore, errorAfter;



  for (j = 0; j < *ngroups; ++j)
    permutation_measure_group[j] = 0;

  GetRNGstate();
  for (indexTree = 0; indexTree < *ntree; ++indexTree){

    predictClassTree( xdata, *n, *p, treemap + 2*idxByNnode, nodestatus + idxByNnode, xbestsplit + idxByNnode, 
                      bestvar + idxByNnode, nodepred + idxByNnode, ndbigtree[indexTree], cat, *nclass, pred, nodex, *maxcat);
    
    // count the number of correct prediction for oob samples
    correctPred = 0;
    nOOB = 0;
    for (i = 0; i < *n; ++i){
      if(inbag[i + indexTree * *n]==0) {
        if((pred[i]==ydata[i]))
          correctPred++;
        nOOB++;
      }
    }

    if(nOOB==0){
      // printf("Error: No OOB samples. Rerun with more trees.\n");
      return;
    }
    errorBefore = 1 - (double)correctPred/nOOB;

    // for each groups
    m = 0;
    for (j = 0; j < *ngroups; ++j){

      //Get the indexes of the current group
      for (k = 0; k < *p; ++k){
        currentGroup[k] = 0;
      }
      for (k = 0; k < nvarGroup[j]; ++k){
        idxCurrentGroup[k] = idxGroup[m + k];
        currentGroup[idxCurrentGroup[k]] = 1;
      }


      //Copy the current group in tx and get idxGroupOOB
      itx=0;
      for(i = 0; i < *n; ++i){
        for (k = 0; k < *p; ++k){
          if(currentGroup[k]==1){
            tx[itx] = xdata[i * *p + k];
            itx++;
          }
          if (inbag[indexTree * *n + i]==0 && currentGroup[k]==1)
            idxGroupOOB[i * *p+k]=1;
          else
            idxGroupOOB[i * *p+k]=0;
        }
      }

      // Permute x for OOB samples
      permuteOOBgroupByIndexes(xdata, *n, *p, nvarGroup[j], idxGroupOOB, nOOB);

      //Predict the modified data
      predictClassTree( xdata, *n, *p, treemap + 2*idxByNnode, nodestatus + idxByNnode, xbestsplit + idxByNnode, 
                        bestvar + idxByNnode, nodepred + idxByNnode, ndbigtree[indexTree], cat, *nclass, predPerm, nodex, *maxcat);

      // Count the correct predictions for oob samples
      correctPredPerm = 0;
      for (i = 0; i < *n; ++i){
        if(inbag[i + indexTree * *n]==0) {
          if(predPerm[i]==ydata[i])
            correctPredPerm++;
        }
      }

      errorAfter = 1 - (double)correctPredPerm/nOOB;
      impTree = errorAfter - errorBefore;
      permutation_measure_group[j] = permutation_measure_group[j] + impTree;

      itx=0;
      for(i = 0; i < *n; ++i){
        for (k = 0; k < *p; ++k){
          if(currentGroup[k]==1){
            xdata[i * *p + k] = tx[itx];
            itx++;
          }
        }
      }

      m += nvarGroup[j];
    }
    idxByNnode += *nrnodes;
  }
  PutRNGstate();

  for (j = 0; j < *ngroups; ++j)
    permutation_measure_group[j] = permutation_measure_group[j] / *ntree;



  free(nodex);
  free(pred);
  free(predPerm);
  free(tx);
  free(idxGroupOOB);
  free(currentGroup);
  free(idxCurrentGroup);
}




// 
// Regression
// 

void predictRegTree(double *x, int nsample, int mdim,
                    int *lDaughter, int *rDaughter, SMALL_INT *nodestatus,
                    double *ypred, double *split, double *nodepred,
                    int *splitVar, int treeSize, int *cat, int maxcat,
                    int *nodex)
{

    int i, j, k, m, *cbestsplit;
    unsigned int npack;
    
    /* decode the categorical splits */
    if (maxcat > 1) {
        cbestsplit = (int *) calloc(maxcat * treeSize, sizeof(int));
        zeroInt(cbestsplit, maxcat * treeSize);
        for (i = 0; i < treeSize; ++i) {
            if (nodestatus[i] != NODE_TERMINAL && cat[splitVar[i] - 1] > 1) {
                npack = (unsigned int) split[i];
                /* unpack `npack' into bits */
                for (j = 0; npack; npack >>= 1, ++j) {
                    cbestsplit[j + i*maxcat] = npack & 1;
                }
            }
        }
    }
    
    for (i = 0; i < nsample; ++i) {
        k = 0;
        while (nodestatus[k] != NODE_TERMINAL) { /* go down the tree */
            m = splitVar[k] - 1;
            if (cat[m] == 1) {
                k = (x[m + i*mdim] <= split[k]) ?
                    lDaughter[k] - 1 : rDaughter[k] - 1;
            } else {
                /* Split by a categorical predictor */
                k = cbestsplit[(int) x[m + i * mdim] - 1 + k * maxcat] ?
                    lDaughter[k] - 1 : rDaughter[k] - 1;
            }
        }
        /* terminal node: assign prediction and move on to next */
        ypred[i] = nodepred[k];
        nodex[i] = k + 1;
    }
    if (maxcat > 1) free(cbestsplit);
}



void R_varImpGroup_Reg( double *permutation_measure_group, double *xdata, double *ydata, int *n, int *p, int *ntree, int *lDaughter, int *rDaughter, 
                        SMALL_INT *nodestatus, double *xbestsplit, int *bestvar, double* nodepred, int *ndbigtree, int *cat, int *maxcat, int *ngroups, 
                        int *nvarGroup, int *maxnvarGroup, int *idxGroup, int *inbag, int *nrnodes)
{


  if(*ngroups==1){
    return;
  }

  int i, j, k, indexTree, nOOB, itx, m;
  int idxByNnode = 0;
  double mseBefore, mseAfter, impTree;
  
  int *nodex = (int *) malloc(*n * sizeof(int));
  double *pred = (double *) malloc(*n * sizeof(double));
  double *predPerm = (double *) malloc(*n * sizeof(double));
  double *tx =  (double *) malloc(*n * *maxnvarGroup * sizeof(double));
  int sizeTx = *n * *maxnvarGroup;

  int *idxCurrentGroup = (int *)malloc(*maxnvarGroup * sizeof(int));
  int *currentGroup = (int *)malloc(*p * sizeof(int));
  int *idxGroupOOB = (int *)malloc(*n * *p * sizeof(int));


  for (j = 0; j < *ngroups; ++j)
    permutation_measure_group[j] = 0;

  GetRNGstate();
  for (indexTree = 0; indexTree < *ntree; ++indexTree){


    predictRegTree( xdata, *n, *p, lDaughter + idxByNnode, rDaughter + idxByNnode, nodestatus + idxByNnode, pred, 
                    xbestsplit + idxByNnode, nodepred + idxByNnode, bestvar + idxByNnode, ndbigtree[indexTree], cat, *maxcat, nodex);

    
    // Compute the MSE between pred and ydata for oob samples
    mseBefore = 0;
    nOOB = 0;
    for (i = 0; i < *n; ++i){
      if(inbag[i + indexTree * *n]==0) {
        mseBefore += (ydata[i] - pred[i])*(ydata[i] - pred[i]);
        nOOB++;
      }
    }
    mseBefore /= nOOB;

    if(nOOB==0){
      return;
    }

    // for each groups
    m = 0;
    for (j = 0; j < *ngroups; ++j){


      //Get the indexes of the current group
      for (k = 0; k < *p; ++k){
        currentGroup[k] = 0;
      }
      for (k = 0; k < nvarGroup[j]; ++k){
        idxCurrentGroup[k] = idxGroup[m + k];
        currentGroup[idxCurrentGroup[k]] = 1;
      }

     
      //Copy the current group in tx and get idxGroupOOB
      itx=0;
      for(i = 0; i < *n; ++i){
        for (k = 0; k < *p; ++k){
          if(currentGroup[k]==1){
            tx[itx] = xdata[i * *p + k];
            itx++;
          }
          if (inbag[indexTree * *n + i]==0 && currentGroup[k]==1)
            idxGroupOOB[i * *p+k]=1;
          else
            idxGroupOOB[i * *p+k]=0;
        }
      }





      // Permute x for OOB samples
      permuteOOBgroupByIndexes(xdata, *n, *p, nvarGroup[j], idxGroupOOB, nOOB);


      //Predict the modified data
      predictRegTree( xdata, *n, *p, lDaughter + idxByNnode, rDaughter + idxByNnode, nodestatus + idxByNnode, predPerm, 
                      xbestsplit + idxByNnode, nodepred + idxByNnode, bestvar + idxByNnode, ndbigtree[indexTree], cat, *maxcat, nodex);

      

      // Compute the MSE between predPerm and ydata for oob samples
      mseAfter = 0;
      for (i = 0; i < *n; ++i){
        if(inbag[i + indexTree * *n]==0)
          mseAfter += (ydata[i] - predPerm[i])*(ydata[i] - predPerm[i]);
      }
      mseAfter /= nOOB;


      impTree = mseAfter - mseBefore;
      permutation_measure_group[j] = permutation_measure_group[j] + impTree;



      // restore the original data x <- tx
      itx=0;
      for(i = 0; i < *n; ++i){
        for (k = 0; k < *p; ++k){
          if(currentGroup[k]==1){
            xdata[i * *p + k] = tx[itx];
            itx++;
          }
        }
      }
    m += nvarGroup[j];
    }
    idxByNnode += *nrnodes;
  }
  PutRNGstate();

  for (j = 0; j < *ngroups; ++j)
    permutation_measure_group[j] = permutation_measure_group[j] / *ntree;

  free(nodex);
  free(pred);
  free(predPerm);
  free(tx);
  free(idxGroupOOB);
  free(currentGroup);
  free(idxCurrentGroup);
}
