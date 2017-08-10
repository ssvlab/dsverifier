#if (_DIMENSION==2)
  #include "2Inverse.h"
#else
  #include "3Inverse.h"
#endif

void multiply(matrixt result,matrixt transform)
{
  cnttype i,j,k;
  control_floatt temp[_DIMENSION][_DIMENSION];
  for (i=0;i<_DIMENSION;i++)
  {
    for (j=0;j<_DIMENSION;j++)
    {
      temp[i][j]=result[i][0]*transform[0][j];
      for (k=1;k<_DIMENSION;k++)
      {
        temp[i][j]+=result[i][k]*transform[k][j];
      }
    }
    for (j=0;j<_DIMENSION;j++) result[i][j]=temp[i][j];    
  }
}

void transformPoint(vectort point,matrixt transform)
{
  cnttype i,j;
  control_floatt temp[_DIMENSION];
  for (i=0;i<_DIMENSION;i++)
  {
    temp[i]=transform[i][0]*point[0];
    for (j=1;j<_DIMENSION;j++) 
    {
      temp[i]+=transform[i][j]*point[j];
    }
  }
  for (j=0;j<_DIMENSION;j++) point[j]=temp[j];    
}

control_floatt findSupport(vectort point,vectort vector)
{
  cnttype i,j,k;
  control_floatt result=0;
  for (i=0;i<_DIMENSION;i++) result+=vector[i]*point[i];
  return result;
}

void powerMatrix(matrixt result,matrixt transform,int pow)
{
  cnttype i,j;
  matrixt temp;
  for (i=0;i<_DIMENSION;i++)
  {
    for (j=0;j<_DIMENSION;j++) temp[i][j]=transform[i][j];
  }
  if (pow&1)
  {
    for (i=0;i<_DIMENSION;i++)
    {
      for (j=0;j<_DIMENSION;j++) result[i][j]=transform[i][j];
    }
  }
  else
  {
    for (i=0;i<_DIMENSION;i++)
    {
      for (j=0;j<_DIMENSION;j++) result[i][j]=0;
      result[i][i]=1;  
    }      
  }
  while (pow >>= 1) {
    multiply(temp,temp);
    if (pow&1) multiply(result,transform);
  }
}

void accelerateInputs()
{
  cnttype i,j;
  matrixt mat_inv,matrix;
  for (i=0;i<_DIMENSION;i++)
  {
    for (j=0;j<_DIMENSION;j++) matrix[i][j]=-loop_cbmc[i][j];
    matrix[i][i]+=_one;
  }
#ifndef __CPROVER
  print_matrix("I-A",matrix);
#endif  
  inverse(mat_inv,matrix);
#ifndef __CPROVER
  print_matrix("inverse",mat_inv);
#endif  
  for (i=0;i<_NUM_INPUT_VERTICES;i++)
  {
    for (j=0;j<_DIMENSION;j++) accel_vertices[i][j]=input_vertices[i][j];
    for (j=0;j<_NUM_VECTORS;j++) accelsupports[i][j]=findSupport(accel_vertices[i],vectors[j]);
    transformPoint(accel_vertices[i],mat_inv);
  }
  
}

#ifdef _NUM_ITERATIONS
void checkIteration(vectort point,matrixt transform,int input_index)
{
  // Warning: point is being transformed here!
  cnttype i;
  transformPoint(point,transform);
#ifndef __CPROVER
    print_vector("point",point);
#endif      
  
  for (i=0;i<_NUM_VECTORS;i++)
  {
    control_floatt value=findSupport(point,vectors[i])+accelsupports[input_index][i];
#ifndef __CPROVER
    printf("%f<%f\n",value,supports[i]);
#endif      
    verify_assume(value<supports[i]);
  }
}

void checkIterations(matrixt transform)
{
  matrixt matrix,temp;
  cnttype i,j,k;
#if (_NUM_INPUT_VERTICES>1)
  #ifndef __CPROVER
    printf("accel inputs\n");
  #endif  
  accelerateInputs();
  #ifndef __CPROVER
    for (i=0;i<_NUM_INPUT_VERTICES;i++) print_vector("input",accel_vertices[i]);
  #endif
  for (k=0;k<_NUM_INPUT_VERTICES;k++)
  {
    for (i=0;i<_NUM_VERTICES;i++)
    {
      #ifndef __CPROVER
        print_vector("init",vertices[i]);
      #endif
      #ifdef POINTS_PER_ITER
        bool skip=true;
        for (j=0;j<_NUM_ITERATIONS;j++)
        {
           if ((iter_vertices[j][0]==i) && (iter_vertices[j][1]==k)) skip=false;
        }
        if (skip) continue;
      #endif  
      for (j=0;j<_DIMENSION;j++) reach_vertices[k][i][j]=vertices[i][j]-accel_vertices[k][j];
      checkIteration(reach_vertices[k][i],transform,k);
    }
  }
#else
  for (j=0;j<_NUM_VECTORS;j++) accelsupports[0][j]=0;
  #ifndef __CPROVER
    printf("no inputs\n");
  #endif
  for (i=0;i<_NUM_VERTICES;i++)
  {
    #ifdef POINTS_PER_ITER
      bool skip=true;
      for (j=0;j<_NUM_ITERATIONS;j++)
      {
         if (iter_vertices[j][0]==i) skip=false;
      }
      if (skip) continue;
    #endif  
    for (j=0;j<_DIMENSION;j++) reach_vertices[0][i][j]=vertices[i][j];
    checkIteration(reach_vertices[0][i],transform,0);
  }
#endif
  for (i=0;i<_DIMENSION;i++)
  {
    for (j=0;j<_DIMENSION;j++) matrix[i][j]=transform[i][j];
  }
  int lastIteration=1;
  for(i=0;i<_NUM_ITERATIONS;i++)
  {
    powerMatrix(temp,transform,iterations[i]-lastIteration);
    multiply(matrix,temp);
    #ifndef __CPROVER
      printf("iter %d\n",iterations[i]);
      print_matrix("matrix",matrix);
    #endif
    #if (_NUM_INPUT_VERTICES>1)
      for (k=0;k<_NUM_INPUT_VERTICES;k++)
      {
        #ifdef POINTS_PER_ITER
          if (iter_vertices[i][1]!=k) continue;
        #endif
        for (j=0;j<_NUM_VERTICES;j++) 
        {
          #ifdef POINTS_PER_ITER
            if (iter_vertices[i][0]!=k) continue;
          #endif
          checkIteration(reach_vertices[k][j],matrix,k);
        }
      }
    #else
      for (j=0;j<_NUM_VERTICES;j++) checkIteration(reach_vertices[0][j],matrix,0);      
    #endif
    lastIteration=iterations[i];
  }
#ifndef __CPROVER
  printf("iters checked\n");
#endif
}
#endif