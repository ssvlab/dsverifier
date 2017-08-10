void inverse(int_matrixt result,int_matrixt transform)
{
  struct intervalt invDet=interval_mult(interval_mult(transform[0][0],transform[1][1]),transform[2][2]);
  invDet=interval_add(invDet,interval_mult(interval_mult(transform[0][1],transform[1][2]),transform[2][0]));
  invDet=interval_add(invDet,interval_mult(interval_mult(transform[0][2],transform[1][0]),transform[2][1]));
  invDet=interval_sub(invDet,interval_mult(interval_mult(transform[0][2],transform[1][1]),transform[2][0]));
  invDet=interval_sub(invDet,interval_mult(interval_mult(transform[0][1],transform[1][0]),transform[2][2]));
  invDet=interval_sub(invDet,interval_mult(interval_mult(transform[0][0],transform[1][2]),transform[2][1]));
  struct intervalt det;
  det.low=1.0/invDet.high;
  det.high=(-1.0)/invDet.low;
  det.high=-det.high;
  
  result[0][0]=interval_sub(interval_mult(transform[1][1],transform[2][2]),interval_mult(transform[1][2],transform[2][1]));
  result[0][1]=interval_sub(interval_mult(transform[1][2],transform[2][0]),interval_mult(transform[1][0],transform[2][2]));
  result[0][2]=interval_sub(interval_mult(transform[1][0],transform[2][1]),interval_mult(transform[1][1],transform[2][0]));
  result[1][0]=interval_sub(interval_mult(transform[0][2],transform[2][1]),interval_mult(transform[0][1],transform[2][2]));
  result[1][1]=interval_sub(interval_mult(transform[0][0],transform[2][2]),interval_mult(transform[0][2],transform[2][0]));
  result[1][2]=interval_sub(interval_mult(transform[0][1],transform[2][0]),interval_mult(transform[0][0],transform[2][1]));
  result[2][0]=interval_sub(interval_mult(transform[0][1],transform[1][2]),interval_mult(transform[0][2],transform[1][1]));
  result[2][1]=interval_sub(interval_mult(transform[0][2],transform[1][0]),interval_mult(transform[0][0],transform[1][2]));
  result[2][2]=interval_sub(interval_mult(transform[0][0],transform[1][1]),interval_mult(transform[0][1],transform[1][0]));
  
  result[0][0]=interval_mult(result[0][0],det);
  result[0][1]=interval_mult(result[0][1],det);
  result[0][2]=interval_mult(result[0][2],det);
  result[1][0]=interval_mult(result[1][0],det);
  result[1][1]=interval_mult(result[1][1],det);
  result[1][2]=interval_mult(result[1][2],det);
  result[2][0]=interval_mult(result[2][0],det);
  result[2][1]=interval_mult(result[2][1],det);
  result[2][2]=interval_mult(result[2][2],det);
}