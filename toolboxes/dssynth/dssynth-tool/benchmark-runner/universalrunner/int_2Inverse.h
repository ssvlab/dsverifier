void inverse(int_matrixt result,int_matrixt transform)
{
  struct intervalt invDet=interval_sub(interval_mult(transform[0][0],transform[1][1]),interval_mult(transform[0][1],transform[1][0]));
  struct intervalt det;
  det.low=1.0/invDet.high;
  det.high=(-1.0)/invDet.low;
  det.high=-det.high;
  result[0][0]=interval_mult(det,transform[1][1]);
  result[1][1]=interval_mult(det,transform[0][0]);  
  result[0][1]=interval_mult(neg_interval(det),transform[1][0]);
  result[1][0]=interval_mult(neg_interval(det),transform[0][1]);
}
