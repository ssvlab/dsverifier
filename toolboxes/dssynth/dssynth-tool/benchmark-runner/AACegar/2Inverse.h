void inverse(matrixt result,matrixt transform)
{
  control_floatt det=1/(transform[0][0]*transform[1][1]-transform[0][1]*transform[1][0]);
  result[0][0]=det*transform[1][1];
  result[0][1]=-det*transform[1][0];
  result[1][0]=-det*transform[0][1];
  result[1][1]=det*transform[0][0];  
}
