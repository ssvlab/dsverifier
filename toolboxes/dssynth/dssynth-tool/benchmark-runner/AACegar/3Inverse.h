void inverse(matrixt result,matrixt transform)
{
  control_floatt det=1/
    (transform[0][0]*transform[1][1]*transform[2][2]+transform[0][1]*transform[1][2]*transform[2][0]+transform[0][2]*transform[1][0]*transform[2][1]
    -transform[0][2]*transform[1][1]*transform[2][0]-transform[0][1]*transform[1][0]*transform[2][2]-transform[0][0]*transform[1][2]*transform[2][1]);
  result[0][0]=(transform[1][1]*transform[2][2]-transform[1][2]*transform[2][1])*det;
  result[0][1]=-(transform[1][0]*transform[2][2]-transform[1][2]*transform[2][0])*det;
  result[0][2]=(transform[1][0]*transform[2][1]-transform[1][1]*transform[2][0])*det;
  result[1][0]=-(transform[0][1]*transform[2][2]-transform[0][2]*transform[2][1])*det;
  result[1][1]=(transform[0][0]*transform[2][2]-transform[0][2]*transform[2][0])*det;
  result[1][2]=-(transform[0][0]*transform[2][1]-transform[0][1]*transform[2][0])*det;
  result[2][0]=(transform[0][1]*transform[1][2]-transform[0][2]*transform[1][1])*det;
  result[2][1]=-(transform[0][0]*transform[1][2]-transform[0][2]*transform[1][0])*det;
  result[2][2]=(transform[0][0]*transform[1][1]-transform[0][1]*transform[1][0])*det;
}