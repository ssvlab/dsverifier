//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "MatrixToString.h"
#include <string>

std::string emptyString;
std::string defaultLeftBracket="[ ";
std::string defaultRightBracket=" ]\n";
std::string cegisLeftBracket="{ ";
std::string cegisRightBracket=" }\n";
std::string defaultNoRightBracket="\n";
std::string defaultSeparator=" , ";
std::string dimensionMismatch="Dimensions do not match";
std::string loadError="Load error";
std::string processingError="Process error";
std::string version="1.0";

namespace abstract {


template <class scalar>
typename MatToStr<scalar>::MatrixS MatToStr<scalar>::ms_emptyMatrix(0,0);

template <class scalar>
MatToStr<scalar> MatToStr<scalar>::ms_defaultLogger(6,defaultLeftBracket,defaultRightBracket,defaultSeparator);

template <class scalar>
MatToStr<scalar> MatToStr<scalar>::ms_defaultDecoder(6,emptyString,defaultNoRightBracket,defaultSeparator);

template <class scalar>
MatToStr<scalar> MatToStr<scalar>::ms_cegisLogger(6,cegisLeftBracket,cegisRightBracket,defaultSeparator);

template <class scalar>
scalar MatToStr<scalar>::ms_one(1);

template <class scalar>
typename MatToStr<scalar>::refScalar MatToStr<scalar>::ms_zero(0);

template <class scalar>
bool MatToStr<scalar>::ms_traceIntervals=true;

template <class scalar>
bool MatToStr<scalar>::ms_useConsole=false;

template <class scalar>
const char MatToStr<scalar>::ms_codes[eNumParameters]={'p','q','v','f','o','s','t','l','m','g','e'};

template <class scalar>
MatToStr<scalar>::MatToStr(int precision,std::string leftBracket,std::string rightBracket,std::string separator) :
    m_zero(0),
    m_precision(precision),
    m_leftBracket(leftBracket),
    m_rightBracket(rightBracket),
    m_separator(separator)
{}

template <class scalar>
MatToStr<scalar>::MatToStr(bool brackets) :
    m_zero(0),
    m_precision(6),
    m_leftBracket(brackets ? "[ " : ""),
    m_rightBracket(brackets ? " ]\n" : "\n"),
    m_separator(" , ")
{}

template <class scalar>
std::string MatToStr<scalar>::MakeWTerm(scalar number,int i,int j)
{
  if (func::isNearZero(number,ms_zero)) return "0";
  if (i==j) return MakeNumber(number,false,12);
  std::stringstream buffer;
  if (func::isNearZero(ms_one-number,ms_zero)) {
    buffer << "w[" << i << "][" << j << "]";
  }
  else if (func::isNearZero(ms_one+number,ms_zero)) {
    buffer << "(-w[" << i << "][" << j << "])";
  }
  else {
    buffer << "(" << MakeNumber(number,false,12) << "*w[" << i << "][" << j << "])";
  }
  return buffer.str();
}

template <class scalar>
std::string MatToStr<scalar>::MakeWRow(MatrixS row,int k)
{
  std::stringstream buffer;
  bool first_done=false;
  buffer << "(";
  if (!func::isNearZero(row.coeff(0,0),ms_zero)) {
    buffer << MakeWTerm(row.coeff(0,0),0,k);
    first_done=true;
  }
  for (int j=1;j<row.cols();j++) {
    if (!func::isNearZero(row.coeff(0,j),ms_zero)) {
      if (first_done) buffer << "+";
      buffer << MakeWTerm(row.coeff(0,j),j,k);
      first_done=true;
    }
  }
  buffer << ")";
  if (!first_done) return "(0)";
  return buffer.str();
}

template <class scalar>
std::string MatToStr<scalar>::MakeSTerm(scalar number,std::string &s)
{
  if (func::isNearZero(number,1e-12)) return "0";
  if (s=="0") return s;
  if (s=="(0)") return "0";
  if (func::isNearZero(ms_one-number,1e-12)) return s;
  if (s=="(1)") return MakeNumber(number,false,12);
  if (s=="(-1)") return MakeNumber(-number,false,12);
  std::stringstream buffer;
  if (func::isNearZero(ms_one+number,ms_zero)) buffer << "(-" << s << ")";
  else buffer << "(" << MakeNumber(number,false,12) << "*" << s << ")";
  return buffer.str();
}

template <class scalar>
std::string MatToStr<scalar>::MakeLSTerm(std::string string,int j,std::string s)
{
  if (s=="0") return s;
  if (s=="(0)") return "0";
  std::stringstream buffer;
  if (s=="(1)") buffer << string << "[" << j << "]";
  else buffer << "(" << string << "[" << j << "]*" << s << ")";
  return buffer.str();
}

template <class scalar>
std::string MatToStr<scalar>::MakeNumber(scalar number,const bool interval,const int precision,const int row)
{
  std::ostringstream stream;
  stream.precision(precision);
  if (row>=0) number/=m_norms.coeff(row);
  if (func::isNearZero(number,ms_zero)) number=0;
  refScalar midPoint=func::toCentre(number);
  refScalar error=func::toUpper(number)-midPoint;
  stream << func::toDouble(midPoint);
  if (interval && (error>ms_zero)) {
    stream.precision((error>abs(midPoint)) ? precision : 3);
    stream << "+" << func::toDouble(error) ;
  }
  return stream.str();
}

inline size_t skipSpaces(const char * pData,size_t pos,size_t length) {
  while (pos<length) {
    if ((pData[pos]!=32) && (pData[pos]!=8)) return pos;
    pos++;
  }
  return pos;
}

inline size_t skipControls(const char * pData,size_t pos,size_t length) {
  while (pos<length) {
    if ((pData[pos]!=32) && (pData[pos]!=8) && (pData[pos]!='\n')) return pos;
    pos++;
  }
  return pos;
}

template <class scalar>
int MatToStr<scalar>::lines(const std::string &str,size_t pos)
{
  return lines(str.data(),pos,str.length());
}

template <class scalar>
int MatToStr<scalar>::lines(const char * pData,size_t pos,const size_t eof)
{
  int result=0;
  while (pos<eof) {
    if ((pData[pos]=='[') || (pData[pos]=='{') || (pData[pos]=='(')) result=0;
    if ((pData[pos]=='\n') || (pData[pos]==';')) result++;
    if ((pData[pos]==']') || (pData[pos]=='}') || (pData[pos]==')')) {
      if ((pos>0) && ((pData[pos-1]=='[') || (pData[pos-1]=='{') || (pData[pos-1]=='('))) return 0;
      return result+1;
    }
    pos++;
  }
  return result;
}

template <class scalar>
int MatToStr<scalar>::cols(const std::string &str,size_t pos)
{
  return cols(str.data(),pos,str.length());
}

template <class scalar>
int MatToStr<scalar>::cols(const char * pData,size_t pos,const size_t eof)
{
  int result=0;
  while (pos<eof) {
    if (pData[pos]=='[') result++;
    if ((pData[pos]=='[') || (pData[pos]=='{') || (pData[pos]=='(')) result=0;
    if ((pData[pos]=='\n') || (pData[pos]==';')) return ++result;
    if ((pData[pos]==']') || (pData[pos]=='}') || (pData[pos]==')')) {
      if ((pos>0) && ((pData[pos-1]=='[') || (pData[pos-1]=='{') || (pData[pos-1]=='('))) return 0;
      return ++result;
    }
    pos++;
  }
  return result;
}

template <class scalar>
bool MatToStr<scalar>::hasInequalities(const std::string &str,size_t pos)
{
  return hasInequalities(str.data(),pos,str.length());
}

template <class scalar>
bool MatToStr<scalar>::hasInequalities(const char * pData,size_t pos,const size_t eof)
{
  pos=skipControls(pData,pos,eof);
  while (pos<eof) {
    if ((pData[pos]=='>') || (pData[pos]=='<')) return true;
    if ((pData[pos]=='\n') || (pData[pos]==';') || pData[pos]==']') return false;
    pos++;
  }
  return false;
}

template <class scalar>
int MatToStr<scalar>::getCommand(commands_t &command,const char * pData,size_t pos,size_t eof)
{
  pos=skipControls(pData,pos,eof);
  switch(pData[pos]) {
  case '+':
    command=ePlusCmd;
    break;
  case '=':
    command=eEqualsCmd;
    break;
  case ':':
    command=eGivenCmd;
      break;
  case '-':
    if ((pos<eof) && (pData[pos+1]=='>')) {
      command=eArrowCmd;
      pos++;
    }
    else command=eMinusCmd;
    break;
  default:
    command=eNoCmd;
    return pos;
  }
  return pos+1;
}

template <class scalar>
int MatToStr<scalar>::getCommand(commands_t &command,const std::string &str,size_t pos)
{
  return getCommand(command,str.data(),pos,str.length());
}

template <class scalar>
int MatToStr<scalar>::StringToMat(MatrixS &matrix,const std::string &str,size_t pos)
{
  return StringToMat(matrix,ms_emptyMatrix,str.data(),pos,str.length());
}

template <class scalar>
int MatToStr<scalar>::StringToMat(MatrixS &directions,MatrixS &supports,const std::string &str,size_t pos)
{
  return StringToMat(directions,supports,str.data(),pos,str.length());
}

template <class scalar>
scalar MatToStr<scalar>::getNumber(const char * pData,size_t &pos,const size_t eof)
{
  pos=skipControls(pData,pos,eof);
  scalar value;
  char *pEnd=func::toScalar(pData+pos,value);
  pos=(int)(pEnd-pData);
  return value;
}

template <class scalar>
int MatToStr<scalar>::StringToMat(MatrixS& matrix,MatrixS &aux,const char * pData,size_t pos,size_t eof)
{
  char * pEnd;
  pos=skipControls(pData,pos,eof);
  if ((pos<eof) && ((pData[pos]=='[') || (pData[pos]=='{') || (pData[pos]=='('))) pos++;
  matrix=MatrixS::Zero(matrix.rows(),matrix.cols());
  if (aux.rows()>0) aux=MatrixS::Zero(aux.rows(),aux.cols());
  for (int row=0;(row<matrix.rows()) && (pos<eof);row++) {
    for (int col=0;(col<matrix.cols()) && (pos<eof);col++) {
      pos=skipSpaces(pData,pos,eof);
      if ((pos<eof) && ((pData[pos]=='\n') || (pData[pos]==';') || (pData[pos]==']') || (pData[pos]=='}') || (pData[pos]==')'))) break;
      scalar value;
      pEnd=func::toScalar(pData+pos,value);
      pos=skipSpaces(pData,(int)(pEnd-pData),eof);
      if ((pData[pos]=='x') || (pData[pos]=='u')) {
        pos++;
        col=std::strtold(pData+pos,&pEnd);
        pos=(int)(pEnd-pData);
        if (func::isZero(value)) value=1.0;
      }
      matrix.coeffRef(row,col)=value;
      pos=skipSpaces(pData,(int)(pEnd-pData),eof);
      if (pData[pos]=='+') {
        scalar error=std::strtold(pData+pos,&pEnd);
        pos=skipSpaces(pData,(int)(pEnd-pData),eof);
      }
      if ((pData[pos]=='\n') || (pData[pos]==';') || (pData[pos]=='<') || (pData[pos]=='>') || (pData[pos]=='=')) break;
      else if (pData[pos]==',') pos++;
      else if (pData[pos]=='&') pos++;
    }
    if ((pData[pos]=='\n') || (pData[pos]==';')) pos++;
    else if ((pData[pos]=='<') || (pData[pos]=='>') || (pData[pos]=='=')) {
      if (aux.rows()<=0) return -pos;
      bool reverse=(pData[pos]=='>');
      pos++;
      pos=skipSpaces(pData,pos,eof);
      scalar value;
      pEnd=func::toScalar(pData+pos,value);
      aux.coeffRef(row,0)=value;
      if (reverse) {
        matrix.row(row)=-matrix.row(row);
        aux.coeffRef(row,0)=-aux.coeffRef(row,0);
      }
      pos=skipSpaces(pData,(int)(pEnd-pData),eof);
      if (pData[pos]=='+') {
        long double error=std::strtold(pData+pos,&pEnd);
        pos=skipSpaces(pData,(int)(pEnd-pData),eof);
      }
      if ((pData[pos]=='\n') || (pData[pos]==';')) pos++;
    }
    pos=skipSpaces(pData,pos,eof);
  }
  if ((pData[pos]==']') || (pData[pos]=='}') || (pData[pos]==')')) pos++;
  return pos;
}

template <class scalar>
int MatToStr<scalar>::StringToDim(ParamMatrix &paramValues,const std::string &str,size_t pos)
{
  return StringToDim(paramValues,str.data(),pos,str.length());
}

template <class scalar>
int MatToStr<scalar>::StringToDim(ParamMatrix &paramValues,const char * pData,size_t pos,size_t eof)
{
  char * pEnd;
  pos=skipSpaces(pData,pos,eof);
  bool found=false;
  for (int j=0;j<eNumParameters;j++) {
    if (pData[pos]==ms_codes[j]) found=true;
  }
  if (!found) return pos;
  for (int j=0;j<eNumParameters;j++) {
    for (int k=0;k<3;k++) paramValues.coeffRef(j,k)=0;
    if (pData[pos]==ms_codes[j]) {
      if (pData[++pos]!='=') return -pos;
      pos++;
      paramValues.coeffRef(j,0)=std::strtold(pData+pos,&pEnd);
      pos=(int)(pEnd-pData);
      pos=skipSpaces(pData,pos,eof);
      if (pData[pos]==':') {
        pos++;
        paramValues.coeffRef(j,1)=std::strtold(pData+pos,&pEnd);
        pos=(int)(pEnd-pData);
        pos=skipSpaces(pData,pos,eof);
        if (pData[pos]==':') {
          pos++;
          paramValues.coeffRef(j,2)=std::strtold(pData+pos,&pEnd);
          pos=(int)(pEnd-pData);
          pos=skipSpaces(pData,pos,eof);
        }
      }
      if (pData[pos]==',') pos++;
      pos=skipSpaces(pData,pos,eof);
    }
  }
  return pos;
}

template <class scalar> std::string MatToStr<scalar>::DimToString(ParamMatrix* paramValues,refScalar error)
{
  std::ostringstream stream;
  for (int i=0;i<eNumParameters;i++) {
    if ((paramValues->coeff(i,0)>0) || (paramValues->coeff(i,1)>0)) {
      stream << ms_codes[i] << "=" << paramValues->coeff(i,0);
      if (paramValues->coeff(i,1)>0) {
        stream << ":" << paramValues->coeff(i,1);
        if (paramValues->coeff(i,2)>1) {
          stream << ":" << paramValues->coeff(i,2);
        }
      }
      stream << ",";
    }
  }
  stream << "e=" << func::toDouble(error) << "\n";
  return stream.str();
}

/// Returns the description of a matrix
template <class scalar> std::string MatToStr<scalar>::MatToString(const MatrixS &matrix,const bool interval,const bool normalised,const bool transpose)
{
  std::string result=m_leftBracket;
  if (normalised) clearNorms(matrix,transpose);
  if (transpose) {
    for (int col=0;col<matrix.cols();col++)
    {
      for (int row=0;row<matrix.rows();row++) result+=MakeNumber(matrix.coeff(row,col),interval,m_precision,normalised ? row : -1)+m_separator; \
      if (result.length()>3) result.resize(result.length()-3);
      result+="\n  ";
    }
  }
  else {
    for (int row=0;row<matrix.rows();row++)
    {
      for (int col=0;col<matrix.cols();col++) result+=MakeNumber(matrix.coeff(row,col),interval,m_precision,normalised ? row : -1)+m_separator; \
      if (result.length()>3) result.resize(result.length()-3);
      result+="\n  ";
    }
  }
  if (result.length()>3) result.resize(result.length()-3);
  result+=m_rightBracket;
  return result;
}

/// Returns the description of a complex matrix
template <class scalar>
std::string MatToStr<scalar>::MatToString(const MatrixC &matrix,const bool interval,const bool normalised,const bool transpose)
{
  std::string result=m_leftBracket;
  if (transpose) {
      for (int i=0;i<matrix.cols();i++)
      {
        for (int j=0;j<matrix.rows();j++)
        {
          result+="("+MakeNumber(matrix.coeff(j,i).real(),interval,m_precision)+",";
          result+=MakeNumber(matrix.coeff(j,i).imag(),interval,m_precision)+")"+m_separator;// , ";
        }
        result.resize(result.length()-m_separator.length());
        result+="\n  ";
      }
  }
  else {
    for (int i=0;i<matrix.rows();i++)
    {
      for (int j=0;j<matrix.cols();j++)
      {
        result+="("+MakeNumber(matrix.coeff(i,j).real(),interval,m_precision)+",";
        result+=MakeNumber(matrix.coeff(i,j).imag(),interval,m_precision)+")"+m_separator;// , ";
      }
      result.resize(result.length()-m_separator.length());
      result+="\n  ";
    }
  }
  if (result.length()>3) result.resize(result.length()-3);
  result+=m_rightBracket;
  return result;
}

/// Returns the description of a complex matrix
template <class scalar>
std::string MatToStr<scalar>::IneToString(const MatrixS &directions,const MatrixS &supports,const bool interval,const bool normalised,const bool transposed)
{
  std::string result=m_leftBracket;
  if (normalised) clearNorms(directions,transposed);
  for (int row=0;row<supports.rows();row++) { //ie transposed ? directions.cols() : pDirections->rows()
    if (transposed) {
        for (int col=0;col<directions.rows();col++) {
          result+=MakeNumber(directions.coeff(col,row),interval,m_precision,normalised ? col : -1)+m_separator;
        }
    }
    else {
      for (int col=0;col<directions.cols();col++) {
        result+=MakeNumber(directions.coeff(row,col),interval,m_precision,normalised ? row : -1)+m_separator;
      }
    }
    if (result.length()>3) result.resize(result.length()-3);
    result+=" < ";
    for (int col=0;col<supports.cols();col++) {
      result+=MakeNumber(supports.coeff(row,col),interval,m_precision,normalised ? row : -1)+m_separator;
    }
    if (result.length()>3) result.resize(result.length()-3);
    result+="\n  ";
  }
  if (result.length()>3) result.resize(result.length()-3);
  if (!result.empty()) result+=m_rightBracket;
  return result;
}

template <class scalar>
void MatToStr<scalar>::clearNorms(const MatrixS &matrix,const bool transpose)
{
  if (transpose) m_norms=matrix.colwise().norm().transpose();
  else m_norms=matrix.rowwise().norm();
  for (int row=0;row<m_norms.rows();row++) {
    if (func::isZero(m_norms.coeff(row)/*,m_zero*/)) m_norms.coeffRef(row)=1;
  }
}

template <class scalar>
std::list<std::string> MatToStr<scalar>::split(const std::string &string,const std::string &separator)
{
  std::list<std::string> result;
  size_t pos=0,found=0;
  while(pos<string.length()) {
    found=string.find(separator,pos);
    if (found==std::string::npos) found=string.length();
    if (found>pos) result.push_back(string.substr(pos,found-pos));
    pos=found+separator.length();
  }
  return result;
}

template <class scalar>
void MatToStr<scalar>::logData(const std::string data,const bool newLine)
{
  if (ms_useConsole) {
    std::cout << data;
    if (newLine) std::cout << std::endl;
    return;
  }
  if (!m_trace.is_open()) m_trace.open("trace.txt",std::ios_base::app);
  if (m_trace.is_open()) {
    m_trace.write(data.data(),data.size());
    if (newLine) m_trace << std::endl;
    m_trace.close();
  }
}

template <class scalar>
void MatToStr<scalar>::logData(scalar number,const std::string title,bool brackets)
{
  if (brackets) {
    if (!title.empty()) logData(title+m_leftBracket,false);
    logData(MakeNumber(number,ms_traceIntervals)+m_rightBracket,false);
  }
  else {
    if (!title.empty()) logData(title,false);
    logData(MakeNumber(number,ms_traceIntervals),false);
  }
}

template <class scalar>
void MatToStr<scalar>::logData(std::vector<int> &vector,const std::string title)
{
  std::stringstream stream;
  stream << title << vector[0];
  for (unsigned int i=1;i<vector.size();i++) stream << " , " << vector[i];
  logData(stream.str());
}

template <class scalar>
void MatToStr<scalar>::logData(const MatrixS &matrix,const std::string title,bool transpose,bool forceNewLine)
{
  if (!title.empty()) logData(title,(transpose ? (matrix.cols()>1) : (matrix.rows()>1)) || forceNewLine);
  logData(MatToString(matrix,ms_traceIntervals,false,transpose),false);
}

template <class scalar>
void MatToStr<scalar>::logNormalisedData(const MatrixS &matrix,int col,const std::string title)
{
  MatrixS normedMatrix=matrix;
  if (col>=0) {
    for (int row=0;row<normedMatrix.rows();row++) {
      if (!func::isZero(normedMatrix.coeff(row,col))) {
        normedMatrix.row(row)/=abs(normedMatrix.coeff(row,col));
      }
    }
  }
  else {
    scalar normValue=normedMatrix.norm();
    normedMatrix/=normValue;
  }
  logData(normedMatrix,title);
}

template <class scalar>
void MatToStr<scalar>::logData(const MatrixC &matrix,const std::string title,bool transpose,bool forceNewLine)
{
  if (!title.empty()) logData(title,(transpose ? (matrix.cols()>1) : (matrix.rows()>1)) || forceNewLine);
  logData(MatToString(matrix,ms_traceIntervals,false,transpose),false);
}

template <class scalar>
void MatToStr<scalar>::logData(const MatrixS &directions,const MatrixS &supports,const std::string title,bool transpose)
{
  if (!title.empty()) logData(title,true);
  logData(IneToString(directions,supports,ms_traceIntervals,false,transpose),false);
}

/// Returns the description of a matrix
template <class scalar> std::string MatToStr<scalar>::MatToC(const std::string name,const MatrixS &matrix,const bool interval,const bool use_cpp)
{
  std::string result;
  if (!name.empty()) result+=name+"={.coeffs";
  result+="={";
  if (matrix.rows()==1) {
    for (int col=0;col<matrix.cols();col++) {
      if (col>0) result+=m_separator;
      result+=MakeNumber(matrix.coeff(0,col),false,m_precision);
    }
  }
  else {
    for (int row=0;row<matrix.rows();row++)
    {
      if (row>0) result+=" , ";
      if (use_cpp) result+=" { ";
      for (int col=0;col<matrix.cols();col++) {
        if (col>0) result+=m_separator;
        result+=MakeNumber(matrix.coeff(row,col),false,m_precision);
      }
      if (use_cpp) result+=" }";
    }
  }
  result+=" }";
  if (interval && !name.empty()) {
    result+=", .uncertainty={";
    if (matrix.rows()==1) {
      for (int col=0;col<matrix.cols();col++) {
        if (col>0) result+=m_separator;
        result+=MakeNumber(interval ? func::toWidth(matrix.coeff(0,col)) : 0,false,m_precision);
      }
    }
    else {
      for (int row=0;row<matrix.rows();row++)
      {
        if (row>0) result+=" , ";
        if (use_cpp) result+=" { ";
        for (int col=0;col<matrix.cols();col++) {
          if (col>0) result+=m_separator;
          result+=MakeNumber(interval ? func::toWidth(matrix.coeff(row,col))/2 : 0,false,m_precision);
        }
        if (use_cpp) result+=" }";
      }
    }
    result+=" }";
  }
  if (!name.empty()) result+="}";
  result+=";\n";
  return result;
}

/// Returns a C boolen statement describing a set of Inequlities
template <class scalar> std::string MatToStr<scalar>::IneToC(const std::string type,const std::string cast,const std::string name,const MatrixS &directions,const MatrixS &supports,const bool interval,const int orBlockSize)
{
  std::stringstream buffer;
  if (orBlockSize<=1) {
    int firstRow=0;
    for (int row=0;row<directions.rows();row++) {
      int firstCol=0;
      for (int col=0;col<directions.cols();col++) {
        scalar coeff=directions.coeff(row,col);
        if (func::isZero(coeff)) firstCol++;
        else break;
      }
      if (firstCol==directions.cols()) {
        firstRow++;
        continue;
      }
      if (row>firstRow)  buffer << " && ";
      buffer << "(";
      for (int col=firstCol;col<directions.cols();col++) {
        scalar coeff=directions.coeff(row,col);
        if ((col>firstCol) && (!func::isNegative(coeff))) buffer << "+";
        if (func::isZero(ms_one+coeff)) {
          buffer << "-";
        }
        else if (func::isNegative(coeff)) {
          buffer << "-" << cast << MakeNumber(-coeff,false,m_precision) << "*";
        }
        else if (!func::isZero(ms_one-coeff)) {
          buffer << cast << MakeNumber(coeff,false,m_precision) << "*";
        }
        buffer << name << "[" << col << "]";
      }
      buffer << "<" << cast << MakeNumber(supports.coeff(row,0),false,m_precision) << ")\n";
    }
  }
  else {
    buffer << "(";
    for (int row=0;row<directions.rows();row++) {
      int firstCol=0;
      for (int col=0;col<directions.cols();col++) {
        scalar coeff=directions.coeff(row,col);
        if (func::isZero(coeff)) firstCol++;
        else break;
      }
      if (firstCol==directions.cols()) continue;
      if (row>0) {
        if ((row%orBlockSize)==0) buffer << ") && (";
        else buffer << " || ";
      }
      buffer << "(";
      for (int col=firstCol;col<directions.cols();col++) {
        scalar coeff=directions.coeff(row,col);
        if ((col>firstCol) && (!func::isNegative(coeff))) buffer << "+";
        if (func::isZero(ms_one+coeff)) {
          buffer << "-";
        }
        else if (func::isNegative(coeff)) {
          buffer << "-" << cast << MakeNumber(-coeff,false,m_precision) << "*";
        }
        else if (!func::isZero(ms_one-coeff)) {
          buffer << cast << MakeNumber(coeff,false,m_precision) << "*";
        }
        buffer << name << "[" << col << "]";
      }
      if (firstCol==directions.cols()) buffer << "1)\n";
      else buffer << "<" << cast << MakeNumber(supports.coeff(row,0),false,m_precision) << ")\n";
    }
    buffer << ")";
    if (buffer.str()=="()") return type+"(1)";
  }
  if (buffer.str().empty()) return type+"(1)";
  return type+"("+buffer.str()+")";
}


#ifdef USE_LDOUBLE
  template class MatToStr<long double>;
  #ifdef USE_INTERVALS
    template class MatToStr<ldinterval>;
  #endif
#endif
#ifdef USE_MPREAL
  template class MatToStr<mpfr::mpreal>;
  #ifdef USE_INTERVALS
    template class MatToStr<mpinterval>;
  #endif
#endif

}
