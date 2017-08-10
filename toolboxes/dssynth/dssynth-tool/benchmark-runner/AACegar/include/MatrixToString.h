//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#ifndef MATRIX_TO_STRING_H
#define MATRIX_TO_STRING_H

#include "Scalar.h"
#include <vector>
#include <fstream>
#include <string>
#include <list>

extern std::string emptyString,defaultLeftBracket,defaultRightBracket,defaultNoRightBracket,defaultSeparator;
extern std::string dimensionMismatch,loadError,processingError;
extern std::string version;

namespace abstract {
typedef Eigen::Matrix<int,eNumParameters,3> ParamMatrix;

template <class scalar> class MatToStr {
public:
    typedef interval_def<scalar> type;
    typedef typename type::scalar_type refScalar;
    typedef typename type::complexS complexS;
    typedef typename type::MatrixS  MatrixS;
    typedef typename type::MatrixC  MatrixC;

    typedef functions<refScalar> func;

    void setPrecision(int precision) { m_precision=precision; }
    MatToStr(const int precision,const std::string leftBracket,const std::string rightBracket,const std::string separator);
    MatToStr(const bool brackets);
    scalar getNumber(const char * pData,size_t &pos=0,const size_t eof=0);
    scalar getNumber(const std::string &str,size_t &pos=0) { return getNumber(str.data(),pos,str.size()); }
    std::string MakeNumber(scalar number,const bool interval=ms_traceIntervals,const int precision=6,const int row=-1);
    std::string MakeSTerm(scalar number,std::string &s);
    std::string MakeLSTerm(std::string string,int j,std::string s);
    std::string MakeWTerm(scalar number,int i,int j);
    std::string MakeWRow(MatrixS row,int k);
    std::string MatToString(const MatrixS &matrix,const bool interval=ms_traceIntervals,const bool normalised=false,const bool transpose=false);
    std::string MatToString(const MatrixC &matrix,const bool interval=ms_traceIntervals,const bool normalised=false,const bool transpose=false);
    std::string IneToString(const MatrixS &directions,const MatrixS &supports,const bool interval=ms_traceIntervals,const bool normalised=false,const bool transposed=false);

    /// Returns a C structure description of a matrix
    std::string MatToC(const std::string name,const MatrixS &matrix,const bool interval=false,const bool use_cpp=true);

    /// Returns a C boolen statement describing a set of Inequlities
    std::string IneToC(const std::string type,const std::string cast,const std::string name,const MatrixS &directions,const MatrixS &supports,const bool interval=false,const int orBlockSize=1);

    int lines(const std::string &str,size_t pos=0);
    int lines(const char * pData,size_t pos,const size_t eof);

    int cols(const std::string &str,size_t pos);
    int cols(const char * pData,size_t pos,const size_t eof);

    bool hasInequalities(const std::string &str,size_t pos=0);
    bool hasInequalities(const char * pData,size_t pos,const size_t eof);
    std::list<std::string> split(const std::string &string,const std::string &separator);
    int StringToMat(MatrixS &matrix,const std::string &str,size_t pos=0);
    int StringToMat(MatrixS &directions,MatrixS &supports,const std::string &str,size_t pos=0);
    int StringToMat(MatrixS &matrix,MatrixS &aux,const char * pData,size_t pos,const size_t eof);
    int StringToDim(ParamMatrix &paramValues,const std::string &str,size_t pos=0);
    int StringToDim(ParamMatrix &paramValues,const char * pData,size_t pos,const size_t eof);
    std::string DimToString(ParamMatrix* paramValues,refScalar error);
    int getCommand(commands_t &command,const char * pData,size_t pos,const size_t eof);
    int getCommand(commands_t &command,const std::string &str,size_t pos=0);
    void logData(const std::string data,const bool newLine=true);
    void logData(scalar number,const std::string title,const bool brackets=false);
    void logData(std::vector<int> &vector,const std::string title);
    void logNormalisedData(const MatrixS &matrix,int col=-1,const std::string title="");
    void logData(const MatrixS &matrix,const std::string title="",const bool transpose=false,bool forceNewLine=false);
    void logData(const MatrixC &matrix,const std::string title="",const bool transpose=false,bool forceNewLine=false);
    void logData(const MatrixS &directions,const MatrixS &supports,const std::string title="",const bool transpose=false);
public:
    void clearNorms(const MatrixS &matrix,const bool transpose);
    MatrixS             m_norms;
    refScalar           m_zero;
private:
    std::ofstream       m_trace;
    ParamMatrix         m_results;
    static const char   ms_codes[eNumParameters];
public:
    int                 m_precision;
    std::string         m_leftBracket;
    std::string         m_rightBracket;
    std::string         m_separator;
    static bool         ms_traceIntervals;
    static bool         ms_useConsole;
    static scalar       ms_one;
    static refScalar    ms_zero;
    static MatrixS      ms_emptyMatrix;
    static MatToStr     ms_defaultLogger;
    static MatToStr     ms_defaultDecoder;
    static MatToStr     ms_cegisLogger;
};

}

#endif //MATRIX_TO_STRING_H
