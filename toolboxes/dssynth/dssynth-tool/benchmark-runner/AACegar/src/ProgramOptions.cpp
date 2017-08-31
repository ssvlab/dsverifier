//Authors: Dario Cattaruzza, Alessandro Abate, Peter Schrammel, Daniel Kroening
//University of Oxford 2016
//This code is supplied under the BSD license agreement (see license.txt)

#include "DynamicalSystem.h"
#include "Synthesiser.h"
#include "ProgramOptions.h"

namespace abstract{

std::string ProgramOptions::getVersion()
{
  return version;
}

void ProgramOptions::help(std::ostream &out)
{
  out << "Axelerator " << getVersion()
      << " Usage" << std::endl << std::endl;
  out << "Command Line Options" << std::endl;
  out << "-ld" << std::endl;
  out << "Long double arithmetic. Uses standard c++ long doubles for all "
         "operations (precision is machine dependent as per c++ specifications). "
         "This is the fastest performance. Results are unsound."
      << std::endl << std::endl;
  out << "-ldi" << std::endl;
  out << "Long double intervals. Soundness is achieved through the use of interval "
          "arithmetic at the cost of around 4x slower speed."
      << std::endl << std::endl;
  out << "-mp" << std::endl;
  out << "Multiple precision arithmetic. # identifies the number of bits in the "
         "mantissa. If not present, the system will default to 256-bit mantissas. "
      << std::endl << std::endl;
  out << "-mpi" << std::endl;
  out << "Multiple precision intervals. Same as mp but soundness is achieved "
         "through the use of interval arithmetic at the cost of around 4x slower speed. "
      << std::endl << std::endl;
  out << "-ii" << std::endl;
  out << "Ignore intervals. This switch instructs the software to output results "
         "as central values without their interval ranges."
      << std::endl << std::endl;
  out << "-u" << std::endl;
  out << "Process unsound results. When this option is selected, unsound results do "
         "are processed and tagged in the result as 'not enough precision'. Otherwise "
         "the software will provide no result."
      << std::endl << std::endl;
  out << "-norm" << std::endl;
  out << "results are given using normalized template directions."
      << std::endl << std::endl;
  out << "-vert" << std::endl;
  out << "results are given as polyhedral vertices as opposed to nequalities."
      << std::endl << std::endl;
  out << "'filename'" << std::endl;
  out << "filename to include in the processing stage. All files included will be "
         "processed according to their data and program options. Results will be "
         "placed in files with extensions corresponding to the numeric options "
         "given (i.e. '.ld','.mp','.ldi','.mpi'). Multiple files will be processed "
         "sequentially but independently of each other."
      << std::endl << std::endl;
  out << "File Format" << std::endl;
  out << "<Header Options (p=,q=,etc)>" << std::endl;
  out << "G -> AX + BU" << std::endl;
  out << std::endl;
  out << "where the header options are as follows:" << std::endl;
  out << "\t p=#" << std::endl;
  out << "\t Dimension of the state space. This parameter is always required." << std::endl;
  out << "\t q=#" << std::endl;
  out << "\t Dimension of the parametric inputs. If not present, the problem is assumed "
         "to have no inputs. Parametric inputs are not deterministic but assumed to have "
         "the same value throughout the progression." << std::endl;
  out << "\t v=#" << std::endl;
  out << "\t Dimension of the control inputs. This parameter will override q if present."
         " Control inputs can take any non-deterministic value at each iteration. "
         "Overapproximations of control inputs are in general less precise." << std::endl;
  out << "\t s=min:max:step" << std::endl;
  out << "\t Number of steps (if missing, the problem is assumed unbounded or an upper "
         "bound will be found based on the guard). If only the min value is supplied, "
         "the problem will be solved for that number of iterations. If all values are "
         "supplied the problem will be solved for all iterations starting at min and "
         "finishing at max (default=min) with a separation given by step (default 1)."
      << std::endl;
  out << "\t l=min:max:step" << std::endl;
  out << "\t Logahedral order for the abstract matrix directions. The number of faces "
         "of the abstract matrix will be (2^l)*(p^2)."
         " The higher the order the more precise the abstraction (also the slower the "
         "performance). If not present, the default value is 2. If max and steop are "
         "provided, the tool will provide the result for each abstraction in the range."
      << std::endl;
  out << "\t t=min:max:step" << std::endl;
  out << "\t Logahedral order for the template directions of the reach tube. The number "
         "of faces of the reach tube will be approximately 4*t*p^2."
         " If not present, the default value is 0 which corresponds to a hypercube."
      << std::endl;
  out << "\t m=#" << std::endl;
  out << "\t Number of bits in the mantissa representation (only valid for mp problems). "
         "If not present, the default is either the precision suppliend in the command "
         "line or 256 in its absence." << std::endl;
  out << std::endl << std::endl;
  out << "A and B are matrices and G,X and U are polyhedral sets (defined as CX < D), "
         "they are all represented in matrix format as follows:" << std::endl;
  out << "For matrices" << std::endl;
  out << "\t[ a11 , a12, ... , a1n" << std::endl
      << "\t  a21 , a22, ... , a2n" << std::endl
      << "\t  ..." << std::endl
      << "\t  ap1 , ap2, ... , apn ]" << std::endl;
  out << "For sets" << std::endl;
  out << "\t[ c11 , c12 , ... , c1p < d1" << std::endl
      << "\t  c21 , c22 , ... , c2p < d2" << std::endl
      << "\t  ..." << std::endl
      << "\t  cm1 , cm2 , ... , cmp < dm ]" << std::endl;
  out << "note that the '<' symbol can also be replaced with a '>' for inequalities"
         " that have a minimum instead of a maximum. The software will negate both "
         "sides and produce a st will all maximums, thus result files will have no "
         "'>' symbols. ie cj1,cj2, ... ,cjp > dj <=> -cj1,-cj2, ... ,-cjp < -dj"
      << std::endl;
  out << "Additionally, when interval arithmetic is selected, result files will be "
         "printed using a central value and an error range. Items marked as "
         "cij+eij indicate an interval [cij-eij  cij+eij]. Note that if there is no "
         "error range (eij=0) the printed value will be cij only" << std::endl;
  out << std::endl;

  out << "A complete file for a 2 dimensional problem could look like this:" << std::endl;
  out << "p=2,q=1" << std::endl;
  out << "[g11 , g12 < h1" << std::endl
      << " g21 , g22 < h2 ]" << std::endl
      << " ->" << std::endl
      << "[a11 , a12" << std::endl
      << " a21 , a22 ]" << std::endl
      << "[x11 , x12 < y1" << std::endl
      << " x21 , x22 < y2" << std::endl
      << " x31 , x32 < y3" << std::endl
      << " x41 , x42 < y4 ]" << std::endl
      << "+" << std::endl
      << "[b11" << std::endl
      << " b21 ]" << std::endl
      << "[u11 < v1" << std::endl
      << " u21 < v2]" << std::endl;
  out << "The result file could look like this:" << std::endl;
  out << "p=2,q=2,s=50,l=2,t=0,e=1.5e-23" << std::endl;
  out << "[r11 + e11 , r12 + e12 < s1" << std::endl
      << " r21 + e21 , r22 + e22 < s2" << std::endl
      << " r31 + e31 , r32 + e32 < s3" << std::endl
      << " r41 + e41 , r42 + e42 < s4 ]" << std::endl;
}

void ProgramOptions::process()
{
  while (types.size()>0) {
    switch(types.back())
    {
#ifdef USE_LDOUBLE
    case eLD: {
  #ifdef USE_SINGLES
      //DynamicalSystem<long double> ldsystem;
      MatToStr<long double>::ms_useConsole=useConsole;
      Synthesiser<long double> ldsystem;
      ldsystem.m_synthType=synthType;
      ldsystem.ms_fullAnswers=!answerOnly;
      if (files.size()>0) ldsystem.processFiles(files,displayType,space,traceIntervals,options);
      else ldsystem.processOptions(options,displayType,space,traceIntervals,true);
  #endif
      break;
    }
    case eLDI: {
  #ifdef USE_INTERVALS
      //DynamicalSystem<ldinterval> ldisystem;
      MatToStr<ldinterval>::ms_useConsole=useConsole;
      MatToStr<ldinterval>::ms_traceIntervals=traceIntervals;
      Synthesiser<ldinterval> ldisystem;
      ldisystem.m_synthType=synthType;
      ldisystem.ms_fullAnswers=!answerOnly;
      if (files.size()>0) ldisystem.processFiles(files,displayType,space,traceIntervals,options);
      else ldisystem.processOptions(options,displayType,space,traceIntervals,true);
  #endif
      break;
    }
#endif
#ifdef USE_MPREAL
    case eMP: {
  #ifdef USE_SINGLES
      //DynamicalSystem<mpfr::mpreal> mpsystem;
      MatToStr<mpfr::mpreal>::ms_useConsole=useConsole;
      Synthesiser<mpfr::mpreal> mpsystem;
      mpsystem.m_synthType=synthType;
      mpsystem.ms_fullAnswers=!answerOnly;
      if (files.size()>0) mpsystem.processFiles(files,displayType,space,traceIntervals,options);
      else mpsystem.processOptions(options,displayType,space,traceIntervals,true);
  #endif
      break;
    }
    case eMPI: {
  #ifdef USE_INTERVALS
      //DynamicalSystem<mpinterval> mpisystem;
      MatToStr<mpinterval>::ms_traceIntervals=traceIntervals;
      MatToStr<mpinterval>::ms_useConsole=useConsole;
      Synthesiser<mpinterval> mpisystem;
      mpisystem.m_synthType=synthType;
      mpisystem.ms_fullAnswers=!answerOnly;
      if (files.size()>0) mpisystem.processFiles(files,displayType,space,traceIntervals,options);
      else mpisystem.processOptions(options,displayType,space,traceIntervals,true);
  #endif
      break;
    }
#endif
    }
    types.pop_back();
  }
}

ProgramOptions::ProgramOptions(int argc, char *argv[]) :
  synthType(eReachTubeSynth),
  displayType(eInequalities),
  space(eNormalSpace)
{
  traceIntervals=true;
  formal=true;
  answerOnly=true;
  useConsole=true;

  //Important!: these have to be in the same order as the enums
  std::string commandOptions[eMaxStr]={"params","dynamics","isense","osense","iosense","guard","iguard","sguard","oguard","init","inputs","templates","arma","control","ref"};
  std::string typeOptions[eAllTypes+1]={"ld","mp","ldi","mpi","all"};
  std::string synthOptions[eCEGISSynth+1]={"reach","init","input","sense","eigen","dynamics","CEGIS"};

  typedef std::map<std::string,optionStr_t> optionNames_t;
  typedef std::map<std::string,numericType_t> typeNames_t;
  typedef std::map<std::string,synthesisType_t> synthNames_t;
  optionNames_t optionNames;
  for (int i=0;i<eMaxStr;i++) {
    optionNames.insert(std::pair<std::string,optionStr_t>(commandOptions[i],(optionStr_t)i));
  }
  typeNames_t typeNames;
  for (int i=0;i<=eAllTypes;i++) {
    typeNames.insert(std::pair<std::string,numericType_t>(typeOptions[i],(numericType_t)i));
  }
  synthNames_t synthNames;
  for (int i=0;i<=eCEGISSynth;i++) {
    synthNames.insert(std::pair<std::string,synthesisType_t>(synthOptions[i],(synthesisType_t)i));
  }

  int precision=256;
  if (argc<2) help(std::cout);
  for (int i=1;i<argc;i++) {
    if (argv[i][0]=='-') {
      int offset=1;
      while(argv[i][offset]=='-') offset++;
      optionNames_t::iterator it=optionNames.find(argv[i]+offset);
      if (it!=optionNames.end()) {
        if (++i<argc) options[it->second]=argv[i];
        continue;
      }
      typeNames_t::iterator typ=typeNames.find(argv[i]+offset);
      if (typ!=typeNames.end()) {
        if (typ->second==eAllTypes) {
          for (int i=0;i<eAllTypes;i++) types.push_back((numericType_t)i);
        }
        else types.push_back(typ->second);
        if (((i+1)<argc) && (argv[i+1][0]!='-')) {
          int result=atoi(argv[++i]);
          if (result>0) precision=result;
          else i--;
        }
#ifdef USE_MPREAL
        functions<mpfr::mpreal>::setDefaultPrec(precision);
        functions<mpfr::mpreal>::ms_formal=formal;
#endif
#ifdef USE_LDOUBLE
        functions<long double>::ms_formal=formal;
#endif
        continue;
      }

      if (strcmp(argv[i]+offset,"version")==0) {
        std::cout << "Axelerator v." << getVersion() << std::endl;
      }
      else if (strcmp(argv[i]+offset,"synth")==0) {
        if (++i<argc) {
          synthNames_t::iterator it=synthNames.find(argv[i]);
          if (it!=synthNames.end()) synthType=it->second;
        }
      }
      else if (strcmp(argv[i]+offset,"trace")==0) useConsole=false;
      else if (strcmp(argv[i]+offset,"eigen")==0) space=eEigenSpace;
      else if (strcmp(argv[i]+offset,"p")==0)     continue;
      else if (strcmp(argv[i]+offset,"u")==0) {
        formal=false;
        functions<mpfr::mpreal>::ms_formal=formal;
        functions<long double>::ms_formal=formal;
      }
      else if (strcmp(argv[i]+offset,"full")==0)  answerOnly=false;
      else if (strcmp(argv[i]+offset,"ii")==0)    traceIntervals=false;
      else if (strcmp(argv[i]+offset,"norm")==0)  displayType=eNormalised;
      else if (strcmp(argv[i]+offset,"vert")==0)  displayType=eVertices;
      else if (strcmp(argv[i]+offset,"ine")==0)   displayType=eInequalities;
      else if (argv[i][offset]=='h')              help(std::cout);
      else {
        std::cout << "unknown parameter " << argv[i] << std::endl;
        std::cout << "use -h for help" << std::endl;
        return;
      }
    }
    else {
      std::ifstream file(argv[i]);
      if (file) files.push_back(argv[i]);
      else std::cout << "unable to open file: " << argv[i] << " [" << i << "]" << std::endl;
    }
  }
  for (optionList_t::iterator it=options.begin();it!=options.end();it++) {
    std::string &data=it->second;
    int pos=data.find("\"");
    while (pos!=std::string::npos) {
      data.erase(pos,1);
      pos=data.find("\"");
    }
  }
  if ((types.size()==0) && (files.size()>0)) {
    types.push_back(eLD);
    types.push_back(eLDI);
    types.push_back(eMP);
    types.push_back(eMPI);
    functions<mpfr::mpreal>::ms_formal=formal;
    functions<long double>::ms_formal=formal;
  }
}

}
