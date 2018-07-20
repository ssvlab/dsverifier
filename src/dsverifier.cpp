/*
 # --------------------------------------------------
 #
 #  Digital-Systems Verifier (DSVerifier)
 #
 # --------------------------------------------------
 #
 #  Federal University of Amazonas - UFAM
 #
 #  Authors: Hussama Ismail - hussamaismail@gmail.com
 #           Iury Bessa - iury.bessa@gmail.com
 #           Lucas Cordeiro - lucasccordeiro@gmail.com
 #
 #  Contributors: Daniel Mello - dani-dmello@hotmail.com
 #                Lennon Chaves <lennon.correach@gmail.com>
 #
 # --------------------------------------------------
 #  Usage:
 #
 #    ./dsverifier file.c or file.ss
 #         --realization DFI
 #         --property STABILITY
 #         --x-size 10
 #         --timeout 3600
 #
 #  Supported Properties:
 #
 #     for digital-systems in transfer function:
 #        OVERFLOW, LIMIT_CYCLE, ZERO_INPUT_LIMIT_CYCLE,
 #        TIMING, ERROR, STABILITY, and MINIMUM_PHASE
 #
 #     for digital-systems using closed-loop in transfer functions:
 #        STABILITY_CLOSED_LOOP, LIMIT_CYCLE_CLOSED_LOOP,
 #        and QUANTIZATION_ERROR_CLOSED_LOOP
 #
 #  Supported Realizations:
 #     DFI, DFII, TDFII,
 #     DDFI, DDFII, TDDFII.
 #
 # --------------------------------------------------
 */

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <complex>
#include <algorithm>
#include <cmath>
#include <exception>
#include <assert.h>
#include <iomanip>
#include <regex>
#include <fstream>
#include <streambuf>
#include <math.h>
#include <cassert>
#include <sstream>
#include <list>
#include <stack>
#include <map>
#include <iterator>


/* eigen dependencies */
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Polynomials>
#include <unsupported/Eigen/MatrixFunctions>

/* boost dependencies */
#include <boost/algorithm/string.hpp>

#include "version.h"

#include "../bmc/core/definitions.h"
#include "../bmc/core/fixed_point.h"
#include "../bmc/core/floating_point.h"
#include "../bmc/core/util.h"
#include "../bmc/core/delta_operator.h"
#include "../bmc/core/initialization.h"
#include "../bmc/core/filter_functions.h"

#include "dsverifier_messages.h"

typedef Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootType RootTypet;
typedef Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType RootsTypet;

const char * properties[] =
{ "OVERFLOW", "LIMIT_CYCLE", "ZERO_INPUT_LIMIT_CYCLE", "ERROR", "TIMING",
    "TIMING_MSP430", "STABILITY", "STABILITY_CLOSED_LOOP",
    "LIMIT_CYCLE_CLOSED_LOOP", "QUANTIZATION_ERROR_CLOSED_LOOP",
    "MINIMUM_PHASE", "QUANTIZATION_ERROR", "CONTROLLABILITY", "OBSERVABILITY",
    "LIMIT_CYCLE_STATE_SPACE", "SAFETY_STATE_SPACE", "FILTER_MAGNITUDE_NON_DET",
    "FILTER_MAGNITUDE_DET", "FILTER_PHASE_DET", "FILTER_PHASE_NON_DET",
    "SETTLING_TIME"};

const char * rounding[] =
{ "ROUNDING", "FLOOR", "CEIL" };
const char * overflow[] =
{ "DETECT_OVERFLOW", "SATURATE", "WRAPAROUND" };
const char * realizations[] =
{ "DFI", "DFII", "TDFII", "DDFI", "DDFII", "TDDFII" };
const char * bmcs[] =
{ "ESBMC", "CBMC" };
const char * connections_mode[] =
{ "SERIES", "FEEDBACK" };
const char * arithmetic_mode[] =
{ "FIXEDBV", "FLOATBV" };
const char * wordlength_mode[] =
{ "16", "32", "64" };
const char * error_mode[] =
{ "ABSOLUTE", "RELATIVE" };

// Related to math expression parser
const int LEFT_ASSOC  = 0;
const int RIGHT_ASSOC = 1;

/* expected parameters */
unsigned int desired_x_size = 0;

double mynondet;

class dsverifier_stringst
{
  public:
    std::string desired_wordlength_mode;
    std::string desired_arithmetic_mode;
    std::string desired_filename;
    std::string desired_property;
    std::string desired_realization;
    std::string desired_connection_mode;
    std::string desired_error_mode;
    std::string desired_rounding_mode;
    std::string desired_overflow_mode;
    std::string desired_timeout;
    std::string desired_bmc;
    std::string desired_function;
    std::string desired_solver;
    std::string desired_macro_parameters;
    std::string desired_ds_id;
};

dsverifier_stringst dsv_strings;

/* state space */
bool stateSpaceVerification = false;
bool closedloop = false;
bool nofwl = false;
bool translate = false;
bool k_induction = false;
digital_system_state_space _controller;
digital_system_state_space _controller_fxp;
double desired_quantization_limit = 0.0;
bool show_counterexample_data = false;
bool preprocess = false;


/*******************************************************************
 Function: replace_all_string

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string replace_all_string(std::string str, const std::string& from,
    const std::string& to)
{
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos)
  {
    str.replace(start_pos, from.length(), to);
    // Handles case where 'to' is a substring of 'from'
    start_pos += to.length();
  }
  return str;
}

/*******************************************************************
 Function: extract_data_from_file

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void extract_data_from_file()
{
  std::ifstream verification_file(dsv_strings.desired_filename);
  bool ds_id_found = false;

  for(std::string current_line; getline(verification_file, current_line);)
  {
    /* removing whitespaces */
    current_line = replace_all_string(current_line, " ", "");
    current_line = replace_all_string(current_line, "\t", "");
    /* check the last comma, and remove it */
    if(current_line.back() == ',')
    {
      current_line.pop_back();
    }

    /* check if there is an desired ds_id and find the region*/
    if(dsv_strings.desired_ds_id.size() != 0)
    {
      std::string::size_type find_desired_ds_id = current_line.find(
          dsv_strings.desired_ds_id, 0);
      if(ds_id_found == false)
      {
        if(find_desired_ds_id != std::string::npos)
          ds_id_found = true;
        else
          continue; /* go to next line */
      }
    }

    std::string::size_type ds_a = current_line.find(".a=", 0);
    if(ds_a != std::string::npos)
    {
      std::vector<std::string> coefficients;
      boost::split(coefficients, current_line, boost::is_any_of(","));
      for(int i = 0; i < coefficients.size(); i++)
      {
        std::string coefficient = coefficients.at(i);
        coefficient = replace_all_string(coefficient, ".a={", "");
        coefficient = replace_all_string(coefficient, "}", "");
        ds.a[i] = std::atof(coefficient.c_str());
        ds.a_size = coefficients.size();
      }
      continue;
    }
    std::string::size_type ds_b = current_line.find(".b=", 0);
    if(ds_b != std::string::npos)
    {
      std::vector<std::string> coefficients;
      boost::split(coefficients, current_line, boost::is_any_of(","));
      for(int i = 0; i < coefficients.size(); i++)
      {
        std::string coefficient = coefficients.at(i);
        coefficient = replace_all_string(coefficient, ".b={", "");
        coefficient = replace_all_string(coefficient, "}", "");
        ds.b[i] = std::atof(coefficient.c_str());
        ds.b_size = coefficients.size();
      }
      continue;
    }
    std::string::size_type ds_sample_time = current_line.find(".sample_time",
        0);
    if(ds_sample_time != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".sample_time=", "");
      ds.sample_time = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type impl_int_bits = current_line.find(".int_bits", 0);
    if(impl_int_bits != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".int_bits=", "");
      impl.int_bits = std::atoi(current_line.c_str());
      continue;
    }
    std::string::size_type impl_frac_bits = current_line.find(".frac_bits", 0);
    if(impl_frac_bits != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".frac_bits=", "");
      impl.frac_bits = std::atoi(current_line.c_str());
      continue;
    }
    std::string::size_type impl_min = current_line.find(".min", 0);
    if(impl_min != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".min=", "");
      impl.min = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type impl_max = current_line.find(".max", 0);
    if(impl_max != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".max=", "");
      impl.max = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type impl_delta = current_line.find(".delta", 0);
    if(impl_delta != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".delta=", "");
      impl.delta = std::atof(current_line.c_str());
      continue;
    }
    // Filter Coefficients
    std::string::size_type filter_Ap = current_line.find(".Ap", 0);
    if(filter_Ap != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".Ap=", "");
      filter.Ap = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_Ac = current_line.find(".Ac", 0);
    if(filter_Ac != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".Ac=", "");
      filter.Ac = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_Ar = current_line.find(".Ar", 0);
    if(filter_Ar != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".Ar=", "");
      filter.Ar = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_wp = current_line.find(".wp", 0);
    if(filter_wp != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".wp=", "");
      filter.wp = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_wc = current_line.find(".wc", 0);
    if(filter_wc != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".wc=", "");
      filter.wc = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_wr = current_line.find(".wr", 0);
    if(filter_wr != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".wr=", "");
      filter.wr = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_w1p = current_line.find(".w1p", 0);
    if(filter_w1p != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".w1p=", "");
      filter.w1p = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_w1c = current_line.find("w1c", 0);
    if(filter_w1c != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".w1c=", "");
      filter.w1c = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_w1r = current_line.find(".w1r", 0);
    if(filter_w1r != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".w1r=", "");
      filter.w1r = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_w2p = current_line.find(".w2p", 0);
    if(filter_w2p != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".w2p=", "");
      filter.w2p = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_w2c = current_line.find(".w2c", 0);
    if(filter_w2c != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".w2c=", "");
      filter.w2c = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_w2r = current_line.find(".w2r", 0);
    if(filter_w2r != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".w2r=", "");
      filter.w2r = std::atof(current_line.c_str());
      continue;
    }
    std::string::size_type filter_type = current_line.find(".type", 0);
    if(filter_type != std::string::npos)
    {
      current_line = replace_all_string(current_line, ".type=", "");
      filter.type = std::atof(current_line.c_str());
      continue;
    }
  }
}

/*******************************************************************
 Function: validate_function

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_function(std::string data)
{
  if(data.empty())
    std::cout << "specify a function name" << std::endl;
  else
    dsv_strings.desired_function = data;
}

/*******************************************************************
 Function: validate_selected_bmc

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_bmc(std::string data)
{
  int length = (sizeof(bmcs) / sizeof(*bmcs));
  for(int i = 0; i < length; i++)
  {
    if(bmcs[i] == data)
    {
      dsv_strings.desired_bmc = data;
      break;
    }
  }
  if(dsv_strings.desired_bmc.size() == 0)
  {
    std::cout << "invalid bmc: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_wordlength_mode

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_wordlength_mode(std::string data)
{
  int length = (sizeof(wordlength_mode) / sizeof(*wordlength_mode));
  for(int i = 0; i < length; i++)
  {
    if(wordlength_mode[i] == data)
    {
      dsv_strings.desired_wordlength_mode = data;
      break;
    }
  }
  if(dsv_strings.desired_wordlength_mode.size() == 0)
  {
    std::cout << "invalid arithmetic-mode: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_arithmetic_mode

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_arithmetic_mode(std::string data)
{
  int length = (sizeof(arithmetic_mode) / sizeof(*arithmetic_mode));
  for(int i = 0; i < length; i++)
  {
    if(arithmetic_mode[i] == data)
    {
      dsv_strings.desired_arithmetic_mode = data;
      break;
    }
  }
  if(dsv_strings.desired_arithmetic_mode.size() == 0)
  {
    std::cout << "invalid arithmetic-mode: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_connection_mode

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_connection_mode(std::string data)
{
  int length = (sizeof(connections_mode) / sizeof(*connections_mode));
  for(int i = 0; i < length; i++)
  {
    if(connections_mode[i] == data)
    {
      dsv_strings.desired_connection_mode = data;
      break;
    }
  }
  if(dsv_strings.desired_connection_mode.size() == 0)
  {
    std::cout << "invalid connection-mode: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_error

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_error_mode(std::string data)
{
  int length = (sizeof(error_mode) / sizeof(*error_mode));
  for(int i = 0; i < length; i++)
  {
    if(error_mode[i] == data)
    {
      dsv_strings.desired_error_mode = data;
      break;
    }
  }
  if(dsv_strings.desired_error_mode.size() == 0)
  {
    std::cout << "invalid error mode: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_rounding_mode

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_rounding_mode(std::string data)
{
  int length = (sizeof(rounding) / sizeof(*rounding));
  for(int i = 0; i < length; i++)
  {
    if(rounding[i] == data)
    {
      dsv_strings.desired_rounding_mode = data;
      break;
    }
  }
  if(dsv_strings.desired_rounding_mode.size() == 0)
  {
    std::cout << "invalid rounding-mode: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_overflow_mode

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_overflow_mode(std::string data)
{
  int length = (sizeof(overflow) / sizeof(*overflow));
  for(int i = 0; i < length; i++)
  {
    if(overflow[i] == data)
    {
      dsv_strings.desired_overflow_mode = data;
      break;
    }
  }
  if(dsv_strings.desired_overflow_mode.size() == 0)
  {
    std::cout << "invalid overflow-mode: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_selected_realization

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_realization(std::string data)
{
  int length = (sizeof(realizations) / sizeof(*realizations));
  for(int i = 0; i < length; i++)
  {
    if(realizations[i] == data)
    {
      dsv_strings.desired_realization = data;
      break;
    }
  }

  if(dsv_strings.desired_realization.size() == 0)
  {
    std::cout << "invalid realization: " << data << std::endl;
    exit(1);
  }

  bool is_delta_realization = (dsv_strings.desired_realization == "DDFI"
       || dsv_strings.desired_realization == "DDFII" ||
       dsv_strings.desired_realization == "TDDFII");

  if(is_delta_realization)
  {
    extract_data_from_file();
    if(impl.delta == 0)
    {
      std::cout << "invalid delta realization: " << impl.delta << std::endl;
      exit(1);
    }
  }
}

/*******************************************************************
 Function: validate_selected_property

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_selected_property(std::string data)
{
  int length = (sizeof(properties) / sizeof(*properties));
  for(int i = 0; i < length; i++)
  {
    if(properties[i] == data)
    {
      dsv_strings.desired_property = data;
      break;
    }
  }
  if(dsv_strings.desired_property.size() == 0)
  {
    std::cout << "invalid property: " << data << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: validate_filename

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void validate_filename(std::string file)
{
  dsverifier_messaget dsv_msg;
  if(file == "--help" || file == "-h")
  {
    dsv_msg.help();
  }
  else if(file.substr(file.size() - 3, std::string::npos) != ".ss")
  {
    std::string::size_type loc = file.find(".c", 0);
    if(loc == std::string::npos)
    {
      std::cout << file << ": failed to figure out type of file" << std::endl;
      exit(1);
    }
  }
  else
  {
    stateSpaceVerification = true;
  }

  dsv_strings.desired_filename = file;
}

/*******************************************************************
 Function: check_required_parameters

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_required_parameters()
{
  if(dsv_strings.desired_bmc.size() == 0)
  {
    // we use CBMC as our default back-end model-checker
    dsv_strings.desired_bmc = "CBMC";
  }
}

/*******************************************************************
 Function: bind_parameters

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void bind_parameters(int argc, char* argv[])
{
  if(argc == 1)
  {
    dsverifier_messaget dsv_msg;
    dsv_msg.help();
    exit(1);
  }
  validate_filename(argv[1]);
  dsverifier_messaget dsv_msg;
  for(int i = 2; i < argc; ++i)
  {
    if(std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")
    {
      dsverifier_messaget dsv_msg;
      dsv_msg.help();
    }
    else if(std::string(argv[i]) == "--realization")
    {
      if(i + 1 < argc)
        validate_selected_realization(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--property")
    {
      if(i + 1 < argc)
        validate_selected_property(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--x-size")
    {
      if(i + 1 < argc)
        desired_x_size = std::atoi(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--wordlength")
    {
      if(i + 1 < argc)
        validate_selected_wordlength_mode(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--unlimited-x-size")
    {
      k_induction = true;
    }
    else if(std::string(argv[i]) == "--connection-mode")
    {
      if(i + 1 < argc)
        validate_selected_connection_mode(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--error-mode")
    {
      if(i + 1 < argc)
        validate_selected_error_mode(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--rounding-mode")
    {
      if(i + 1 < argc)
        validate_selected_rounding_mode(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--overflow-mode")
    {
      if(i + 1 < argc)
        validate_selected_overflow_mode(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--timeout")
    {
      if(i + 1 < argc)
        dsv_strings.desired_timeout = argv[++i];
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--bmc")
    {
      if(i + 1 < argc)
        validate_selected_bmc(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--arithmetic-mode")
    {
      if(i + 1 < argc)
        validate_selected_arithmetic_mode(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--function")
    {
      if(i + 1 < argc)
        validate_function(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--solver")
    {
      if(i + 1 < argc)
        dsv_strings.desired_solver = argv[++i];
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--limit")
    {
      if(i + 1 < argc)
        desired_quantization_limit = std::stod(argv[++i]);
      else
        dsv_msg.show_required_argument_message(argv[i]);
    }
    else if(std::string(argv[i]) == "--closed-loop")
    {
      closedloop = true;
    }
    else if(std::string(argv[i]) == "--no-fwl")
    {
      nofwl = true;
    }
    else if(std::string(argv[i]) == "--tf2ss")
    {
      translate = true;
    }
    else if(std::string(argv[i]) == "--preprocess")
    {
      preprocess = true;
    }
    else if(std::string(argv[i]) == "--show-ce-data")
    {
      show_counterexample_data = true;
    }
    else
    {
      /* get macro parameters */
      std::string parameter = argv[i];
      std::string::size_type loc = parameter.find("-D", 0);
      if(loc != std::string::npos)
      {
        dsv_strings.desired_macro_parameters += " " + parameter;
        /* check if there is an desired benchmark */
        std::string::size_type find_desired_ds_id =
          parameter.find("-DDS_ID=", 0);
        if(find_desired_ds_id != std::string::npos)
        {
          std::vector<std::string> parts;
          boost::split(parts, parameter, boost::is_any_of("="));
          dsv_strings.desired_ds_id = "DS_ID==" + parts.at(1);
        }
      }
      else
      {
        std::cout << "invalid parameter: " << argv[i] << std::endl;
        exit(1);
      }
    }
  }

  check_required_parameters();
}

/*******************************************************************
 Function: execute_command_line

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string execute_command_line(std::string command)
{
  FILE *pipe = popen(command.c_str(), "r");
  if(!pipe)
    return "ERROR";
  char buffer[128];
  std::string result = "";
  while(!feof(pipe))
  {
    if(fgets(buffer, 128, pipe) != NULL)
    {
      std::cout << buffer;
      result += buffer;
    }
  }
  pclose(pipe);
  return result;
}

/*******************************************************************
 Function: prepare_bmc_command_line

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string prepare_bmc_command_line()
{
  char * dsverifier_home;
  dsverifier_home = getenv("DSVERIFIER_HOME");
  if(dsverifier_home == NULL)
  {
    std::cout << std::endl << "[ERROR] - You must set DSVERIFIER_HOME "
        "environment variable." << std::endl;
    exit(1);
  }
  std::string bmc_path = std::string(dsverifier_home) + "/bmc";
  std::string model_checker_path = std::string(dsverifier_home)
      + "/model-checker";
  std::string command_line;
  if(!(preprocess))
  {
    if(dsv_strings.desired_bmc == "ESBMC")
    {
      if(k_induction)
      {
        command_line = "gcc -E " + dsv_strings.desired_filename
            + " -DK_INDUCTION_MODE=K_INDUCTION -DBMC=ESBMC -I " + bmc_path;
      }
      else
      {
        command_line =
            model_checker_path + "/esbmc " + dsv_strings.desired_filename
                + " --no-bounds-check --no-pointer-check  "
                  "--no-div-by-zero-check -DBMC=ESBMC -I "
                + bmc_path;
      }
      if(dsv_strings.desired_timeout.size() > 0)
      {
        command_line += " --timeout " + dsv_strings.desired_timeout;
      }
    }
    else if(dsv_strings.desired_bmc == "CBMC")
    {
      command_line = model_checker_path + "/cbmc " +
          dsv_strings.desired_filename +
          " --stop-on-fail -DBMC=CBMC -I " + bmc_path;
    }
  }
  else if(preprocess)
  {
    command_line = "gcc -E " + dsv_strings.desired_filename;

    if(dsv_strings.desired_bmc == "ESBMC")
    {
      command_line += " -DBMC=ESBMC -I " + bmc_path;

      if(k_induction)
      {
        command_line += " -DK_INDUCTION_MODE=K_INDUCTION ";
      }
    }
    if(dsv_strings.desired_bmc == "CBMC")
    {
      command_line += " -DBMC=CBMC -I " + bmc_path;
    }
  }

  if(dsv_strings.desired_function.size() > 0)
    command_line += " --function " + dsv_strings.desired_function;

  if(dsv_strings.desired_solver.size() > 0)
  {
    if(!preprocess)
      command_line += " --" + dsv_strings.desired_solver;
  }

  if(dsv_strings.desired_realization.size() > 0)
    command_line += " -DREALIZATION=" + dsv_strings.desired_realization;

  if(dsv_strings.desired_property.size() > 0)
    command_line += " -DPROPERTY=" + dsv_strings.desired_property;

  if(dsv_strings.desired_connection_mode.size() > 0)
    command_line += " -DCONNECTION_MODE=" + dsv_strings.desired_connection_mode;

  if(!dsv_strings.desired_arithmetic_mode.compare("FLOATBV"))
    command_line += " --floatbv -DARITHMETIC=FLOATBV";
  else
    command_line += " --fixedbv -DARITHMETIC=FIXEDBV";

  if(dsv_strings.desired_wordlength_mode.size() > 0)
    command_line += " --" + dsv_strings.desired_wordlength_mode;

  if(dsv_strings.desired_error_mode.size() > 0)
    command_line += " -DERROR_MODE=" + dsv_strings.desired_error_mode;

  if(dsv_strings.desired_rounding_mode.size() > 0)
    command_line += " -DROUNDING_MODE=" + dsv_strings.desired_rounding_mode;

  if(dsv_strings.desired_overflow_mode.size() > 0)
    command_line += " -DOVERFLOW_MODE=" + dsv_strings.desired_overflow_mode;

  if(desired_x_size > 0)
    command_line += " -DX_SIZE=" + std::to_string(desired_x_size);

  command_line += dsv_strings.desired_macro_parameters;

  return command_line;
}

/*******************************************************************
 Function: prepare_bmc_command_line_ss

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

std::string prepare_bmc_command_line_ss()
{
  char * dsverifier_home;
  dsverifier_home = getenv("DSVERIFIER_HOME");
  if(dsverifier_home == NULL)
  {
    std::cout << std::endl
        << "[ERROR] - You must set DSVERIFIER_HOME environment variable."
        << std::endl;
    exit(1);
  }
  std::string command_line;
  std::string bmc_path = std::string(dsverifier_home) + "/bmc";
  std::string model_checker_path = std::string(dsverifier_home)
      + "/model-checker";

  if(dsv_strings.desired_bmc == "ESBMC")
  {
    command_line =
        model_checker_path
            + "/esbmc input.c --no-bounds-check --no-pointer-check "
              "--no-div-by-zero-check --smt-during-symex  --smt-symex-guard --z3 -DBMC=ESBMC -I "
            + bmc_path;

    if(dsv_strings.desired_timeout.size() > 0)
      command_line += " --timeout " + dsv_strings.desired_timeout;
  }
  else if(dsv_strings.desired_bmc == "CBMC")
  {
    command_line = model_checker_path
        + "/cbmc --stop-on-fail input.c -DBMC=CBMC -I " + bmc_path;
  }

  if(dsv_strings.desired_property.size() > 0)
    command_line += " -DPROPERTY=" + dsv_strings.desired_property;

  if(desired_x_size > 0)
    command_line += " -DK_SIZE=" + std::to_string(desired_x_size);

  command_line += dsv_strings.desired_macro_parameters;

  return command_line;
}

digital_system ds;
implementation impl;
filter_parameters filter;

/*******************************************************************
 Function: get_fxp_value

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

int get_fxp_value(std::string exp)
{
  std::vector<std::string> tokens;
  boost::split(tokens, exp, boost::is_any_of("="));

  return std::atoi(tokens[1].c_str());
}

/*******************************************************************
 Function: extract_regexp_data_for_vector

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void extract_regexp_data_for_vector(std::string src, std::regex & regexp,
    std::vector<double> & vector, unsigned int & factor)
{
  double value, value_num, value_den;
  std::sregex_iterator next(src.begin(), src.end(), regexp);
  std::sregex_iterator end;
  while(next != end)
  {
    std::smatch match = *next;
    value_num = (double) get_fxp_value(match.str());
    value_den = (double) factor;
    value = (double)(value_num / value_den);
    vector.push_back(value);
    next++;
  }
}

/*******************************************************************
 Function: print_counterexample_data_for_state_space

 Inputs:

 Outputs:

 Purpose: print counterexample data for state-space systems

 \*******************************************************************/

void print_counterexample_data_for_state_space()
{
  try
  {
    dsverifier_messaget dsv_msg;
    std::cout << std::endl << "Counterexample Data:" << std::endl;

    std::cout << "  Property = " << dsv_strings.desired_property << std::endl;
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Numerator Plant", ds.b, ds.b_size);
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Denominator Plant", ds.a, ds.a_size);
    std::cout << "  Numerator Plant Size = " << ds.b_size << std::endl;
    std::cout << "  Denominator Plant Size = " << ds.a_size << std::endl;

    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Numerator Controller", ds.b, ds.b_size);
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Denominator Controller", ds.a, ds.a_size);
    std::cout << "  Numerator Controller Size = " << ds.b_size << std::endl;
    std::cout << "  Denominator Controller Size = " << ds.a_size << std::endl;

    std::cout << "  X Size = " << desired_x_size << std::endl;
    std::cout << "  Sample Time = " << ds.sample_time << std::endl;
    std::cout << "  Implementation = " << "<" << impl.int_bits << ","
      << impl.frac_bits << ">" << std::endl;
    std::cout << "  Realization = " << dsv_strings.desired_realization
      << std::endl;
    std::cout << "  Dynamic Range = " << "{" << impl.min << "," << impl.max
      << "}" << std::endl;
  } catch(std::regex_error& e)
  {
    std::cout
      << "[ERROR] It was not able to process the counterexample data. :("
      << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: print_counterexample_data_for_closed_loop

 Inputs:

 Outputs:

 Purpose: print counterexample data for closed-loop systems in transfer-function format

 \*******************************************************************/

void print_counterexample_data_for_closed_loop()
{
  try
  {
    dsverifier_messaget dsv_msg;
    std::cout << std::endl << "Counterexample Data:" << std::endl;

    std::cout << "  Property = " << dsv_strings.desired_property << std::endl;
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Numerator Plant", ds.b, ds.b_size);
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Denominator Plant", ds.a, ds.a_size);
    std::cout << "  Numerator Plant Size = " << ds.b_size << std::endl;
    std::cout << "  Denominator Plant Size = " << ds.a_size << std::endl;

    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Numerator Controller", ds.b, ds.b_size);
    dsv_msg.cplus_print_array_elements_ignoring_empty(
        "  Denominator Controller", ds.a, ds.a_size);
    std::cout << "  Numerator Controller Size = " << ds.b_size << std::endl;
    std::cout << "  Denominator Controller Size = " << ds.a_size << std::endl;

    std::cout << "  X Size = " << desired_x_size << std::endl;
    std::cout << "  Sample Time = " << ds.sample_time << std::endl;
    std::cout << "  Implementation = " << "<" << impl.int_bits << ","
        << impl.frac_bits << ">" << std::endl;
    std::cout << "  Realization = " << dsv_strings.desired_realization
      << std::endl;
    std::cout << "  Dynamic Range = " << "{" << impl.min << "," << impl.max
        << "}" << std::endl;
  } catch(std::regex_error& e)
  {
    std::cout
        << "[ERROR] It was not able to process the counterexample data. :("
        << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: print_counterexample_data_for_restricted_properties

 Inputs:

 Outputs:

 Purpose: print counterexample data for stability/minimum phase properties for ESBMC

 \*******************************************************************/

void print_counterexample_data_for_restricted_properties()
{
  try
  {
    dsverifier_messaget dsv_msg;
    std::cout << std::endl << "Counterexample Data:" << std::endl;

    bool is_delta_realization = (dsv_strings.desired_realization == "DDFI" ||
        dsv_strings.desired_realization == "DDFII" ||
        dsv_strings.desired_realization == "TDDFII");

    std::cout << "  Property = " << dsv_strings.desired_property << std::endl;
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Numerator ", ds.b, ds.b_size);
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Denominator ", ds.a, ds.a_size);
    std::cout << "  Numerator Size = " << ds.b_size << std::endl;
    std::cout << "  Denominator Size = " << ds.a_size << std::endl;

    if(is_delta_realization)
      std::cout << "  Delta: " << impl.delta << std::endl;

    std::cout << "  X Size = " << desired_x_size << std::endl;
    std::cout << "  Sample Time = " << ds.sample_time << std::endl;
    std::cout << "  Implementation = " << "<" << impl.int_bits << ","
        << impl.frac_bits << ">" << std::endl;
    std::cout << "  Realization = " << dsv_strings.desired_realization
      << std::endl;
    std::cout << "  Dynamic Range = " << "{" << impl.min << "," << impl.max
        << "}" << std::endl;
  }
  catch(std::regex_error& e)
  {
    std::cout
        << "[ERROR] It was not able to process the counterexample data. :("
        << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: print_counterexample_data

 Inputs:

 Outputs:

 Purpose: print counterexample data for overflow/stability/minimum phase/limit-cycle properties for CBMC

 \*******************************************************************/

void print_counterexample_data(std::string counterexample)
{
  std::vector<double> numerator;
  std::vector<double> denominator;
  std::vector<double> inputs;
  std::vector<double> outputs;
  std::vector<double> initial_states;
  unsigned int factor = pow(2, impl.frac_bits);

  std::size_t verification_failed = counterexample.find("VERIFICATION FAILED");
  if(verification_failed == std::string::npos)
    return;

  if(dsv_strings.desired_bmc == "ESBMC")
  {
    std::cout << std::endl
        << "[INFO] DSVerifier cannot extract the counterexample data."
        << std::endl;
    return;
  }

  try
  {
    /* process quantized numerator data */
    std::regex num_fxp_regexp("b_fxp\\[[0-9]+l?\\]=-?[0-9]+l?");
    extract_regexp_data_for_vector(counterexample, num_fxp_regexp, numerator,
        factor);

    /* process quantized denominator data */
    std::regex den_fxp_regexp("a_fxp\\[[0-9]+l?\\]=-?[0-9]+l?");
    extract_regexp_data_for_vector(counterexample, den_fxp_regexp, denominator,
        factor);

    /* process input data */
    std::regex input_regexp(" x\\[[0-9]+l?\\]=-?[0-9]+l?");
    extract_regexp_data_for_vector(counterexample, input_regexp, inputs,
        factor);

    /* process output data */
    std::regex output_regexp("y\\[[0-9]+l?\\]=-?[0-9]+l?");
    extract_regexp_data_for_vector(counterexample, output_regexp, outputs,
        factor);

    /* process initial states data */
    std::regex initial_states_regexp("y0\\[[0-9]+l?\\]=-?[0-9]+l?");
    extract_regexp_data_for_vector(counterexample, initial_states_regexp,
        initial_states, factor);
    if(initial_states.size() == 0)
    {
      std::regex initial_states_regexp_df2("w0\\[[0-9]+l?\\]=-?[0-9]+l?");
      extract_regexp_data_for_vector(counterexample, initial_states_regexp_df2,
          initial_states, factor);
    }

    bool is_delta_realization = (dsv_strings.desired_realization == "DDFI"
      || dsv_strings.desired_realization == "DDFII" ||
      dsv_strings.desired_realization == "TDDFII");

    // TODO: extend this counterexample data to closed-loop systems
    dsverifier_messaget dsv_msg;
    std::cout << std::endl << "Counterexample Data:" << std::endl;

    std::cout << "  Property = " << dsv_strings.desired_property << std::endl;
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Numerator ", ds.b, ds.b_size);
    dsv_msg.cplus_print_array_elements_ignoring_empty(
      "  Denominator ", ds.a, ds.a_size);

    if(is_delta_realization)
      std::cout << "  Delta: " << impl.delta << std::endl;

    std::cout << "  X Size = " << desired_x_size << std::endl;
    std::cout << "  Sample Time = " << ds.sample_time << std::endl;
    std::cout << "  Implementation = " << "<" << impl.int_bits << ","
        << impl.frac_bits << ">" << std::endl;

    dsv_msg.cplus_print_array_elements_ignoring_empty(
        "  Numerator (fixed-point)",
        &numerator[0], numerator.size());
    dsv_msg.cplus_print_array_elements_ignoring_empty(
        "  Denominator (fixed-point)",
        &denominator[0], denominator.size());

    std::cout << "  Realization = " << dsv_strings.desired_realization
      << std::endl;
    std::cout << "  Dynamic Range = " << "{" << impl.min << "," << impl.max
      << "}" << std::endl;

    dsv_msg.cplus_print_array_elements_ignoring_empty("  Initial States",
        &initial_states[0], initial_states.size());
    dsv_msg.cplus_print_array_elements_ignoring_empty("  Inputs", &inputs[0],
        inputs.size());
    dsv_msg.cplus_print_array_elements_ignoring_empty("  Outputs",
        &outputs[inputs.size()], outputs.size() - inputs.size());
  } catch(std::regex_error& e)
  {
    std::cout
        << "[ERROR] It was not able to process the counterexample data. :("
        << std::endl;
    exit(1);
  }
}

/*******************************************************************
 Function: get_roots_from_polynomial

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

int get_roots_from_polynomial(double polynomial[], int poly_size,
    std::vector<RootTypet> & roots)
{
  unsigned int size = poly_size;

  /* coefficients */
  std::vector<double> coefficients_vector;
  for(int i = 0; i < poly_size; i++)
    coefficients_vector.push_back(polynomial[i]);

  /* remove leading zeros */
  std::vector<double>::iterator it = coefficients_vector.begin();
  while(it != coefficients_vector.end())
  {
    if(*it != 0.0)
      break;
    it = coefficients_vector.erase(it);
  }

  size = coefficients_vector.size();

  /* check if there is any element left on the vector */
  if(!size)
  {
    std::cout << std::endl << "No remaining elements in polynomial vector";
    throw std::runtime_error("tla");
  }

  /* check the polynomial order */
  if(coefficients_vector.size() >= 3)
  {
    Eigen::VectorXd coefficients(coefficients_vector.size());

    /* copy elements from the list to the array - insert in reverse order */
    unsigned int i = 0;
    for(it = coefficients_vector.begin(); it != coefficients_vector.end();
        ++it, ++i)
    {
      coefficients[size - i - 1] = *it;
    }

    /* eigen solver object */
    Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;

    /* solve denominator using QR decomposition */
    solver.compute(coefficients);

    RootsTypet solver_roots = solver.roots();

    for(unsigned int i = 0; i < solver_roots.rows(); ++i)
      roots.push_back(solver_roots[i]);
  }
  else if(coefficients_vector.size() == 2)
  {
    double root = -coefficients_vector.at(1) / coefficients_vector.at(0);
    roots.push_back(root);
  }

  return 0;
}

/*******************************************************************
 Function: check_delta_stability_margin

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool check_delta_stability_margin(std::vector<RootTypet> roots)
{
  std::cout << "checking delta stability margin" << std::endl;
  bool stable = true;
  for(unsigned int i = 0; i < roots.size(); i++)
  {
    std::complex<double> eig = roots.at(i);
    eig.real(eig.real() * impl.delta);
    eig.imag(eig.imag() * impl.delta);
    eig.real(eig.real() + 1);
    if((std::abs(eig) < 1) == false)
    {
      stable = false;
      break;
    }
  }

  return stable;
}

/*******************************************************************
 Function: check_shift_stability_margin

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool check_shift_stability_margin(std::vector<RootTypet> roots)
{
  std::cout << "checking shift stability margin" << std::endl;
  bool stable = true;
  for(unsigned int i = 0; i < roots.size(); i++)
  {
    std::complex<double> eig = roots.at(i);
    if((std::abs(eig) < 1) == false)
    {
      stable = false;
      break;
    }
  }

  return stable;
}

/*******************************************************************
 Function: show_implementation_parameters

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void show_implementation_parameters()
{
  std::cout << std::endl << "implementation int_bits: " << impl.int_bits
      << std::endl;
  std::cout << "implementation frac_bits: " << impl.frac_bits << std::endl;
  std::cout << "implementation delta: " << impl.delta << std::endl;
}

/*******************************************************************
 Function: check_stability_shift_domain_using_jury

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_stability_shift_domain_using_jury()
{
  dsverifier_messaget dsv_msg;
  show_implementation_parameters();
  std::cout << std::endl;
  double sa_fxp[MAX_DSORDER];
  dsv_msg.cplus_print_array_elements("original denominator",
    ds.a, ds.a_size);
  fxp_t a_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  fxp_to_double_array(sa_fxp, a_fxp, ds.a_size);
  dsv_msg.cplus_print_array_elements("quantized denominator",
    sa_fxp, ds.a_size);
  bool is_stable = check_stability(sa_fxp, ds.a_size);
  if(is_stable)
  {
    dsv_msg.show_verification_successful();
  }
  else
  {
    dsv_msg.show_verification_failed();
  }
}

/*******************************************************************
 Function: check_minimum_phase_shift_domain_using_jury

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_minimum_phase_shift_domain_using_jury()
{
  dsverifier_messaget dsv_msg;
  show_implementation_parameters();
  std::cout << std::endl;
  double sb_fxp[MAX_DSORDER];
  dsv_msg.cplus_print_array_elements("original numerator",
    ds.b, ds.b_size);
  fxp_t b_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
  fxp_to_double_array(sb_fxp, b_fxp, ds.b_size);
  dsv_msg.cplus_print_array_elements("quantized denominator",
    sb_fxp, ds.b_size);
  bool is_stable = check_stability(sb_fxp, ds.b_size);
  if(is_stable)
  {
    dsv_msg.show_verification_successful();
  }
  else
  {
    dsv_msg.show_verification_failed();
  }
}

/*******************************************************************
 Function: check_stability_shift_domain_using_eigen

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_stability_shift_domain_using_eigen()
{
  dsverifier_messaget dsv_msg;
  show_implementation_parameters();
  std::cout << std::endl;
  double sa_fxp[MAX_DSORDER];
  dsv_msg.cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  fxp_t a_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  fxp_to_double_array(sa_fxp, a_fxp, ds.a_size);
  dsv_msg.cplus_print_array_elements("quantized denominator",
    sa_fxp, ds.a_size);
  std::vector<RootTypet> poly_roots;
  get_roots_from_polynomial(sa_fxp, ds.a_size, poly_roots);
  bool is_stable = check_shift_stability_margin(poly_roots);

  if(is_stable)
    dsv_msg.show_verification_successful();
  else
    dsv_msg.show_verification_failed();
}

/*******************************************************************
 Function: check_minimum_phase_shift_domain

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_minimum_phase_shift_domain()
{
  dsverifier_messaget dsv_msg;
  show_implementation_parameters();
  std::cout << std::endl;
  double sb_fxp[MAX_DSORDER];
  dsv_msg.cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  fxp_t b_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
  fxp_to_double_array(sb_fxp, b_fxp, ds.b_size);
  dsv_msg.cplus_print_array_elements("quantized numerator", sb_fxp, ds.b_size);
  bool is_stable = check_stability(sb_fxp, ds.b_size);

  if(is_stable)
    dsv_msg.show_verification_successful();
  else
    dsv_msg.show_verification_failed();
}

/*******************************************************************
 Function: check_stability_delta_domain

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_stability_delta_domain()
{
  dsverifier_messaget dsv_msg;
  show_implementation_parameters();
  std::cout << std::endl;
  double db[MAX_DSORDER];
  double da[MAX_DSORDER];
  fxp_t a_fxp[MAX_DSORDER];
  dsv_msg.cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  dsv_msg.cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  get_delta_transfer_function_with_base(ds.b, db, ds.b_size, ds.a, da,
      ds.a_size, impl.delta);
  dsv_msg.cplus_print_array_elements("delta denominator", da, ds.a_size);
  dsv_msg.cplus_print_array_elements("delta numerator", db, ds.b_size);
  fxp_t da_fxp[MAX_DSORDER];
  try
  {
    fxp_double_to_fxp_array(da, da_fxp, ds.a_size);
  } catch(int e)
  {
    std::cout
        << "An overflow occurs after delta transformation"
        << std::endl;
    dsv_msg.show_verification_failed();
    exit(1);
  }
  if((da[0] != 0) && (da_fxp[0] == 0))
  {
    std::cout << std::endl;
    std::cout << "ds.a[0] = " << std::to_string(da[0]) << " ----> "
        << std::to_string(da_fxp[0]) << std::endl;
    dsv_msg.show_delta_not_representable();
    dsv_msg.show_underflow_message();
    dsv_msg.show_verification_error();
    exit(1);
  }
  double da_qtz[MAX_DSORDER];
  fxp_to_double_array(da_qtz, da_fxp, ds.a_size);
  dsv_msg.cplus_print_array_elements("quantized delta denominator",
    da_qtz, ds.a_size);
  std::vector<RootTypet> poly_roots;
  get_roots_from_polynomial(da_qtz, ds.a_size, poly_roots);
  bool is_stable = check_delta_stability_margin(poly_roots);

  if(is_stable)
    dsv_msg.show_verification_successful();
  else
    dsv_msg.show_verification_failed();
}

/*******************************************************************
 Function: check_if_file_exists

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

bool check_if_file_exists(const std::string & name)
{
  if(FILE *file = fopen(name.c_str(), "r"))
  {
    fclose(file);
    return true;
  }
  else
  {
    return false;
  }
}

/*******************************************************************
 Function: check_minimum_phase_delta_domain

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_minimum_phase_delta_domain()
{
  dsverifier_messaget dsv_msg;
  show_implementation_parameters();
  std::cout << std::endl;
  double db[MAX_DSORDER];
  double da[MAX_DSORDER];
  dsv_msg.cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  dsv_msg.cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  get_delta_transfer_function_with_base(ds.b, db, ds.b_size, ds.a, da,
      ds.a_size, impl.delta);
  dsv_msg.cplus_print_array_elements("delta numerator", db, ds.b_size);
  dsv_msg.cplus_print_array_elements("delta denominator", da, ds.a_size);
  fxp_t db_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(db, db_fxp, ds.b_size);

  if((db[0] != 0) && (db_fxp[0] == 0))
  {
    std::cout << std::endl;
    std::cout << "ds.b[0] = " << std::to_string(db[0]) << " ----> "
        << std::to_string(db_fxp[0]) << std::endl;
    dsv_msg.show_delta_not_representable();
    dsv_msg.show_underflow_message();
    dsv_msg.show_verification_error();
    exit(1);
  }

  double db_qtz[MAX_DSORDER];
  fxp_to_double_array(db_qtz, db_fxp, ds.b_size);
  dsv_msg.cplus_print_array_elements("quantized delta numerator",
    db_qtz, ds.b_size);
  std::vector<RootTypet> poly_roots;
  get_roots_from_polynomial(db_qtz, ds.b_size, poly_roots);
  bool is_stable = check_delta_stability_margin(poly_roots);

  if(is_stable)
    dsv_msg.show_verification_successful();
  else
    dsv_msg.show_verification_failed();
}

// VERIFICATION OF SETTLING TIME


/*******************************************************************
 Function: y_k

 Inputs:

 Outputs:

 Purpose: Calculate the system's output

 \*******************************************************************/
double y_k(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
        Eigen::MatrixXd D, double u, int k, Eigen::MatrixXd x0)
{
  Eigen::MatrixXd y;
  y = C * A.pow(k) * x0;
  for(int m = 0; m <= (k - 1); m++)
  {
    y += (C * A.pow(k - m - 1) * B * u) + D * u;
  }
  return y(0, 0);
}

/*******************************************************************
 Function: y_ss

 Inputs:

 Outputs:

 Purpose: Calculate the steady state output

 \*******************************************************************/
double y_ss(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
        Eigen::MatrixXd D, double u)
{
  double yss;
  Eigen::MatrixXd AUX;
  Eigen::MatrixXd AUX2;
  Eigen::MatrixXd AUX3;
  Eigen::MatrixXd Id;

  // get the expression y_ss=(C(I-A)^(-1)B+D)u
  Id.setIdentity(A.rows(), A.cols());
  AUX = Id - A;
  AUX3 = AUX.inverse();

  AUX2 = (C * AUX3 * B + D);
  yss = AUX2(0, 0) * u;

  return yss;
}

/*******************************************************************
 Function: isSameSign

 Inputs:

 Outputs:

 Purpose: Check if two variables are both positive or both negative

 \*******************************************************************/
bool isSameSign(double a, double b)
{
  if(((a >= 0) && (b >= 0)) || ((a <= 0) && (b <= 0)))
    return true;
  else
    return false;
}

/*******************************************************************
 Function: check_state_space_stability

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/
int check_state_space_stability()
{
  Eigen::MatrixXd matrixA(_controller.nStates,_controller.nStates);
  int i, j;

  for(i = 0; i < _controller.nStates; ++i)
  {
    for(j = 0; j < _controller.nStates; ++j)
    {
      matrixA(i, j) = (double)_controller.A[i][j];
    }
  }

  std::complex<double> lambda;
  std::complex<double> margem(1, 0);
  double v;

  dsverifier_messaget dsv_msg;
  Eigen::VectorXcd eivals = matrixA.eigenvalues();
  for(i = 0; i < _controller.nStates; ++i)
  {
    lambda = eivals[i];
    v = std::sqrt(lambda.real()*lambda.real()+lambda.imag()*lambda.imag());

    if(v > 1.0)
    {
      std::cout << "unstable: " << std::endl;
      return 0; // unstable system
    }
  }
  return 1; // stable system
}

/*******************************************************************
 Function: isEigPos

 Inputs:

 Outputs:

 Purpose: Check if the eigenvalues are real and positive

 \*******************************************************************/
bool isEigPos(Eigen::MatrixXd A)
{
  int isStable = check_state_space_stability();
  std::complex<double> lambda;
  bool status;
  Eigen::VectorXcd eivals = A.eigenvalues();
  for(int i = 0; i < _controller.nStates; ++i)
  {
    lambda = eivals[i];
    if((lambda.real() >= 0) && (lambda.imag() == 0))
      status = true;
    else
    {
      status = false;
      break;
    }
  }
  if((isStable == 1) && (status == true))
    return true;
  else
    return false;
}


/*******************************************************************
 Function: peak_output

 Inputs:

 Outputs:

 Purpose: Calculate the first peak value of the output

 \*******************************************************************/
void peak_output(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
      Eigen::MatrixXd D, Eigen::MatrixXd x0, double *out, double yss, double u, double p)
{
  double greater, cmp, o, v, inf, sup;
  int i = 0, dim;
  dim = _controller.nStates;
  greater = fabs(y_k(A, B, C, D, u, i, x0));
  o = y_k(A, B, C, D, u, i+1, x0);
  cmp = fabs(o);
  if(isEigPos(A) == true)// In the future this block will not be necessary
  {
	v = y_k(A, B, C, D, u, i, x0);
	if(yss > 0)
	{
	  inf = (yss - (yss * (p/100)));
	  sup = (yss * (p/100) + yss);
	}
	else
	{
	  sup = (yss - (yss * (p/100)));
	  inf = (yss * (p/100) + yss);
	}
    while(!((v < sup) && (v > inf)))
    {
      ++i;
      v = y_k(A, B, C, D, u, i, x0);
    }
	out[1] = v;
	out[0] = i+1;
  }
  else
  {
    while((cmp >= greater))
    {
      if(greater < cmp)
      {
        greater = cmp;
        out[1] = o;
        out[0] = i+2;
      }
      else
      {
        out[1] = o;
        out[0] = i+2;
      }
      if(!isSameSign(yss, out[1]))
      {
        greater = 0;
      }
      ++i;
      o = y_k(A, B, C, D, u, i+1, x0);
      cmp = fabs(o);
    }
  }
}

/*******************************************************************
 Function: cplxMag

 Inputs:

 Outputs:

 Purpose: Get the magnitude of a complex number

 \*******************************************************************/
double cplxMag(double real, double imag)
{
  return sqrt(real * real + imag * imag);
}

/*******************************************************************
 Function: maxMagEigVal

 Inputs:

 Outputs:

 Purpose: Calculate the magnitude of the maximum eigenvalue

 \*******************************************************************/
double maxMagEigVal(Eigen::MatrixXd A)
{
  double _real, _imag;
  double maximum = 0, aux;

  Eigen::VectorXcd eivals = A.eigenvalues();
  for(int i = 0; i < _controller.nStates; ++i)
  {
    _real = eivals[i].real();
    _imag = eivals[i].imag();
    aux = cplxMag(_real, _imag);
    if(aux > maximum)
    {
      maximum = aux;
    }
  }
  return maximum;
}

/*******************************************************************
 Function: c_bar

 Inputs:

 Outputs:

 Purpose: Calculate the variable c_bar needed to check settling time

 \*******************************************************************/
double c_bar(double yp, double yss, double lambmax, int kp)
{
  double cbar;
  cbar = (yp-yss)/(pow(lambmax, kp));
  return cbar;
}

/*******************************************************************
 Function: log_b

 Inputs:

 Outputs:

 Purpose: Calculate the log

 \*******************************************************************/
double log_b(double base, double x)
{
  return (double) (log(x) / log(base));
}

/*******************************************************************
 Function: k_bar

 Inputs:

 Outputs:

 Purpose: Calculate the variable k_bar needed to check settling time

 \*******************************************************************/
int k_bar(double lambdaMax, double p, double cbar, double yss, int order)
{
  double k_ss, x;
  x = fabs((p * yss) / (100 * cbar));
  k_ss = log_b(lambdaMax, x);
  return ceil(k_ss)+order;
}

/*******************************************************************
 Function: check_settling_time

 Inputs:

 Outputs:

 Purpose: Check if a given settling time satisfies to a system

 \*******************************************************************/
int check_settling_time(Eigen::MatrixXd A, Eigen::MatrixXd B,
        Eigen::MatrixXd C, Eigen::MatrixXd D, Eigen::MatrixXd x0,
        double u, double tsr, double p, double ts)
{
  double peakV[2];
  double yss, yp, lambMax, cbar, output, inf, sup, v;
  int kbar, kp, i = 0;
  yss = y_ss(A, B, C, D, u);
  if(yss > 0)
  {
    inf = (yss - (yss * (p/100)));
    sup = (yss * (p/100) + yss);
  }
  else
  {
    sup = (yss - (yss * (p/100)));
    inf = (yss * (p/100) + yss);
  }
/*  if(isEigPos(A) == true)
  {
	v = y_k(A, B, C, D, u, i, x0);
    while(!((v < sup) && (v > inf)))
    {
      ++i;
      v = y_k(A, B, C, D, u, i, x0);
    }
	kbar = i+1;
	if(tsr < kbar * ts)
      return 0;
  }
  else
  {*/
    peak_output(A, B, C, D, x0, peakV, yss, u, p);
    yp = (double) peakV[1];
    kp = (int) peakV[0];
    lambMax = maxMagEigVal(A);
    std::cout << "lambMax=" << lambMax << std::endl;
    std::cout << "yp=" << yp << std::endl;
    std::cout << "kp=" << kp << std::endl;
    std::cout << "yss=" << yss << std::endl;

    if(isEigPos(A) == true)
    {
      std::cout << "cbar=N/A" << std::endl;
      kbar = kp;
    }
    else
    {
      cbar = c_bar(yp, yss, lambMax, kp);
      std::cout << "cbar=" << cbar << std::endl;
      kbar = k_bar(lambMax, p, cbar, yss, (int)A.rows());
    }
    if(!(kbar * ts < tsr))
    {
      i = ceil(tsr / ts);
      output = y_k(A, B, C, D, u, i-1, x0);
      while(i <= kbar)
      {
  	    if(!((output <= sup) && (output >= inf)))
        {
          std::cout << "kbar=" << kbar << std::endl;
          return 0;
        }
        ++i;
        output = y_k(A, B, C, D, u, i-1, x0);
      }
    }
//  }
  std::cout << "kbar=" << kbar << std::endl;
  return 1;
}

/*******************************************************************
 Function: verify_settling_time

 Inputs:

 Outputs:

 Purpose: Verify the settling time property

 \*******************************************************************/
void verify_state_space_settling_time(void)
{
  double peakV[2];
  double yss, yp, tp, lambMax, cbar, ts, tsr, p, u;
  int i, j, kbar, k_ss;
  dsverifier_messaget dsv_msg;
  _controller_fxp = _controller;

  tsr = _controller.tsr;

  ts = _controller.ts;

  p = _controller.p;

  u = (double)_controller.inputs[0][0];

  Eigen::MatrixXd A(_controller.nStates, _controller.nStates);
  Eigen::MatrixXd B(_controller.nStates, 1);
  Eigen::MatrixXd C(1, _controller.nStates);
  Eigen::MatrixXd D(1, 1);
  Eigen::MatrixXd x0(_controller.nStates, 1);

  for(i = 0; i < _controller.nStates; ++i)
  {
    for(j = 0; j < _controller.nStates; ++j)
    {
      A(i, j) = _controller.A[i][j];
    }
  }

  for(i = 0; i < _controller.nStates; ++i)
  {
    for(j = 0; j < 1; ++j)
    {
      B(i, j) = _controller.B[i][j];
    }
  }

  for(i = 0; i < 1; ++i)
  {
    for(j = 0; j < _controller.nStates; ++j)
    {
      C(i, j) = _controller.C[i][j];
    }
  }

  for(i = 0; i < 1; ++i)
  {
    for(j = 0; j < 1; ++j)
    {
      D(i, j) = _controller.D[i][j];
    }
  }

  for(i = 0; i < _controller.nStates; ++i)
  {
    for(j = 0; j < 1; ++j)
    {
      x0(i, j) = _controller.x0[i][j];
    }
  }

  int isStable = check_state_space_stability();
  if(isStable)
  {
    if(!check_settling_time(A, B, C, D, x0, u, tsr, p, ts))
    {
      dsv_msg.show_verification_failed();
      exit(0);
    }
    else
    {
      dsv_msg.show_verification_successful();
    }
  }
  else
  {
    std::cout << "The system is unstable."<< std::endl;

    Eigen::VectorXcd eivals = A.eigenvalues();
    std::cout << "The eigenvalues of A:" << std::endl << eivals << std::endl;
    dsv_msg.show_verification_failed();
    exit(0);
  }
}

/*******************************************************************
 Function: verify_state_space_stability

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/
void verify_state_space_stability()
{
  int isStable = check_state_space_stability();
  dsverifier_messaget dsv_msg;

  if(isStable)
  {
    dsv_msg.show_verification_successful(); // stable system
  }
  else
  {
  dsv_msg.show_verification_failed(); // unstable system
  exit(0);
  }
}

/*******************************************************************
 Function: generates_mag_response

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define LOWPASS 1
#define HIGHPASS 2
#define PASSBAND 3
#define SINE_precision 7

void generates_mag_response(double* num, int lnum, double* den, int lden,
    double* res, int N)
{
  double w;
  int m, i;
  double out_numRe[N + 1];
  double out_numIm[N + 1];
  double out_denRe[N + 1];
  double out_denIm[N + 1];
  double old_out_Re;
  double zero_test;
  for(w = 0, i = 0; w <= M_PI; w += M_PI / N, ++i)
  {
    out_numRe[i] = num[0];
    out_numIm[i] = 0;
    for(m = 1; m < lnum; ++m)
    {
      old_out_Re = out_numRe[i];
      out_numRe[i] = cosTyl(w, SINE_precision) * out_numRe[i]
          - sinTyl(w, SINE_precision) * out_numIm[i] + num[m];
      out_numIm[i] = sinTyl(w, SINE_precision) * old_out_Re
          + cosTyl(w, SINE_precision) * out_numIm[i];
    }
    out_denRe[i] = den[0];
    out_denIm[i] = 0;

    for(m = 1; m < lden; ++m)
    {
      old_out_Re = out_denRe[i];
      out_denRe[i] = cosTyl(w, SINE_precision) * out_denRe[i]
          - sinTyl(w, SINE_precision) * out_denIm[i] + den[m];
      out_denIm[i] = sinTyl(w, SINE_precision) * old_out_Re
          + cosTyl(w, SINE_precision) * out_denIm[i];
    }

    res[i] = sqrt(out_numRe[i] * out_numRe[i] + out_numIm[i] * out_numIm[i]);
    zero_test = sqrt(out_denRe[i] * out_denRe[i] + out_denIm[i] * out_denIm[i]);
    res[i] = res[i] / zero_test;
  }
}

/*******************************************************************
 Function: check_filter_magnitude_det

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_filter_magnitude_det()
{
  int freq_response_samples = 10000;
  double w;
  double w_incr = 1.0 / freq_response_samples;
  double res[freq_response_samples + 1];
  double _res[freq_response_samples + 1];
  int i, j;
  int cuttof_freq_index;
  bool response_is_valid = true;
  /* generates magnitude response of the quantized TF,
   * placing the result in the "res" array */
  generates_mag_response(ds.b, ds.b_size, ds.a, ds.a_size, res,
      freq_response_samples);

  /* quantize "a" array using fxp */
  fxp_t a_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  double _a[MAX_DSORDER];
  fxp_to_double_array(_a, a_fxp, ds.a_size);

  /* quantize "b" array using fxp */
  fxp_t b_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
  double _b[MAX_DSORDER];
  fxp_to_double_array(_b, b_fxp, ds.b_size);

  /* generates magnitude response of the quantized TF,
   * placing the result in the "_res" array */
  generates_mag_response(_b, ds.b_size, _a, ds.a_size, _res,
      freq_response_samples);

  printf("\nOriginal floating-point input coefficients are");

  printf("\n a= {");
  for(i = 0; i < ds.a_size; i++)
    printf("%.60lf, ", ds.a[i]);
  printf("}\n");

  printf("\n b= {");
  for(i = 0; i < ds.b_size; i++)
    printf("%.60lf, ", ds.b[i]);
  printf("}\n");

  printf("\nNew coefficients after fixed-point conversion are");

  printf("\n _a= {");
  for(i = 0; i < ds.a_size; i++)
    printf("%.60lf, ", _a[i]);
  printf("}\n");

  printf("\n _b= {");
  for(i = 0; i < ds.a_size; i++)
    printf("%.60lf, ", _b[i]);
  printf("}\n");

  if(filter.type == LOWPASS)
  {
    // lowpass
    if((filter.wp == 0) && (filter.wr == 0))
    {
      filter.wp = filter.wc - w_incr;
      filter.wr = filter.wc + w_incr;
    }

    if(filter.Ar == 0)
      filter.Ar = 1;

    for(i = 0, w = 0; (w <= 1.0); ++i, w += w_incr)
    {
      printf("sample: %d\n", i);
      printf("w= %f res = %.32lf\n", w, res[i]);
      printf("w=  %f _res= %.32lf\n", w, _res[i]);
      if((w < filter.wp) || (doubleComparisson(filter.wp, w, 0.0000001)))
      {
        if(!(_res[i] >= filter.Ap))
        {
          printf("|----------------Passband Failure-------------|");
          response_is_valid = false;
          break;
        }
      }
      if(doubleComparisson(filter.wc, w, 0.000001) && (filter.wc != 0))
      {
        if(!((_res[i]) < (filter.Ac)))
        {
          printf("|-------------Cutoff Frequency Failure--------|");
          response_is_valid = false;
          break;
        }
      }
      if(((w > filter.wr) || (doubleComparisson(filter.wr, w, 0.0000001)))
          && (w <= 1) && (filter.wr != 0))
      {
        if(!(_res[i] <= filter.Ar))
        {
          printf("|----------------Stopband Failure-------------|");
          response_is_valid = false;
          break;
        }
      }
    }
  }
  else if(filter.type == HIGHPASS)
  {
    // highpass
    if((filter.wp == 0) && (filter.wr == 0))
    {
      filter.wp = filter.wc + w_incr;
      filter.wr = filter.wc - w_incr;
    }

    if(filter.Ar == 0)
      filter.Ar = 1;

    for(i = 0, w = 0; (w <= 1.0); ++i, w += w_incr)
    {
      printf("sample: %d\n", i);
      printf("w= %f res = %.32lf\n", w, res[i]);
      printf("w=  %f _res= %.32lf\n", w, _res[i]);
      if((w < filter.wr) || (doubleComparisson(filter.wr, w, 0.0000001)))
      {
        if(!(_res[i] <= filter.Ar))
        {
          printf("|----------------Stopband Failure-------------|");
          response_is_valid = false;
          break;
        }
      }
      if(doubleComparisson(filter.wc, w, 0.0000001) && (filter.wc != 0))
      {
        if(!((_res[i]) < (filter.Ac)))
        {
          printf("|-------------Cutoff Frequency Failure--------|");
          response_is_valid = false;
          break;
        }
      }
      if(((w > filter.wp) || (doubleComparisson(filter.wp, w, 0.0000001)))
          && (w <= 1) && (filter.wr != 0))
      {
        if(!(_res[i] >= filter.Ap))
        {
          printf("|----------------Passband Failure-------------|");
          response_is_valid = false;
          break;
        }
      }
    }
  }
  else if(filter.type == PASSBAND)
  {
    if(filter.Ar == 0)
      filter.Ar = 1;

    printf("wc = %.32lf\n", filter.wc);
    printf("w1r = %.32lf\n", filter.w1r);
    printf("w2r = %.32lf\n", filter.w2r);
    printf("w1c = %.32lf\n", filter.w1c);
    printf("w2c = %.32lf\n", filter.w2c);
    printf("w1p = %.32lf\n", filter.w1p);
    printf("w2p = %.32lf\n", filter.w2p);

    for(i = 0, w = 0; (w <= 1.0); ++i, w += w_incr)
    {
      printf("sample: %d\n", i);
      printf("w= %f res = %.32lf\n", w, res[i]);
      printf("w=  %f _res= %.32lf\n", w, _res[i]);
      if(((w < filter.w1r) || (doubleComparisson(filter.w1r, w, 0.0000001)))
          && (filter.w1r != 0))
      {
        if(!(_res[i] <= filter.Ar))
        {
          printf("filter.w1r = %f \n", filter.w1r);
          printf("|----------------Stopband Failure1-------------|");
          response_is_valid = false;
          break;
        }
      }
      if(((w < filter.w1c) || (doubleComparisson(filter.w1c, w, 0.0000001)))
          && ((w > (filter.w1c - w_incr))
              || (doubleComparisson(filter.w1c - w_incr, w, 0.0000001)))
          && (filter.w1c != 0))
      {
        printf(" Entrou em wc1 = %f\n", w);
        if(!(_res[i] <= filter.Ac))
        {
          printf("|-------------Cutoff Frequency Failure1--------|");
          response_is_valid = false;
          break;
        }
      }
      if(((w > filter.w1p) || (doubleComparisson(filter.w1p, w, 0.0000001)))
          && ((w < filter.w2p) ||
          (doubleComparisson(filter.w2p, w, 0.0000001))))
      {
        if(!(_res[i] >= filter.Ap))
        {
          printf("|----------------Passband Failure1-------------|");
          response_is_valid = false;
          break;
        }
      }
      if(((w > filter.w2c) || (doubleComparisson(filter.w2c, w, 0.0000001)))
          && (w < (filter.w2c + w_incr)
              || (doubleComparisson(filter.w2c + w_incr, w, 0.0000001)))
          && (filter.w2c != 0))
      {
        printf(" Entrou em wc2 = %f\n", w);
        if(!(_res[i] <= filter.Ac))
        {
          printf("|-------------Cutoff Frequency Failure2--------|");
          response_is_valid = false;
          break;
        }
      }
      if(((w > filter.w2r)) && (w <= 1) && (filter.w2r != 0))
      {
        if(!(_res[i] <= filter.Ar))
        {
          printf("|----------------Stopband Failure2-------------|");
          response_is_valid = false;
          break;
        }
      }
    }
  }

  dsverifier_messaget dsv_msg;

  if(response_is_valid == false)
    dsv_msg.show_verification_failed();
  else
    dsv_msg.show_verification_successful();
}

/*******************************************************************
 Function: generates_mag_response

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void resp_phase(double* num, int lnum, double* den, int lden, double* res,
    int N)
{
  double w;
  int m, i;
  double out_numRe[N + 1], old_out_r;
  double out_numIm[N + 1];
  double out_denRe[N + 1], out_denIm[N + 1];
  for(w = 0, i = 0; w <= M_PI; w += M_PI / N, ++i)
  {
    out_numRe[i] = num[0];
    out_numIm[i] = 0;
    for(m = 1; m < lnum; ++m)
    {
      old_out_r = out_numRe[i];
      out_numRe[i] = cos(w) * out_numRe[i] - sin(w) * out_numIm[i] + num[m];
      out_numIm[i] = sin(w) * old_out_r + cos(w) * out_numIm[i];
    }

    out_denRe[i] = den[0];
    out_denIm[i] = 0;
    for(m = 1; m < lden; ++m)
    {
      old_out_r = out_denRe[i];
      out_denRe[i] = cos(w) * out_denRe[i] - sin(w) * out_denIm[i] + den[m];
      out_denIm[i] = sin(w) * old_out_r + cos(w) * out_denIm[i];
    }

    res[i] = atan(out_numIm[i] / out_numRe[i]); // numerator abs
    res[i] = res[i] - atan(out_denIm[i] / out_denRe[i]); // den abs
  }
}

/*******************************************************************
 Function: check_filter_phase_det

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_filter_phase_det(void)
{
  int freq_response_samples = 100;
  double w;
  double w_incr = 1.0 / freq_response_samples;
  double res[freq_response_samples + 1];
  double _res[freq_response_samples + 1];
  int i, j;
  bool response_is_valid = true;

  /* quantize "a" array using fxp */
  fxp_t a_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  double _a[MAX_DSORDER];
  fxp_to_double_array(_a, a_fxp, ds.a_size);

  /* quantize "b" array using fxp */
  fxp_t b_fxp[MAX_DSORDER];
  fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
  double _b[MAX_DSORDER];
  fxp_to_double_array(_b, b_fxp, ds.b_size);

  /* generates magnitude response of the floating point TF,
   * placing the result in the "res" array */
  resp_phase(ds.b, ds.b_size, ds.a, ds.a_size, res, freq_response_samples);

  /* generates magnitude response of the quantized TF,
   * placing the result in the "_res" array */
  resp_phase(_b, ds.b_size, _a, ds.a_size, _res, freq_response_samples);

  /* generates magnitude response, placing the result in the "res" array*/

  float diff = 0.3;

  for(i = 0, w = 0; (w <= 1.0); ++i, w += w_incr)
  {
    if(!(fabs(res[i] - _res[i]) <= diff))
    {
      printf("|-------------Phase Failure------------|");
      response_is_valid = false;
      break;
    }
  }

  dsverifier_messaget dsv_msg;

  if(response_is_valid == false)
    dsv_msg.show_verification_failed();
  else
    dsv_msg.show_verification_successful();
}

/*******************************************************************
 Function: check_file_exists

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void check_file_exists()
{
  /* check if the specified file exists */
  if(check_if_file_exists(dsv_strings.desired_filename) == false)
  {
    std::cout << "file " << dsv_strings.desired_filename
      << ": failed to open input file" << std::endl;
    exit(1);
  }
}

template<typename T>
bool isNumber(T x){
   std::string s;
   std::stringstream ss;
   ss << x;
   ss >>s;
   if(s.empty() || std::isspace(s[0]) || std::isalpha(s[0])) return false ;
   char * p ;
   strtod(s.c_str(), &p) ;
   return (*p == 0) ;
}

// Map the different operators: +, -, *, / etc
typedef std::map< std::string, std::pair< int,int >> OpMap;
typedef std::vector<std::string>::const_iterator cv_iter;
typedef std::string::iterator s_iter;

const OpMap::value_type assocs[] =
    {  OpMap::value_type( "+", std::make_pair<int,int>(0, int(LEFT_ASSOC))),
       OpMap::value_type( "-", std::make_pair<int,int>(0, int(LEFT_ASSOC))),
       OpMap::value_type( "*", std::make_pair<int,int>(5, int(LEFT_ASSOC))),
       OpMap::value_type( "/", std::make_pair<int,int>(5, int(LEFT_ASSOC)))};

const OpMap opmap( assocs, assocs + sizeof( assocs ) / sizeof( assocs[ 0 ] ) );

// Test if token is an pathensesis
bool isParenthesis( const std::string& token)
{
  return token == "(" || token == ")";
}

// Test if token is an operator
bool isOperator( const std::string& token)
{
  return token == "+" || token == "-" ||
         token == "*" || token == "/";
}

// Test associativity of operator token
bool isAssociative( const std::string& token, const int& type)
{
  const std::pair<int,int> p = opmap.find( token )->second;
  return p.second == type;
}

bool isLetters(std::string input)
{
  int uppercaseChar;
  for (int i = 0; i < input.size(); i++)
  {
    uppercaseChar = toupper(input[i]); //Convert character to upper case version of character
	if (uppercaseChar < 'A' || uppercaseChar > 'Z') //If character is not A-Z
    {
	  return false;
    }
  }
  //At this point, we have gone through every character and checked it.
  return true; //Return true since every character had to be A-Z
}

// Compare precedence of operators.
int cmpPrecedence( const std::string& token1, const std::string& token2 )
{
  const std::pair<int,int> p1 = opmap.find( token1 )->second;
  const std::pair<int,int> p2 = opmap.find( token2 )->second;
  return p1.first - p2.first;
}

// Convert infix expression format into reverse Polish notation
bool infixToRPN( const std::vector<std::string>& inputTokens,
                 const int& size,
                 std::vector<std::string>& strArray )
{
  bool success = true;
  std::list<std::string> out;
  std::stack<std::string> stack;
  // While there are tokens to be read
  for ( int i = 0; i < size; i++ )
  {
    // Read the token
    const std::string token = inputTokens[ i ];
    // If token is an operator
    if ( isOperator( token ) )
    {
      // While there is an operator token, o2, at the top of the stack AND
      // either o1 is left-associative AND its precedence is equal to that of o2,
      // OR o1 has precedence less than that of o2,
      const std::string o1 = token;
      if ( !stack.empty() )
      {
        std::string o2 = stack.top();
        while ( isOperator( o2 ) &&
                ( ( isAssociative( o1, LEFT_ASSOC ) &&  cmpPrecedence( o1, o2 ) == 0 ) ||
                ( cmpPrecedence( o1, o2 ) < 0 ) ) )
        {
          // pop o2 off the stack, onto the output queue;
          stack.pop();
          out.push_back( o2 );
          if ( !stack.empty() )
            o2 = stack.top();
          else
            break;
        }
      }
      // push o1 onto the stack.
      stack.push( o1 );
    }
    // If the token is a left parenthesis, then push it onto the stack.
    else if ( token == "(" )
    {
      // Push token to top of the stack
      stack.push( token );
    }
    // If token is a right bracket ')'
    else if ( token == ")" )
    {
      // Until the token at the top of the stack is a left parenthesis,
      // pop operators off the stack onto the output queue.
      std::string topToken  = stack.top();
      while ( topToken != "(" )
      {
        out.push_back(topToken );
        stack.pop();
        if ( stack.empty() ) break;
        topToken = stack.top();
      }
      // Pop the left parenthesis from the stack, but not onto the output queue.
      if ( !stack.empty() ) stack.pop();
      // If the stack runs out without finding a left parenthesis,
      // then there are mismatched parentheses.
      if ( topToken != "(" )
      {
        return false;
      }
    }
    // If the token is a number, then add it to the output queue.
    else
    {
      out.push_back( token );
    }
  }
  // While there are still operator tokens in the stack:
  while ( !stack.empty() )
  {
    const std::string stackToken = stack.top();
    // If the operator token on the top of the stack is a parenthesis,
    // then there are mismatched parentheses.
    if ( isParenthesis( stackToken )   )
    {
      return false;
    }
    // Pop the operator onto the output queue./
    out.push_back( stackToken );
    stack.pop();
  }
  strArray.assign( out.begin(), out.end() );
  return success;
}


double RPNtoDouble( std::vector<std::string> tokens )
{
  std::stack<std::string> st;
  double output;
  // For each token
  for( int i = 0; i < (int) tokens.size(); ++i )
  {
    const std::string token = tokens[ i ];
//    if(isJustOneLetter(token))
//      result = strtod( token.c_str(), NULL );
    // If the token is a value push it onto the stack
    if( !isOperator(token) )
    {
//      if(isJustOneLetter(token))
//        d2 = m;
//      else
        st.push(token);
    }
    else
    {
      double result =  0.0;
      // Token is an operator: pop top two entries
      const std::string val2 = st.top();
      st.pop();
      double d2;
      if(isLetters(val2))
      {
    	d2 = mynondet;
      }
      else
        d2 = strtod( val2.c_str(), NULL );
      if( !st.empty() )
      {
        const std::string val1 = st.top();
        st.pop();
        double d1;
        if(isLetters(val1))
        {
		  d1 = mynondet;
        }
        else
          d1 = strtod( val1.c_str(), NULL );
        //Get the result
        result = token == "+" ? d1 + d2 :
                 token == "-" ? d1 - d2 :
                 token == "*" ? d1 * d2 :
                                 d1 / d2;
      }
      else
      {
        if ( token == "-" )
          result = d2 * -1;
        else
        {
          result = d2;
        }
      }
       // Push result onto stack
       std::ostringstream s;
       s << result;
       st.push( s.str() );
     }
    }
  if(isLetters(st.top()))
    output = mynondet;
  else
    output = strtod( st.top().c_str(), NULL );
  return output;
}

std::vector<std::string> getExpressionTokens( const std::string& expression )
{
  std::vector<std::string> tokens;
  std::string str = "";
  for( int i = 0; i < (int) expression.length(); ++i )
  {
    const std::string token( 1, expression[ i ] );
    if( isOperator( token ) || isParenthesis( token ) )
    {
      if( !str.empty() )
      {
        tokens.push_back( str ) ;
      }
      str = "";
      tokens.push_back( token );
    }
    else
    {
      // Append the numbers
      if( !token.empty() && token != " " )
      {
        str.append( token );
      }
      else
      {
        if( str != "" )
        {
          tokens.push_back( str );
          str = "";
        }
      }
    }
  }
  return tokens;
}

// Print iterators in a generic way
template<typename T, typename InputIterator>
void Print( const std::string& message,
             const InputIterator& itbegin,
             const InputIterator& itend,
             const std::string& delimiter)
{
  std::cout << message << std::endl;
  std::copy(itbegin,
              itend,
              std::ostream_iterator<T>(std::cout, delimiter.c_str()));
  std::cout << std::endl;
}

double parserToValidNumber(std::string s)
{
//  Print<char, s_iter>( "Input expression:", s.begin(), s.end(), "" );
  double d;
  // Tokenize input expression
  std::vector<std::string> tokens = getExpressionTokens( s );
  // Evaluate feasible expressions
  std::vector<std::string> rpn;
  if(infixToRPN( tokens, tokens.size(), rpn ) )
  {
    d = RPNtoDouble( rpn );
//    Print<std::string, cv_iter>( "RPN tokens:  ", rpn.begin(), rpn.end(), " " );
  }
  else
  {
   std::cout << "Mis-match in parentheses" << std::endl;
  }
  return d;
}

/*******************************************************************
 Function: extract_data_from_ss_file

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void extract_data_from_ss_file()
{
  std::ifstream verification_file(dsv_strings.desired_filename);
  std::string current_line;
  getline(verification_file, current_line);

  /* getting implementation specifications */
  std::string str_bits;
  int i;
  for(i = 0; current_line[i] != '<'; i++)
  {
    // just increment the variable i
  }
  i++;
  for(; current_line[i] != ','; i++)
    str_bits.push_back(current_line[i]);
  impl.int_bits = std::stoi(str_bits);
  str_bits.clear();
  i++;
  for(; current_line[i] != '>'; i++)
    str_bits.push_back(current_line[i]);
  impl.frac_bits = std::stoi(str_bits);
  str_bits.clear();

  getline(verification_file, current_line); // range

  std::string str_range;
  for(i = 0; current_line[i] != '['; i++)
  {
    // just increment the variable i
  }
  i++;
  for(; current_line[i] != ','; i++)
    str_range.push_back(current_line[i]);
  impl.min = std::stoi(str_range);
  str_range.clear();
  i++;
  for(; current_line[i] != ']'; i++)
    str_range.push_back(current_line[i]);
  impl.max = std::stoi(str_range);
  str_range.clear();

  getline(verification_file, current_line); // states

  for(i = 0; current_line[i] != '='; i++)
  {
    // just increment the variable i
  }

  i++;
  i++;

  for(; current_line[i] != ';'; i++)
    str_bits.push_back(current_line[i]);

  int states = std::stoi(str_bits);
  str_bits.clear();

  getline(verification_file, current_line); // inputs

  for(i = 0; current_line[i] != '='; i++)
  {
    // just increment the variable i
  }

  i++;
  i++;

  for(; current_line[i] != ';'; i++)
    str_bits.push_back(current_line[i]);

  int inputs = std::stoi(str_bits);
  str_bits.clear();

  getline(verification_file, current_line); // outputs

  for(i = 0; current_line[i] != '='; i++)
  {
    // just increment the variable i
  }

  i++;
  i++;

  for(; current_line[i] != ';'; i++)
    str_bits.push_back(current_line[i]);

  int outputs = std::stoi(str_bits);
  str_bits.clear();

  /* Updating _controller */
  _controller.nStates = states;
  _controller.nInputs = inputs;
  _controller.nOutputs = outputs;

  /* Initializing matrix A */
  getline(verification_file, current_line); // matrix A
  str_bits.clear();

  for(i = 0; current_line[i] != '['; i++)
  {
    // just increment the variable i
  }
  i++;

  int lines = 0;
  int columns = 0;

  for(; current_line[i] != ']'; i++)
  {
    if(current_line[i] != ',' && current_line[i] != ';')
    {
      str_bits.push_back(current_line[i]);
    }
    else if(current_line[i] == ';')
    {
      if(isNumber(str_bits))
      {
        _controller.A[lines][columns] = std::stod(str_bits);
//        std::cout << _controller.A[lines][columns] << std::endl;
      }
      else
      {
        _controller.A[lines][columns] = parserToValidNumber(str_bits);
      }
      lines++;
      columns = 0;
      str_bits.clear();
    }
    else
    {
      if(isNumber(str_bits))
      {
        _controller.A[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.A[lines][columns] = parserToValidNumber(str_bits);
      }
      columns++;
      str_bits.clear();
    }
  }

  if(isNumber(str_bits))
  {
    _controller.A[lines][columns] = std::stod(str_bits);
  }
  else
  {
    _controller.A[lines][columns] = parserToValidNumber(str_bits);
  }
  str_bits.clear();
//  std::cout << "A[0][1]=" << _controller.A[0][1] << std::endl;
  /* Initializing matrix B */

  getline(verification_file, current_line); // matrix B
  str_bits.clear();
  for(i = 0; current_line[i] != '['; i++)
  {
    // just increment the variable i
  }
  i++;

  lines = 0;
  columns = 0;

  for(; current_line[i] != ']'; i++)
  {
    if(current_line[i] != ',' && current_line[i] != ';')
    {
      str_bits.push_back(current_line[i]);
    }
    else if(current_line[i] == ';')
    {
      if(isNumber(str_bits))
      {
        _controller.B[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.B[lines][columns] = parserToValidNumber(str_bits);
      }
      lines++;
      columns = 0;
      str_bits.clear();
    }
    else
    {
      if(isNumber(str_bits))
      {
        _controller.B[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.B[lines][columns] = parserToValidNumber(str_bits);
      }
      columns++;
      str_bits.clear();
    }
  }
  if(isNumber(str_bits))
  {
    _controller.B[lines][columns] = std::stod(str_bits);
  }
  else
  {
    _controller.B[lines][columns] = parserToValidNumber(str_bits);
  }
  str_bits.clear();

  /* Initializing matrix C */
  getline(verification_file, current_line); // matrix C
  str_bits.clear();
  for(i = 0; current_line[i] != '['; i++)
  {
    // just increment the variable i
  }
  i++;
  lines = 0;
  columns = 0;

  for(; current_line[i] != ']'; i++)
  {
    if(current_line[i] != ',' && current_line[i] != ';')
    {
      str_bits.push_back(current_line[i]);
    }
    else if(current_line[i] == ';')
    {
      if(isNumber(str_bits))
      {
        _controller.C[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.C[lines][columns] = parserToValidNumber(str_bits);
      }
      lines++;
      columns = 0;
      str_bits.clear();
    }
    else
    {
      if(isNumber(str_bits))
      {
    	_controller.C[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.C[lines][columns] = parserToValidNumber(str_bits);
      }
      columns++;
      str_bits.clear();
    }
  }

  if(isNumber(str_bits))
  {
    _controller.C[lines][columns] = std::stod(str_bits);
  }
  else
  {
    _controller.C[lines][columns] = parserToValidNumber(str_bits);
  }
  str_bits.clear();

  /* initialising matrix D */

  getline(verification_file, current_line); // matrix D
  str_bits.clear();
  for(i = 0; current_line[i] != '['; i++)
  {
    // just count the variable i
  }
  i++;
  lines = 0;
  columns = 0;
  for(; current_line[i] != ']'; i++)
  {
    if(current_line[i] != ',' && current_line[i] != ';')
    {
      str_bits.push_back(current_line[i]);
    }
    else if(current_line[i] == ';')
    {
      if(isNumber(str_bits))
      {
        _controller.D[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.D[lines][columns] = parserToValidNumber(str_bits);
      }
      lines++;
      columns = 0;
      str_bits.clear();
    }
    else
    {
      if(isNumber(str_bits))
      {
        _controller.D[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.D[lines][columns] = parserToValidNumber(str_bits);
      }
      columns++;
      str_bits.clear();
    }
  }

  if(isNumber(str_bits))
  {
    _controller.D[lines][columns] = std::stod(str_bits);
  }
  else
  {
    _controller.D[lines][columns] = parserToValidNumber(str_bits);
  }
  str_bits.clear();

  /* initialising matrix x0 */

  getline(verification_file, current_line); // matrix x0
  str_bits.clear();
  for(i = 0; current_line[i] != '['; i++)
  {
    // just increment the variable i
  }
  i++;

  lines = 0;
  columns = 0;

  for(; current_line[i] != ']'; i++)
  {
    if(current_line[i] != ',' && current_line[i] != ';')
    {
      str_bits.push_back(current_line[i]);
    }
    else if(current_line[i] == ';')
    {
      if(isNumber(str_bits))
      {
        _controller.x0[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.x0[lines][columns] = parserToValidNumber(str_bits);
      }
      lines++;
      columns = 0;
      str_bits.clear();
    }
    else
    {
      if(isNumber(str_bits))
      {
        _controller.x0[lines][columns] = std::stod(str_bits);
      }
      else
      {
        _controller.x0[lines][columns] = parserToValidNumber(str_bits);
      }
      columns++;
      str_bits.clear();
    }
  }

  if(isNumber(str_bits))
  {
    _controller.x0[lines][columns] = std::stod(str_bits);
  }
  else
  {
    _controller.x0[lines][columns] = parserToValidNumber(str_bits);
  }
  str_bits.clear();

  /* initialising matrix Inputs */

  getline(verification_file, current_line); // matrix inputs
  str_bits.clear();
  for(i = 0; current_line[i] != '['; i++)
  {
    // just count the variable i
  };
  i++;
  lines = 0;
  columns = 0;
  for(; current_line[i] != ']'; i++)
  {
    if(current_line[i] != ',' && current_line[i] != ';')
    {
      str_bits.push_back(current_line[i]);
    }
    else if(current_line[i] == ';')
    {
      _controller.inputs[lines][columns] = std::stod(str_bits);
      lines++;
      columns = 0;
      str_bits.clear();
    }
    else
    {
      _controller.inputs[lines][columns] = std::stod(str_bits);
      columns++;
      str_bits.clear();
    }
  }

  _controller.inputs[lines][columns] = std::stod(str_bits);
  str_bits.clear();

  if(dsv_strings.desired_property == "SETTLING_TIME")
  {
    getline(verification_file, current_line); // tsr

    for(i = 0; current_line[i] != '='; i++)
    {
      // just increment the variable i
    }

    i++;
    i++;

    for(; current_line[i] != ';'; i++)
      str_bits.push_back(current_line[i]);

      double tsr = std::stod(str_bits);
      str_bits.clear();

      getline(verification_file, current_line); // ts

      for(i = 0; current_line[i] != '='; i++)
      {
        // just increment the variable i
      }

      i++;
      i++;

      for(; current_line[i] != ';'; i++)
        str_bits.push_back(current_line[i]);

        double ts = std::stod(str_bits);
        str_bits.clear();

        getline(verification_file, current_line); // p

        for(i = 0; current_line[i] != '='; i++)
        {
          // just increment the variable i
        }

        i++;
        i++;

        for(; current_line[i] != ';'; i++)
          str_bits.push_back(current_line[i]);

          double p = std::stod(str_bits);
          str_bits.clear();

          /* Updating _controller */
          _controller.tsr = tsr;
          _controller.ts = ts;
          _controller.p = p;
    }

  if(closedloop)
  {
    getline(verification_file, current_line); // matrix controller
    str_bits.clear();
    for(i = 0; current_line[i] != '['; i++)
      {};

    i++;
    lines = 0;
    columns = 0;
    for(; current_line[i] != ']'; i++)
    {
      if(current_line[i] != ',' && current_line[i] != ';')
      {
        str_bits.push_back(current_line[i]);
      }
      else if(current_line[i] == ';')
      {
        if(isNumber(str_bits))
        {
          _controller.K[lines][columns] = std::stod(str_bits);
        }
        else
        {
        // TODO: to be implemented
        }
        lines++;
        columns = 0;
        str_bits.clear();
      }
      else
      {
        if(isNumber(str_bits))
    	{
    	  _controller.K[lines][columns] = std::stod(str_bits);
    	}
    	else
    	{
    	// TODO: to be implemented
    	}
        columns++;
        str_bits.clear();
      }
    }
    if(isNumber(str_bits))
    {
      _controller.K[lines][columns] = std::stod(str_bits);
    }
    else
    {
    // TODO: to be implemented
    }
    str_bits.clear();
  }

#if DEBUG_DSV
  print_matrix(_controller.K, 1, states);
  print_matrix(_controller.B, states, inputs);
  print_matrix(_controller.C, outputs, states);
  print_matrix(_controller.D, outputs, inputs);
#endif
}

/*******************************************************************
 Function: state_space_parser

 Inputs:

 Outputs:

 Purpose:

 \*******************************************************************/

void state_space_parser()
{
  std::string verification_file;

  std::string tmp;
  std::ostringstream cf_value_precision;
  unsigned int i, j;
  cf_value_precision.precision(64);

  verification_file =
      "#include <dsverifier.h>\n digital_system_state_space _controller;"
      "\n implementation impl = {\n .int_bits = ";
  verification_file.append(std::to_string(impl.int_bits));
  verification_file.append(",\n .frac_bits = ");
  verification_file.append(std::to_string(impl.frac_bits));
  verification_file.append(",\n .min = ");
  verification_file.append(std::to_string(impl.min));
  verification_file.append(",\n .max = ");
  verification_file.append(std::to_string(impl.max));
  verification_file.append("};\n int nStates = ");
  verification_file.append(std::to_string(_controller.nStates));
  verification_file.append(";\n int nInputs = ");
  verification_file.append(std::to_string(_controller.nInputs));
  verification_file.append(";\n int nOutputs = ");
  verification_file.append(std::to_string(_controller.nOutputs));
  verification_file.append(";\n double error_limit = ");
  cf_value_precision << std::fixed << desired_quantization_limit;
  verification_file.append(cf_value_precision.str());
  verification_file.append(
      ";\n int closed_loop = " + std::string(closedloop ? "1" : "0"));
  verification_file.append(";\n void initials(){\n");

  for(i = 0; i < _controller.nStates; i++)
  {
    for(j = 0; j < _controller.nStates; j++)
    {
      verification_file.append("\t_controller.A[");
      verification_file.append(std::to_string(i));
      verification_file.append("][");
      verification_file.append(std::to_string(j));
      verification_file.append("] = ");
      verification_file.append(std::to_string(_controller.A[i][j]));
      verification_file.append(";\n");
    }
  }

  for(i = 0; i < _controller.nStates; i++)
  {
    for(j = 0; j < _controller.nInputs; ++j)
    {
      verification_file.append("\t_controller.B[");
      verification_file.append(std::to_string(i));
      verification_file.append("][");
      verification_file.append(std::to_string(j));
      verification_file.append("] = ");
      verification_file.append(std::to_string(_controller.B[i][j]));
      verification_file.append(";\n");
    }
  }

  for(i = 0; i < _controller.nOutputs; i++)
  {
    for(j = 0; j < _controller.nStates; j++)
    {
      verification_file.append("\t_controller.C[");
      verification_file.append(std::to_string(i));
      verification_file.append("][");
      verification_file.append(std::to_string(j));
      verification_file.append("] = ");
      verification_file.append(std::to_string(_controller.C[i][j]));
      verification_file.append(";\n");
    }
  }

  for(i = 0; i < _controller.nOutputs; i++)
  {
    for(j = 0; j < _controller.nInputs; j++)
    {
      verification_file.append("\t_controller.D[");
      verification_file.append(std::to_string(i));
      verification_file.append("][");
      verification_file.append(std::to_string(j));
      verification_file.append("] = ");
      verification_file.append(std::to_string(_controller.D[i][j]));
      verification_file.append(";\n");
    }
  }

  for(i = 0; i < _controller.nInputs; i++)
  {
    for(j = 0; j < 1; j++)
    {
      verification_file.append("\t_controller.inputs[");
      verification_file.append(std::to_string(i));
      verification_file.append("][");
      verification_file.append(std::to_string(j));
      verification_file.append("] = ");
      verification_file.append(std::to_string(_controller.inputs[i][j]));
      verification_file.append(";\n");
    }
  }

  if(closedloop)
  {
    for(i = 0; i < _controller.nOutputs; i++)
    {
      for(j = 0; j < _controller.nStates; j++)
      {
        verification_file.append("\t_controller.K[");
        verification_file.append(std::to_string(i));
        verification_file.append("][");
        verification_file.append(std::to_string(j));
        verification_file.append("] = ");
        cf_value_precision.str(std::string());
        cf_value_precision << std::fixed << _controller.K[i][j];
        verification_file.append(cf_value_precision.str());
        verification_file.append(";\n");
      }
    }
  }

  verification_file.append("}\n");

  std::ofstream myfile("input.c");

  if(myfile.is_open())
  {
    myfile << verification_file;
    myfile.close();
  }
  else
    std::cout << "Unable to open file";
}

/*******************************************************************
 Function: closed_loop

 Inputs: None

 Outputs:

 Purpose: Compute A-B*K and C-D*K

 \*******************************************************************/

void closed_loop()
{
  fxp_t K_fxp[LIMIT][LIMIT];
  double result1[LIMIT][LIMIT];

  int i, j, k;
  for(i = 0; i < LIMIT; i++)
    for(j = 0; j < LIMIT; j++)
      result1[i][j] = 0;

//  for(j = 0; j < _controller.nStates; j++)
//    std::cout << _controller.K[0][j] << std::endl;

  if(nofwl!=true)
  {
    for(i = 0; i < _controller.nStates; i++)
    {
      K_fxp[0][i] = fxp_double_to_fxp(_controller.K[0][i]);
      _controller.K[0][i] = fxp_to_double(K_fxp[0][i]);
    }
    nofwl = true;

//    for(j = 0; j < _controller.nStates; j++)
//      std::cout << _controller.K[0][j] << std::endl;

  }
//  for(i = 0; i < _controller.nStates; i++)
//    for(j = 0; j < _controller.nStates; j++)
//      std::cout << _controller.A[i][j] << std::endl;

  // B*K
  double_matrix_multiplication(_controller.nStates, _controller.nInputs,
      _controller.nInputs, _controller.nStates, _controller.B, _controller.K,
      result1);

  double_sub_matrix(_controller.nStates, _controller.nStates,_controller.A,
      result1, _controller.A);

//  for(i = 0; i < _controller.nStates; i++)
//    for(j = 0; j < _controller.nStates; j++)
//      std::cout << _controller.A[i][j] << std::endl;

  for(i = 0; i < LIMIT; i++)
    for(j = 0; j < LIMIT; j++)
      result1[i][j] = 0;

  // D*K
  double_matrix_multiplication(_controller.nOutputs, _controller.nInputs,
      _controller.nInputs, _controller.nStates, _controller.D, _controller.K,
      result1);

  double_sub_matrix(_controller.nOutputs, _controller.nStates, _controller.C,
      result1, _controller.C);
  closedloop = false;
}

/*******************************************************************
 Function: tf2ss

 Inputs:

 Outputs:

 Purpose: This function converts a transfer function into a state space
 It only works for SISO systems

 \*******************************************************************/

void tf2ss()
{
  unsigned int i, j;

  _controller.nStates = ds.b_size - 1;
  _controller.nInputs = 1;
  _controller.nOutputs = 1;

  for(i = 0; i < _controller.nStates - 1; i++)
  {
    for(j = 0; j < _controller.nStates; j++)
    {
      if(j == i + 1)
        _controller.A[i][j] = 1;
      else
        _controller.A[i][j] = 0;
    }
  }

  for(j = 0; j < _controller.nStates; j++)
    _controller.A[_controller.nStates - 1][j] = -ds.b[_controller.nStates - j];

  for(i = 0; i < _controller.nStates - 1; i++)
    for(j = 0; j < _controller.nInputs; j++)
      _controller.B[i][j] = 0;

  /* for SISO systems */
  _controller.B[_controller.nStates - 1][0] = 1;

  for(i = 0; i < _controller.nOutputs; i++)
    for(j = 0; j < _controller.nStates; j++)
      _controller.C[i][j] = ds.a[_controller.nStates - j]
          - (ds.b[_controller.nStates - j] * ds.a[0]);

  for(i = 0; i < _controller.nOutputs; i++)
    for(j = 0; j < _controller.nInputs; j++)
      _controller.D[i][j] = ds.a[0];
}

/*******************************************************************
 Function: main

 Inputs: 

 Outputs:

 Purpose:

 \*******************************************************************/

/* main function */
int main(int argc, char* argv[])
{
  /* without overflow */
  set_overflow_mode = NONE;

  bind_parameters(argc, argv);
  dsverifier_messaget dsv_msg;

  if(dsv_strings.desired_property == "QUANTIZATION_ERROR"
      && desired_quantization_limit == 0.0)
    dsv_msg.show_required_argument_message("QUANTIZATION_ERROR");

  check_file_exists();

  std::cout << "Running: Digital Systems Verifier (DSVerifier)" << std::endl;

  if(translate)
  {
    extract_data_from_file();
    tf2ss();
    state_space_parser();
    exit(0);
  }

  if(preprocess)
  {
    std::string command_line_preprocess = prepare_bmc_command_line();
    execute_command_line(command_line_preprocess);
    exit(0);
  }

  if(dsv_strings.desired_rounding_mode == "ROUNDING")
  {
	  rounding_mode = ROUNDING;
  }
  else if(dsv_strings.desired_rounding_mode == "FLOOR")
  {
	  rounding_mode = FLOOR;
  }
  else if(dsv_strings.desired_rounding_mode == "CEIL")
  {
	  rounding_mode = CEIL;
  }
  else
  {
	  rounding_mode = FLOOR;
  }

  if(stateSpaceVerification)
  {
    extract_data_from_ss_file();

    if(closedloop && dsv_strings.desired_property != "QUANTIZATION_ERROR")
      closed_loop();

    if(dsv_strings.desired_property == "STABILITY")
    {
      std::cout << "Checking stability..." << std::endl;
      verify_state_space_stability();
      exit(0);
    }
    else if(dsv_strings.desired_property == "SETTLING_TIME")
    {
      if(closedloop==true){
        closed_loop();
      }
      std::cout << "Checking settling time..." << std::endl;
      verify_state_space_settling_time();
      exit(0);
    }
    else if(dsv_strings.desired_property == "QUANTIZATION_ERROR"
        || dsv_strings.desired_property == "SAFETY_STATE_SPACE"
        || dsv_strings.desired_property == "CONTROLLABILITY"
        || dsv_strings.desired_property == "OBSERVABILITY"
        || dsv_strings.desired_property == "LIMIT_CYCLE_STATE_SPACE")
    {
      state_space_parser();
      std::string command_line = prepare_bmc_command_line_ss();
      std::cout << "Back-end Verification: " << command_line << std::endl;
      execute_command_line(command_line);
      std::cout << "mynondet=" << mynondet << std::endl;
      exit(0);
    }
    else
    {
      std::cout << "There is no check!" << std::endl;
    }
  }
  else
  {
    bool is_delta_realization = (dsv_strings.desired_realization == "DDFI"
        || dsv_strings.desired_realization == "DDFII" ||
        dsv_strings.desired_realization == "TDDFII");

    bool is_restricted_property = (dsv_strings.desired_property == "STABILITY"
        || dsv_strings.desired_property == "MINIMUM_PHASE");

    bool is_restricted_counterexample = ((
        dsv_strings.desired_property == "OVERFLOW"
        && dsv_strings.desired_bmc == "ESBMC") ||
        dsv_strings.desired_property == "STABILITY"
        || dsv_strings.desired_property == "MINIMUM_PHASE");

    bool is_closed_loop_counterexample = (dsv_strings.desired_property
        == "STABILITY_CLOSED_LOOP"
        || dsv_strings.desired_property == "LIMIT_CYCLE_CLOSED_LOOP"
        || dsv_strings.desired_property == "QUANTIZATION_ERROR_CLOSED_LOOP");

    bool is_state_space_counterexample = (
        dsv_strings.desired_property == "STABILITY"
        || dsv_strings.desired_property == "OBSERVABILITY"
        || dsv_strings.desired_property == "CONTROLLABILITY"
        || dsv_strings.desired_property == "QUANTIZATION_ERROR");

    extract_data_from_file();

    if(!((is_delta_realization && is_restricted_property)
        || (dsv_strings.desired_property == "FILTER_MAGNITUDE_DET")
        || (dsv_strings.desired_property == "FILTER_PHASE_DET")))
    {
      std::string command_line = prepare_bmc_command_line();
      std::cout << "Back-end Verification: " << command_line << std::endl;
      std::string counterexample;

      if(k_induction)
      {
        char *dsverifier_home = getenv("DSVERIFIER_HOME");
        std::string model_checker_path = std::string(dsverifier_home)
            + "/model-checker";
        command_line += " > temp.c";
        execute_command_line(command_line);
        command_line = model_checker_path
            + "/esbmc temp.c --clang-frontend --k-induction --boolector";
        counterexample = execute_command_line(command_line);
      }
      else
      {
        counterexample = execute_command_line(command_line);
      }

      if(show_counterexample_data)
      {
        if(is_restricted_counterexample)
        {
          print_counterexample_data_for_restricted_properties();
        }
        else if(is_closed_loop_counterexample)
        {
          // print_counterexample_data_for_closed_loop();
        }
        else if(is_state_space_counterexample)
        {
          // print_counterexample_data_for_state_space();
        }
        else
        {
          print_counterexample_data(counterexample);
        }
      }

      exit(0);
    }
    else
    {
      try
      {
        initialization();
        if(dsv_strings.desired_property == "STABILITY")
          check_stability_delta_domain();
        else if(dsv_strings.desired_property == "MINIMUM_PHASE")
          check_minimum_phase_delta_domain();
        else if(dsv_strings.desired_property == "FILTER_MAGNITUDE_DET")
          check_filter_magnitude_det();
        else if(dsv_strings.desired_property == "FILTER_PHASE_DET")
          check_filter_phase_det();

        if(show_counterexample_data)
          print_counterexample_data_for_restricted_properties();

        exit(0);
      } catch(std::exception & e)
      {
        std::cout << std::endl << "An unexpected error occurred in DSVerifier"
            << std::endl;
      }
    }
  }
}
