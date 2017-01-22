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
#
# --------------------------------------------------
#
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

/* eigen dependencies */
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Polynomials>

/* boost dependencies */
#include <boost/algorithm/string.hpp>


typedef bool _Bool;

void __DSVERIFIER_assume(_Bool expression)
{
	/* nothing to do */
}

void __DSVERIFIER_assert(_Bool expression)
{
  if (expression == 0)
  {
    throw 0;
  }
}

#include "version.h"

#include "../bmc/core/definitions.h"
#include "../bmc/core/fixed-point.h"
#include "../bmc/core/util.h"
#include "../bmc/core/delta-operator.h"
#include "../bmc/core/initialization.h"

#include "print_messages.h"

typedef Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootType RootType;
typedef Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType RootsType;

const char * properties [] = { "OVERFLOW", "LIMIT_CYCLE",
		"ZERO_INPUT_LIMIT_CYCLE", "ERROR",
		"TIMING", "TIMING_MSP430", "STABILITY", "STABILITY_CLOSED_LOOP",
		"LIMIT_CYCLE_CLOSED_LOOP", "QUANTIZATION_ERROR_CLOSED_LOOP",
		"MINIMUM_PHASE", "QUANTIZATION_ERROR", "CONTROLLABILITY",
		"OBSERVABILITY", "LIMIT_CYCLE_STATE_SPACE", "SAFETY_STATE_SPACE",
		"FILTER_MAGNITUDE_NON_DET", "FILTER_MAGNITUDE_DET"};

const char * rounding [] = { "ROUNDING", "FLOOR", "CEIL" };
const char * overflow [] = { "DETECT_OVERFLOW", "SATURATE", "WRAPAROUND" };
const char * realizations [] = { "DFI", "DFII", "TDFII", "DDFI", "DDFII", "TDDFII" };
const char * bmcs [] = { "ESBMC", "CBMC" };
const char * connections_mode [] = { "SERIES", "FEEDBACK" };
const char * error_mode [] = { "ABSOLUTE", "RELATIVE" };

/* expected parameters */
unsigned int desired_x_size = 0;
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

/* state space */
bool stateSpaceVerification = false;
bool closedloop = false;
bool translate = false;
bool k_induction = false;
digital_system_state_space _controller;
double desired_quantization_limit = 0.0;
bool show_counterexample_data = false;

/*******************************************************************\

Function: replace_all_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string replace_all_string(
  std::string str,
  const std::string& from,
  const std::string& to)
{
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos)
  {
	str.replace(start_pos, from.length(), to);
	start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
  }
  return str;
}

/*******************************************************************\

Function: extract_data_from_file

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void extract_data_from_file()
{
  std::ifstream verification_file(desired_filename);
  bool ds_id_found = false;

  for(std::string current_line; getline( verification_file, current_line );)
  {
	/* removing whitespaces */
	current_line = replace_all_string(current_line, " ", "");
	current_line = replace_all_string(current_line, "\t", "");
	/* check the last comma, and remove it */
	if (current_line.back() == ',')
	{
	  current_line.pop_back();
	}

	/* check if there is an desired ds_id and find the region*/
	if (desired_ds_id.size() != 0)
	{
	  std::string::size_type find_desired_ds_id = current_line.find(desired_ds_id, 0);
	  if (ds_id_found == false)
	  {
        if (find_desired_ds_id != std::string::npos)
		  ds_id_found = true;
        else
          continue; /* go to next line */
	  }
	}

    std::string::size_type ds_a = current_line.find(".a=", 0);
	if (ds_a != std::string::npos)
	{
	  std::vector<std::string> coefficients;
      boost::split(coefficients, current_line, boost::is_any_of(","));
      for(int i=0; i< coefficients.size(); i++)
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
	if (ds_b != std::string::npos)
	{
	  std::vector<std::string> coefficients;
	  boost::split(coefficients, current_line, boost::is_any_of(","));
	  for(int i=0; i< coefficients.size(); i++)
	  {
        std::string coefficient = coefficients.at(i);
		coefficient = replace_all_string(coefficient, ".b={", "");
		coefficient = replace_all_string(coefficient, "}", "");
		ds.b[i] = std::atof(coefficient.c_str());
		ds.b_size = coefficients.size();
      }
	  continue;
	}
	std::string::size_type ds_sample_time = current_line.find(".sample_time", 0);
	if (ds_sample_time != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".sample_time=", "");
	  ds.sample_time = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type impl_int_bits = current_line.find(".int_bits", 0);
	if (impl_int_bits != std::string::npos)
	{
      current_line = replace_all_string(current_line, ".int_bits=", "");
	  impl.int_bits = std::atoi(current_line.c_str());
	  continue;
	}
	std::string::size_type impl_frac_bits = current_line.find(".frac_bits", 0);
	if (impl_frac_bits != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".frac_bits=", "");
	  impl.frac_bits = std::atoi(current_line.c_str());
	  continue;
	}
	std::string::size_type impl_min = current_line.find(".min", 0);
	if (impl_min != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".min=", "");
	  impl.min = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type impl_max = current_line.find(".max", 0);
	if (impl_max != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".max=", "");
	  impl.max = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type impl_delta = current_line.find(".delta", 0);
	if (impl_delta != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".delta=", "");
	  impl.delta = std::atof(current_line.c_str());
	  continue;
	}
	//FILTER COEFFICIENTS 
	std::string::size_type filter_Ap = current_line.find(".Ap", 0);
	if (filter_Ap != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".Ap=", "");
	  filter.Ap = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type filter_Ac = current_line.find(".Ac", 0);
	if (filter_Ac != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".Ac=", "");
	  filter.Ac = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type filter_Ar = current_line.find(".Ar", 0);
	if (filter_Ar != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".Ar=", "");
	  filter.Ar = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type filter_wp = current_line.find(".wp", 0);
	if (filter_wp != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".wp=", "");
	  filter.wp = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type filter_wc = current_line.find(".wc", 0);
	if (filter_wc != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".wc=", "");
	  filter.wc = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type filter_wr = current_line.find(".wr", 0);
	if (filter_wr != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".wr=", "");
	  filter.wr = std::atof(current_line.c_str());
	  continue;
	}
	std::string::size_type filter_type = current_line.find(".type", 0);
	if (filter_type != std::string::npos)
	{
	  current_line = replace_all_string(current_line, ".type=", "");
	  filter.type = std::atof(current_line.c_str());
	  continue;
	}
  }
}

/*******************************************************************\

Function: validate_function

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_function(std::string data)
{
  if (data.empty())
    std::cout << "specify a function name" << std::endl;
  else
    desired_function = data;
}

/*******************************************************************\

Function: validate_selected_bmc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_selected_bmc(std::string data)
{
  int length = (sizeof(bmcs)/sizeof(*bmcs));
  for(int i=0; i<length; i++)
  {
	if (bmcs[i] == data)
	{
	  desired_bmc = data;
	  break;
	}
  }
  if (desired_bmc.size() == 0)
  {
	std::cout << "invalid bmc: " << data << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: validate_selected_connection_mode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_selected_connection_mode(std::string data)
{
  int length = (sizeof(connections_mode)/sizeof(*connections_mode));
  for(int i=0; i<length; i++)
  {
	if (connections_mode[i] == data)
	{
	  desired_connection_mode = data;
	  break;
	}
  }
  if (desired_connection_mode.size() == 0)
  {
	std::cout << "invalid connection-mode: " << data << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: validate_selected_error

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_selected_error_mode(std::string data)
{
  int length = (sizeof(error_mode)/sizeof(*error_mode));
  for(int i=0; i<length; i++)
  {
	if (error_mode[i] == data)
	{
	  desired_error_mode = data;
	  break;
	}
  }
  if (desired_error_mode.size() == 0)
  {
	std::cout << "invalid error mode: " << data << std::endl;
	exit(1);
  }
}


/*******************************************************************\

Function: validate_selected_rounding_mode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_selected_rounding_mode(std::string data)
{
  int length = (sizeof(rounding)/sizeof(*rounding));
  for(int i=0; i<length; i++)
  {
    if (rounding[i] == data)
	{
	  desired_rounding_mode = data;
	  break;
	}
  }
  if (desired_rounding_mode.size() == 0)
  {
	std::cout << "invalid rounding-mode: " << data << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: validate_selected_overflow_mode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_selected_overflow_mode(std::string data)
{
  int length = (sizeof(overflow)/sizeof(*overflow));
  for(int i=0; i<length; i++)
  {
	if (overflow[i] == data)
	{
	  desired_overflow_mode = data;
	  break;
	}
  }
  if (desired_overflow_mode.size() == 0)
  {
	std::cout << "invalid overflow-mode: " << data << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: validate_selected_realization

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_selected_realization(std::string data)
{
  int length = (sizeof(realizations)/sizeof(*realizations));
  for(int i=0; i<length; i++)
  {
	if (realizations[i] == data)
	{
	  desired_realization = data;
	  break;
	}
  }

  if (desired_realization.size() == 0)
  {
	std::cout << "invalid realization: " << data << std::endl;
	exit(1);
  }

  bool is_delta_realization = (desired_realization == "DDFI" || desired_realization == "DDFII" || desired_realization == "TDDFII");

  if (is_delta_realization)
  {
	extract_data_from_file();
	if (impl.delta == 0)
	{
	  std::cout << "invalid delta realization: " << impl.delta << std::endl;
	  exit(1);
	}
  }
}

/*******************************************************************\

Function: validate_selected_property

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/


void validate_selected_property(std::string data)
{
  int length = (sizeof(properties)/sizeof(*properties));
  for(int i=0; i<length; i++)
  {
	if (properties[i] == data)
	{
	  desired_property = data;
	  break;
	  }
  }
  if (desired_property.size() == 0)
  {
    std::cout << "invalid property: " << data << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: validate_filename

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void validate_filename(std::string file)
{
  if (file == "--help" || file == "-h")
  {
	help();
  }
  else if(file.substr(file.size()-3, std::string::npos) != ".ss")
  {
	std::string::size_type loc = file.find(".c", 0 );
	if( loc == std::string::npos )
	{
	  std::cout << file << ": failed to figure out type of file" << std::endl;
	  exit(1);
	}
  }
  else
  {
	stateSpaceVerification = true;
  }

  desired_filename = file;
}

/*******************************************************************\

Function: check_required_parameters

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_required_parameters()
{
  if (desired_bmc.size() == 0)
  {
	//we use CBMC as our default back-end model-checker
	desired_bmc = "CBMC";
  }
}

/*******************************************************************\

Function: bind_parameters

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bind_parameters(int argc, char* argv[])
{
  if (argc == 1)
  {
	help();
	exit(1);
  }
  validate_filename(argv[1]);
  for (int i = 2; i < argc; ++i)
  {
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")
	{
	  help();
	}
	else if (std::string(argv[i]) == "--realization")
	{
	  if (i + 1 < argc)
	    validate_selected_realization(argv[++i]);
	  else
        show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--property")
	{
	  if (i + 1 < argc)
        validate_selected_property(argv[++i]);
	  else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--x-size")
	{
	  if (i + 1 < argc)
        desired_x_size = std::atoi(argv[++i]);
      else
        show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--unlimited-x-size")
	{
	  k_induction = true;
	}
	else if (std::string(argv[i]) == "--connection-mode")
	{
	  if (i + 1 < argc)
        validate_selected_connection_mode(argv[++i]);
      else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--error-mode")
	{
      if (i + 1 < argc)
        validate_selected_error_mode(argv[++i]);
	  else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--rounding-mode")
	{
	  if (i + 1 < argc)
		validate_selected_rounding_mode(argv[++i]);
	  else
        show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--overflow-mode")
	{
	  if (i + 1 < argc)
        validate_selected_overflow_mode(argv[++i]);
      else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--timeout")
	{
      if (i + 1 < argc)
		desired_timeout = argv[++i];
      else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--bmc")
	{
	  if (i + 1 < argc)
        validate_selected_bmc(argv[++i]);
      else
        show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--function")
	{
	  if (i + 1 < argc)
        validate_function(argv[++i]);
      else
        show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--solver")
	{
	  if (i + 1 < argc)
		desired_solver = argv[++i];
	  else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--limit")
	{
	  if (i + 1 < argc)
		desired_quantization_limit = std::stod(argv[++i]);
	  else
		show_required_argument_message(argv[i]);
	}
	else if (std::string(argv[i]) == "--closed-loop")
	{
	  closedloop = true;
	}
	else if (std::string(argv[i]) == "--tf2ss")
	{
	  translate = true;
	}
	else if (std::string(argv[i]) == "--show-ce-data")
	{
	  show_counterexample_data = true;
	}
	else
	{
      /* get macro parameters */
	  std::string parameter = argv[i];
	  std::string::size_type loc = parameter.find("-D", 0 );
	  if( loc != std::string::npos )
	  {
		desired_macro_parameters += " " + parameter;
		/* check if there is an desired benchmark */
		std::string::size_type find_desired_ds_id = parameter.find("-DDS_ID=", 0 );
		if( find_desired_ds_id != std::string::npos )
		{
		  std::vector<std::string> parts;
		  boost::split(parts, parameter, boost::is_any_of("="));
		  desired_ds_id = "DS_ID==" + parts.at(1);
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

/*******************************************************************\

Function: execute_command_line

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string execute_command_line(std::string command)
{
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return "ERROR";
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

/*******************************************************************\

Function: prepare_bmc_command_line

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string prepare_bmc_command_line()
{
  char * dsverifier_home;
  dsverifier_home = getenv("DSVERIFIER_HOME");
  if (dsverifier_home == NULL)
  {
	std::cout << std::endl << "[ERROR] - You must set DSVERIFIER_HOME "
	  "environment variable." << std::endl;
	exit(1);
  }
  std::string bmc_path = std::string(dsverifier_home) + "/bmc";
  std::string model_checker_path = std::string(dsverifier_home) + "/model-checker";
  std::string command_line;
  if (desired_bmc == "ESBMC")
  {
	command_line = model_checker_path + "/esbmc " + desired_filename +
	  " --no-bounds-check --no-pointer-check --no-div-by-zero-check -DBMC=ESBMC -I " +
	  bmc_path;
	if (k_induction)
	{
	  command_line += " --k-induction --unlimited-k-steps -DK_INDUCTION_MODE=K_INDUCTION ";
	}
	if (desired_timeout.size() > 0)
	{
	  command_line += " --timeout " + desired_timeout;
	}
  }
  else if (desired_bmc == "CBMC")
  {
	command_line =  model_checker_path + "/cbmc " + desired_filename +
	  " --fixedbv --stop-on-fail -DBMC=CBMC -I " + bmc_path;
  }

  if (desired_function.size() > 0)
	command_line += " --function " + desired_function;

  if (desired_solver.size() > 0)
	command_line += " --" + desired_solver;

  if (desired_realization.size() > 0)
	command_line += " -DREALIZATION=" + desired_realization;

  if (desired_property.size() > 0)
	command_line += " -DPROPERTY=" + desired_property;

  if (desired_connection_mode.size() > 0)
	command_line += " -DCONNECTION_MODE=" + desired_connection_mode;

  if (desired_error_mode.size() > 0)
    command_line += " -DERROR_MODE=" + desired_error_mode;

  if (desired_rounding_mode.size() > 0)
    command_line += " -DROUNDING_MODE=" + desired_rounding_mode;

  if (desired_overflow_mode.size() > 0)
    command_line += " -DOVERFLOW_MODE=" + desired_overflow_mode;

  if (desired_x_size > 0)
    command_line += " -DX_SIZE=" + std::to_string(desired_x_size);

  command_line += desired_macro_parameters;

  return command_line;
}

/*******************************************************************\

Function: prepare_bmc_command_line_ss

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string prepare_bmc_command_line_ss()
{
  char * dsverifier_home;
  dsverifier_home = getenv("DSVERIFIER_HOME");
  if (dsverifier_home == NULL)
  {
	std::cout << std::endl << "[ERROR] - You must set DSVERIFIER_HOME environment variable." << std::endl;
	exit(1);
  }
  std::string command_line;
  std::string bmc_path = std::string(dsverifier_home) + "/bmc";
  std::string model_checker_path = std::string(dsverifier_home) + "/model-checker";

  if (desired_bmc == "ESBMC")
  {
	command_line = model_checker_path + "/esbmc input.c --no-bounds-check --no-pointer-check --no-div-by-zero-check -DBMC=ESBMC -I "
	+ bmc_path;

	if (desired_timeout.size() > 0)
	  command_line += " --timeout " + desired_timeout;
  }
  else if (desired_bmc == "CBMC")
  {
	command_line = model_checker_path + "/cbmc --fixedbv --stop-on-fail input.c -DBMC=CBMC -I " + bmc_path;
  }

  if (desired_property.size() > 0)
	command_line += " -DPROPERTY=" + desired_property;

  if (desired_x_size > 0)
	command_line += " -DK_SIZE=" + std::to_string(desired_x_size);

  command_line += desired_macro_parameters;

  return command_line;
}

digital_system ds;
implementation impl;
filter_parameters filter; 

/*******************************************************************\

Function: get_fxp_value

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_fxp_value(std::string exp)
{
  std::vector<std::string> tokens;
  boost::split(tokens, exp,boost::is_any_of("="));

  return std::atoi(tokens[1].c_str());
}

/*******************************************************************\

Function: extract_regexp_data_for_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void extract_regexp_data_for_vector(std::string src, std::regex & regexp, std::vector<double> & vector, unsigned int & factor)
{
  std::sregex_iterator next(src.begin(), src.end(), regexp);
  std::sregex_iterator end;
  while (next != end)
  {
	std::smatch match = *next;
	double value = (double) get_fxp_value(match.str()) / (double) factor;
	vector.push_back(value);
	next++;
  }
}

/*******************************************************************\

Function: print_counterexample_data_for_restricted_properties

  Inputs:

 Outputs:

 Purpose: print counterexample data for overflow/stability/minimum phase properties

\*******************************************************************/

void print_counterexample_data_for_restricted_properties()
{
  try
  {
	std::cout << std::endl << "Counterexample Data:" << std::endl;

	bool is_delta_realization = (desired_realization == "DDFI" || desired_realization == "DDFII" || desired_realization == "TDDFII");
	
	std::cout << "  Property = " << desired_property << std::endl;
	cplus_print_array_elements_ignoring_empty("  Numerator ", ds.b, ds.b_size);
	cplus_print_array_elements_ignoring_empty("  Denominator ", ds.a, ds.a_size);
	std::cout << "  Numerator Size = " << ds.b_size << std::endl;
	std::cout << "  Denominator Size = " << ds.a_size << std::endl;

	if (is_delta_realization)
		std::cout << "  Delta: " << impl.delta << std::endl;
	
	std::cout << "  X Size = " << desired_x_size << std::endl;
	std::cout << "  Sample Time = " << ds.sample_time << std::endl;
	std::cout << "  Implementation = " << "<" << impl.int_bits << "," << impl.frac_bits << ">" << std::endl;
	std::cout << "  Realization = " << desired_realization << std::endl;
	std::cout << "  Dynamic Range = " << "{" << impl.min << "," << impl.max << "}" << std::endl;

  }
  catch (std::regex_error& e)
  {
	std::cout << "[ERROR] It was not able to process the counterexample data. :(" << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: print_counterexample_data

  Inputs:

 Outputs:

 Purpose:

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
  if (verification_failed == std::string::npos)
	return;

  if (desired_bmc == "ESBMC")
  {
	std::cout << std::endl <<
	  "[INFO] Unfortunately, DSVerifier is unable to extract the counterexample data."
	  << std::endl;
	return;
  }

  try
  {
	/* process quantized numerator data */
	std::regex num_fxp_regexp("b_fxp\\[[0-9]+l?\\]=-?[0-9]+l?");
	extract_regexp_data_for_vector(counterexample, num_fxp_regexp, numerator, factor);

	/* process quantized denominator data */
	std::regex den_fxp_regexp("a_fxp\\[[0-9]+l?\\]=-?[0-9]+l?");
	extract_regexp_data_for_vector(counterexample, den_fxp_regexp, denominator, factor);

	/* process input data */
	std::regex input_regexp(" x\\[[0-9]+l?\\]=-?[0-9]+l?");
	extract_regexp_data_for_vector(counterexample, input_regexp, inputs, factor);

	/* process output data */
	std::regex output_regexp("y\\[[0-9]+l?\\]=-?[0-9]+l?");
	extract_regexp_data_for_vector(counterexample, output_regexp, outputs, factor);

	/* process initial states data */
	std::regex initial_states_regexp("y0\\[[0-9]+l?\\]=-?[0-9]+l?");
	extract_regexp_data_for_vector(counterexample, initial_states_regexp, initial_states, factor);
	if (initial_states.size() == 0)
	{
	  std::regex initial_states_regexp_df2("w0\\[[0-9]+l?\\]=-?[0-9]+l?");
	  extract_regexp_data_for_vector(counterexample, initial_states_regexp_df2, initial_states, factor);
	}

	bool is_delta_realization = (desired_realization == "DDFI" || desired_realization == "DDFII" || desired_realization == "TDDFII");

	//TODO: extend this counterexample data to closed-loop systems
	std::cout << std::endl << "Counterexample Data:" << std::endl;

	std::cout << "  Property = " << desired_property << std::endl;
	cplus_print_array_elements_ignoring_empty("  Numerator ", ds.b, ds.b_size);
	cplus_print_array_elements_ignoring_empty("  Denominator ", ds.a, ds.a_size);

	if (is_delta_realization)
	  std::cout << "  Delta: " << impl.delta << std::endl;

	std::cout << "  X Size = " << desired_x_size << std::endl;
	std::cout << "  Sample Time = " << ds.sample_time << std::endl;
	std::cout << "  Implementation = " << "<" << impl.int_bits << "," << impl.frac_bits << ">" << std::endl;

	cplus_print_array_elements_ignoring_empty("  Numerator (fixed-point)", &numerator[0], numerator.size());
	cplus_print_array_elements_ignoring_empty("  Denominator (fixed-point)", &denominator[0], denominator.size());

	std::cout << "  Realization = " << desired_realization << std::endl;
	std::cout << "  Dynamic Range = " << "{" << impl.min << "," << impl.max << "}" << std::endl;

	cplus_print_array_elements_ignoring_empty("  Initial States", &initial_states[0], initial_states.size());
	cplus_print_array_elements_ignoring_empty("  Inputs", &inputs[0], inputs.size());
	cplus_print_array_elements_ignoring_empty("  Outputs", &outputs[inputs.size()], outputs.size() - inputs.size());
  }
  catch (std::regex_error& e)
  {
	std::cout << "[ERROR] It was not able to process the counterexample data. :(" << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: get_roots_from_polynomial

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_roots_from_polynomial(double polynomial[], int poly_size, std::vector<RootType> & roots)
{
  unsigned int size = poly_size;

  /* coefficients */
  std::vector<double> coefficients_vector;
  for (int i=0; i< poly_size; i++)
	coefficients_vector.push_back(polynomial[i]);

  /* remove leading zeros */
  std::vector<double>::iterator it=coefficients_vector.begin();
  while(it != coefficients_vector.end())
  {
	if(*it != 0.0)
	  break;
	it=coefficients_vector.erase(it);
  }

  size=coefficients_vector.size();

  /* check if there is any element left on the vector */
  if(!size)
  {
	std::cout << std::endl << "No remaining elements in polynomial vector";
	throw std::runtime_error ("tla");
  }

  /* check the polynomial order */
  if (coefficients_vector.size() >= 3)
  {
	Eigen::VectorXd coefficients(coefficients_vector.size());

	/* copy elements from the list to the array - insert in reverse order */
	unsigned int i=0;
	for(it=coefficients_vector.begin();
		it!=coefficients_vector.end();
		++it, ++i)
	{
	  coefficients[size-i-1] = *it;
	}

	/* eigen solver object */
	Eigen::PolynomialSolver<double, Eigen::Dynamic> solver;

	/* solve denominator using QR decomposition */
	solver.compute(coefficients);

	RootsType solver_roots = solver.roots();

	for(unsigned int i=0; i<solver_roots.rows(); ++i)
	  roots.push_back(solver_roots[i]);

  }
  else if (coefficients_vector.size() == 2)
  {
	double root = - coefficients_vector.at(1) / coefficients_vector.at(0);
	roots.push_back(root);
  }

  return 0;
}

/*******************************************************************\

Function: check_delta_stability_margin

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool check_delta_stability_margin(std::vector<RootType> roots)
{
  std::cout << "checking delta stability margin" << std::endl;
  bool stable = true;
  for(unsigned int i=0; i<roots.size(); i++)
  {
	std::complex<double> eig = roots.at(i);
	eig.real(eig.real() * impl.delta);
	eig.imag(eig.imag() * impl.delta);
	eig.real(eig.real() + 1);
	if ((std::abs(eig) < 1) == false)
	{
	  stable = false;
	  break;
	}
  }

  return stable;
}

/*******************************************************************\

Function: check_shift_stability_margin

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool check_shift_stability_margin(std::vector<RootType> roots)
{
  std::cout << "checking shift stability margin" << std::endl;
  bool stable = true;
  for(unsigned int i=0; i<roots.size(); i++)
  {
	std::complex<double> eig = roots.at(i);
	if ((std::abs(eig) < 1) == false)
	{
	  stable = false;
	  break;
	}
  }

  return stable;
}

/*******************************************************************\

Function: show_implementation_parameters

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_implementation_parameters()
{
  std::cout << std::endl << "implementation int_bits: " << impl.int_bits << std::endl;
  std::cout << "implementation frac_bits: " << impl.frac_bits << std::endl;
  std::cout << "implementation delta: " << impl.delta << std::endl;
}

/*******************************************************************\

Function: check_stability_shift_domain_using_jury

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_stability_shift_domain_using_jury()
{
  show_implementation_parameters();
  std::cout << std::endl;
  double sa_fxp[ds.a_size];
  cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  fxp_t a_fxp[ds.a_size];
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  fxp_to_double_array(sa_fxp, a_fxp, ds.a_size);
  cplus_print_array_elements("quantized denominator", sa_fxp, ds.a_size);
  bool is_stable = check_stability(sa_fxp, ds.a_size);
  if (is_stable)
  {
	show_verification_successful();
  }
  else
  {
	show_verification_failed();
  }
}

/*******************************************************************\

Function: check_minimum_phase_shift_domain_using_jury

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_minimum_phase_shift_domain_using_jury()
{
  show_implementation_parameters();
  std::cout << std::endl;
  double sb_fxp[ds.b_size];
  cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  fxp_t b_fxp[ds.b_size];
  fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
  fxp_to_double_array(sb_fxp, b_fxp, ds.b_size);
  cplus_print_array_elements("quantized denominator", sb_fxp, ds.b_size);
  bool is_stable = check_stability(sb_fxp, ds.b_size);
  if (is_stable)
  {
	show_verification_successful();
  }
  else
  {
	show_verification_failed();
  }
}

/*******************************************************************\

Function: check_stability_shift_domain_using_eigen

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_stability_shift_domain_using_eigen()
{
  show_implementation_parameters();
  std::cout << std::endl;
  double sa_fxp[ds.a_size];
  cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  fxp_t a_fxp[ds.a_size];
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  fxp_to_double_array(sa_fxp, a_fxp, ds.a_size);
  cplus_print_array_elements("quantized denominator", sa_fxp, ds.a_size);
  std::vector<RootType> poly_roots;
  get_roots_from_polynomial(sa_fxp, ds.a_size, poly_roots);
  bool is_stable = check_shift_stability_margin(poly_roots);

  if (is_stable)
	show_verification_successful();
  else
	show_verification_failed();
}

/*******************************************************************\

Function: check_minimum_phase_shift_domain

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_minimum_phase_shift_domain()
{
  show_implementation_parameters();
  std::cout << std::endl;
  double sb_fxp[ds.b_size];
  cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  fxp_t b_fxp[ds.b_size];
  fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
  fxp_to_double_array(sb_fxp, b_fxp, ds.b_size);
  cplus_print_array_elements("quantized numerator", sb_fxp, ds.b_size);
  bool is_stable = check_stability(sb_fxp, ds.b_size);

  if (is_stable)
	show_verification_successful();
  else
	show_verification_failed();
}

/*******************************************************************\

Function: check_stability_delta_domain

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_stability_delta_domain()
{
  show_implementation_parameters();
  std::cout << std::endl;
  double db[ds.b_size];
  double da[ds.a_size];
  fxp_t a_fxp[ds.a_size];
  cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
  get_delta_transfer_function_with_base(ds.b, db, ds.b_size, ds.a, da, ds.a_size, impl.delta);
  cplus_print_array_elements("delta denominator", da, ds.a_size);
  cplus_print_array_elements("delta numerator", db, ds.b_size);
  fxp_t da_fxp[ds.a_size];
  try
  {
	fxp_double_to_fxp_array(da, da_fxp, ds.a_size);
  }
  catch (int e)
  {
	std::cout << "an fixed-point arithmetic overflow occurs after delta transformation" << std::endl;
	show_verification_failed();
	exit(1);
  }
  if ((da[0] != 0) && (da_fxp[0] == 0))
  {
	std::cout << std::endl;
	std::cout << "ds.a[0] = "<< std::to_string(da[0]) << " ----> " << std::to_string(da_fxp[0]) << std::endl;
	show_delta_not_representable();
	show_underflow_message();
	show_verification_error();
	exit(1);
  }
  double da_qtz[ds.a_size];
  fxp_to_double_array(da_qtz, da_fxp, ds.a_size);
  cplus_print_array_elements("quantized delta denominator", da_qtz, ds.a_size);
  std::vector<RootType> poly_roots;
  get_roots_from_polynomial(da_qtz, ds.a_size, poly_roots);
  bool is_stable = check_delta_stability_margin(poly_roots);

  if (is_stable)
	show_verification_successful();
  else
	show_verification_failed();
}

/*******************************************************************\

Function: check_if_file_exists

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool check_if_file_exists (const std::string & name)
{
  if (FILE *file = fopen(name.c_str(), "r"))
  {
	fclose(file);
	return true;
  }
  else
  {
	return false;
  }
}

/*******************************************************************\

Function: check_minimum_phase_delta_domain

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_minimum_phase_delta_domain()
{
  show_implementation_parameters();
  std::cout << std::endl;
  double db[ds.b_size];
  double da[ds.a_size];
  cplus_print_array_elements("original numerator", ds.b, ds.b_size);
  cplus_print_array_elements("original denominator", ds.a, ds.a_size);
  get_delta_transfer_function_with_base(ds.b, db, ds.b_size, ds.a, da, ds.a_size, impl.delta);
  cplus_print_array_elements("delta numerator", db, ds.b_size);
  cplus_print_array_elements("delta denominator", da, ds.a_size);
  fxp_t db_fxp[ds.b_size];
  fxp_double_to_fxp_array(db, db_fxp, ds.b_size);

  if ((db[0] != 0) && (db_fxp[0] == 0))
  {
	std::cout << std::endl;
	std::cout << "ds.b[0] = "<< std::to_string(db[0]) << " ----> " << std::to_string(db_fxp[0]) << std::endl;
	show_delta_not_representable();
	show_underflow_message();
	show_verification_error();
	exit(1);
  }

  double db_qtz[ds.b_size];
  fxp_to_double_array(db_qtz, db_fxp, ds.b_size);
  cplus_print_array_elements("quantized delta numerator", db_qtz, ds.b_size);
  std::vector<RootType> poly_roots;
  get_roots_from_polynomial(db_qtz, ds.b_size, poly_roots);
  bool is_stable = check_delta_stability_margin(poly_roots);

  if (is_stable)
	show_verification_successful();
  else
	show_verification_failed();
}

/*******************************************************************\

Function: check_state_space_stability

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_state_space_stability()
{
  Eigen::MatrixXd matrixA = Eigen::MatrixXd::Ones(_controller.nStates,_controller.nStates);
  int i, j;

  for(i=0; i<_controller.nStates;i++)
  {
	for(j=0; j<_controller.nStates;j++)
	{
	  matrixA(i,j) = _controller.A[i][j]; //fxp_double_to_fxp(A[i][j]);
	}
  }

  std::complex< double > lambda;
  std::complex< double > margem(1,0);

  for(i=0; i<_controller.nStates; i++ )
  {
	lambda = matrixA.eigenvalues()[i];
	std::cout << "abs(lambda): " << std::abs(lambda) << std::endl;
	double v = std::abs(lambda);
	if( v > 1.0 )
	{
		show_verification_failed(); //unstable
		exit(0);
	}
  }

  show_verification_successful(); //stable
}

/*******************************************************************\

Function: generates_mag_response

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
#define M_PI 3.14159265358979323846
#define LOWPASS 1
#define HIGHPASS 2

void generates_mag_response(double* num, int lnum, double* den, int lden, double* res, int N) {

	double w;
	int m, i;
	double out_numRe[N + 1];
	double out_numIm[N + 1];
	double out_denRe[N + 1];
	double out_denIm[N + 1];
	double old_out_Re;
	double zero_test;
	for (w = 0, i = 0; w <= M_PI; w += M_PI / N, ++i) {
		out_numRe[i] = num[0];
		out_numIm[i] = 0;
		for (m = 1; m < lnum; ++m) {
			old_out_Re = out_numRe[i];
			out_numRe[i] = cos(w) * out_numRe[i] - sin(w) * out_numIm[i] + num[m];
			out_numIm[i] = sin(w) * old_out_Re + cos(w) * out_numIm[i];
		}
		out_denRe[i] = den[0];
		out_denIm[i] = 0;

		for (m = 1; m < lden; ++m) {
			old_out_Re = out_denRe[i];
			out_denRe[i] = cos(w) * out_denRe[i] - sin(w) * out_denIm[i] + den[m];
			out_denIm[i] = sin(w) * old_out_Re + cos(w) * out_denIm[i];
		}

		res[i] = sqrt(out_numRe[i] * out_numRe[i] + out_numIm[i] * out_numIm[i]); 
	    zero_test = sqrt(out_denRe[i] * out_denRe[i] + out_denIm[i] * out_denIm[i]);
	 	//if (zero_test != 0)
		res[i] = res[i] / zero_test;
	}
}

/*******************************************************************\

Function: check_filter_magnitude_det

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_filter_magnitude_det()
{
    int freq_response_samples = 100;
	double w;
	double w_incr = 1.0 / freq_response_samples;
  	double res[freq_response_samples+1];
  	int i,j;
  	bool response_is_valid = true;

 	/* generates magnitude response of the quantized TF, placing the result in the "res" array*/	
 	generates_mag_response(ds.b, ds.b_size, ds.a, ds.a_size, res, freq_response_samples);


  	/* quantize "a" array using fxp */
	fxp_t a_fxp[ds.a_size];
	fxp_double_to_fxp_array(ds.a, a_fxp, ds.a_size);
	double _a[ds.a_size];
	fxp_to_double_array(_a, a_fxp, ds.a_size);


  	/* quantize "b" array using fxp */
	fxp_t b_fxp[ds.b_size];
	fxp_double_to_fxp_array(ds.b, b_fxp, ds.b_size);
	double _b[ds.b_size];
	fxp_to_double_array(_b, b_fxp, ds.b_size);


    /* generates magnitude response of the quantized TF, placing the result in the "res" array*/
    generates_mag_response(ds.b, ds.b_size, ds.a, ds.a_size, res, freq_response_samples);	

	if (filter.type == LOWPASS) { //lowpass
		for (i = 0, w = 0; (w <= 1.0); ++i, w += w_incr) {
			if (w <= filter.wp) {
				if(!(res[i] >= filter.Ap)) 
				{
					printf("|----------------Passband Failure-------------|");
					response_is_valid = false;
					break;
				} 
			} else if (w == filter.wc) {
				if(!(res[i] <= filter.Ac)) 
				{
					printf("|-------------Cutoff Frequency Failure--------|");
					response_is_valid = false;
					break; 
				}
			} else if ((w >= filter.wr) && (w <= 1)) {
				if(!(res[i] <= filter.Ar))
				{
					printf("|----------------Stopband Failure-------------|");
					response_is_valid = false;
					break; 
				} 
			}
		}

	} else if (filter.type == HIGHPASS) { //highpass
		for (i = 0, w = 0; (w <= 1.0); ++i, w += w_incr) {
			if (w <= filter.wr) {
				if(!(res[i] <= filter.Ar)) 
				{
					printf("|----------------Stopband Failure-------------|");
					response_is_valid = false;
					break; 
				}
			} else if (w == filter.wc) {
				if(!(res[i] <= filter.Ac)) 
				{
					printf("|-------------Cutoff Frequency Failure--------|");
					response_is_valid = false;
					break; 
				}
			} else if ((w > filter.wp) && (w <= 1)) {
				if(!(res[i] >= filter.Ap))
				{
					printf("|----------------Passband Failure-------------|");
					response_is_valid = false;
					break;
				} 
			}
		}
	} 
	
	if (response_is_valid == false)
		show_verification_failed();
	else 
		show_verification_successful();
}

/*******************************************************************\

Function: check_file_exists

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void check_file_exists()
{
  /* check if the specified file exists */
  if (check_if_file_exists(desired_filename) == false)
  {
	std::cout << "file " << desired_filename <<
	  ": failed to open input file" << std::endl;
	exit(1);
  }
}

/*******************************************************************\

Function: extract_data_from_ss_file

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void extract_data_from_ss_file()
{
  std::ifstream verification_file(desired_filename);
  std::string current_line;
  getline( verification_file, current_line );

  /* getting implementation specifications */
  std::string str_bits;
  int i;
  for(i = 0; current_line[i] != '<';i++);
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

  getline( verification_file, current_line ); // range

  std::string str_range;
  for(i = 0; current_line[i] != '[';i++);
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

  getline( verification_file, current_line ); // states

  for(i = 0; current_line[i] != '=';i++){}

  i++; i++;

  for(; current_line[i] != ';'; i++)
	str_bits.push_back(current_line[i]);

  int states = std::stoi(str_bits);
  str_bits.clear();

  getline( verification_file, current_line ); // inputs

  for(i = 0; current_line[i] != '=';i++){}

  i++; i++;

  for(; current_line[i] != ';'; i++)
	str_bits.push_back(current_line[i]);

  int inputs = std::stoi(str_bits);
  str_bits.clear();

  getline( verification_file, current_line ); // outputs

  for(i = 0; current_line[i] != '=';i++){}

  i++; i++;

  for(; current_line[i] != ';'; i++)
	str_bits.push_back(current_line[i]);

  int outputs = std::stoi(str_bits);
  str_bits.clear();

  /* Updating _controller */
  _controller.nStates = states;
  _controller.nInputs = inputs;
  _controller.nOutputs = outputs;

  /* initialising matrix A */
  getline( verification_file, current_line ); // matrix A
  str_bits.clear();

  for(i = 0; current_line[i] != '[';i++);
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
	  _controller.A[lines][columns] = std::stod(str_bits);
	  lines++;
	  columns = 0;
	  str_bits.clear();
	}
	else
	{
	  _controller.A[lines][columns] = std::stod(str_bits);
	  columns++;
	  str_bits.clear();
	}
  }

  _controller.A[lines][columns] = std::stod(str_bits);
  str_bits.clear();

  /* initialising matrix B */

  getline( verification_file, current_line ); // matrix B
  str_bits.clear();
  for(i = 0; current_line[i] != '[';i++);
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
	  _controller.B[lines][columns] = std::stod(str_bits);
	  lines++;
	  columns = 0;
	  str_bits.clear();
	}
	else
	{
	  _controller.B[lines][columns] = std::stod(str_bits);
	  columns++;
	  str_bits.clear();
	}
  }

  _controller.B[lines][columns] = std::stod(str_bits);
  str_bits.clear();

  /* initialising matrix C */
  getline( verification_file, current_line ); // matrix C
  str_bits.clear();
  for(i = 0; current_line[i] != '[';i++);
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
	  _controller.C[lines][columns] = std::stod(str_bits);
	  lines++;
	  columns = 0;
	  str_bits.clear();
	}
	else
	{
	  _controller.C[lines][columns] = std::stod(str_bits);
	  columns++;
	  str_bits.clear();
	}
  }

  _controller.C[lines][columns] = std::stod(str_bits);
  str_bits.clear();

  /* initialising matrix D */

  getline( verification_file, current_line ); // matrix D
  str_bits.clear();
  for(i = 0; current_line[i] != '[';i++);
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
	  _controller.D[lines][columns] = std::stod(str_bits);
	  lines++;
	  columns = 0;
	  str_bits.clear();
	}
	else
	{
	  _controller.D[lines][columns] = std::stod(str_bits);
	  columns++;
	  str_bits.clear();
	}
  }

  _controller.D[lines][columns] = std::stod(str_bits);
  str_bits.clear();

  /* initialising matrix Inputs */

  getline( verification_file, current_line ); // matrix inputs
  str_bits.clear();
  for(i = 0; current_line[i] != '[';i++);
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

  if(closedloop)
  {
	getline( verification_file, current_line ); // matrix controller
	str_bits.clear();
	for(i = 0; current_line[i] != '[';i++);
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
		_controller.K[lines][columns] = std::stod(str_bits);
		lines++;
		columns = 0;
		str_bits.clear();
	  }
	  else
	  {
		_controller.K[lines][columns] = std::stod(str_bits);
		columns++;
		str_bits.clear();
      }
    }

	_controller.K[lines][columns] = std::stod(str_bits);
  }

#if DEBUG_DSV
	print_matrix(_controller.K,1,states);
	print_matrix(_controller.B,states,inputs);
	print_matrix(_controller.C,outputs,states);
	print_matrix(_controller.D,outputs,inputs);
#endif
}

/*******************************************************************\

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

  verification_file = "#include <dsverifier.h>\n digital_system_state_space _controller;\n implementation impl = {\n .int_bits = ";
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
  cf_value_precision  << std::fixed << desired_quantization_limit;
  verification_file.append(cf_value_precision.str());
  verification_file.append(";\n int closed_loop = "+std::string(closedloop ? "1": "0"));
  verification_file.append(";\n void initials(){\n");

  for (i=0; i<_controller.nStates; i++)
  {
	for (j=0; j<_controller.nStates; j++)
	{
	  verification_file.append("\t_controller.A[");
	  verification_file.append(std::to_string(i));
	  verification_file.append("][");
	  verification_file.append(std::to_string(j));
	  verification_file.append("] = ");
	  cf_value_precision.str(std::string());
	  cf_value_precision << std::fixed << _controller.A[i][j];
	  verification_file.append(cf_value_precision.str());
	  verification_file.append(";\n");
	}
  }

  for (i=0; i<_controller.nStates; i++)
  {
	for (j=0; j<_controller.nInputs; j++)
	{
	  verification_file.append("\t_controller.B[");
	  verification_file.append(std::to_string(i));
	  verification_file.append("][");
	  verification_file.append(std::to_string(j));
	  verification_file.append("] = ");
	  cf_value_precision.str(std::string());
	  cf_value_precision << std::fixed << _controller.B[i][j];
	  verification_file.append(cf_value_precision.str());
	  verification_file.append(";\n");
	}
  }

  for (i=0; i<_controller.nOutputs; i++)
  {
	for (j=0; j<_controller.nStates; j++)
	{
	  verification_file.append("\t_controller.C[");
	  verification_file.append(std::to_string(i));
	  verification_file.append("][");
	  verification_file.append(std::to_string(j));
	  verification_file.append("] = ");
	  cf_value_precision.str(std::string());
	  cf_value_precision << std::fixed << _controller.C[i][j];
	  verification_file.append(cf_value_precision.str());
	  verification_file.append(";\n");
	}
  }

  for (i=0; i<_controller.nOutputs; i++)
  {
	for (j=0; j<_controller.nInputs; j++)
	{
	  verification_file.append("\t_controller.D[");
	  verification_file.append(std::to_string(i));
	  verification_file.append("][");
	  verification_file.append(std::to_string(j));
	  verification_file.append("] = ");
	  cf_value_precision.str(std::string());
	  cf_value_precision << std::fixed << _controller.D[i][j];
	  verification_file.append(cf_value_precision.str());
	  verification_file.append(";\n");
	}
  }

  for (i=0; i<_controller.nInputs; i++)
  {
	for (j=0; j < 1; j++)
	{
	  verification_file.append("\t_controller.inputs[");
	  verification_file.append(std::to_string(i));
	  verification_file.append("][");
	  verification_file.append(std::to_string(j));
	  verification_file.append("] = ");
	  cf_value_precision.str(std::string());
	  cf_value_precision << std::fixed << _controller.inputs[i][j];
	  verification_file.append(cf_value_precision.str());
	  verification_file.append(";\n");
	}
  }

  if(closedloop)
  {
	for (i=0; i<_controller.nOutputs; i++)
	{
	  for (j=0; j<_controller.nStates; j++)
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

  std::ofstream myfile ("input.c");

  if (myfile.is_open())
  {
	myfile << verification_file;
	myfile.close();
  }
  else
    std::cout << "Unable to open file";
}

/*******************************************************************\

Function: closed_loop

  Inputs: None

 Outputs:

 Purpose: Compute A-B*K and C-D*K

\*******************************************************************/

void closed_loop()
{
  double result1[LIMIT][LIMIT];

  int i, j, k;
  for(i=0; i<LIMIT;i++)
	for(j=0; j<LIMIT;j++)
	  result1[i][j]=0;

  // B*K
  double_matrix_multiplication(
	_controller.nStates,
	_controller.nInputs,
    _controller.nInputs,
    _controller.nStates,
    _controller.B,
    _controller.K,
    result1);

  double_sub_matrix(
	_controller.nStates,
	_controller.nStates,
	_controller.A,
	result1,
	_controller.A);

  for(i=0; i<LIMIT;i++)
	for(j=0; j<LIMIT;j++)
	  result1[i][j]=0;

  //D*K
  double_matrix_multiplication(
	_controller.nOutputs,
	_controller.nInputs,
    _controller.nInputs,
    _controller.nStates,
    _controller.D,
    _controller.K,
    result1);

  double_sub_matrix(
    _controller.nOutputs,
	_controller.nStates,
	_controller.C,
	result1,
	_controller.C);
}

/*******************************************************************\

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

  for (i=0; i<_controller.nStates - 1; i++)
  {
	for (j=0; j<_controller.nStates; j++)
	{
	  if(j == i + 1)
	    _controller.A[i][j] = 1;
	  else
	    _controller.A[i][j] = 0;
	}
  }

  for (j=0; j<_controller.nStates; j++)
    _controller.A[_controller.nStates - 1][j] = - ds.b[_controller.nStates - j];

  for (i=0; i<_controller.nStates - 1; i++)
	for (j=0; j<_controller.nInputs; j++)
	  _controller.B[i][j] = 0;

  /* for SISO systems */
  _controller.B[_controller.nStates - 1][0] = 1;

  for (i=0; i<_controller.nOutputs; i++)
	for (j=0; j<_controller.nStates; j++)
	  _controller.C[i][j] = ds.a[_controller.nStates - j] - (ds.b[_controller.nStates - j] * ds.a[0]);

  for (i=0; i<_controller.nOutputs; i++)
	for (j=0; j<_controller.nInputs; j++)
	  _controller.D[i][j] = ds.a[0];
}

/*******************************************************************\

Function: main

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

/* main function */
int main(int argc, char* argv[])
{
  /* without overflow */
  overflow_mode = NONE;

  bind_parameters(argc, argv);

  if(desired_property == "QUANTIZATION_ERROR" && desired_quantization_limit == 0.0)
    show_required_argument_message("QUANTIZATION_ERROR");

  check_file_exists();

  std::cout << "Running: Digital Systems Verifier (DSVerifier)" << std::endl;

  if(translate)
  {
	extract_data_from_file();
	tf2ss();
	state_space_parser();
	exit(0);
  }

  if (stateSpaceVerification)
  {
	extract_data_from_ss_file();

	if(closedloop && desired_property != "QUANTIZATION_ERROR")
	  closed_loop();

	if( desired_property == "STABILITY" )
	{
	  std::cout << "Checking stability..." << std::endl;
	  check_state_space_stability();
	  exit(0);
	}
	else if(desired_property == "QUANTIZATION_ERROR" ||
			desired_property == "SAFETY_STATE_SPACE" ||
			desired_property == "CONTROLLABILITY" ||
			desired_property == "OBSERVABILITY" ||
			desired_property == "LIMIT_CYCLE_STATE_SPACE")
	{
	  state_space_parser();
	  std::string command_line = prepare_bmc_command_line_ss();
	  std::cout << "Back-end Verification: " << command_line << std::endl;
	  execute_command_line(command_line);
	  exit(0);
	}
	else
	{
	  std::cout << "There is no check!" << std::endl;
	}
  }
  else
  {
	bool is_delta_realization = (desired_realization == "DDFI" ||
	  desired_realization == "DDFII" || desired_realization == "TDDFII");

	bool is_restricted_property = (desired_property == "STABILITY" ||
	  desired_property == "MINIMUM_PHASE");

	bool is_counterexample_property = ((desired_property == "OVERFLOW" && desired_bmc == "ESBMC") ||
	  desired_property == "STABILITY" || desired_property == "MINIMUM_PHASE");

	extract_data_from_file();

  	if (!((is_delta_realization && is_restricted_property)||(desired_property == "FILTER_MAGNITUDE_DET")))
	{
	  std::string command_line = prepare_bmc_command_line();
	  std::cout << "Back-end Verification: " << command_line << std::endl;
	  std::string counterexample = execute_command_line(command_line);
	  if (show_counterexample_data)
	  {
	    if(is_counterexample_property)
		{
          print_counterexample_data_for_restricted_properties();
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
		if (desired_property == "STABILITY")
			check_stability_delta_domain();
		else if (desired_property == "MINIMUM_PHASE")
			check_minimum_phase_delta_domain();
		else if (desired_property == "FILTER_MAGNITUDE_DET")
			check_filter_magnitude_det();

        if(show_counterexample_data)
          print_counterexample_data_for_restricted_properties();

		exit(0);

		}
	  catch(std::exception & e)
	  {
		std::cout << std::endl << "An unexpected error occurred in DSVerifier" << std::endl;
	  }
    }
  }
}