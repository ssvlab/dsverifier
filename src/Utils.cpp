/**
 * \file Utils.cpp
 *
 * \brief //TODO
 *
 * Authors: Felipe R. Monteiro <rms.felipe@gmail.com>
 *
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.md', which is part of this source code package.
 */

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
              "--no-div-by-zero-check -DBMC=ESBMC -I "
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
