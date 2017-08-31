/**
 * \file Stringst.h
 *
 * \brief //TODO
 *
 * Authors: Felipe R. Monteiro <rms.felipe@gmail.com>
 *
 * This file is subject to the terms and conditions defined in
 * file 'LICENSE.md', which is part of this source code package.
 */

#ifndef SRC_STRINGST_H_
#define SRC_STRINGST_H_

class Stringst
{
private:
    std::string desiredWordlengthMode;
    std::string desiredArithmeticMode;
    std::string desiredFilename;
    std::string desiredProperty;
    std::string desiredRealization;
    std::string desiredConnectionMode;
    std::string desiredErrorMode;
    std::string desiredRoundingMode;
    std::string desiredOverflowMode;
    std::string desiredTimeout;
    std::string desiredBMC;
    std::string desiredFunction;
    std::string desiredSolver;
    std::string desiredMacroParameters;
    std::string desiredDsId;

public:
    Stringst();
    virtual ~Stringst();
    std::string getDesiredArithmeticMode() const;
    void setDesiredArithmeticMode(std::string desiredArithmeticMode);
    std::string getDesiredBmc() const;
    void setDesiredBmc(std::string desiredBmc);
    std::string getDesiredConnectionMode() const;
    void setDesiredConnectionMode(std::string desiredConnectionMode);
    std::string getDesiredDsId() const;
    void setDesiredDsId(std::string desiredDsId);
    std::string getDesiredErrorMode() const;
    void setDesiredErrorMode(std::string desiredErrorMode);
    std::string getDesiredFilename() const;
    void setDesiredFilename(std::string desiredFilename);
    std::string getDesiredFunction() const;
    void setDesiredFunction(std::string desiredFunction);
    std::string getDesiredMacroParameters() const;
    void setDesiredMacroParameters(std::string desiredMacroParameters);
    std::string getDesiredOverflowMode() const;
    void setDesiredOverflowMode(std::string desiredOverflowMode);
    std::string getDesiredProperty() const;
    void setDesiredProperty(std::string desiredProperty);
    std::string getDesiredRealization() const;
    void setDesiredRealization(std::string desiredRealization);
    std::string getDesiredRoundingMode() const;
    void setDesiredRoundingMode(std::string desiredRoundingMode);
    std::string getDesiredSolver() const;
    void setDesiredSolver(std::string desiredSolver);
    std::string getDesiredTimeout() const;
    void setDesiredTimeout(std::string desiredTimeout);
    std::string getDesiredWordlengthMode() const;
    void setDesiredWordlengthMode(std::string desiredWordlengthMode);
};

#endif /* SRC_STRINGST_H_ */
