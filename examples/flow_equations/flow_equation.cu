#include "../../thrust_code/include/flow_equation_interface/flow_equation.hpp"

#include "lorentz_attractor/lorentz_attractor_flow_equation.hpp"

FlowEquationsWrapper *FlowEquationsWrapper::make_flow_equation(const std::string theory)
{
	if (theory == "lorentz_attractor")
		return new LorentzAttractorFlowEquations(0);
	else
	{
		std:: cout << "ERROR: Flow equation not known" << std::endl;
		std::exit(EXIT_FAILURE);
	}
}