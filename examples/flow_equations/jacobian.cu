#include "../../thrust_code/include/flow_equation_interface/jacobian_equation.hpp"

#include "lorentz_attractor/lorentz_attractor_jacobian.hpp"

JacobianWrapper *JacobianWrapper::make_jacobian(const std::string theory)
{
	if (theory == "lorentz_attractor")
		return new LorentzAttractorJacobianEquations(0);
	else
	{
		std:: cout << "ERROR: Jacobian equation not known" << std::endl;
		std::exit(EXIT_FAILURE);
	}
}