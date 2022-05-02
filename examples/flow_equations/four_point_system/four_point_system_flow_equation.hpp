#ifndef PROJECT_FOURPOINTSYSTEMFLOWEQUATION_HPP
#define PROJECT_FOURPOINTSYSTEMFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include <flow_equation_interface/flow_equation.hpp>


struct FourPointSystemFlowEquation0 : public FlowEquation
{
	FourPointSystemFlowEquation0(const cudaT k) : k_(k),
		const_expr0_(-2 * (pow(M_PI, -1))),
		const_expr1_((1*1.0/180) * (pow(M_PI, -1))),
		const_expr2_((1*1.0/12) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemFlowEquation1 : public FlowEquation
{
	FourPointSystemFlowEquation1(const cudaT k) : k_(k),
		const_expr0_(-1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(1*1.0/2),
		const_expr3_(-2*1.0/15),
		const_expr4_(-1*1.0/90),
		const_expr5_(1*1.0/18),
		const_expr6_((6*1.0/5) * (pow(M_PI, -1))),
		const_expr7_((-1*1.0/240) * (pow(M_PI, -1))),
		const_expr8_((1*1.0/6) * (pow(M_PI, -1))),
		const_expr9_((1*1.0/8) * (pow(M_PI, -1))),
		const_expr10_((1*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
	const cudaT const_expr6_;
	const cudaT const_expr7_;
	const cudaT const_expr8_;
	const cudaT const_expr9_;
	const cudaT const_expr10_;
};


struct FourPointSystemFlowEquation2 : public FlowEquation
{
	FourPointSystemFlowEquation2(const cudaT k) : k_(k),
		const_expr0_(1*1.0/5),
		const_expr1_(2*1.0/5),
		const_expr2_(2*1.0/15),
		const_expr3_(1*1.0/2),
		const_expr4_(1*1.0/15),
		const_expr5_((1*1.0/13387716) * (pow(M_PI, -1))),
		const_expr6_((-2859388*1.0/2657205) * (pow(M_PI, -2))),
		const_expr7_((-1*1.0/12754584) * (pow(M_PI, -2))),
		const_expr8_((1*1.0/191318760) * (pow(M_PI, -2))),
		const_expr9_((-1*1.0/63772920) * (pow(M_PI, -2))),
		const_expr10_((-1*1.0/4251528) * (pow(M_PI, -2))),
		const_expr11_((1*1.0/382637520) * (pow(M_PI, -2))),
		const_expr12_((-1*1.0/3188646) * (pow(M_PI, -2))),
		const_expr13_((1*1.0/19131876) * (pow(M_PI, -2))),
		const_expr14_((-1*1.0/19131876) * (pow(M_PI, -2))),
		const_expr15_((-1*1.0/76527504) * (pow(M_PI, -2))),
		const_expr16_((-32830375*1.0/4251528) * (pow(M_PI, -2))),
		const_expr17_((2125764*1.0/6815761) * M_PI)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
	const cudaT const_expr6_;
	const cudaT const_expr7_;
	const cudaT const_expr8_;
	const cudaT const_expr9_;
	const cudaT const_expr10_;
	const cudaT const_expr11_;
	const cudaT const_expr12_;
	const cudaT const_expr13_;
	const cudaT const_expr14_;
	const cudaT const_expr15_;
	const cudaT const_expr16_;
	const cudaT const_expr17_;
};


struct FourPointSystemFlowEquation3 : public FlowEquation
{
	FourPointSystemFlowEquation3(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(-2*1.0/15),
		const_expr3_(-1*1.0/90),
		const_expr4_(1*1.0/18),
		const_expr5_((1*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
};


struct FourPointSystemFlowEquation4 : public FlowEquation
{
	FourPointSystemFlowEquation4(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_((-2859388*1.0/2657205) * (pow(M_PI, -2))),
		const_expr2_((-1*1.0/12754584) * (pow(M_PI, -2))),
		const_expr3_((1*1.0/191318760) * (pow(M_PI, -2))),
		const_expr4_((-1*1.0/63772920) * (pow(M_PI, -2))),
		const_expr5_((-1*1.0/4251528) * (pow(M_PI, -2))),
		const_expr6_((1*1.0/382637520) * (pow(M_PI, -2))),
		const_expr7_((-1*1.0/3188646) * (pow(M_PI, -2))),
		const_expr8_((1*1.0/19131876) * (pow(M_PI, -2))),
		const_expr9_((-1*1.0/19131876) * (pow(M_PI, -2))),
		const_expr10_((-1*1.0/76527504) * (pow(M_PI, -2))),
		const_expr11_((-32830375*1.0/4251528) * (pow(M_PI, -2))),
		const_expr12_((2125764*1.0/6815761) * M_PI)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
	const cudaT const_expr6_;
	const cudaT const_expr7_;
	const cudaT const_expr8_;
	const cudaT const_expr9_;
	const cudaT const_expr10_;
	const cudaT const_expr11_;
	const cudaT const_expr12_;
};


class FourPointSystemFlowEquations : public FlowEquationsWrapper
{
public:
	FourPointSystemFlowEquations(const cudaT k) : k_(k)
	{
		flow_equations_ = std::vector<std::shared_ptr<FlowEquation>> {
			std::make_shared<FourPointSystemFlowEquation0>(k_),
			std::make_shared<FourPointSystemFlowEquation1>(k_),
			std::make_shared<FourPointSystemFlowEquation2>(k_),
			std::make_shared<FourPointSystemFlowEquation3>(k_),
			std::make_shared<FourPointSystemFlowEquation4>(k_)
		};
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int dim_index) override
	{
		(*flow_equations_[dim_index])(derivatives, variables);
	}

	size_t get_dim() override
	{
		return dim_;
	}

	static std::string model_;
	static size_t dim_;
	static std::string explicit_variable_;
	static std::vector<std::string> explicit_functions_;

	json get_json() const override
	{
		return json {
			{"model", model_},
			{"dim", dim_},
			{"explicit_variable", explicit_variable_},
			{"explicit_functions", explicit_functions_}
		};
	}

private:
	const cudaT k_;
	std::vector<std::shared_ptr<FlowEquation>> flow_equations_;
};

#endif //PROJECT_FOURPOINTSYSTEMFLOWEQUATION_HPP
