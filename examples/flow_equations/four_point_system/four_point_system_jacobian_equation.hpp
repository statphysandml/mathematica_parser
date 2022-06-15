#ifndef PROJECT_FOURPOINTSYSTEMJACOBIAN_HPP
#define PROJECT_FOURPOINTSYSTEMJACOBIAN_HPP

#include <math.h>
#include <tuple>

#include <odesolver/flow_equations/jacobian_equation.hpp>


struct FourPointSystemJacobianEquation0 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation0(const cudaT k) : k_(k),
		const_expr0_((1*1.0/2) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation1 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation1(const cudaT k) : k_(k),
		const_expr0_((16*1.0/3) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation2 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation2(const cudaT k) : k_(k),
		const_expr0_(4 * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation3 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation3(const cudaT k) : k_(k),
		const_expr0_((1*1.0/6) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation4 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation4(const cudaT k) : k_(k),
		const_expr0_(pow(M_PI, -1))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation5 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation5(const cudaT k) : k_(k),
		const_expr0_(-1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(1*1.0/2),
		const_expr3_((1*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
};


struct FourPointSystemJacobianEquation6 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation6(const cudaT k) : k_(k),
		const_expr0_(-1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(1*1.0/2),
		const_expr3_(57 * M_PI),
		const_expr4_((1*1.0/570) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
};


struct FourPointSystemJacobianEquation7 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation7(const cudaT k) : k_(k),
		const_expr0_((4*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation8 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation8(const cudaT k) : k_(k),
		const_expr0_(-3*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((1*1.0/1140) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation9 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation9(const cudaT k) : k_(k),
		const_expr0_(-1*1.0/2),
		const_expr1_(1*1.0/2),
		const_expr2_((1*1.0/228) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation10 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation10(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((1*1.0/66938580) * (pow(M_PI, -1))),
		const_expr3_((-1*1.0/20447283) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
};


struct FourPointSystemJacobianEquation11 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation11(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((-1*1.0/114059340739845) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation12 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation12(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(91247472591876 * M_PI),
		const_expr3_((-1*1.0/228118681479690) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
};


struct FourPointSystemJacobianEquation13 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation13(const cudaT k) : k_(k),
		const_expr0_(-1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(1*1.0/2),
		const_expr3_((1*1.0/1368712088878140) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
};


struct FourPointSystemJacobianEquation14 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation14(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((-1*1.0/456237362959380) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation15 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation15(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((1*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation16 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation16(const cudaT k) : k_(k),
		const_expr0_((8*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation17 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation17(const cudaT k) : k_(k),
		const_expr0_((-8*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation18 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation18(const cudaT k) : k_(k),
		const_expr0_(4*1.0/15),
		const_expr1_(2*1.0/3),
		const_expr2_(4*1.0/3),
		const_expr3_(-47*1.0/2),
		const_expr4_(-1*1.0/2),
		const_expr5_(3*1.0/2),
		const_expr6_((1*1.0/19) * (pow(M_PI, -1)))
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
};


struct FourPointSystemJacobianEquation19 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation19(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_((1*1.0/114) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
};


struct FourPointSystemJacobianEquation20 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation20(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((1*1.0/20447283) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation21 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation21(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_((1*1.0/102236415) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct FourPointSystemJacobianEquation22 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation22(const cudaT k) : k_(k),
		const_expr0_((2*1.0/102236415) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointSystemJacobianEquation23 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation23(const cudaT k) : k_(k),
		const_expr0_(-1*1.0/2),
		const_expr1_(3*1.0/2),
		const_expr2_(1*1.0/2),
		const_expr3_((1*1.0/408945660) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
};


struct FourPointSystemJacobianEquation24 : public odesolver::flowequations::JacobianEquation
{
	FourPointSystemJacobianEquation24(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_(2453673960 * M_PI),
		const_expr2_((1*1.0/1226836980) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


class FourPointSystemJacobianEquations : public odesolver::flowequations::JacobianEquationsWrapper
{
public:
	FourPointSystemJacobianEquations(const cudaT k) : k_(k)
	{
		jacobian_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::JacobianEquation>> {
			std::make_shared<FourPointSystemJacobianEquation0>(k),
			std::make_shared<FourPointSystemJacobianEquation1>(k),
			std::make_shared<FourPointSystemJacobianEquation2>(k),
			std::make_shared<FourPointSystemJacobianEquation3>(k),
			std::make_shared<FourPointSystemJacobianEquation4>(k),
			std::make_shared<FourPointSystemJacobianEquation5>(k),
			std::make_shared<FourPointSystemJacobianEquation6>(k),
			std::make_shared<FourPointSystemJacobianEquation7>(k),
			std::make_shared<FourPointSystemJacobianEquation8>(k),
			std::make_shared<FourPointSystemJacobianEquation9>(k),
			std::make_shared<FourPointSystemJacobianEquation10>(k),
			std::make_shared<FourPointSystemJacobianEquation11>(k),
			std::make_shared<FourPointSystemJacobianEquation12>(k),
			std::make_shared<FourPointSystemJacobianEquation13>(k),
			std::make_shared<FourPointSystemJacobianEquation14>(k),
			std::make_shared<FourPointSystemJacobianEquation15>(k),
			std::make_shared<FourPointSystemJacobianEquation16>(k),
			std::make_shared<FourPointSystemJacobianEquation17>(k),
			std::make_shared<FourPointSystemJacobianEquation18>(k),
			std::make_shared<FourPointSystemJacobianEquation19>(k),
			std::make_shared<FourPointSystemJacobianEquation20>(k),
			std::make_shared<FourPointSystemJacobianEquation21>(k),
			std::make_shared<FourPointSystemJacobianEquation22>(k),
			std::make_shared<FourPointSystemJacobianEquation23>(k),
			std::make_shared<FourPointSystemJacobianEquation24>(k)
		};
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int row_idx, const int col_idx) override
	{
		(*jacobian_equations_[row_idx * dim_ + col_idx])(derivatives, variables);
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int matrix_idx) override
	{
		(*jacobian_equations_[matrix_idx])(derivatives, variables);
	}

	size_t get_dim() override
	{
		return dim_;
	}

	static std::string model_;
	static size_t dim_;

private:
	const cudaT k_;
	std::vector<std::shared_ptr<odesolver::flowequations::JacobianEquation>> jacobian_equations_;
};

#endif //PROJECT_FOURPOINTSYSTEMJACOBIAN_HPP
