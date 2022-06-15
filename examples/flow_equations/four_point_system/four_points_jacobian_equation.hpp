#ifndef PROJECT_FOURPOINTSJACOBIANEQUATION_HPP
#define PROJECT_FOURPOINTSJACOBIANEQUATION_HPP

#include <math.h>
#include <tuple>

#include <odesolver/flow_equations/jacobian_equation.hpp>


struct FourPointsJacobianEquation0 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation0(const cudaT k) : k_(k),
		const_expr0_((1*1.0/2) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation1 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation1(const cudaT k) : k_(k),
		const_expr0_((16*1.0/3) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation2 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation2(const cudaT k) : k_(k),
		const_expr0_(4 * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation3 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation3(const cudaT k) : k_(k),
		const_expr0_((1*1.0/6) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation4 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation4(const cudaT k) : k_(k),
		const_expr0_(pow(M_PI, -1))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation5 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation5(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation6 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation6(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation7 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation7(const cudaT k) : k_(k),
		const_expr0_((4*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation8 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation8(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation9 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation9(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation10 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation10(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation11 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation11(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation12 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation12(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation13 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation13(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation14 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation14(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation15 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation15(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation16 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation16(const cudaT k) : k_(k),
		const_expr0_((8*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation17 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation17(const cudaT k) : k_(k),
		const_expr0_((-8*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation18 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation18(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation19 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation19(const cudaT k) : k_(k),
		const_expr0_(1*1.0/2),
		const_expr1_((1*1.0/114) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
};


struct FourPointsJacobianEquation20 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation20(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation21 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation21(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation22 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation22(const cudaT k) : k_(k),
		const_expr0_((2*1.0/102236415) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct FourPointsJacobianEquation23 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation23(const cudaT k) : k_(k),
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


struct FourPointsJacobianEquation24 : public odesolver::flowequations::JacobianEquation
{
	FourPointsJacobianEquation24(const cudaT k) : k_(k),
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


class FourPointsJacobianEquations : public odesolver::flowequations::JacobianEquationsWrapper
{
public:
	FourPointsJacobianEquations(const cudaT k) : k_(k)
	{
		jacobian_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::JacobianEquation>> {
			std::make_shared<FourPointsJacobianEquation0>(k),
			std::make_shared<FourPointsJacobianEquation1>(k),
			std::make_shared<FourPointsJacobianEquation2>(k),
			std::make_shared<FourPointsJacobianEquation3>(k),
			std::make_shared<FourPointsJacobianEquation4>(k),
			std::make_shared<FourPointsJacobianEquation5>(k),
			std::make_shared<FourPointsJacobianEquation6>(k),
			std::make_shared<FourPointsJacobianEquation7>(k),
			std::make_shared<FourPointsJacobianEquation8>(k),
			std::make_shared<FourPointsJacobianEquation9>(k),
			std::make_shared<FourPointsJacobianEquation10>(k),
			std::make_shared<FourPointsJacobianEquation11>(k),
			std::make_shared<FourPointsJacobianEquation12>(k),
			std::make_shared<FourPointsJacobianEquation13>(k),
			std::make_shared<FourPointsJacobianEquation14>(k),
			std::make_shared<FourPointsJacobianEquation15>(k),
			std::make_shared<FourPointsJacobianEquation16>(k),
			std::make_shared<FourPointsJacobianEquation17>(k),
			std::make_shared<FourPointsJacobianEquation18>(k),
			std::make_shared<FourPointsJacobianEquation19>(k),
			std::make_shared<FourPointsJacobianEquation20>(k),
			std::make_shared<FourPointsJacobianEquation21>(k),
			std::make_shared<FourPointsJacobianEquation22>(k),
			std::make_shared<FourPointsJacobianEquation23>(k),
			std::make_shared<FourPointsJacobianEquation24>(k)
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

#endif //PROJECT_FOURPOINTSJACOBIANEQUATION_HPP
