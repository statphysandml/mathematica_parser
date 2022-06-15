#include "four_points_flow_equation.hpp"

std::string FourPointsFlowEquations::model_ = "four_points";
size_t FourPointsFlowEquations::dim_ = 5;
std::string FourPointsFlowEquations::explicit_variable_ = "k";
std::vector<std::string> FourPointsFlowEquations::explicit_functions_ = {"mu", "Lam3", "Lam4", "g3", "g4"};


struct comp_func_four_points0
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_points0(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (thrust::get<3>(t) * const_expr0_) + (-2 * thrust::get<1>(t)) + (thrust::get<3>(t) * (210 + (-960 * thrust::get<0>(t)) + (1920 * (pow(thrust::get<0>(t), 2)))) * (pow((1 + thrust::get<1>(t)), -3)) * const_expr1_) + (thrust::get<2>(t) * (-24 + (48 * thrust::get<4>(t))) * (pow((1 + thrust::get<1>(t)), -2)) * const_expr2_);
	}
};


void FourPointsFlowEquation0::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[4].end(), variables[3].end(), variables[2].end(), derivatives.end())), comp_func_four_points0(const_expr0_, const_expr1_, const_expr2_));
}


struct comp_func_four_points6
{
	const cudaT const_expr0_;
	const cudaT const_expr10_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;

	comp_func_four_points6(const cudaT const_expr0, const cudaT const_expr10, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4, const cudaT const_expr5)
		: const_expr0_(const_expr0), const_expr10_(const_expr10), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4), const_expr5_(const_expr5) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<0>(t) * (-1 + (const_expr0_ * (pow(thrust::get<3>(t), -1)) * ((2 * thrust::get<3>(t)) + (((-5 * (pow(thrust::get<3>(t), 2))) + (-47 * (pow(thrust::get<3>(t), const_expr2_)) * (pow(thrust::get<2>(t), const_expr1_)) * (pow((1 + thrust::get<1>(t)), -2))) + ((pow(thrust::get<3>(t), 2)) * ((const_expr3_ * (299 + (-1780 * thrust::get<0>(t)) + (3640 * (pow(thrust::get<0>(t), 2))) + (-2336 * (pow(thrust::get<0>(t), 3)))) * (pow((1 + thrust::get<1>(t)), -5))) + (const_expr4_ * (-1470 + (14880 * thrust::get<0>(t)) + (-20280 * (pow(thrust::get<0>(t), 2))) + (-103680 * (pow(thrust::get<0>(t), 3)))) * (pow((1 + thrust::get<1>(t)), -4))))) + (thrust::get<3>(t) * thrust::get<2>(t) * ((16 * (1 + (-3 * thrust::get<0>(t))) * thrust::get<4>(t) * (pow((1 + thrust::get<1>(t)), -4))) + (const_expr5_ * (360 + (-48 * ((30 * thrust::get<0>(t)) + (-59 * thrust::get<4>(t)))) + (-1440 * thrust::get<0>(t) * thrust::get<4>(t))) * (pow((1 + thrust::get<1>(t)), -3)))))) * const_expr10_))));
	}
};


struct comp_func_four_points5
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr9_;

	comp_func_four_points5(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr9)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr9_(const_expr9) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), const_expr0_)) * (pow(thrust::get<3>(t), const_expr1_)) * (8 + (-24 * thrust::get<0>(t))) * (pow((1 + thrust::get<2>(t)), -2)) * const_expr9_;
	}
};


struct comp_func_four_points4
{
	const cudaT const_expr8_;

	comp_func_four_points4(const cudaT const_expr8)
		: const_expr8_(const_expr8) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<3>(t) * ((24 * thrust::get<0>(t)) + (-96 * thrust::get<1>(t) * thrust::get<0>(t))) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr8_;
	}
};


struct comp_func_four_points3
{
	const cudaT const_expr7_;

	comp_func_four_points3(const cudaT const_expr7)
		: const_expr7_(const_expr7) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<1>(t) * (132 + (-720 * thrust::get<0>(t)) + (960 * (pow(thrust::get<0>(t), 2))) + (-480 * (pow(thrust::get<0>(t), 3)))) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr7_;
	}
};


struct comp_func_four_points2
{
	const cudaT const_expr6_;

	comp_func_four_points2(const cudaT const_expr6)
		: const_expr6_(const_expr6) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1)
	{
		return val1 * const_expr6_;
	}
};


struct comp_func_four_points1
{
	comp_func_four_points1()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<2>(t) + thrust::get<3>(t) + thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<4>(t);
	}
};


void FourPointsFlowEquation1::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	dev_vec inter_med_vec4(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[2].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[4].end(), variables[3].end(), variables[2].end(), inter_med_vec4.end())), comp_func_four_points6(const_expr0_, const_expr10_, const_expr1_, const_expr2_, const_expr3_, const_expr4_, const_expr5_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec1.end())), comp_func_four_points5(const_expr0_, const_expr1_, const_expr9_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[0].end(), variables[4].end(), inter_med_vec0.end())), comp_func_four_points4(const_expr8_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec3.end())), comp_func_four_points3(const_expr7_));
	thrust::transform(variables[3].begin(), variables[3].end(), inter_med_vec2.begin(), comp_func_four_points2(const_expr6_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), inter_med_vec4.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), inter_med_vec3.end(), inter_med_vec4.end(), derivatives.end())), comp_func_four_points1());
}


struct comp_func_four_points20
{
	const cudaT const_expr13_;

	comp_func_four_points20(const cudaT const_expr13)
		: const_expr13_(const_expr13) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * (-357694770 + (4 * thrust::get<1>(t) * (141599312 + (-401719146 * thrust::get<0>(t)))) + (703069912 * thrust::get<0>(t)) + ((pow(thrust::get<1>(t), 2)) * (1324702194 + (501442416 * thrust::get<0>(t))))) * (pow((1 + thrust::get<3>(t)), -4)) * const_expr13_;
	}
};


struct comp_func_four_points19
{
	const cudaT const_expr12_;

	comp_func_four_points19(const cudaT const_expr12)
		: const_expr12_(const_expr12) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<2>(t) * (16061481 + (8 * thrust::get<0>(t) * (-5610604 + (5355213 * thrust::get<0>(t))))) * (pow((1 + thrust::get<1>(t)), -4)) * const_expr12_;
	}
};


struct comp_func_four_points18
{
	const cudaT const_expr11_;

	comp_func_four_points18(const cudaT const_expr11)
		: const_expr11_(const_expr11) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * (-3238317810 + (-10 * (pow(thrust::get<1>(t), 2)) * (-777791658 + (-1171611936 * thrust::get<0>(t)))) + (80 * thrust::get<1>(t) * (89438368 + (-101648442 * thrust::get<0>(t)))) + (1625502880 * thrust::get<0>(t))) * (pow((1 + thrust::get<3>(t)), -4)) * const_expr11_;
	}
};


struct comp_func_four_points17
{
	const cudaT const_expr10_;
	const cudaT const_expr3_;

	comp_func_four_points17(const cudaT const_expr10, const cudaT const_expr3)
		: const_expr10_(const_expr10), const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), const_expr3_)) * (pow(thrust::get<3>(t), const_expr3_)) * ((8937232 * thrust::get<0>(t) * (-4 + (9 * thrust::get<0>(t)))) + (-1784609 * (-17 + (32 * thrust::get<0>(t))))) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr10_;
	}
};


struct comp_func_four_points16
{
	const cudaT const_expr9_;

	comp_func_four_points16(const cudaT const_expr9)
		: const_expr9_(const_expr9) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * (-970172435 + (1633705330 * thrust::get<0>(t)) + (-40 * thrust::get<1>(t) * (-87715448 + (222044563 * thrust::get<0>(t)))) + (8 * (pow(thrust::get<1>(t), 2)) * (-531595195 + (1662847997 * thrust::get<0>(t))))) * (pow((1 + thrust::get<3>(t)), -5)) * const_expr9_;
	}
};


struct comp_func_four_points15
{
	const cudaT const_expr9_;

	comp_func_four_points15(const cudaT const_expr9)
		: const_expr9_(const_expr9) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * ((-26769135 * (17 + (8 * thrust::get<1>(t) * (-8 + (9 * thrust::get<1>(t)))))) + (2 * (353519805 + (4 * thrust::get<1>(t) * (-514449355 + (742510961 * thrust::get<1>(t))))) * thrust::get<0>(t))) * (pow((1 + thrust::get<3>(t)), -5)) * const_expr9_;
	}
};


struct comp_func_four_points14
{
	const cudaT const_expr8_;

	comp_func_four_points14(const cudaT const_expr8)
		: const_expr8_(const_expr8) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), 2)) * (pow(thrust::get<3>(t), -1)) * (1502649636 + (-5687823600 * thrust::get<0>(t)) + (-2679298800 * (pow(thrust::get<0>(t), 2))) + (38630196192 * (pow(thrust::get<0>(t), 3))) + (-69322630560 * (pow(thrust::get<0>(t), 4)))) * (pow((1 + thrust::get<2>(t)), -5)) * const_expr8_;
	}
};


struct comp_func_four_points13
{
	const cudaT const_expr7_;

	comp_func_four_points13(const cudaT const_expr7)
		: const_expr7_(const_expr7) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), 2)) * (pow(thrust::get<3>(t), -1)) * (112533531 + (-855576992 * thrust::get<0>(t)) + (3683259968 * (pow(thrust::get<0>(t), 2))) + (-7947008128 * (pow(thrust::get<0>(t), 3))) + (6385327072 * (pow(thrust::get<0>(t), 4)))) * (pow((1 + thrust::get<2>(t)), -6)) * const_expr7_;
	}
};


struct comp_func_four_points12
{
	const cudaT const_expr6_;

	comp_func_four_points12(const cudaT const_expr6)
		: const_expr6_(const_expr6) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (pow(val1, 2)) * (pow(val2, -1)) * const_expr6_;
	}
};


struct comp_func_four_points11
{
	comp_func_four_points11()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<3>(t) + thrust::get<2>(t) + thrust::get<7>(t) + thrust::get<0>(t) + thrust::get<8>(t) + thrust::get<1>(t) + thrust::get<5>(t) + thrust::get<6>(t) + thrust::get<4>(t);
	}
};


struct comp_func_four_points10
{
	const cudaT const_expr16_;

	comp_func_four_points10(const cudaT const_expr16)
		: const_expr16_(const_expr16) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return val2 * (pow((1 + val1), -2)) * const_expr16_;
	}
};


struct comp_func_four_points9
{
	const cudaT const_expr15_;

	comp_func_four_points9(const cudaT const_expr15)
		: const_expr15_(const_expr15) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<2>(t) * (-90445640 + (-367789656 * thrust::get<0>(t)) + (-1235175840 * (pow(thrust::get<0>(t), 2)))) * (pow((1 + thrust::get<1>(t)), -3)) * const_expr15_;
	}
};


struct comp_func_four_points8
{
	const cudaT const_expr14_;
	const cudaT const_expr3_;

	comp_func_four_points8(const cudaT const_expr14, const cudaT const_expr3)
		: const_expr14_(const_expr14), const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), const_expr3_)) * (pow(thrust::get<3>(t), const_expr3_)) * (-273938712 + (-45067812 * thrust::get<0>(t)) + (2 * (258770766 + (-8966844 * thrust::get<0>(t))) * thrust::get<0>(t))) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr14_;
	}
};


struct comp_func_four_points7
{
	const cudaT const_expr0_;
	const cudaT const_expr17_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;

	comp_func_four_points7(const cudaT const_expr0, const cudaT const_expr17, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4, const cudaT const_expr5)
		: const_expr0_(const_expr0), const_expr17_(const_expr17), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4), const_expr5_(const_expr5) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = (((20852576 * (pow(thrust::get<7>(t), 2)) * (pow(thrust::get<6>(t), -1))) + (const_expr0_ * (pow(thrust::get<7>(t), 2)) * (pow(thrust::get<6>(t), -1)) * (-40238058 + (250557792 * thrust::get<0>(t)) + (-360274560 * (pow(thrust::get<0>(t), 2))) + (-705294720 * (pow(thrust::get<0>(t), 3))) + (1488964032 * (pow(thrust::get<0>(t), 4)))) * (pow((1 + thrust::get<4>(t)), -5))) + (const_expr1_ * thrust::get<7>(t) * (15762012 + (-272657520 * (pow(thrust::get<0>(t), 2)) * thrust::get<8>(t)) + (-33775740 * ((2 * thrust::get<0>(t)) + thrust::get<8>(t))) + (120091520 * thrust::get<0>(t) * (thrust::get<0>(t) + (2 * thrust::get<8>(t))))) * (pow((1 + thrust::get<4>(t)), -4))) + (const_expr2_ * (pow(thrust::get<7>(t), const_expr3_)) * (pow(thrust::get<6>(t), const_expr3_)) * (-39405030 + (715645920 * thrust::get<0>(t)) + (-2502309120 * (pow(thrust::get<0>(t), 2)))) * (pow((1 + thrust::get<4>(t)), -3))) + (const_expr4_ * thrust::get<6>(t) * (-50663610 + (180137280 * thrust::get<8>(t)) + (-1699046160 * (pow(thrust::get<8>(t), 2)))) * (pow((1 + thrust::get<4>(t)), -3))) + (const_expr3_ * thrust::get<6>(t) * (35782296 + (-98340024 * thrust::get<0>(t))) * (pow((1 + thrust::get<4>(t)), -2)))) * const_expr5_) + (-1 * (pow(thrust::get<6>(t), -1)) * thrust::get<8>(t) * ((2 * thrust::get<6>(t)) + (thrust::get<6>(t) * (thrust::get<1>(t) + thrust::get<2>(t) + thrust::get<3>(t) + thrust::get<5>(t)) * const_expr17_)));
	}
};


void FourPointsFlowEquation2::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	dev_vec inter_med_vec4(derivatives.size());
	dev_vec inter_med_vec5(derivatives.size());
	dev_vec inter_med_vec6(derivatives.size());
	dev_vec inter_med_vec7(derivatives.size());
	dev_vec inter_med_vec8(derivatives.size());
	dev_vec inter_med_vec9(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec5.end())), comp_func_four_points20(const_expr13_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[4].end(), inter_med_vec7.end())), comp_func_four_points19(const_expr12_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec6.end())), comp_func_four_points18(const_expr11_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec2.end())), comp_func_four_points17(const_expr10_, const_expr3_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec9.end())), comp_func_four_points16(const_expr9_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec3.end())), comp_func_four_points15(const_expr9_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec8.end())), comp_func_four_points14(const_expr8_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec1.end())), comp_func_four_points13(const_expr7_));
	thrust::transform(variables[3].begin(), variables[3].end(), variables[4].begin(), inter_med_vec4.begin(), comp_func_four_points12(const_expr6_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec3.begin(), inter_med_vec2.begin(), inter_med_vec1.begin(), inter_med_vec4.begin(), inter_med_vec5.begin(), inter_med_vec6.begin(), inter_med_vec7.begin(), inter_med_vec8.begin(), inter_med_vec9.begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec3.end(), inter_med_vec2.end(), inter_med_vec1.end(), inter_med_vec4.end(), inter_med_vec5.end(), inter_med_vec6.end(), inter_med_vec7.end(), inter_med_vec8.end(), inter_med_vec9.end(), inter_med_vec3.end())), comp_func_four_points11());
	thrust::transform(variables[0].begin(), variables[0].end(), variables[4].begin(), inter_med_vec2.begin(), comp_func_four_points10(const_expr16_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[4].end(), inter_med_vec1.end())), comp_func_four_points9(const_expr15_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec0.end())), comp_func_four_points8(const_expr14_, const_expr3_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), variables[0].begin(), inter_med_vec3.begin(), variables[4].begin(), variables[3].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), variables[0].end(), inter_med_vec3.end(), variables[4].end(), variables[3].end(), variables[2].end(), derivatives.end())), comp_func_four_points7(const_expr0_, const_expr17_, const_expr1_, const_expr2_, const_expr3_, const_expr4_, const_expr5_));
}


struct comp_func_four_points21
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;

	comp_func_four_points21(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4, const cudaT const_expr5)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4), const_expr5_(const_expr5) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (2 * thrust::get<3>(t)) + (((-5 * (pow(thrust::get<3>(t), 2))) + (-47 * (pow(thrust::get<3>(t), const_expr0_)) * (pow(thrust::get<2>(t), const_expr1_)) * (pow((1 + thrust::get<1>(t)), -2))) + ((pow(thrust::get<3>(t), 2)) * ((const_expr2_ * (299 + (-1780 * thrust::get<0>(t)) + (3640 * (pow(thrust::get<0>(t), 2))) + (-2336 * (pow(thrust::get<0>(t), 3)))) * (pow((1 + thrust::get<1>(t)), -5))) + (const_expr3_ * (-1470 + (14880 * thrust::get<0>(t)) + (-20280 * (pow(thrust::get<0>(t), 2))) + (-103680 * (pow(thrust::get<0>(t), 3)))) * (pow((1 + thrust::get<1>(t)), -4))))) + (thrust::get<3>(t) * thrust::get<2>(t) * ((16 * (1 + (-3 * thrust::get<0>(t))) * thrust::get<4>(t) * (pow((1 + thrust::get<1>(t)), -4))) + (const_expr4_ * (360 + (-48 * ((30 * thrust::get<0>(t)) + (-59 * thrust::get<4>(t)))) + (-1440 * thrust::get<0>(t) * thrust::get<4>(t))) * (pow((1 + thrust::get<1>(t)), -3)))))) * const_expr5_);
	}
};


void FourPointsFlowEquation3::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[4].end(), variables[3].end(), variables[2].end(), derivatives.end())), comp_func_four_points21(const_expr0_, const_expr1_, const_expr2_, const_expr3_, const_expr4_, const_expr5_));
}


struct comp_func_four_points35
{
	const cudaT const_expr8_;

	comp_func_four_points35(const cudaT const_expr8)
		: const_expr8_(const_expr8) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * (-357694770 + (4 * thrust::get<1>(t) * (141599312 + (-401719146 * thrust::get<0>(t)))) + (703069912 * thrust::get<0>(t)) + ((pow(thrust::get<1>(t), 2)) * (1324702194 + (501442416 * thrust::get<0>(t))))) * (pow((1 + thrust::get<3>(t)), -4)) * const_expr8_;
	}
};


struct comp_func_four_points34
{
	const cudaT const_expr7_;

	comp_func_four_points34(const cudaT const_expr7)
		: const_expr7_(const_expr7) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<2>(t) * (16061481 + (8 * thrust::get<0>(t) * (-5610604 + (5355213 * thrust::get<0>(t))))) * (pow((1 + thrust::get<1>(t)), -4)) * const_expr7_;
	}
};


struct comp_func_four_points33
{
	const cudaT const_expr6_;

	comp_func_four_points33(const cudaT const_expr6)
		: const_expr6_(const_expr6) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * (-3238317810 + (-10 * (pow(thrust::get<1>(t), 2)) * (-777791658 + (-1171611936 * thrust::get<0>(t)))) + (80 * thrust::get<1>(t) * (89438368 + (-101648442 * thrust::get<0>(t)))) + (1625502880 * thrust::get<0>(t))) * (pow((1 + thrust::get<3>(t)), -4)) * const_expr6_;
	}
};


struct comp_func_four_points32
{
	const cudaT const_expr0_;
	const cudaT const_expr5_;

	comp_func_four_points32(const cudaT const_expr0, const cudaT const_expr5)
		: const_expr0_(const_expr0), const_expr5_(const_expr5) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), const_expr0_)) * (pow(thrust::get<3>(t), const_expr0_)) * ((8937232 * thrust::get<0>(t) * (-4 + (9 * thrust::get<0>(t)))) + (-1784609 * (-17 + (32 * thrust::get<0>(t))))) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr5_;
	}
};


struct comp_func_four_points31
{
	const cudaT const_expr4_;

	comp_func_four_points31(const cudaT const_expr4)
		: const_expr4_(const_expr4) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * (-970172435 + (1633705330 * thrust::get<0>(t)) + (-40 * thrust::get<1>(t) * (-87715448 + (222044563 * thrust::get<0>(t)))) + (8 * (pow(thrust::get<1>(t), 2)) * (-531595195 + (1662847997 * thrust::get<0>(t))))) * (pow((1 + thrust::get<3>(t)), -5)) * const_expr4_;
	}
};


struct comp_func_four_points30
{
	const cudaT const_expr4_;

	comp_func_four_points30(const cudaT const_expr4)
		: const_expr4_(const_expr4) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * ((-26769135 * (17 + (8 * thrust::get<1>(t) * (-8 + (9 * thrust::get<1>(t)))))) + (2 * (353519805 + (4 * thrust::get<1>(t) * (-514449355 + (742510961 * thrust::get<1>(t))))) * thrust::get<0>(t))) * (pow((1 + thrust::get<3>(t)), -5)) * const_expr4_;
	}
};


struct comp_func_four_points29
{
	const cudaT const_expr3_;

	comp_func_four_points29(const cudaT const_expr3)
		: const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), 2)) * (pow(thrust::get<3>(t), -1)) * (1502649636 + (-5687823600 * thrust::get<0>(t)) + (-2679298800 * (pow(thrust::get<0>(t), 2))) + (38630196192 * (pow(thrust::get<0>(t), 3))) + (-69322630560 * (pow(thrust::get<0>(t), 4)))) * (pow((1 + thrust::get<2>(t)), -5)) * const_expr3_;
	}
};


struct comp_func_four_points28
{
	const cudaT const_expr2_;

	comp_func_four_points28(const cudaT const_expr2)
		: const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), 2)) * (pow(thrust::get<3>(t), -1)) * (112533531 + (-855576992 * thrust::get<0>(t)) + (3683259968 * (pow(thrust::get<0>(t), 2))) + (-7947008128 * (pow(thrust::get<0>(t), 3))) + (6385327072 * (pow(thrust::get<0>(t), 4)))) * (pow((1 + thrust::get<2>(t)), -6)) * const_expr2_;
	}
};


struct comp_func_four_points27
{
	const cudaT const_expr1_;

	comp_func_four_points27(const cudaT const_expr1)
		: const_expr1_(const_expr1) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (pow(val1, 2)) * (pow(val2, -1)) * const_expr1_;
	}
};


struct comp_func_four_points26
{
	comp_func_four_points26()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<1>(t) + thrust::get<8>(t) + thrust::get<3>(t) + thrust::get<5>(t) + thrust::get<2>(t) + thrust::get<7>(t) + thrust::get<4>(t) + thrust::get<6>(t) + thrust::get<0>(t);
	}
};


struct comp_func_four_points25
{
	const cudaT const_expr11_;

	comp_func_four_points25(const cudaT const_expr11)
		: const_expr11_(const_expr11) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return val2 * (pow((1 + val1), -2)) * const_expr11_;
	}
};


struct comp_func_four_points24
{
	const cudaT const_expr10_;

	comp_func_four_points24(const cudaT const_expr10)
		: const_expr10_(const_expr10) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<2>(t) * (-90445640 + (-367789656 * thrust::get<0>(t)) + (-1235175840 * (pow(thrust::get<0>(t), 2)))) * (pow((1 + thrust::get<1>(t)), -3)) * const_expr10_;
	}
};


struct comp_func_four_points23
{
	const cudaT const_expr0_;
	const cudaT const_expr9_;

	comp_func_four_points23(const cudaT const_expr0, const cudaT const_expr9)
		: const_expr0_(const_expr0), const_expr9_(const_expr9) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<1>(t), const_expr0_)) * (pow(thrust::get<3>(t), const_expr0_)) * (-273938712 + (-45067812 * thrust::get<0>(t)) + (2 * (258770766 + (-8966844 * thrust::get<0>(t))) * thrust::get<0>(t))) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr9_;
	}
};


struct comp_func_four_points22
{
	const cudaT const_expr12_;

	comp_func_four_points22(const cudaT const_expr12)
		: const_expr12_(const_expr12) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (2 * thrust::get<1>(t)) + (thrust::get<1>(t) * (thrust::get<0>(t) + thrust::get<3>(t) + thrust::get<4>(t) + thrust::get<2>(t)) * const_expr12_);
	}
};


void FourPointsFlowEquation4::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	dev_vec inter_med_vec4(derivatives.size());
	dev_vec inter_med_vec5(derivatives.size());
	dev_vec inter_med_vec6(derivatives.size());
	dev_vec inter_med_vec7(derivatives.size());
	dev_vec inter_med_vec8(derivatives.size());
	dev_vec inter_med_vec9(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec3.end())), comp_func_four_points35(const_expr8_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[4].end(), inter_med_vec7.end())), comp_func_four_points34(const_expr7_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec5.end())), comp_func_four_points33(const_expr6_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec8.end())), comp_func_four_points32(const_expr0_, const_expr5_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec0.end())), comp_func_four_points31(const_expr4_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[3].begin(), variables[0].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[3].end(), variables[0].end(), inter_med_vec6.end())), comp_func_four_points30(const_expr4_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec4.end())), comp_func_four_points29(const_expr3_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec9.end())), comp_func_four_points28(const_expr2_));
	thrust::transform(variables[3].begin(), variables[3].end(), variables[4].begin(), inter_med_vec2.begin(), comp_func_four_points27(const_expr1_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec3.begin(), inter_med_vec2.begin(), inter_med_vec0.begin(), inter_med_vec4.begin(), inter_med_vec5.begin(), inter_med_vec6.begin(), inter_med_vec7.begin(), inter_med_vec8.begin(), inter_med_vec9.begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec3.end(), inter_med_vec2.end(), inter_med_vec0.end(), inter_med_vec4.end(), inter_med_vec5.end(), inter_med_vec6.end(), inter_med_vec7.end(), inter_med_vec8.end(), inter_med_vec9.end(), inter_med_vec1.end())), comp_func_four_points26());
	thrust::transform(variables[0].begin(), variables[0].end(), variables[4].begin(), inter_med_vec3.begin(), comp_func_four_points25(const_expr11_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[4].end(), inter_med_vec2.end())), comp_func_four_points24(const_expr10_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[3].begin(), variables[0].begin(), variables[4].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[3].end(), variables[0].end(), variables[4].end(), inter_med_vec0.end())), comp_func_four_points23(const_expr0_, const_expr9_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), variables[4].begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), variables[4].end(), inter_med_vec1.end(), inter_med_vec2.end(), inter_med_vec3.end(), derivatives.end())), comp_func_four_points22(const_expr12_));
}

