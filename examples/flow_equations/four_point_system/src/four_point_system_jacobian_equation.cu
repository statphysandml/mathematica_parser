#include <four_point_system/four_point_system_jacobian_equation.hpp>

std::string FourPointSystemJacobianEquations::model_ = "four_point_system";
size_t FourPointSystemJacobianEquations::dim_ = 5;


struct comp_func_four_point_system0
{
	const cudaT const_expr0_;

	comp_func_four_point_system0(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -2 + ((pow((1 + thrust::get<3>(t)), -4)) * ((thrust::get<2>(t) * (-7 + (32 * (1 + (-2 * thrust::get<4>(t))) * thrust::get<4>(t)))) + (-8 * thrust::get<0>(t) * (-1 + (2 * thrust::get<1>(t))) * (1 + thrust::get<3>(t)))) * const_expr0_);
	}
};


void FourPointSystemJacobianEquation0::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system0(const_expr0_));
}


struct comp_func_four_point_system1
{
	const cudaT const_expr0_;

	comp_func_four_point_system1(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<1>(t) * (-1 + (4 * thrust::get<2>(t))) * (pow((1 + thrust::get<0>(t)), -3)) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation1::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[3].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[3].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system1(const_expr0_));
}


struct comp_func_four_point_system2
{
	const cudaT const_expr0_;

	comp_func_four_point_system2(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return val2 * (pow((1 + val1), -2)) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation2::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), variables[4].begin(), derivatives.begin(), comp_func_four_point_system2(const_expr0_));
}


struct comp_func_four_point_system3
{
	const cudaT const_expr0_;

	comp_func_four_point_system3(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (-12 + ((7 + (32 * val2 * (-1 + (2 * val2)))) * (pow((1 + val1), -3)))) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation3::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), comp_func_four_point_system3(const_expr0_));
}


struct comp_func_four_point_system4
{
	const cudaT const_expr0_;

	comp_func_four_point_system4(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return const_expr0_ * (-2 + (4 * val2)) * (pow((1 + val1), -2));
	}
};


void FourPointSystemJacobianEquation4::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), variables[2].begin(), derivatives.begin(), comp_func_four_point_system4(const_expr0_));
}


struct comp_func_four_point_system5
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;

	comp_func_four_point_system5(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<2>(t), const_expr0_)) * (pow((1 + thrust::get<3>(t)), -6)) * ((15 * (pow(thrust::get<0>(t), const_expr1_)) * (-38 + (67 * thrust::get<4>(t))) * (pow((1 + thrust::get<3>(t)), 3))) + (-30 * (pow(thrust::get<2>(t), const_expr2_)) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((114 * thrust::get<1>(t) * (1 + thrust::get<3>(t))) + (-1 * thrust::get<4>(t) * (15 + (590 * thrust::get<1>(t)) + (15 * thrust::get<3>(t)) + (574 * thrust::get<1>(t) * thrust::get<3>(t)))) + (12 * (pow(thrust::get<4>(t), 2)) * (5 + (9 * thrust::get<1>(t)) + (5 * (1 + thrust::get<1>(t)) * thrust::get<3>(t)))))) + ((pow(thrust::get<2>(t), const_expr1_)) * ((627 * (1 + thrust::get<3>(t))) + (5 * thrust::get<4>(t) * (-885 + (-586 * thrust::get<3>(t)) + (4 * thrust::get<4>(t) * (425 + (-686 * thrust::get<4>(t)) + (2312 * (pow(thrust::get<4>(t), 2))) + (4 * (-5 + (8 * thrust::get<4>(t) * (7 + (54 * thrust::get<4>(t))))) * thrust::get<3>(t))))))))) * const_expr3_;
	}
};


void FourPointSystemJacobianEquation5::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system5(const_expr0_, const_expr1_, const_expr2_, const_expr3_));
}


struct comp_func_four_point_system6
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;

	comp_func_four_point_system6(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<2>(t), const_expr0_)) * (pow((1 + thrust::get<3>(t)), -5)) * ((-1005 * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 3))) + (-20 * (pow(thrust::get<2>(t), const_expr2_)) * (1 + thrust::get<3>(t)) * (((pow((1 + thrust::get<3>(t)), 4)) * const_expr3_) + (thrust::get<0>(t) * (15 + (586 * thrust::get<1>(t)) + (15 * thrust::get<3>(t)) + (574 * thrust::get<1>(t) * thrust::get<3>(t)) + (-24 * thrust::get<4>(t) * (5 + (8 * thrust::get<1>(t)) + (5 * (1 + thrust::get<1>(t)) * thrust::get<3>(t)))))))) + ((pow(thrust::get<2>(t), const_expr1_)) * (2138 + (-1680 * (pow(thrust::get<4>(t), 2)) * (-9 + (4 * thrust::get<3>(t)))) + (80 * thrust::get<4>(t) * (-84 + (5 * thrust::get<3>(t)))) + (-256 * (pow(thrust::get<4>(t), 3)) * (343 + (270 * thrust::get<3>(t)))) + (5 * thrust::get<3>(t) * (368 + (15 * thrust::get<3>(t) * (10 + (thrust::get<3>(t) * (10 + (thrust::get<3>(t) * (5 + thrust::get<3>(t)))))))))))) * const_expr4_;
	}
};


void FourPointSystemJacobianEquation6::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system6(const_expr0_, const_expr1_, const_expr2_, const_expr3_, const_expr4_));
}


struct comp_func_four_point_system7
{
	const cudaT const_expr0_;

	comp_func_four_point_system7(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), -4)) * ((57 * (1 + thrust::get<0>(t))) + (thrust::get<2>(t) * (-293 + (-287 * thrust::get<0>(t)) + (6 * thrust::get<2>(t) * (8 + (5 * thrust::get<0>(t))))))) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation7::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system7(const_expr0_));
}


struct comp_func_four_point_system8
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system8(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<2>(t), const_expr0_)) * (pow((1 + thrust::get<0>(t)), -5)) * ((15 * (pow(thrust::get<1>(t), const_expr1_)) * (-38 + (67 * thrust::get<3>(t))) * (pow((1 + thrust::get<0>(t)), 3))) + ((pow(thrust::get<2>(t), const_expr1_)) * ((-1120 * (pow(thrust::get<3>(t), 3)) * (-9 + (4 * thrust::get<0>(t)))) + (80 * (pow(thrust::get<3>(t), 2)) * (-84 + (5 * thrust::get<0>(t)))) + (-128 * (pow(thrust::get<3>(t), 4)) * (343 + (270 * thrust::get<0>(t)))) + (57 * (1 + thrust::get<0>(t)) * (13 + (24 * thrust::get<0>(t) * (2 + thrust::get<0>(t)) * (2 + (thrust::get<0>(t) * (2 + thrust::get<0>(t))))))) + (2 * thrust::get<3>(t) * (2138 + (5 * thrust::get<0>(t) * (368 + (15 * thrust::get<0>(t) * (10 + (thrust::get<0>(t) * (10 + (thrust::get<0>(t) * (5 + thrust::get<0>(t)))))))))))))) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation8::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[3].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system8(const_expr0_, const_expr1_, const_expr2_));
}


struct comp_func_four_point_system9
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system9(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<2>(t), const_expr0_)) * (pow((1 + thrust::get<3>(t)), -4)) * ((-9 * (pow(thrust::get<0>(t), const_expr1_)) * (-38 + (67 * thrust::get<4>(t))) * (pow((1 + thrust::get<3>(t)), 2))) + (8 * (pow(thrust::get<2>(t), const_expr1_)) * ((114 * thrust::get<1>(t) * (1 + thrust::get<3>(t))) + (-1 * thrust::get<4>(t) * (15 + (586 * thrust::get<1>(t)) + (15 * thrust::get<3>(t)) + (574 * thrust::get<1>(t) * thrust::get<3>(t)))) + (12 * (pow(thrust::get<4>(t), 2)) * (5 + (8 * thrust::get<1>(t)) + (5 * (1 + thrust::get<1>(t)) * thrust::get<3>(t))))))) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation9::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system9(const_expr0_, const_expr1_, const_expr2_));
}


struct comp_func_four_point_system14
{
	comp_func_four_point_system14()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<1>(t), 2)) * (87158987 + (-250441606 * thrust::get<0>(t)) + (8 * thrust::get<2>(t) * ((3 * (-67448349 + (39498775 * thrust::get<0>(t)))) + (thrust::get<2>(t) * (1437041213 + (55818725 * thrust::get<0>(t)) + (2 * thrust::get<2>(t) * (-1892461901 + (1919359561 * thrust::get<2>(t)) + (59 * (-6820303 + (12239165 * thrust::get<2>(t))) * thrust::get<0>(t)))))))));
	}
};


struct comp_func_four_point_system13
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_four_point_system13(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -6 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<1>(t), const_expr1_)) * (pow((1 + thrust::get<0>(t)), 2)) * (15318099 + (45656452 * thrust::get<0>(t)) + (4 * thrust::get<3>(t) * (3527699 + (-19686405 * thrust::get<0>(t)) + (3 * thrust::get<3>(t) * (-6453845 + (249079 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system12
{
	comp_func_four_point_system12()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<1>(t), 2)) * (pow((1 + thrust::get<0>(t)), 2)) * (204371563 + ((pow(thrust::get<2>(t), 2)) * (33939672 + (-308793960 * thrust::get<0>(t)))) + (5 * thrust::get<0>(t) * (34874168 + (19698225 * thrust::get<0>(t)))) + (-2 * thrust::get<2>(t) * (225513035 + (45973707 * thrust::get<0>(t)))));
	}
};


struct comp_func_four_point_system11
{
	comp_func_four_point_system11()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -1 * thrust::get<2>(t) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * (19809651 + (-124579062 * thrust::get<1>(t)) + (4 * thrust::get<4>(t) * (-344680080 + (thrust::get<4>(t) * (1343716751 + (-2042942830 * thrust::get<1>(t)))) + (953515386 * thrust::get<1>(t)))) + (-692814214 * thrust::get<3>(t)) + (4 * ((261448352 * thrust::get<1>(t)) + (thrust::get<4>(t) * (308050240 + (571199341 * thrust::get<4>(t)) + (48 * (-13982433 + (7550336 * thrust::get<4>(t))) * thrust::get<1>(t))))) * thrust::get<3>(t)));
	}
};


struct comp_func_four_point_system10
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;

	comp_func_four_point_system10(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = ((pow(thrust::get<0>(t), -1)) * (pow((1 + thrust::get<7>(t)), -6)) * ((-30 * (pow(thrust::get<6>(t), 2)) * (-6706343 + (16 * thrust::get<8>(t) * (2609977 + (2 * thrust::get<8>(t) * (-1876430 + (-3673410 * thrust::get<8>(t)) + (7755021 * (pow(thrust::get<8>(t), 2))))))))) + (32 * thrust::get<6>(t) * thrust::get<0>(t) * ((-187643 * (21 + (10 * thrust::get<8>(t) * (-9 + (16 * thrust::get<8>(t)))))) + (5 * (1688787 + (4 * thrust::get<8>(t) * (-3002288 + (3408219 * thrust::get<8>(t))))) * thrust::get<3>(t))) * (1 + thrust::get<7>(t))) + (60 * (pow(thrust::get<6>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr1_)) * (1313501 + (16 * thrust::get<8>(t) * (-1490929 + (5213144 * thrust::get<8>(t))))) * (pow((1 + thrust::get<7>(t)), 2))) + (30 * (pow(thrust::get<0>(t), 2)) * (pow((1 + thrust::get<7>(t)), 2)) * (-4274929 + (8 * thrust::get<3>(t) * (-750572 + (7079359 * thrust::get<3>(t)))) + (-5963716 * thrust::get<7>(t)) + (16390004 * thrust::get<8>(t) * (1 + thrust::get<7>(t)))))) * const_expr2_) + ((pow(thrust::get<0>(t), -1)) * thrust::get<3>(t) * (pow((1 + thrust::get<7>(t)), -7)) * (thrust::get<2>(t) + thrust::get<1>(t) + thrust::get<4>(t) + thrust::get<5>(t)) * const_expr3_);
	}
};


void FourPointSystemJacobianEquation10::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[3].begin(), variables[1].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[3].end(), variables[1].end(), inter_med_vec3.end())), comp_func_four_point_system14());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[1].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[3].end(), variables[1].end(), inter_med_vec2.end())), comp_func_four_point_system13(const_expr0_, const_expr1_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[2].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[2].end(), inter_med_vec0.end())), comp_func_four_point_system12());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec1.end())), comp_func_four_point_system11());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), inter_med_vec0.begin(), inter_med_vec1.begin(), variables[2].begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), inter_med_vec0.end(), inter_med_vec1.end(), variables[2].end(), inter_med_vec2.end(), inter_med_vec3.end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system10(const_expr0_, const_expr1_, const_expr2_, const_expr3_));
}


struct comp_func_four_point_system18
{
	comp_func_four_point_system18()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 2 * thrust::get<2>(t) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((115103685719070 * (1 + thrust::get<3>(t))) + (20 * thrust::get<1>(t) * ((-10040787 * thrust::get<1>(t) * (-8730291 + (9321622 * thrust::get<3>(t)))) + (88 * (-571875192758 + (255640730419 * thrust::get<3>(t)))))) + (thrust::get<4>(t) * ((-409257549223360 * (1 + thrust::get<3>(t))) + (3 * thrust::get<1>(t) * ((2975048 * thrust::get<1>(t) * (-976169399 + (226510080 * thrust::get<3>(t)))) + (5 * (504191436061413 + (274363798502845 * thrust::get<3>(t)))))))));
	}
};


struct comp_func_four_point_system17
{
	comp_func_four_point_system17()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -24 * (pow(thrust::get<1>(t), 2)) * ((17788979447497 * (1 + thrust::get<0>(t))) + (1859405 * thrust::get<3>(t) * (-29774297 + (23699265 * thrust::get<0>(t)))) + (10 * thrust::get<2>(t) * ((-5115719365292 * (1 + thrust::get<0>(t))) + (371881 * thrust::get<3>(t) * (241367493 + (11163745 * thrust::get<0>(t)))))) + (8 * (pow(thrust::get<2>(t), 3)) * ((52856369685981 * (1 + thrust::get<0>(t))) + (1859405 * thrust::get<3>(t) * (343963618 + (144422147 * thrust::get<0>(t)))))) + (-6 * (pow(thrust::get<2>(t), 2)) * ((25037084615010 * (1 + thrust::get<0>(t))) + (371881 * thrust::get<3>(t) * (1644117897 + (402397877 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system16
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_four_point_system16(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 40 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 2)) * ((-20323631463938 * (1 + thrust::get<3>(t))) + (142126174250336 * thrust::get<4>(t) * (1 + thrust::get<3>(t))) + (6693858 * thrust::get<4>(t) * thrust::get<1>(t) * (-4778114 + (249079 * thrust::get<3>(t)))) + (-3346929 * thrust::get<1>(t) * (758609 + (6562135 * thrust::get<3>(t)))));
	}
};


struct comp_func_four_point_system15
{
	const cudaT const_expr2_;

	comp_func_four_point_system15(const cudaT const_expr2)
		: const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = (pow(thrust::get<1>(t), -1)) * (pow((1 + thrust::get<4>(t)), -6)) * (thrust::get<5>(t) + thrust::get<0>(t) + thrust::get<2>(t) + thrust::get<3>(t)) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation11::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec2.end())), comp_func_four_point_system18());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[3].begin(), variables[1].begin(), variables[2].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[3].end(), variables[1].end(), variables[2].end(), inter_med_vec1.end())), comp_func_four_point_system17());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec0.end())), comp_func_four_point_system16(const_expr0_, const_expr1_));
	thrust::transform(variables[0].begin(), variables[0].end(), variables[4].begin(), inter_med_vec3.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 418913812698915 * (pow(val2, 2)) * (pow((1 + val1), 4)); });
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), variables[4].begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), variables[0].begin(), inter_med_vec3.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), variables[4].end(), inter_med_vec1.end(), inter_med_vec2.end(), variables[0].end(), inter_med_vec3.end(), derivatives.end())), comp_func_four_point_system15(const_expr2_));
}


struct comp_func_four_point_system23
{
	comp_func_four_point_system23()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = -1115643 * (pow(thrust::get<1>(t), 2)) * (130409755 + (-89131340 * thrust::get<0>(t)) + (16 * ((5 * (pow(thrust::get<2>(t), 2)) * (241367493 + (11163745 * thrust::get<0>(t)))) + (5 * thrust::get<2>(t) * (-29774297 + (23699265 * thrust::get<0>(t)))) + (10 * (pow(thrust::get<2>(t), 4)) * (343963618 + (144422147 * thrust::get<0>(t)))) + (-2 * (pow(thrust::get<2>(t), 3)) * (1644117897 + (402397877 * thrust::get<0>(t)))) + (4289082 * (pow(thrust::get<0>(t), 2)) * (15 + (thrust::get<0>(t) * (20 + (thrust::get<0>(t) * (15 + (thrust::get<0>(t) * (6 + thrust::get<0>(t))))))))))));
	}
};


struct comp_func_four_point_system22
{
	const cudaT const_expr2_;

	comp_func_four_point_system22(const cudaT const_expr2)
		: const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = 5 * thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), 2)) * (((pow((1 + thrust::get<0>(t)), 4)) * const_expr2_) + (thrust::get<1>(t) * ((160652592 * (pow(thrust::get<2>(t), 2)) * (718663 + (4288805 * thrust::get<0>(t)))) + (8 * thrust::get<2>(t) * (163674634714241 + (113599186087265 * thrust::get<0>(t)))) + (-9 * (22851632995047 + (thrust::get<0>(t) * (27096679526074 + (12208992685375 * thrust::get<0>(t)))))))));
	}
};


struct comp_func_four_point_system21
{
	comp_func_four_point_system21()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<2>(t) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((-160 * thrust::get<4>(t) * ((44 * (571875192758 + (-255640730419 * thrust::get<3>(t)))) + (10040787 * thrust::get<1>(t) * (-8730291 + (9321622 * thrust::get<3>(t)))))) + (15 * (-7470553046365 + (-113475063262391 * thrust::get<3>(t)) + (2975048 * thrust::get<1>(t) * (13686929 + (130724176 * thrust::get<3>(t)))))) + (6 * (pow(thrust::get<4>(t), 2)) * ((5950096 * thrust::get<1>(t) * (-976169399 + (226510080 * thrust::get<3>(t)))) + (5 * (504191436061413 + (274363798502845 * thrust::get<3>(t)))))));
	}
};


struct comp_func_four_point_system20
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_four_point_system20(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 5578215 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<1>(t), const_expr1_)) * (pow((1 + thrust::get<0>(t)), 2)) * (91610749 + (182625808 * thrust::get<0>(t)) + (48 * thrust::get<3>(t) * (-758609 + (-6562135 * thrust::get<0>(t)) + (thrust::get<3>(t) * (-4778114 + (249079 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system19
{
	const cudaT const_expr3_;

	comp_func_four_point_system19(const cudaT const_expr3)
		: const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = (pow(thrust::get<0>(t), -1)) * (pow((1 + thrust::get<5>(t)), -6)) * (thrust::get<3>(t) + thrust::get<2>(t) + thrust::get<4>(t) + thrust::get<1>(t)) * const_expr3_;
	}
};


void FourPointSystemJacobianEquation12::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[3].begin(), variables[1].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[3].end(), variables[1].end(), inter_med_vec0.end())), comp_func_four_point_system23());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[2].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[2].end(), inter_med_vec3.end())), comp_func_four_point_system22(const_expr2_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec1.end())), comp_func_four_point_system21());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[1].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[3].end(), variables[1].end(), inter_med_vec2.end())), comp_func_four_point_system20(const_expr0_, const_expr1_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), inter_med_vec3.end(), variables[0].end(), derivatives.end())), comp_func_four_point_system19(const_expr3_));
}


struct comp_func_four_point_system26
{
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system26(const cudaT const_expr1, const cudaT const_expr2)
		: const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 1115643 * thrust::get<1>(t) * ((-15 * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 2)) * (91610749 + (182625808 * thrust::get<3>(t)) + (48 * thrust::get<4>(t) * (-758609 + (-6562135 * thrust::get<3>(t)) + (thrust::get<4>(t) * (-4778114 + (249079 * thrust::get<3>(t)))))))) + (-6 * (pow(thrust::get<2>(t), const_expr2_)) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((5 * (-61357561 + (54747716 * thrust::get<1>(t)) + (-346407107 * thrust::get<3>(t)) + (522896704 * thrust::get<1>(t) * thrust::get<3>(t)))) + (2 * (pow(thrust::get<4>(t), 2)) * (5946066345 + (-7809355192 * thrust::get<1>(t)) + (5 * (571199341 + (362416128 * thrust::get<1>(t))) * thrust::get<3>(t)))) + (-80 * thrust::get<4>(t) * (26766752 + (-38506280 * thrust::get<3>(t)) + (9 * thrust::get<1>(t) * (-8730291 + (9321622 * thrust::get<3>(t)))))))) + (12 * (pow(thrust::get<2>(t), const_expr1_)) * (130409755 + (-89131340 * thrust::get<3>(t)) + (16 * ((5 * (pow(thrust::get<4>(t), 2)) * (241367493 + (11163745 * thrust::get<3>(t)))) + (5 * thrust::get<4>(t) * (-29774297 + (23699265 * thrust::get<3>(t)))) + (10 * (pow(thrust::get<4>(t), 4)) * (343963618 + (144422147 * thrust::get<3>(t)))) + (-2 * (pow(thrust::get<4>(t), 3)) * (1644117897 + (402397877 * thrust::get<3>(t)))) + (4289082 * (pow(thrust::get<3>(t), 2)) * (15 + (thrust::get<3>(t) * (20 + (thrust::get<3>(t) * (15 + (thrust::get<3>(t) * (6 + thrust::get<3>(t))))))))))))));
	}
};


struct comp_func_four_point_system25
{
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system25(const cudaT const_expr1, const cudaT const_expr2)
		: const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -40894566 * (1 + thrust::get<3>(t)) * ((6 * (pow(thrust::get<2>(t), const_expr1_)) * (6706343 + (-16 * thrust::get<4>(t) * (2609977 + (2 * thrust::get<4>(t) * (-1876430 + (-3673410 * thrust::get<4>(t)) + (7755021 * (pow(thrust::get<4>(t), 2))))))))) + (-4 * (pow(thrust::get<2>(t), const_expr2_)) * thrust::get<0>(t) * (3940503 + (-68164380 * (pow(thrust::get<4>(t), 2)) * thrust::get<1>(t)) + (-8443935 * ((2 * thrust::get<4>(t)) + thrust::get<1>(t))) + (30022880 * thrust::get<4>(t) * (thrust::get<4>(t) + (2 * thrust::get<1>(t))))) * (1 + thrust::get<3>(t))) + (5 * (pow(thrust::get<0>(t), const_expr1_)) * (1313501 + (16 * thrust::get<4>(t) * (-1490929 + (5213144 * thrust::get<4>(t))))) * (pow((1 + thrust::get<3>(t)), 2))) + (-104262880 * (pow(thrust::get<2>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 5))));
	}
};


struct comp_func_four_point_system24
{
	const cudaT const_expr0_;
	const cudaT const_expr3_;

	comp_func_four_point_system24(const cudaT const_expr0, const cudaT const_expr3)
		: const_expr0_(const_expr0), const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<3>(t), const_expr0_)) * (pow(thrust::get<0>(t), -1)) * (pow((1 + thrust::get<4>(t)), -6)) * (thrust::get<1>(t) + thrust::get<2>(t)) * const_expr3_;
	}
};


void FourPointSystemJacobianEquation13::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec1.end())), comp_func_four_point_system26(const_expr1_, const_expr2_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec0.end())), comp_func_four_point_system25(const_expr1_, const_expr2_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), inter_med_vec0.begin(), inter_med_vec1.begin(), variables[3].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), inter_med_vec0.end(), inter_med_vec1.end(), variables[3].end(), variables[0].end(), derivatives.end())), comp_func_four_point_system24(const_expr0_, const_expr3_));
}


struct comp_func_four_point_system30
{
	comp_func_four_point_system30()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 2 * (pow(thrust::get<1>(t), 2)) * ((240 * (pow(thrust::get<2>(t), 2)) * ((-5115719365292 * (1 + thrust::get<0>(t))) + (371881 * thrust::get<3>(t) * (241367493 + (11163745 * thrust::get<0>(t)))))) + (48 * thrust::get<2>(t) * ((17788979447497 * (1 + thrust::get<0>(t))) + (1859405 * thrust::get<3>(t) * (-29774297 + (23699265 * thrust::get<0>(t)))))) + (96 * (pow(thrust::get<2>(t), 4)) * ((52856369685981 * (1 + thrust::get<0>(t))) + (1859405 * thrust::get<3>(t) * (343963618 + (144422147 * thrust::get<0>(t)))))) + (-96 * (pow(thrust::get<2>(t), 3)) * ((25037084615010 * (1 + thrust::get<0>(t))) + (371881 * thrust::get<3>(t) * (1644117897 + (402397877 * thrust::get<0>(t)))))) + (6815761 * (1 + thrust::get<0>(t)) * (32012411 + (52131440 * thrust::get<0>(t) * (5 + (thrust::get<0>(t) * (10 + (thrust::get<0>(t) * (10 + (thrust::get<0>(t) * (5 + thrust::get<0>(t))))))))))) + (1115643 * thrust::get<3>(t) * (130409755 + (4 * thrust::get<0>(t) * (-22282835 + (17156328 * thrust::get<0>(t) * (15 + (thrust::get<0>(t) * (20 + (thrust::get<0>(t) * (15 + (thrust::get<0>(t) * (6 + thrust::get<0>(t))))))))))))));
	}
};


struct comp_func_four_point_system29
{
	comp_func_four_point_system29()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 10 * (pow(thrust::get<1>(t), 2)) * (pow((1 + thrust::get<0>(t)), 2)) * ((167565525079566 * thrust::get<2>(t) * (pow((1 + thrust::get<0>(t)), 2))) + (-20447283 * (1 + thrust::get<0>(t)) * (2418929 + (2981858 * thrust::get<0>(t)))) + (53550864 * (pow(thrust::get<3>(t), 3)) * (718663 + (4288805 * thrust::get<0>(t)))) + ((pow(thrust::get<3>(t), 2)) * (654698538856964 + (454396744349060 * thrust::get<0>(t)))) + (-9 * thrust::get<3>(t) * (22851632995047 + (thrust::get<0>(t) * (27096679526074 + (12208992685375 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system28
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_four_point_system28(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 5 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 2)) * ((17905017778522 * (1 + thrust::get<3>(t))) + (1115643 * thrust::get<1>(t) * (91610749 + (182625808 * thrust::get<3>(t)))) + (16 * (pow(thrust::get<4>(t), 2)) * ((71063087125168 * (1 + thrust::get<3>(t))) + (3346929 * thrust::get<1>(t) * (-4778114 + (249079 * thrust::get<3>(t)))))) + (-16 * thrust::get<4>(t) * ((20323631463938 * (1 + thrust::get<3>(t))) + (3346929 * thrust::get<1>(t) * (758609 + (6562135 * thrust::get<3>(t)))))));
	}
};


struct comp_func_four_point_system27
{
	const cudaT const_expr2_;

	comp_func_four_point_system27(const cudaT const_expr2)
		: const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<0>(t), -2)) * (pow((1 + thrust::get<2>(t)), -6)) * (thrust::get<3>(t) + thrust::get<4>(t) + thrust::get<1>(t)) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation14::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[3].begin(), variables[1].begin(), variables[2].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[3].end(), variables[1].end(), variables[2].end(), inter_med_vec0.end())), comp_func_four_point_system30());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), inter_med_vec2.end())), comp_func_four_point_system29());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec1.end())), comp_func_four_point_system28(const_expr0_, const_expr1_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), inter_med_vec0.begin(), variables[0].begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), inter_med_vec0.end(), variables[0].end(), inter_med_vec1.end(), inter_med_vec2.end(), derivatives.end())), comp_func_four_point_system27(const_expr2_));
}


struct comp_func_four_point_system31
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system31(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<3>(t)), -6)) * ((282 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 3))) + (12 * thrust::get<2>(t) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((-15 * (1 + thrust::get<3>(t))) + (-2 * thrust::get<1>(t) * (67 + (59 * thrust::get<3>(t)))) + (12 * thrust::get<4>(t) * (5 + (9 * thrust::get<1>(t)) + (5 * (1 + thrust::get<1>(t)) * thrust::get<3>(t)))))) + (-2 * (pow(thrust::get<2>(t), 2)) * (-201 + (98 * thrust::get<3>(t)) + (4 * thrust::get<4>(t) * (197 + (-248 * thrust::get<3>(t)) + (2 * thrust::get<4>(t) * (-286 + (169 * thrust::get<3>(t)) + (4 * thrust::get<4>(t) * (289 + (216 * thrust::get<3>(t))))))))))) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation15::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system31(const_expr0_, const_expr1_, const_expr2_));
}


struct comp_func_four_point_system32
{
	const cudaT const_expr0_;

	comp_func_four_point_system32(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<2>(t) * (pow((1 + thrust::get<3>(t)), -5)) * ((-30 * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * (5 + (8 * thrust::get<1>(t)) + (5 * (1 + thrust::get<1>(t)) * thrust::get<3>(t)))) + (thrust::get<2>(t) * (135 + (-310 * thrust::get<3>(t)) + (thrust::get<4>(t) * (-975 + (845 * thrust::get<3>(t)) + (24 * thrust::get<4>(t) * (343 + (270 * thrust::get<3>(t))))))))) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation16::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system32(const_expr0_));
}


struct comp_func_four_point_system33
{
	const cudaT const_expr0_;

	comp_func_four_point_system33(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<2>(t) * thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), -4)) * (-65 + (-59 * thrust::get<0>(t)) + (6 * thrust::get<3>(t) * (8 + (5 * thrust::get<0>(t))))) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation17::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[3].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system33(const_expr0_));
}


struct comp_func_four_point_system34
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
	const cudaT const_expr6_;

	comp_func_four_point_system34(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4, const cudaT const_expr5, const cudaT const_expr6)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4), const_expr5_(const_expr5), const_expr6_(const_expr6) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 2 + (((-10 * thrust::get<2>(t)) + (const_expr0_ * thrust::get<2>(t) * (-299 + (4 * thrust::get<4>(t) * (445 + (-910 * thrust::get<4>(t)) + (584 * (pow(thrust::get<4>(t), 2)))))) * (pow((1 + thrust::get<3>(t)), -5))) + (const_expr1_ * thrust::get<2>(t) * (49 + (4 * thrust::get<4>(t) * (-124 + (thrust::get<4>(t) * (169 + (864 * thrust::get<4>(t))))))) * (pow((1 + thrust::get<3>(t)), -4))) + (16 * thrust::get<0>(t) * (1 + (-3 * thrust::get<4>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<3>(t)), -4))) + (const_expr2_ * thrust::get<0>(t) * (15 + (118 * thrust::get<1>(t)) + (-60 * thrust::get<4>(t) * (1 + thrust::get<1>(t)))) * (pow((1 + thrust::get<3>(t)), -3))) + (const_expr3_ * (pow(thrust::get<2>(t), const_expr4_)) * (pow(thrust::get<0>(t), const_expr5_)) * (pow((1 + thrust::get<3>(t)), -2)))) * const_expr6_);
	}
};


void FourPointSystemJacobianEquation18::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system34(const_expr0_, const_expr1_, const_expr2_, const_expr3_, const_expr4_, const_expr5_, const_expr6_));
}


struct comp_func_four_point_system35
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_four_point_system35(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<3>(t)), -4)) * ((120 * thrust::get<2>(t) * (1 + thrust::get<3>(t))) + (-423 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr0_)) * (pow((1 + thrust::get<3>(t)), 2))) + (16 * thrust::get<2>(t) * thrust::get<1>(t) * (65 + (59 * thrust::get<3>(t)))) + (-96 * thrust::get<2>(t) * thrust::get<4>(t) * (5 + (8 * thrust::get<1>(t)) + (5 * (1 + thrust::get<1>(t)) * thrust::get<3>(t))))) * const_expr1_;
	}
};


void FourPointSystemJacobianEquation19::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system35(const_expr0_, const_expr1_));
}


struct comp_func_four_point_system40
{
	comp_func_four_point_system40()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<1>(t), 2)) * (87158987 + (-250441606 * thrust::get<0>(t)) + (8 * thrust::get<2>(t) * ((3 * (-67448349 + (39498775 * thrust::get<0>(t)))) + (thrust::get<2>(t) * (1437041213 + (55818725 * thrust::get<0>(t)) + (2 * thrust::get<2>(t) * (-1892461901 + (1919359561 * thrust::get<2>(t)) + (59 * (-6820303 + (12239165 * thrust::get<2>(t))) * thrust::get<0>(t)))))))));
	}
};


struct comp_func_four_point_system39
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_four_point_system39(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -6 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<1>(t), const_expr1_)) * (pow((1 + thrust::get<0>(t)), 2)) * (15318099 + (45656452 * thrust::get<0>(t)) + (4 * thrust::get<3>(t) * (3527699 + (-19686405 * thrust::get<0>(t)) + (3 * thrust::get<3>(t) * (-6453845 + (249079 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system38
{
	comp_func_four_point_system38()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<1>(t), 2)) * (pow((1 + thrust::get<0>(t)), 2)) * (204371563 + ((pow(thrust::get<2>(t), 2)) * (33939672 + (-308793960 * thrust::get<0>(t)))) + (5 * thrust::get<0>(t) * (34874168 + (19698225 * thrust::get<0>(t)))) + (-2 * thrust::get<2>(t) * (225513035 + (45973707 * thrust::get<0>(t)))));
	}
};


struct comp_func_four_point_system37
{
	comp_func_four_point_system37()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -1 * thrust::get<2>(t) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * (19809651 + (-124579062 * thrust::get<1>(t)) + (4 * thrust::get<4>(t) * (-344680080 + (thrust::get<4>(t) * (1343716751 + (-2042942830 * thrust::get<1>(t)))) + (953515386 * thrust::get<1>(t)))) + (-692814214 * thrust::get<3>(t)) + (4 * ((261448352 * thrust::get<1>(t)) + (thrust::get<4>(t) * (308050240 + (571199341 * thrust::get<4>(t)) + (48 * (-13982433 + (7550336 * thrust::get<4>(t))) * thrust::get<1>(t))))) * thrust::get<3>(t)));
	}
};


struct comp_func_four_point_system36
{
	const cudaT const_expr2_;

	comp_func_four_point_system36(const cudaT const_expr2)
		: const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<3>(t)), -7)) * (thrust::get<1>(t) + thrust::get<0>(t) + thrust::get<4>(t) + thrust::get<2>(t)) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation20::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	dev_vec inter_med_vec0(derivatives.size());
	dev_vec inter_med_vec1(derivatives.size());
	dev_vec inter_med_vec2(derivatives.size());
	dev_vec inter_med_vec3(derivatives.size());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[3].begin(), variables[1].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[3].end(), variables[1].end(), inter_med_vec2.end())), comp_func_four_point_system40());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[3].begin(), variables[1].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[3].end(), variables[1].end(), inter_med_vec3.end())), comp_func_four_point_system39(const_expr0_, const_expr1_));
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[2].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[2].end(), inter_med_vec0.end())), comp_func_four_point_system38());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), inter_med_vec1.end())), comp_func_four_point_system37());
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), variables[0].begin(), inter_med_vec3.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), variables[0].end(), inter_med_vec3.end(), derivatives.end())), comp_func_four_point_system36(const_expr2_));
}


struct comp_func_four_point_system41
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system41(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<3>(t)), -6)) * ((120 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 2)) * (-758609 + (-6562135 * thrust::get<3>(t)) + (thrust::get<4>(t) * (-9556228 + (498158 * thrust::get<3>(t)))))) + (2 * thrust::get<2>(t) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((thrust::get<4>(t) * (5946066345 + (-7809355192 * thrust::get<1>(t)) + (5 * (571199341 + (362416128 * thrust::get<1>(t))) * thrust::get<3>(t)))) + (-20 * (26766752 + (-38506280 * thrust::get<3>(t)) + (9 * thrust::get<1>(t) * (-8730291 + (9321622 * thrust::get<3>(t)))))))) + (-8 * (pow(thrust::get<2>(t), 2)) * ((5 * (-29774297 + (23699265 * thrust::get<3>(t)))) + (2 * thrust::get<4>(t) * (1206837465 + (55818725 * thrust::get<3>(t)) + (thrust::get<4>(t) * (-4932353691 + (6879272360 * thrust::get<4>(t)) + (59 * (-20460909 + (48956660 * thrust::get<4>(t))) * thrust::get<3>(t))))))))) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation21::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system41(const_expr0_, const_expr1_, const_expr2_));
}


struct comp_func_four_point_system42
{
	const cudaT const_expr0_;

	comp_func_four_point_system42(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<0>(t) * (pow((1 + thrust::get<3>(t)), -5)) * ((5 * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * (60209401 + (15324569 * thrust::get<3>(t)) + (24 * thrust::get<1>(t) * (718663 + (4288805 * thrust::get<3>(t)))))) + (thrust::get<2>(t) * (68434645 + (653620880 * thrust::get<3>(t)) + (4 * thrust::get<4>(t) * (392863095 + (-976169399 * thrust::get<4>(t)) + (30 * (-13982433 + (7550336 * thrust::get<4>(t))) * thrust::get<3>(t))))))) * const_expr0_;
	}
};


void FourPointSystemJacobianEquation22::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system42(const_expr0_));
}


struct comp_func_four_point_system43
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;

	comp_func_four_point_system43(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<2>(t), const_expr0_)) * (pow((1 + thrust::get<3>(t)), -6)) * ((5 * (pow(thrust::get<0>(t), const_expr1_)) * (pow((1 + thrust::get<3>(t)), 2)) * (91610749 + (182625808 * thrust::get<3>(t)) + (48 * thrust::get<4>(t) * (-758609 + (-6562135 * thrust::get<3>(t)) + (thrust::get<4>(t) * (-4778114 + (249079 * thrust::get<3>(t)))))))) + (2 * (pow(thrust::get<2>(t), const_expr2_)) * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * ((5 * (-61357561 + (54747716 * thrust::get<1>(t)) + (-346407107 * thrust::get<3>(t)) + (522896704 * thrust::get<1>(t) * thrust::get<3>(t)))) + (2 * (pow(thrust::get<4>(t), 2)) * (5946066345 + (-7809355192 * thrust::get<1>(t)) + (5 * (571199341 + (362416128 * thrust::get<1>(t))) * thrust::get<3>(t)))) + (-80 * thrust::get<4>(t) * (26766752 + (-38506280 * thrust::get<3>(t)) + (9 * thrust::get<1>(t) * (-8730291 + (9321622 * thrust::get<3>(t)))))))) + (-4 * (pow(thrust::get<2>(t), const_expr1_)) * (130409755 + (-89131340 * thrust::get<3>(t)) + (16 * ((5 * (pow(thrust::get<4>(t), 2)) * (241367493 + (11163745 * thrust::get<3>(t)))) + (5 * thrust::get<4>(t) * (-29774297 + (23699265 * thrust::get<3>(t)))) + (10 * (pow(thrust::get<4>(t), 4)) * (343963618 + (144422147 * thrust::get<3>(t)))) + (-2 * (pow(thrust::get<4>(t), 3)) * (1644117897 + (402397877 * thrust::get<3>(t)))) + (4289082 * (pow(thrust::get<3>(t), 2)) * (15 + (thrust::get<3>(t) * (20 + (thrust::get<3>(t) * (15 + (thrust::get<3>(t) * (6 + thrust::get<3>(t)))))))))))))) * const_expr3_;
	}
};


void FourPointSystemJacobianEquation23::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system43(const_expr0_, const_expr1_, const_expr2_, const_expr3_));
}


struct comp_func_four_point_system44
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_four_point_system44(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<3>(t)), -5)) * (((pow((1 + thrust::get<3>(t)), 5)) * const_expr1_) + (20 * thrust::get<0>(t) * (1 + thrust::get<3>(t)) * (-442988327 + (144 * (pow(thrust::get<1>(t), 2)) * (718663 + (4288805 * thrust::get<3>(t)))) + (12 * thrust::get<1>(t) * (60209401 + (15324569 * thrust::get<3>(t)))) + (-5 * thrust::get<3>(t) * (109144786 + (59094675 * thrust::get<3>(t)))))) + (45 * (pow(thrust::get<2>(t), const_expr0_)) * (pow(thrust::get<0>(t), const_expr0_)) * (1 + thrust::get<3>(t)) * (91610749 + (182625808 * thrust::get<3>(t)) + (48 * thrust::get<4>(t) * (-758609 + (-6562135 * thrust::get<3>(t)) + (thrust::get<4>(t) * (-4778114 + (249079 * thrust::get<3>(t)))))))) + (6 * thrust::get<2>(t) * ((5 * (-61357561 + (54747716 * thrust::get<1>(t)) + (-346407107 * thrust::get<3>(t)) + (522896704 * thrust::get<1>(t) * thrust::get<3>(t)))) + (2 * (pow(thrust::get<4>(t), 2)) * (5946066345 + (-7809355192 * thrust::get<1>(t)) + (5 * (571199341 + (362416128 * thrust::get<1>(t))) * thrust::get<3>(t)))) + (-80 * thrust::get<4>(t) * (26766752 + (-38506280 * thrust::get<3>(t)) + (9 * thrust::get<1>(t) * (-8730291 + (9321622 * thrust::get<3>(t))))))))) * const_expr2_;
	}
};


void FourPointSystemJacobianEquation24::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[3].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[3].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_four_point_system44(const_expr0_, const_expr1_, const_expr2_));
}

