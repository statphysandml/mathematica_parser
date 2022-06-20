import os
from mathematicaparser.odevisualization.thrust_meta_programmer import ThrustMetaProgrammer


class JacobianEquationMetaProgrammer(ThrustMetaProgrammer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.make_dir_not_exists(kwargs.get("project_path") + "/include/" + self.theory_name + "/")
        self.header_file = kwargs.get(
            "project_path") + "/include/" + self.theory_name + "/" + self.theory_name + "_jacobian_equation.hpp"
        self.make_dir_not_exists(kwargs.get("project_path") + "/src/")
        self.source_file = kwargs.get("project_path") + "/src/" + self.theory_name + "_jacobian_equation.cu"

    def write_hpp_header(self):
        ifndef_name = ''.join([item.upper() for item in self.theory_name.split(sep="_")])

        with open(self.header_file, "w") as f:
            f.write('#ifndef PROJECT_' + ifndef_name + 'JACOBIANEQUATION_HPP\n'
                    '#define PROJECT_' + ifndef_name + 'JACOBIANEQUATION_HPP\n\n'
                    '#include <math.h>\n'
                    '#include <tuple>\n\n'
                    '#include <odesolver/flow_equations/jacobian_equation.hpp>\n\n')

    def write_hpp_footer(self):
        ifndef_name = ''.join([item.upper() for item in self.theory_name.split(sep="_")])

        with open(self.header_file, "a") as f:
            f.write('\nclass ' + self.class_name + self.base_struct_name + 's : public odesolver::flowequations::' + self.base_struct_name + 'sWrapper\n'
                    '{\n'
                    'public:\n'
                    '\t' + self.class_name + self.base_struct_name + 's(const cudaT k) : k_(k)\n'
                    '\t{\n'
                    '\t\tjacobian_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::' + self.base_struct_name + '>> {\n')
            for dim_index in range(pow(self.dim, 2)-1):
                f.write('\t\t\tstd::make_shared<' + self.class_name + self.base_struct_name + str(dim_index) + '>(k),\n')
            f.write('\t\t\tstd::make_shared<' + self.class_name + self.base_struct_name + str(pow(self.dim, 2)-1) + '>(k)\n\t\t};\n')
            f.write('\t}\n\n'
                    '\tvoid operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, '
                    'const int row_idx, const int col_idx) override\n'
                    '\t{\n'
                    '\t\t(*jacobian_equations_[row_idx * dim_ + col_idx])(derivatives, variables);\n'
                    '\t}\n\n'
                    '\tvoid operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, '
                    'const int matrix_idx) override\n'
                    '\t{\n'
                    '\t\t(*jacobian_equations_[matrix_idx])(derivatives, variables);\n'
                    '\t}\n\n'
                    '\tsize_t get_dim() override\n'
                    '\t{\n'
                    '\t\treturn dim_;\n'
                    '\t}\n\n'
                    '\tstatic std::string model_;\n'
                    '\tstatic size_t dim_;\n'
                    '\n'
                    'private:\n'
                    '\tconst cudaT k_;\n'
                    '\tstd::vector<std::shared_ptr<odesolver::flowequations::' + self.base_struct_name + '>> jacobian_equations_;\n'
                    '};\n\n'
                    '#endif //PROJECT_' + ifndef_name + 'JACOBIANEQUATION_HPP\n')

    def write_cu_header(self):
        with open(self.source_file, "w") as f:
            f.write('#include <' + self.theory_name + '/' + self.theory_name + '_jacobian_equation.hpp>\n\n')
            f.write('std::string ' + self.class_name + self.base_struct_name + 's::model_ = "' + self.theory_name + '";\n'
                    'size_t ' + self.class_name + self.base_struct_name + 's::dim_ = ' + str(self.dim) + ';\n\n')
