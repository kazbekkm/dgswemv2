#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "general_definitions.hpp"
#include "preprocessor/input_parameters.hpp"

#include "simulation/serial/simulation_base.hpp"

#include "problem/definitions.hpp"
#include "problem/serial_functions.hpp"

namespace Serial {
template <typename ProblemType>
class Simulation : public SimulationBase {
  private:
    uint n_steps;

    typename ProblemType::ProblemDiscretizationType discretization;
    typename ProblemType::ProblemGlobalDataType global_data;

    typename ProblemType::ProblemStepperType stepper;
    typename ProblemType::ProblemWriterType writer;
    typename ProblemType::ProblemParserType parser;

    typename ProblemType::ProblemInputType problem_input;

  public:
    Simulation() = default;
    Simulation(const std::string& input_string);

    void Run() override;
    void ComputeL2Residual() override;
    void Finalize() override;
};

template <typename ProblemType>
Simulation<ProblemType>::Simulation(const std::string& input_string) {
    InputParameters<typename ProblemType::ProblemInputType> input(input_string);

    ProblemType::initialize_problem_parameters(input.problem_input);

    input.read_mesh();  // read mesh meta data
    input.read_bcis();  // read bc data

    ProblemType::preprocess_mesh_data(input);

    this->n_steps = (uint)std::ceil(input.stepper_input.run_time / input.stepper_input.dt);

    this->discretization.mesh = typename ProblemType::ProblemMeshType(input.polynomial_order);
    this->stepper             = typename ProblemType::ProblemStepperType(input.stepper_input);
    this->writer              = typename ProblemType::ProblemWriterType(input.writer_input);
    this->parser              = typename ProblemType::ProblemParserType(input);

    this->problem_input = input.problem_input;

    if (this->writer.WritingLog()) {
        this->writer.StartLog();

        this->writer.GetLogFile() << "Starting simulation with p=" << input.polynomial_order << " for "
                                  << input.mesh_input.mesh_data.mesh_name << " mesh" << std::endl
                                  << std::endl;
    }

    this->discretization.initialize(input, this->writer);
}

template <typename ProblemType>
void Simulation<ProblemType>::Run() {
    ProblemType::preprocessor_serial(this->discretization, this->global_data, this->stepper, this->problem_input);

    if (this->writer.WritingLog()) {
        this->writer.GetLogFile() << std::endl << "Launching Simulation!" << std::endl << std::endl;
    }

    if (this->writer.WritingOutput()) {
        this->writer.WriteFirstStep(this->stepper, this->discretization.mesh);
    }

    for (uint step = 1; step <= this->n_steps; ++step) {
        ProblemType::step_serial(this->discretization, this->global_data, this->stepper, this->writer, this->parser);
    }
}

template <typename ProblemType>
void Simulation<ProblemType>::ComputeL2Residual() {
    double ze_residual_L2 = 0;
    double qx_residual_L2 = 0;
    double qy_residual_L2 = 0;

    this->discretization.mesh.CallForEachElement(
        [this, &ze_residual_L2, &qx_residual_L2, &qy_residual_L2](auto& elt) { 
            auto l2 = ProblemType::compute_residual_L2(this->stepper, elt);
            ze_residual_L2 += l2[0]; 
            qx_residual_L2 += l2[1]; 
            qy_residual_L2 += l2[2]; 
        });

    std::cout << "ze L2 error: " << std::setprecision(15) << std::sqrt(ze_residual_L2) << std::endl;
    std::cout << "qx L2 error: " << std::setprecision(15) << std::sqrt(qx_residual_L2) << std::endl;
    std::cout << "qy L2 error: " << std::setprecision(15) << std::sqrt(qy_residual_L2) << std::endl;
}

template <typename ProblemType>
void Simulation<ProblemType>::Finalize() {
    ProblemType::finalize_simulation(this->global_data);
}
}
#endif
