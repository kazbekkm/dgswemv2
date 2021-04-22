#ifndef SIMULATION_OMPI_HPP
#define SIMULATION_OMPI_HPP

#include <omp.h>

#include "general_definitions.hpp"
#include "preprocessor/input_parameters.hpp"
#include "utilities/file_exists.hpp"
#include "sim_unit_ompi.hpp"

template <typename ProblemType>
class OMPISimulation : public OMPISimulationBase {
  private:
    uint n_steps;

    std::vector<std::unique_ptr<OMPISimulationUnit<ProblemType>>> sim_units;
    typename ProblemType::ProblemGlobalDataType global_data;

    typename ProblemType::ProblemStepperType stepper;

  public:
    OMPISimulation() = default;
    OMPISimulation(const std::string& input_string);

    void Run() override;
    void ComputeL2Residual() override;
    void Finalize() override;
};

template <typename ProblemType>
OMPISimulation<ProblemType>::OMPISimulation(const std::string& input_string) {
    int locality_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &locality_id);

    InputParameters<typename ProblemType::ProblemInputType> input(input_string);

    this->n_steps = (uint)std::ceil(input.stepper_input.run_time / input.stepper_input.dt);

    this->stepper = typename ProblemType::ProblemStepperType(input.stepper_input);

    std::string submesh_file_prefix =
        input.mesh_input.mesh_file_name.substr(0, input.mesh_input.mesh_file_name.find_last_of('.')) + "_" +
        std::to_string(locality_id) + '_';
    std::string submesh_file_postfix = input.mesh_input.mesh_file_name.substr(
        input.mesh_input.mesh_file_name.find_last_of('.'), input.mesh_input.mesh_file_name.size());

    uint submesh_id = 0;

    while (Utilities::file_exists(submesh_file_prefix + std::to_string(submesh_id) + submesh_file_postfix)) {
        this->sim_units.emplace_back(new OMPISimulationUnit<ProblemType>(input_string, locality_id, submesh_id));

        ++submesh_id;
    }

    if (this->sim_units.empty()) {
        std::cerr << "Warning: MPI Rank " << locality_id << " has not been assigned any work. This may inidicate\n"
                  << "         poor partitioning and imply degraded performance." << std::endl;
    }
}

template <typename ProblemType>
void OMPISimulation<ProblemType>::Run() {
#pragma omp parallel
    {
        uint n_threads, thread_id, sim_per_thread, begin_sim_id, end_sim_id;

        n_threads = (uint)omp_get_num_threads();
        thread_id = (uint)omp_get_thread_num();

        sim_per_thread = (this->sim_units.size() + n_threads - 1) / n_threads;

        begin_sim_id = sim_per_thread * thread_id;
        end_sim_id   = std::min(sim_per_thread * (thread_id + 1), (uint)this->sim_units.size());

        ProblemType::preprocessor_ompi(this->sim_units, this->global_data, this->stepper, begin_sim_id, end_sim_id);

        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            if (this->sim_units[su_id]->writer.Deserializing()) {
                this->sim_units[su_id]->writer.Deserialize(this->sim_units[su_id]->discretization.mesh);
            }
        }

        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            if (this->sim_units[su_id]->writer.WritingLog()) {
                this->sim_units[su_id]->writer.GetLogFile() << std::endl
                                                            << "Launching Simulation!" << std::endl
                                                            << std::endl;
            }

            if (this->sim_units[su_id]->writer.WritingOutput()) {
                this->sim_units[su_id]->writer.WriteFirstStep(this->stepper,
                                                              this->sim_units[su_id]->discretization.mesh);
            }
        }

        for (uint step = 1; step <= this->n_steps; ++step) {
            ProblemType::step_ompi(this->sim_units, this->global_data, this->stepper, begin_sim_id, end_sim_id);
        }

        for (uint su_id = begin_sim_id; su_id < end_sim_id; ++su_id) {
            if (this->sim_units[su_id]->writer.Serializing()) {
                this->sim_units[su_id]->writer.Serialize(this->sim_units[su_id]->discretization.mesh);
            }
        }
    }  // close omp parallel region
}

template <typename ProblemType>
void OMPISimulation<ProblemType>::ComputeL2Residual() {
    int locality_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &locality_id);

    double ze_global_l2{0};
    double qx_global_l2{0};
    double qy_global_l2{0};
    double hc_global_l2{0};

    double ze_residual_L2 = 0;
    double qx_residual_L2 = 0;
    double qy_residual_L2 = 0;
    double hc_residual_L2 = 0;

    for (auto& sim_unit : this->sim_units) {
        sim_unit->discretization.mesh.CallForEachElement(
        [this, &ze_residual_L2, &qx_residual_L2, &qy_residual_L2,  &hc_residual_L2](auto& elt) { 
            auto l2 = ProblemType::compute_residual_L2(this->stepper, elt);
            ze_residual_L2 += l2[0]; 
            qx_residual_L2 += l2[1]; 
            qy_residual_L2 += l2[2]; 
            hc_residual_L2 += l2[3]; 
        });
    }


    MPI_Reduce(&ze_residual_L2, &ze_global_l2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&qx_residual_L2, &qx_global_l2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&qy_residual_L2, &qy_global_l2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hc_residual_L2, &hc_global_l2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (locality_id == 0) {
        std::cout << "ze L2 error: " << std::setprecision(15) << std::sqrt(ze_global_l2) << std::endl;
        std::cout << "qx L2 error: " << std::setprecision(15) << std::sqrt(qx_global_l2) << std::endl;
        std::cout << "qy L2 error: " << std::setprecision(15) << std::sqrt(qy_global_l2) << std::endl;
        std::cout << "hc L2 error: " << std::setprecision(15) << std::sqrt(hc_global_l2) << std::endl;
    }
}

template <typename ProblemType>
void OMPISimulation<ProblemType>::Finalize() {
    ProblemType::finalize_simulation(this->global_data);
}

#endif