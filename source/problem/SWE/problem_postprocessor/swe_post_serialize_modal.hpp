#ifndef SWE_POST_SERIALIZE_MODAL_HPP
#define SWE_POST_SERIALIZE_MODAL_HPP

namespace SWE {
template <typename MeshType>
void serialize_modal_data(MeshType& mesh, const std::string& output_path) {
    std::string file_name = output_path + mesh.GetMeshName() + "_serialized_modal_data.txt";
    std::ofstream file(file_name);
    mesh.CallForEachElement([&file](auto& elt) {
        for (uint var = 0; var < SWE::n_variables; ++var) {
            for (uint dof = 0; dof < elt.data.get_ndof(); ++dof) {
                file << elt.data.state[0].q(var, dof) << std::endl;
            }
        }
        if (SWE::SedimentTransport::bed_update) {
            for (uint dof = 0; dof < elt.data.get_ndof(); ++dof) {
                file << elt.data.state[0].aux(0, dof) << std::endl;
            }
        }
    });
    file.close();
}

template <typename MeshType>
void deserialize_modal_data(MeshType& mesh, const std::string& output_path) {
    std::string file_name = output_path + mesh.GetMeshName() + "_serialized_modal_data.txt";
    std::ifstream file(file_name);
    mesh.CallForEachElement([&file](auto& elt) {
        for (uint var = 0; var < SWE::n_variables; ++var) {
            for (uint dof = 0; dof < elt.data.get_ndof(); ++dof) {
                file >> elt.data.state[0].q(var, dof);
            }
        }
        if (SWE::SedimentTransport::bed_update) {
            for (uint dof = 0; dof < elt.data.get_ndof(); ++dof) {
                file >> elt.data.state[0].aux(0, dof);
            }
        }
    });
    file.close();
}
}

#endif