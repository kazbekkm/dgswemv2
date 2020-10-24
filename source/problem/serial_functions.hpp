#ifndef PROBLEM_SERIAL_FUNCTIONS_HPP
#define PROBLEM_SERIAL_FUNCTIONS_HPP

#ifdef SWE_SUPPORT
#ifdef RKDG_SUPPORT

#include "problem/SWE/discretization_RKDG/kernels_preprocessor/rkdg_swe_pre_serial.hpp"
#include "problem/SWE/discretization_RKDG/kernels_processor/rkdg_swe_proc_serial_step.hpp"

#endif
#ifdef EHDG_SUPPORT

#include "problem/SWE/discretization_EHDG/kernels_preprocessor/ehdg_swe_pre_serial.hpp"
#include "problem/SWE/discretization_EHDG/kernels_processor/ehdg_swe_proc_serial_step.hpp"

#endif
#ifdef IHDG_SUPPORT

#include "problem/SWE/discretization_IHDG/kernels_preprocessor/ihdg_swe_pre_serial.hpp"
#include "problem/SWE/discretization_IHDG/kernels_processor/ihdg_swe_proc_serial_step.hpp"

#endif
#endif

#ifdef GN_SUPPORT
#ifdef EHDG_SUPPORT

#include "problem/Green-Naghdi/discretization_EHDG/kernels_preprocessor/ehdg_gn_pre_serial.hpp"
#include "problem/Green-Naghdi/discretization_EHDG/kernels_processor/ehdg_gn_proc_serial_step.hpp"

#endif
#endif

#endif