#ifndef OPENMM_BROOK_CALC_GBSAOBC_FORCE_KERNEL_H_
#define OPENMM_BROOK_CALC_GBSAOBC_FORCE_KERNEL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "kernels.h"
#include "../../reference/src/SimTKUtilities/SimTKOpenMMRealType.h"
#include "BrookGbsa.h"
#include "OpenMMBrookInterface.h"

namespace OpenMM {

/**
 * This kernel is invoked to calculate the OBC forces acting on the system.
 */

class BrookCalcGBSAOBCForceKernel : public CalcGBSAOBCForceKernel {

   public:
  
      /**
       * BrookCalcGBSAOBCForceKernel constructor
       */

      BrookCalcGBSAOBCForceKernel( std::string name, const Platform& platform, OpenMMBrookInterface& openMMBrookInterface, System& system );
  
      /**
       * BrookCalcGBSAOBCForceKernel destructor
       */

      ~BrookCalcGBSAOBCForceKernel();
  
      /**
       * Initialize the kernel, setting up the values of all the force field parameters.
       * 
       * @param system     system this kernel will be applied to
       * @param force      GBSAOBCForce this kernel will be used for
       *
       */

      void initialize( const System& system, const GBSAOBCForce& force );
  
      /**
       * Execute the kernel to calculate the forces.
       * 
       * @param positions   particle coordiantes
       * @param forces      output forces
       *
       */

      void executeForces( OpenMMContextImpl& context );
  
      /**
       * Execute the kernel to calculate the energy.
       * 
       * @param positions   particle positions
       *
       * @return  potential energy due to the NonbondedForce
       * Currently always return 0.0 since energies not calculated on gpu
       *
       */

      double executeEnergy( OpenMMContextImpl& context );

      /** 
       * Set log file reference
       * 
       * @param  log file reference
       *
       * @return DefaultReturnValue
       *
       */
      
      int setLog( FILE* log );

      /* 
       * Get contents of object
       *
       * @param level of dump
       *
       * @return string containing contents
       *
       * */
      
      std::string getContents( int level ) const;

      /** 
       * Get log file reference
       * 
       * @return  log file reference
       *
       */
      
      FILE* getLog( void ) const;
      
      /** 
       * Get Brook GBSA reference
       * 
       * @return  Brook GBSA reference
       *
       */
      
      BrookGbsa& getBrookGbsa( void ) const;
      
   private:
   
      // log file reference

      FILE* _log;

      // number of particles

      int _numberOfParticles;
   
      // interface

      OpenMMBrookInterface& _openMMBrookInterface;

      // System reference

      System& _system;

};

} // namespace OpenMM

#endif /* OPENMM_BROOK_CALC_GBSAOBC_FORCE_KERNEL_H_ */
