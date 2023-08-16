import numpy as np
import os
import opensim as osim

from datetime import datetime
from pathlib import Path

from assistive_arm.moco_helpers import get_model


def main():
    model = get_model(model_type="muscle", enable_assist=True)
    model = osim.Model("./moco/models/base/LaiUhlrich2022_scaled.osim")

    inverse = osim.MocoInverse()

    modelProcessor = osim.ModelProcessor(model)
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    inverse.setModel(modelProcessor)
    inverse.setKinematics(osim.TableProcessor("./moco/motions/Sit-to-stand_correct_1.mot"))
    inverse.set_initial_time(0)
    inverse.set_final_time(3.033)
    inverse.set_mesh_interval(0.02)

    inverse.set_kinematics_allow_extra_columns(True)

    solution = inverse.solve()
    solution.getMocoSolution().write("sitToStand_inverse_solution.sto")


if __name__ == "__main__":
    main()