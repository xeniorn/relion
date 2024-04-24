import os
import sys
import subprocess
import argparse

import json

default_vars_json=r"""
{
 "RELION_QSUB_COMMAND": "bash",
 "RELION_QUEUE_USE": "true",
 "RELION_QUEUE_NAME": "auto",
 "RELION_QSUB_TEMPLATE": "/appdata/vbc-tools/clausen/scripts/cbe/relion/relion5GUI_masterSubmissionScript_latest.sh",
 "RELION_QSUB_EXTRA_COUNT": "5",
 "RELION_QSUB_EXTRA1": "qos name (short, medium, long)", 
 "RELION_QSUB_EXTRA1_DEFAULT": "auto",
 "RELION_QSUB_EXTRA1_HELP": "(default auto = short) which qos the job will run in, which controls the max length of job and resources it can consume", 
 "RELION_QSUB_EXTRA2": "required memory per MPI task [GB]", 
 "RELION_QSUB_EXTRA2_DEFAULT": "auto",
 "RELION_QSUB_EXTRA2_HELP": "how much memory to reserve for each MPI task, in GB. Default 13 GB seems to work well. Very large box sizes and some jobs might need more.", 
 "RELION_QSUB_EXTRA3": "Job name", 
 "RELION_QSUB_EXTRA3_DEFAULT": "auto",
 "RELION_QSUB_EXTRA3_HELP": "Replace the default job name in the cluster submission system with this name", 
 "RELION_QSUB_EXTRA4": "Time limit ([d-]hh:mm:ss)", 
 "RELION_QSUB_EXTRA4_DEFAULT": "auto",
 "RELION_QSUB_EXTRA4_HELP": "Here you can set the lower time limit than the maximum of each qos, which will often make your job start earlier than it would otherwise", 
 "RELION_QSUB_EXTRA5": "override sbatch parameters", 
 "RELION_QSUB_EXTRA5_DEFAULT": "",
 "RELION_QSUB_EXTRA5_HELP": "These parameters will be passed verbatim to the sbatch command (e.g. if you want to override any of the parameters manually, add --hold, --dependency:afterany:JOBID, --comment='this is my job' etc", 
 "RELION_ALLOW_CHANGE_MINIMUM_DEDICATED": "0",
 "RELION_CTFFIND_EXECUTABLE": "/software/f2022/software/ctffind/4.1.14-foss-2022b/bin/ctffind",
 "RELION_GCTF_EXECUTABLE": "/software/f2022/software/gctf/1.06/bin/gctf",
 "RELION_MOTIONCOR2_EXECUTABLE": "/software/f2022/software/motioncor2/1.6.4-gcccore-12.2.0/bin/motioncor2",
 "RELION_IMOD_WRAPPER_EXECUTABLE": "",
 "RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE": ""
}
"""

def run(image_path: str,
        apptainer_exe: str,
        relion_gui_exe: str) -> None:

    default_var_dict: dict[str,str] = json.loads(default_vars_json)

    for name,value in default_var_dict.items():
        os.environ[name] = value

    subprocess.run([apptainer_exe, "run", image_path, relion_gui_exe])


def run_from_commandline():
    
    parser = argparse.ArgumentParser(description="Run relion!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_required = parser.add_argument_group("Required")
    parser_required.add_argument('--image-path', type=str, help='Path to the apptainer image to run.', required=True)

    parser_deployment = parser.add_argument_group("Deployment-specific options", "Options that might be different in different deployments, not related to the behavior of the app itself.")    
    parser_deployment.add_argument('--apptainer-exe', type=str, default='apptainer', help='Command for running apptainer (aka singularity) itself.')
    parser_deployment.add_argument('--relion-gui-exe', type=str, default='relion', help='Command for running the relion gui inside the (apptainer) container.')

    args = parser.parse_args()

    run(image_path = args.image_path,
        apptainer_exe = args.apptainer_exe,
        relion_gui_exe = args.relion_gui_exe
        )
    


if __name__ == "__main__":
    run_from_commandline()
