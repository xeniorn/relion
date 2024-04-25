import os
import sys
import subprocess
import argparse

import json

default_vars_json=r"""
{
 "RELION_STACK_BUFFER": "0", 
 "RELION_SCRATCH_DIR": "auto",
 "RELION_SHELL": "bash",
 "RELION_MPI_RUN": "srun",
 "RELION_QSUB_NRMPI": "1",
 "RELION_MPI_MAX": "128",
 "RELION_QSUB_NRTHREADS": "1",
 "RELION_THREAD_MAX": "",
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
 "RELION_MINIMUM_DEDICATED": "1",
 "RELION_ALLOW_CHANGE_MINIMUM_DEDICATED": "0",
 "RELION_CTFFIND_EXECUTABLE": "/software/f2022/software/ctffind/4.1.14-foss-2022b/bin/ctffind",
 "RELION_GCTF_EXECUTABLE": "/software/f2022/software/gctf/1.06/bin/gctf",
 "RELION_MOTIONCOR2_EXECUTABLE": "/software/f2022/software/motioncor2/1.6.4-gcccore-12.2.0/bin/motioncor2",
 "RELION_RESMAP_EXECUTABLE": "",
 "RELION_IMOD_WRAPPER_EXECUTABLE": "",
 "RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE": "",
 "RELION_PDFVIEWER_EXECUTABLE": "",
}
"""

def run(image_path: str,
        apptainer_exe: str,
        relion_gui_exe: str,
        use_gpu: bool) -> None:
    """
    Runs the image according to input parameters
    """

    default_var_dict: dict[str,str] = json.loads(default_vars_json)

    for name,value in default_var_dict.items():
        os.environ[name] = value

    use_gpu_arg: str | None = "--nv" if use_gpu else None 

    args: list[str|None] = [apptainer_exe, "run", use_gpu_arg, image_path, relion_gui_exe]
    final_args: list[str] = [arg for arg in args if arg is not None]

    # apptainer run --nv image.sif reion
    print(final_args)
    subprocess.run(final_args)


def run_from_commandline():
    """
    Parses command line args and passes it on to the actual run command.
    """
    
    parser = argparse.ArgumentParser(description="Run relion!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser_required = parser.add_argument_group("Required")
    parser_required.add_argument('--image-path', type=str, help='Path to the apptainer image to run.', required=True)

    parser_optional = parser.add_argument_group("Optional")
    parser_optional.add_argument('--use-gpu', type=bool, action="store", default=True, help='should gpus be used', required=False)

    parser_deployment = parser.add_argument_group("Deployment-specific options", "Options that might be different in different deployments, not related to the behavior of the app itself.")
    parser_deployment.add_argument('--apptainer-exe', type=str, default='apptainer', help='Command for running apptainer (aka singularity) itself.', required=False)
    parser_deployment.add_argument('--relion-gui-exe', type=str, default='relion', help='Command for running the relion gui inside the (apptainer) container.', required=False)

    parser_misc = parser.add_argument_group("Misc")
    parser_misc_verbosity = parser_misc.add_mutually_exclusive_group(required=False)
    parser_misc_verbosity.add_argument('--quiet', action="store_true", default=False)
    parser_misc_verbosity.add_argument('--verbose', action="store_true", default=False)

    args = parser.parse_args()

    run(image_path = args.image_path,
        apptainer_exe = args.apptainer_exe,
        relion_gui_exe = args.relion_gui_exe,
        use_gpu = args.use_gpu
        )
    


if __name__ == "__main__":
    run_from_commandline()
