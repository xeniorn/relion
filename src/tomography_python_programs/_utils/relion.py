import inspect
from pathlib import Path
from typing import Callable

import makefun
import typer

from .file import write_empty_file

JOB_SUCCESS_FILENAME = 'RELION_JOB_EXIT_SUCCESS'
JOB_FAILURE_FILENAME = 'RELION_JOB_EXIT_FAILURE'
ABORT_JOB_NOW_FILENAME = 'RELION_JOB_ABORT_NOW'
JOB_ABORTED_FILENAME = 'RELION_JOB_ABORTED'


def write_job_success_file(job_directory: Path) -> None:
    """Write a file indicating job success."""
    output_file = job_directory / JOB_SUCCESS_FILENAME
    write_empty_file(output_file)


def write_job_failure_file(job_directory: Path) -> None:
    """Write a file indicating job failure."""
    output_file = job_directory / JOB_FAILURE_FILENAME
    write_empty_file(output_file)


def _check_for_abort_job_now_file(job_directory: Path) -> bool:
    """Check for the presence of a file indicating job termination."""
    output_file = job_directory / ABORT_JOB_NOW_FILENAME
    return output_file.exists()


def _write_job_aborted_file(job_directory: Path) -> None:
    """Write a file indicating that the job was succesfully terminated."""
    output_file = job_directory / JOB_ABORTED_FILENAME
    write_empty_file(output_file)


def abort_job_if_necessary(job_directory: Path) -> None:
    """Abort a job if file indicating job should be terminated is found."""
    if _check_for_abort_job_now_file(job_directory) is True:
        _write_job_aborted_file(job_directory)


PIPELINE_CONTROL_KEYWORD_ARGUMENT = inspect.Parameter(
    'pipeline_control',
    kind=inspect.Parameter.KEYWORD_ONLY,
    default=typer.Option(None, '--pipeline_control'),
    annotation=Path,
)


def relion_pipeline_job(func: Callable) -> Callable:
    """Decorator which turns a function into a RELION pipeline-aware job.

    Specifically
    - a file indicating job success will be written upon completion
    - a file indicating job failure will be written upon failure
    - a 'pipeline_control: Path' keyword argument will be added to the function
      resulting in an autogenerated CLI option '--pipeline_control'
    """
    function_signature = inspect.signature(func)
    new_signature = makefun.add_signature_parameters(
        function_signature, last=[PIPELINE_CONTROL_KEYWORD_ARGUMENT]
    )

    @makefun.wraps(func, new_sig=new_signature)
    def pipeline_job(*args, **kwargs):
        job_directory = kwargs.get('output_directory', None)
        if job_directory is not None:
            job_directory.mkdir(parents=True, exist_ok=True)
        try:
            pipeline_directory = kwargs.pop(PIPELINE_CONTROL_KEYWORD_ARGUMENT.name)
            func(*args, **kwargs)
            if job_directory is not None and pipeline_directory is not None:
                write_job_success_file(job_directory)
        except BaseException:
            if job_directory is not None and pipeline_directory is not None:
                write_job_failure_file(job_directory)
            raise
    return pipeline_job
