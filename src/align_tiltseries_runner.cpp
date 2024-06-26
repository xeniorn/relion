/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#include "src/align_tiltseries_runner.h"

void AlignTiltseriesRunner::read(int argc, char **argv, int rank)
{
	parser.setCommandLine(argc, argv);
	int gen_section = parser.addSection("General options");
	fn_in = parser.getOption("--i", "STAR file with all input tomograms, or a unix wildcard to all tomogram files, e.g. \"mics/*.mrc\"");
	fn_out = parser.getOption("--o", "Directory, where all output files will be stored", "AlignTiltSeries/");
	continue_old = parser.checkOption("--only_do_unfinished", "Only estimate CTFs for those tomograms for which there is not yet a logfile with Final values.");
	do_at_most = textToInteger(parser.getOption("--do_at_most", "Only process up to this number of (unprocessed) tomograms.", "-1"));
    fn_imodwrapper_exe = parser.getOption("--wrapper_executable", "Alister Burt's wrapper script to call IMOD/AreTomo (default is set through $RELION_IMOD_WRAPPER_EXECUTABLE)", "");

    int fid_section = parser.addSection("IMOD fiducial-based alignment options");
    do_imod_fiducials = parser.checkOption("--imod_fiducials", "Use IMOD's fiducial-based alignment method");
    fiducial_diam = textToFloat(parser.getOption("--fiducial_diameter", "Diameter of the fiducials (in nm)", "10"));

    int pat_section = parser.addSection("IMOD patch-tracking alignment options");
    do_imod_patchtrack = parser.checkOption("--imod_patchtrack", "OR: Use IMOD's patrick-tracking alignment method");
    patch_overlap = textToFloat(parser.getOption("--patch_overlap", "Overlap between the patches (in %)", "10."));
    patch_size = textToFloat(parser.getOption("--patch_size", "Patch size (in A)", "10."));

    int aretomo_section = parser.addSection("AreTomo alignment options");
    do_aretomo = parser.checkOption("--aretomo", "OR: Use AreTomo's alignment method");
    aretomo_resolution = textToFloat(parser.getOption("--aretomo_resolution", "Resolution (in A) of the tilt series output by AreTomo. Has little bearing on processing.", "10"));
    aretomo_thickness = textToFloat(parser.getOption("--aretomo_thickness", "Thickness (in A) for AreTomo alignment", "2000"));
    do_aretomo_tiltcorrect = parser.checkOption("--aretomo_tiltcorrect", "Specify to correct the tilt angle offset in the tomogram (AreTomo -TiltCor option; default=false)");
    gpu_ids = parser.getOption("--gpu", "Device ids for each MPI-thread, e.g 0:1:2:3", "");

    patch_overlap = textToFloat(parser.getOption("--patch_overlap", "Overlap between the patches (in %)", "10"));
    patch_size = textToInteger(parser.getOption("--patch_size", "Patch size (in unbinned pixels)", "10"));

    int exp_section = parser.addSection("Expert options");
    other_wrapper_args  = parser.getOption("--other_wrapper_args", "Additional command-line arguments that will be passed onto the wrapper.", "");

    // Initialise verb for non-parallel execution
	verb = 1;

	// Check for errors in the command-line option
	if (parser.checkForErrors())
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
}

void AlignTiltseriesRunner::usage()
{
	parser.writeUsage(std::cout);
}

void AlignTiltseriesRunner::initialise(bool is_leader)
{
	// Get the Imod wrapper executable
	if (fn_imodwrapper_exe == "")
	{
		char *penv;
		penv = getenv("RELION_IMOD_WRAPPER_EXECUTABLE");
		if (penv != NULL)
			fn_imodwrapper_exe = (std::string)penv;
	}

    int i = 0;
    if (do_imod_fiducials) i++;
    if (do_imod_patchtrack) i++;
    if (do_aretomo) i++;
    if (i != 1) REPORT_ERROR("ERROR: you need to specify one of these options: --imod_fiducials or --imod_patchtrack or --aretomo");

	// Make sure fn_out ends with a slash
	if (fn_out[fn_out.length()-1] != '/')
		fn_out += "/";

    // Check if this is a TomographyExperiment starfile, if not raise an error
    if (!tomogramSet.read(fn_in, 1))
    {
        REPORT_ERROR("ERROR: the input file is not a valid tilt series star file");
    }

	idx_tomograms_all.clear();
	idx_tomograms.clear();
	bool warned = false;
	for (long int itomo = 0; itomo < tomogramSet.size(); itomo++)
	{
        FileName fn_star;
        tomogramSet.globalTable.getValue(EMDL_TOMO_TILT_SERIES_STARFILE, fn_star, itomo);
        FileName fn_newstar = getOutputFileWithNewUniqueDate(fn_star, fn_out);
        tomogramSet.globalTable.setValue(EMDL_TOMO_TILT_SERIES_STARFILE, fn_newstar, itomo);

		bool process_this = true;
        bool ignore_this = false;
        if (continue_old)
		{
			if (checkImodWrapperResults(itomo))
			{
				process_this = false; // already done
			}
		}

		if (do_at_most >= 0 && idx_tomograms.size() >= do_at_most)
		{
			if (process_this) {
				ignore_this = true;
				process_this = false;
				if (!warned)
				{
					warned = true;
					std::cout << "NOTE: processing of some tomograms will be skipped as requested by --do_at_most" << std::endl;
				}
			}
			// If this tomogram has already been processed, the result should be included in the output.
			// So ignore_this remains false.
		}

		if (process_this)
		{
			idx_tomograms.push_back(itomo);
		}

		if (!ignore_this)
		{
			idx_tomograms_all.push_back(itomo);
		}
	}

	if (is_leader && do_at_most >= 0 )
	{
		std::cout << tomogramSet.size() << " tomograms were given in the input tomogram set, but we process only ";
		std::cout  << do_at_most << " tomograms as specified in --do_at_most." << std::endl;
	}

    if (do_aretomo)
    {
        if (gpu_ids.length() > 0)
            untangleDeviceIDs(gpu_ids, allThreadIDs);
        else if (verb>0)
            std::cout << "--gpu_ids not specified, threads will automatically be mapped to devices."<< std::endl;
    }

    if (verb > 0)
	{
        std::cout << " Using IMOD wrapper executable in: " << fn_imodwrapper_exe << std::endl;
		std::cout << " to align tilt series for the following tomograms: " << std::endl;
		if (continue_old)
			std::cout << " (skipping all tomograms for which a logfile with Final values already exists " << std::endl;
		for (unsigned  int  i = 0; i < idx_tomograms.size(); ++i)
			std::cout << "  * " << tomogramSet.getTomogramName(idx_tomograms[i]) << std::endl;
	}
}

void AlignTiltseriesRunner::run()
{

    int barstep;
    if (verb > 0)
    {
        std::cout << " Aligning tilt series ..." << std::endl;
        init_progress_bar(idx_tomograms.size());
        barstep = XMIPP_MAX(1, idx_tomograms.size() / 60);
    }

    std::vector<std::string> alltomonames;
    for (long int itomo = 0; itomo < idx_tomograms.size(); itomo++)
    {

        // Abort through the pipeline_control system
        if (pipeline_control_check_abort_job())
            exit(RELION_EXIT_ABORTED);

        executeImodWrapper(idx_tomograms[itomo]);

        if (verb > 0 && itomo % barstep == 0)
            progress_bar(itomo);
    }

    if (verb > 0)
        progress_bar(idx_tomograms.size());

	joinImodWrapperResults();
}


void AlignTiltseriesRunner::executeImodWrapper(long idx_tomo, int rank)
{

    RFLOAT angpix = tomogramSet.getTiltSeriesPixelSize(idx_tomo);

    std::string command = fn_imodwrapper_exe + " ";
    // Make sure the methods are the first argument to the program!
    if (do_imod_fiducials)
    {
        command += " IMOD:fiducials";
        command += " --nominal-fiducial-diameter-nanometers " + floatToString(fiducial_diam);
    }
    else if (do_imod_patchtrack)
    {
        command += " IMOD:patch-tracking";
        command += " --patch-size-angstroms " + integerToString(patch_size);
        command += " --patch-overlap-percentage " + floatToString(patch_overlap);
    }
    else if (do_aretomo)
    {
        command += " AreTomo";
        command += " --output-resolution " + floatToString(aretomo_resolution);
        command += " --alignment-thickness " + floatToString(aretomo_thickness);
    }
    command += " --tilt-series-star-file " + fn_in;
    command += " --tomogram-name " + tomogramSet.getTomogramName(idx_tomo);
    command += " --output-directory " + fn_out;

    if (do_aretomo_tiltcorrect) command += " --tilt-angle-offset-correction";

    if (gpu_ids.length() > 0)
    {
        if (rank >= allThreadIDs.size())
            REPORT_ERROR("ERROR: not enough MPI nodes specified for the GPU IDs.");

        command += " --gpu-ids ";
        for (int igpu = 0; igpu < allThreadIDs[rank].size(); igpu++)
        {
            command += allThreadIDs[rank][igpu];
            if (igpu < allThreadIDs[rank].size()-1) command += ":";
        }
    }

    if (other_wrapper_args.length() > 0)
        command += " " + other_wrapper_args;

    if (system(command.c_str()))
		std::cerr << "WARNING: there was an error in executing: " << command << std::endl;

}

bool AlignTiltseriesRunner::checkImodWrapperResults(long idx_tomo)
{
    FileName fn_star;
    tomogramSet.globalTable.getValue(EMDL_TOMO_TILT_SERIES_STARFILE, fn_star, idx_tomo);

    MetaDataTable MDtomo;
    MDtomo.read(fn_star);
    return (MDtomo.containsLabel(EMDL_TOMO_XSHIFT_ANGST) &&
            MDtomo.containsLabel(EMDL_TOMO_YSHIFT_ANGST) &&
            MDtomo.containsLabel(EMDL_TOMO_XTILT)  &&
            MDtomo.containsLabel(EMDL_TOMO_YTILT)  &&
            MDtomo.containsLabel(EMDL_TOMO_ZROT) );

}

bool AlignTiltseriesRunner::checkEtomoDirectiveFile(long idx_tomo, FileName &filename)
{
    FileName fn_tomo = tomogramSet.getTomogramName(idx_tomo);
    filename = fn_out + "external/" + fn_tomo + "/" + fn_tomo + ".edf";
    return exists(filename);

}

void AlignTiltseriesRunner::joinImodWrapperResults()
{
    // Check again the STAR file exists and has the right labels
    // Also check for the presence of any eTomoDirective files

    MetaDataTable MDout;
    MDout.setName("global");
    bool any_edf = false;

    for (long itomo = 0; itomo < tomogramSet.size(); itomo++)
    {
        if (checkImodWrapperResults(itomo))
        {
            MDout.addObject(tomogramSet.globalTable.getObject(itomo));
            FileName fn_edf;
            if (checkEtomoDirectiveFile(itomo, fn_edf))
            {
                any_edf = true;
                MDout.setValue(EMDL_TOMO_ETOMO_DIRECTIVE_FILE, fn_edf, itomo);
            }
            else
            {
                MDout.setValue(EMDL_TOMO_ETOMO_DIRECTIVE_FILE, std::string("undefined"), itomo);
            }
        }
        else if (verb)
        {
            std::cerr << "WARNING: cannot find tilt series alignment parameters in " << tomogramSet.getTomogramName(itomo) << std::endl;
        }
    }
    if (!any_edf) MDout.deactivateLabel(EMDL_TOMO_ETOMO_DIRECTIVE_FILE);

    MDout.write(fn_out + "aligned_tilt_series.star");

    if (verb > 0)
    {
        std::cout << " Done! Written out: " << fn_out <<  "aligned_tilt_series.star" << std::endl;
    }

}
