#include "reconstruct_tomogram.h"
#include <src/jaz/tomography/projection/projection.h>
#include <src/jaz/tomography/extraction.h>
#include <src/jaz/tomography/reconstruction.h>
#include <src/jaz/tomography/tomogram.h>
#include <src/jaz/image/normalization.h>
#include <src/jaz/image/centering.h>
#include <src/jaz/gravis/t4Matrix.h>
#include <src/jaz/util/log.h>
#include <src/args.h>
#include <src/parallel.h>

#include <omp.h>

using namespace gravis;


void TomoBackprojectProgram::readParameters(int argc, char *argv[])
{
	IOParser parser;

	parser.setCommandLine(argc, argv);

	optimisationSet.read(
		parser,
		true,             // optimisation set
		false,   false,   // particles
		true,    true,    // tomograms
		false,   false,   // trajectories
		false,   false,   // manifolds
		false,   false);  // reference

	int gen_section = parser.addSection("General options");

	tomoName = parser.getOption("--tn", "Tomogram name", "*");
	outFn = parser.getOption("--o", "Output filename (or output directory in case of reconstructing multiple tomograms)");
 	do_even_odd_tomograms = parser.checkOption("--generate_split_tomograms", "Reconstruct tomograms from even/odd movie frames or tilt image index for denoising");

    w = textToInteger(parser.getOption("--w", "Width"));
	h = textToInteger(parser.getOption("--h", "Height" ));
	d = textToInteger(parser.getOption("--d", "Thickness"));

	applyWeight = !parser.checkOption("--no_weight", "Do not perform weighting in Fourier space using a Wiener filter");
	applyPreWeight = parser.checkOption("--pre_weight", "Pre-weight the 2D slices prior to backprojection");
    FourierCrop = parser.checkOption("--Fc", "Downsample the 2D images by Fourier cropping");
    do_only_unfinished = parser.checkOption("--only_do_unfinished", "Only reconstruct those tomograms that haven't finished yet");
    SNR = textToDouble(parser.getOption("--SNR", "SNR assumed by the Wiener filter", "10"));

	applyCtf = parser.checkOption("--ctf", "Perform CTF correction");
    doWiener = !parser.checkOption("--skip_wiener", "Do multiply images with CTF, but don't divide by CTF^2 in Wiener filter");

    if (!doWiener) applyCtf = true;

    zeroDC = !parser.checkOption("--keep_mean", "Do not zero the DC component of each frame");

	taperDist = textToDouble(parser.getOption("--td", "Tapering distance", "0.0"));
	taperFalloff = textToDouble(parser.getOption("--tf", "Tapering falloff", "0.0"));

    // SHWS & Aburt 19Jul2022: use zero-origins from relion-4.1 onwards....
    x0 = textToDouble(parser.getOption("--x0", "X origin", "0.0"));
    y0 = textToDouble(parser.getOption("--y0", "Y origin", "0.0"));
    z0 = textToDouble(parser.getOption("--z0", "Z origin", "0.0"));

	spacing = textToDouble(parser.getOption("--bin", "Binning", "1.0"));
    angpix_spacing = textToDouble(parser.getOption("--binned_angpix", "OR: desired pixel size after binning", "-1"));

    tiltAngleOffset = textToDouble(parser.getOption("--tiltangle_offset", "Offset applied to all tilt angles (in deg)", "0"));
    BfactorPerElectronDose = textToDouble(parser.getOption("--bfactor_per_edose", "B-factor dose-weighting per electron/A^2 dose (default is use Niko's model)", "0"));
    n_threads = textToInteger(parser.getOption("--j", "Number of threads", "1"));

    do_2dproj = parser.checkOption("--do_proj", "Use this to skip calculation of 2D projection of the tomogram along the Z-axis");
    centre_2dproj = textToInteger(parser.getOption("--centre_proj", "Central Z-slice for 2D projection (in tomogram pixels from the middle)", "0"));
    thickness_2dproj = textToInteger(parser.getOption("--thickness_proj", "Thickness of the 2D projection (in tomogram pixels)", "10"));

	Log::readParams(parser);

	if (parser.checkForErrors())
	{
		REPORT_ERROR("Errors encountered on the command line (see above), exiting...");
	}
	
	if (applyPreWeight)
	{
		applyWeight = false;
	}

	ZIO::ensureParentDir(outFn);
}
void TomoBackprojectProgram::initialise(bool verbose)
{
    if (!tomogramSet.read(optimisationSet.tomograms))
        REPORT_ERROR("ERROR: there was a problem reading the tomogram set");

    tomoIndexTodo.clear();

    if (tomoName == "*")
    {
        do_multiple = true;

        for (int idx = 0; idx < tomogramSet.size(); idx++)
        {
            if (do_only_unfinished && exists(getOutputFileName(idx,false,false)))
                continue;
            tomoIndexTodo.push_back(idx);
        }

        if (outFn[outFn.size()-1] != '/') outFn += '/';
        FileName fn_dir = outFn + "tomograms/";

    }
    else
    {
        int myidx = tomogramSet.getTomogramIndex(tomoName);
        if (myidx < 0) REPORT_ERROR("ERROR: cannot find specific tomogram name \"" + tomoName + "\" in the input star file");
        tomoIndexTodo.push_back(myidx);
        do_multiple = false;
    }

    if (verbose)
    {
        std::cout << " + Reconstructing " << tomoIndexTodo.size() << " tomograms: " << std::endl;
        for (int idx = 0; idx < tomoIndexTodo.size(); idx++)
        {
            std::cout << "  - " << tomogramSet.getTomogramName(tomoIndexTodo[idx]) << std::endl;
        }
        if (fabs(tiltAngleOffset) > 0.)
        {
            std::cout << " + Applying a tilt angle offset of " << tiltAngleOffset << " degrees" << std::endl;
        }

        if (do_2dproj)
        {
            std::cout << " + Making 2D projections " << std::endl;
            std::cout << "    - centered at " << centre_2dproj << " tomogram pixels from the centre of the tomogram" << std::endl;
            std::cout << "    - and a thickness of " << thickness_2dproj << " tomogram pixels" << std::endl;
        }
    }

}

void TomoBackprojectProgram::run(int rank, int size)
{
    long my_first_idx, my_last_idx;
    divide_equally(tomoIndexTodo.size(), size, rank , my_first_idx, my_last_idx);

    int barstep, nr_todo = my_last_idx-my_first_idx+1;
    if (rank == 0)
    {
        std::cout << " + Reconstructing ... " << std::endl;
        init_progress_bar(nr_todo);
        barstep = XMIPP_MAX(1, nr_todo / 60);
    }
    for (long idx = my_first_idx; idx <= my_last_idx; idx++)
    {
        // Abort through the pipeline_control system
        if (pipeline_control_check_abort_job())
            exit(RELION_EXIT_ABORTED);

    if (do_even_odd_tomograms)
	{
		reconstructOneTomogram(tomoIndexTodo[idx],true,false); // true/false indicates to reconstruct tomogram from even frames
		reconstructOneTomogram(tomoIndexTodo[idx],false,true); // false/true indicates from odd frames
	}
	else
	{
		reconstructOneTomogram(tomoIndexTodo[idx],false,false);
	}
	
        if (rank == 0 && idx % barstep == 0)
            progress_bar(idx);
    }

    if (rank == 0) progress_bar(nr_todo);

}


void TomoBackprojectProgram::writeOutput(bool do_all_metadata)
{
    // If we were doing multiple tomograms, then also write the updated tomograms.star.
    if (do_multiple)
    {
        // for MPI program
        if (do_all_metadata) setMetaDataAllTomograms();

        tomogramSet.write(outFn + "tomograms.star");

        std::cout << " Written out: " << outFn << "tomograms.star" << std::endl;

    }
    else if (fabs(tiltAngleOffset) > 0.)
    {
        // Also write out the modified metadata file
        int idx=tomoIndexTodo[0];
        FileName fn_star;
        tomogramSet.globalTable.getValue(EMDL_TOMO_TILT_SERIES_STARFILE, fn_star, idx);
        FileName fn_newstar = getOutputFileWithNewUniqueDate(fn_star, outFn);
        tomogramSet.tomogramTables[idx].write(fn_newstar);

    }

}

void TomoBackprojectProgram::initialiseCtfScaleFactors(int tomoIndex, Tomogram &tomogram)
{
    // Skip initialisation of scale factor if it is already present in the tilt series star file
    if (tomogramSet.tomogramTables[tomoIndex].containsLabel(EMDL_CTF_SCALEFACTOR))
        return;

    const int fc = tomogram.frameCount;
    for (int f = 0; f < fc; f++)
    {
        RFLOAT ytilt;
        tomogramSet.tomogramTables[tomoIndex].getValueSafely(EMDL_TOMO_YTILT, ytilt, f);
        RFLOAT scale = cos(DEG2RAD(ytilt));
        tomogramSet.tomogramTables[tomoIndex].setValue(EMDL_CTF_SCALEFACTOR, scale, f);
        tomogram.centralCTFs[f].scale = scale;
    }
}

void TomoBackprojectProgram::reconstructOneTomogram(int tomoIndex, bool doEven, bool doOdd)
{
    Tomogram tomogram;

    if (doEven)
    {
    	tomogram = tomogramSet.loadTomogram(tomoIndex, true, true, false, w, h, d);
    }
    else if (doOdd)
    {
    	tomogram = tomogramSet.loadTomogram(tomoIndex, true, false, true, w, h, d);
    }
    else
    {
        tomogram = tomogramSet.loadTomogram(tomoIndex, true, false, false, w, h, d);
    }

    // Initialise CTF scale factors to cosine(tilt) if they're not present yet
    initialiseCtfScaleFactors(tomoIndex, tomogram);

	if (zeroDC) Normalization::zeroDC_stack(tomogram.stack);
	
	const int fc = tomogram.frameCount;

	BufferedImage<float> stackAct;
	std::vector<d4Matrix> projAct(fc);

	double pixelSizeAct = tomogramSet.getTiltSeriesPixelSize(tomoIndex);

    if (angpix_spacing > 0.)
    {
        spacing = angpix_spacing / pixelSizeAct;
    }

    if (fabs(tiltAngleOffset) > 0.)
    {
        tomogramSet.applyTiltAngleOffset(tomoIndex, tiltAngleOffset);
    }

    if (!tomogram.hasMatrices) REPORT_ERROR("ERROR; tomograms do not have tilt series alignment parameters to calculate projectionMatrices!");

    const int w1 = w / spacing + 0.5;
	const int h1 = h / spacing + 0.5;
	const int t1 = d / spacing;

	if (std::abs(spacing - 1.0) < 1e-2)
	{
		projAct = tomogram.projectionMatrices;
		stackAct = tomogram.stack;
	}
	else
	{
		for (int f = 0; f < fc; f++)
		{
			projAct[f] = tomogram.projectionMatrices[f] / spacing;
			projAct[f](3,3) = 1.0;
		}
		
		if (std::abs(spacing - 1.0) > 1e-2)
		{
			if (!do_multiple) Log::print("Resampling image stack");
			
			if (FourierCrop)
			{
				stackAct = Resampling::FourierCrop_fullStack(
						tomogram.stack, spacing, n_threads, true);
			}
			else
			{
				stackAct = Resampling::downsampleFiltStack_2D_full(
						tomogram.stack, spacing, n_threads);
			}
			
			pixelSizeAct *= spacing;

            tomogramSet.globalTable.setValue(EMDL_TOMO_TOMOGRAM_BINNING, spacing, tomoIndex);
		}
		else
		{
			stackAct = tomogram.stack;
		}
	}
	
	const int w_stackAct = stackAct.xdim;
	const int h_stackAct = stackAct.ydim;
	const int wh_stackAct = w_stackAct/2 + 1;
	
	
	d3Vector orig(x0, y0, z0);
	BufferedImage<float> out(w1, h1, t1);
	out.fill(0.f);
	
	BufferedImage<float> psfStack;

	if (applyCtf)
	{
		// modulate stackAct with CTF (mind the spacing)
		
		psfStack.resize(w_stackAct, h_stackAct, fc);
		BufferedImage<fComplex> debug(wh_stackAct, h_stackAct, fc);
		
		#pragma omp parallel for num_threads(n_threads)
		for (int f = 0; f < fc; f++)
		{
			BufferedImage<float> frame = stackAct.getSliceRef(f);
			
			BufferedImage<fComplex> frameFS;
			FFT::FourierTransform(frame, frameFS, FFT::Both);
			
			CTF ctf = tomogram.centralCTFs[f];
			
			
			BufferedImage<fComplex> ctf2ImageFS(wh_stackAct, h_stackAct);
			
			const double box_size_x = pixelSizeAct * w_stackAct;
			const double box_size_y = pixelSizeAct * h_stackAct;
			
			for (int y = 0; y < h_stackAct;  y++)
			for (int x = 0; x < wh_stackAct; x++)
			{
				const double xA = x / box_size_x;
				const double yA = (y < h_stackAct/2? y : y - h_stackAct) / box_size_y;
				
                const float c = ctf.getCTF(xA, yA, false, false,
                                           true, false, 0.0, false);

				
				ctf2ImageFS(x,y) = fComplex(c*c,0);
				frameFS(x,y) *= c;
			}
			
			FFT::inverseFourierTransform(frameFS, frame, FFT::Both);
			stackAct.getSliceRef(f).copyFrom(frame);
			
			FFT::inverseFourierTransform(ctf2ImageFS, frame, FFT::Both);
			psfStack.getSliceRef(f).copyFrom(frame);
			
			debug.getSliceRef(f).copyFrom(ctf2ImageFS);
		}
	}	
	
	if (applyPreWeight)
	{
		stackAct = RealSpaceBackprojection::preWeight(stackAct, projAct, n_threads);
	}

    if (!do_multiple) Log::print("Backprojecting");
	
	RealSpaceBackprojection::backproject(
		stackAct, projAct, out, n_threads,
		orig, spacing, RealSpaceBackprojection::Linear, taperFalloff, taperDist);
	
	
	if ((applyWeight || applyCtf) && doWiener)
	{
		BufferedImage<float> psf(w1, h1, t1);
		psf.fill(0.f);
		
		if (applyCtf)
		{
			RealSpaceBackprojection::backproject(
					psfStack, projAct, psf, n_threads, 
					orig, spacing, RealSpaceBackprojection::Linear, taperFalloff, taperDist);
		}
		else
		{
			RealSpaceBackprojection::backprojectPsf(
					stackAct, projAct, psf, n_threads, orig, spacing);
		}
		
		Reconstruction::correct3D_RS(out, psf, out, 1.0 / SNR, n_threads);
	}

    if (!do_multiple) Log::print("Writing output");

    const double samplingRate = tomogramSet.getTiltSeriesPixelSize(tomoIndex) * spacing;

    if (doEven)
    	out.write(getOutputFileName(tomoIndex, true, false), samplingRate);
    else if (doOdd)
    	out.write(getOutputFileName(tomoIndex, false, true), samplingRate);
    else 
        out.write(getOutputFileName(tomoIndex, false, false), samplingRate);

    // Also add the tomogram sizes and name to the tomogramSet
    tomogramSet.globalTable.setValue(EMDL_TOMO_SIZE_X, w, tomoIndex);
    tomogramSet.globalTable.setValue(EMDL_TOMO_SIZE_Y, h, tomoIndex);
    tomogramSet.globalTable.setValue(EMDL_TOMO_SIZE_Z, d, tomoIndex);

    if (doEven)
    	tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_HALF1_FILE_NAME, getOutputFileName(tomoIndex, true, false), tomoIndex);
    else if (doOdd)
        tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_HALF2_FILE_NAME, getOutputFileName(tomoIndex, false, true), tomoIndex);
    else  
        tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_FILE_NAME, getOutputFileName(tomoIndex, false, false), tomoIndex);

    if (do_2dproj)
    {
        BufferedImage<float> proj(w1, h1);
        proj.fill(0.f);
        int minz = out.zdim/2 + centre_2dproj - thickness_2dproj/2;
        int maxz = out.zdim/2 + centre_2dproj + thickness_2dproj/2;
        for (int z = 0; z < out.zdim; z++)
        {
            if (z >= minz && z <= maxz)
            {
                for (int y = 0; y < out.ydim; y++)
                    for (int x = 0; x < out.xdim; x++)
                        proj(x, y) += out(x, y, z);
            }
        }
        if (doEven)
            proj.write(getOutputFileName(tomoIndex, true, false, true), samplingRate);
        else if (doOdd)
            proj.write(getOutputFileName(tomoIndex, false, true, true), samplingRate);
        else
            proj.write(getOutputFileName(tomoIndex, false, false, true), samplingRate);

        if (doEven)
            tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_PROJ2D_HALF1_FILE_NAME, getOutputFileName(tomoIndex, true, false, true), tomoIndex);
        else if (doOdd)
            tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_PROJ2D_HALF2_FILE_NAME, getOutputFileName(tomoIndex, false, true, true), tomoIndex);
        else
            tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_PROJ2D_FILE_NAME, getOutputFileName(tomoIndex, false, false, true), tomoIndex);

    }


}

void TomoBackprojectProgram::setMetaDataAllTomograms()
{

    for (int tomoIndex = 0; tomoIndex < tomogramSet.size(); tomoIndex++)
    {

        // SHWS 19apr2023: need to do this again for all tomograms: after completion of MPI job, leader does not know about the tomograms of the followers.
        Tomogram tomogram;
        tomogram = tomogramSet.loadTomogram(tomoIndex, false);
        if (!tomogram.hasMatrices) REPORT_ERROR("ERROR: tomograms do not have tilt series alignment parameters to calculate projectionMatrices");

        if (fabs(tiltAngleOffset) > 0.)
        {
            tomogramSet.applyTiltAngleOffset(tomoIndex, tiltAngleOffset);
        }

        double pixelSizeAct = tomogramSet.getTiltSeriesPixelSize(tomoIndex);
        if (angpix_spacing > 0.) spacing = angpix_spacing / pixelSizeAct;
        if (std::abs(spacing - 1.0) > 1e-2)
            tomogramSet.globalTable.setValue(EMDL_TOMO_TOMOGRAM_BINNING, spacing, tomoIndex);

        // Also add the tomogram sizes and name to the tomogramSet
        tomogramSet.globalTable.setValue(EMDL_TOMO_SIZE_X, w, tomoIndex);
        tomogramSet.globalTable.setValue(EMDL_TOMO_SIZE_Y, h, tomoIndex);
        tomogramSet.globalTable.setValue(EMDL_TOMO_SIZE_Z, d, tomoIndex);

        // And the Bfactor per e/A^2 dose, if provided
        if (BfactorPerElectronDose > 0.)
            tomogramSet.globalTable.setValue(EMDL_CTF_BFACTOR_PERELECTRONDOSE, BfactorPerElectronDose, tomoIndex);

        if (do_even_odd_tomograms)
        {
            tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_HALF1_FILE_NAME,
                                             getOutputFileName(tomoIndex, true, false), tomoIndex);
            tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_HALF2_FILE_NAME,
                                             getOutputFileName(tomoIndex, false, true), tomoIndex);
        }
        else
        {
            tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_FILE_NAME,
                                             getOutputFileName(tomoIndex, false, false), tomoIndex);
        }

        if (do_2dproj)
        {
            if (do_even_odd_tomograms)
            {
                tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_PROJ2D_HALF1_FILE_NAME, getOutputFileName(tomoIndex, true, false, true), tomoIndex);
                tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_PROJ2D_HALF2_FILE_NAME, getOutputFileName(tomoIndex, false, true, true), tomoIndex);
            }
            else
            {
                tomogramSet.globalTable.setValue(EMDL_TOMO_RECONSTRUCTED_TOMOGRAM_PROJ2D_FILE_NAME, getOutputFileName(tomoIndex, false, false, true), tomoIndex);
            }
        }
    }

}

FileName TomoBackprojectProgram::getOutputFileName(int index, bool nameEven, bool nameOdd, bool is_2dproj)
{
    // If we're reconstructing many tomograms, or the output filename is a directory: use standardized output filenames
    FileName fn_result = outFn;

    std::string dirname = (is_2dproj) ? "projections/" : "tomograms/";
    if (do_even_odd_tomograms)
    {
		if (nameEven)
		{
			fn_result += dirname + "rec_" + tomogramSet.getTomogramName(index)+"_half1.mrc";
		}
		else if (nameOdd)
		{
			fn_result += dirname + "rec_" + tomogramSet.getTomogramName(index)+"_half2.mrc";
		}
    }
	else
	{
	    fn_result += dirname + "rec_" + tomogramSet.getTomogramName(index)+".mrc";
	}

    if (!exists(fn_result.beforeLastOf("/"))) mktree(fn_result.beforeLastOf("/"));

    return fn_result;

}
