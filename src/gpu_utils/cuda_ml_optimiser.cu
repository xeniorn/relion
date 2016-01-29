#include <sys/time.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include "src/gpu_utils/cuda_projector.h"
#include "src/gpu_utils/cuda_projector.cuh"
#include "src/gpu_utils/cuda_projector_plan.h"
#include "src/gpu_utils/cuda_benchmark_utils.cuh"
#include "src/gpu_utils/cuda_ml_optimiser.h"
#include "src/gpu_utils/cuda_kernels/helper.cuh"
#include "src/gpu_utils/cuda_kernels/diff2.cuh"
#include "src/gpu_utils/cuda_kernels/wavg.cuh"
#include "src/gpu_utils/cuda_helper_functions.cuh"
#include "src/gpu_utils/cuda_mem_utils.h"
#include "src/complex.h"
#include <fstream>
#include <cuda_runtime.h>
#include "src/parallel.h"
#include <signal.h>
#include <map>

#ifdef CUDA_FORCESTL
#include "src/gpu_utils/cuda_utils_stl.cuh"
#else
#include "src/gpu_utils/cuda_utils_cub.cuh"
#endif

static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;


void getFourierTransformsAndCtfs(long int my_ori_particle,
		OptimisationParamters &op,
		SamplingParameters &sp,
		MlOptimiser *baseMLO,
		MlOptimiserCuda *cudaMLO
		)
{
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_FT);
#endif
	//FourierTransformer transformer;
	CUSTOM_ALLOCATOR_REGION_NAME("GFTCTF");

	for (int ipart = 0; ipart < baseMLO->mydata.ori_particles[my_ori_particle].particles_id.size(); ipart++)
	{
		CUDA_CPU_TIC("init");
		FileName fn_img;
		Image<RFLOAT> img, rec_img;
		MultidimArray<Complex > Fimg;
		MultidimArray<Complex > Faux(cudaMLO->transformer.fFourier,true);
		MultidimArray<RFLOAT> Fctf;

		// Get the right line in the exp_fn_img strings (also exp_fn_recimg and exp_fn_ctfs)
		int istop = 0;
		for (long int ii = baseMLO->exp_my_first_ori_particle; ii < my_ori_particle; ii++)
			istop += baseMLO->mydata.ori_particles[ii].particles_id.size();
		istop += ipart;

		// What is my particle_id?
		long int part_id = baseMLO->mydata.ori_particles[my_ori_particle].particles_id[ipart];
		// Which group do I belong?
		int group_id =baseMLO->mydata.getGroupId(part_id);

		// Get the norm_correction
		RFLOAT normcorr = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NORM);

		// Get the optimal origin offsets from the previous iteration
		Matrix1D<RFLOAT> my_old_offset(2), my_prior(2);
		XX(my_old_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_XOFF);
		YY(my_old_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_YOFF);
		XX(my_prior)      = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_XOFF_PRIOR);
		YY(my_prior)      = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_YOFF_PRIOR);
		// Uninitialised priors were set to 999.
		if (XX(my_prior) > 998.99 && XX(my_prior) < 999.01)
			XX(my_prior) = 0.;
		if (YY(my_prior) > 998.99 && YY(my_prior) < 999.01)
			YY(my_prior) = 0.;

		if (baseMLO->mymodel.data_dim == 3)
		{
			my_old_offset.resize(3);
			my_prior.resize(3);
			ZZ(my_old_offset) = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_ZOFF);
			ZZ(my_prior)      = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_ZOFF_PRIOR);
			// Unitialised priors were set to 999.
			if (ZZ(my_prior) > 998.99 && ZZ(my_prior) < 999.01)
				ZZ(my_prior) = 0.;
		}
		CUDA_CPU_TOC("init");

		CUDA_CPU_TIC("nonZeroProb");
		if (baseMLO->mymodel.orientational_prior_mode != NOPRIOR && !(baseMLO->do_skip_align ||baseMLO-> do_skip_rotate))
		{
			// First try if there are some fixed prior angles
			RFLOAT prior_rot = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_ROT_PRIOR);
			RFLOAT prior_tilt = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_TILT_PRIOR);
			RFLOAT prior_psi = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_PSI_PRIOR);

			// If there were no defined priors (i.e. their values were 999.), then use the "normal" angles
			if (prior_rot > 998.99 && prior_rot < 999.01)
				prior_rot = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_ROT);
			if (prior_tilt > 998.99 && prior_tilt < 999.01)
				prior_tilt = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_TILT);
			if (prior_psi > 998.99 && prior_psi < 999.01)
				prior_psi = DIRECT_A2D_ELEM(baseMLO->exp_metadata,op. metadata_offset + ipart, METADATA_PSI);

			////////// TODO TODO TODO
			////////// How does this work now: each particle has a different sampling object?!!!
			// Select only those orientations that have non-zero prior probability
			baseMLO->sampling.selectOrientationsWithNonZeroPriorProbability(prior_rot, prior_tilt, prior_psi,
					sqrt(baseMLO->mymodel.sigma2_rot), sqrt(baseMLO->mymodel.sigma2_tilt), sqrt(baseMLO->mymodel.sigma2_psi),
					op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);

			long int nr_orients = baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior) * baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior);
			if (nr_orients == 0)
			{
				std::cerr << " sampling.NrDirections()= " << baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior)
						<< " sampling.NrPsiSamplings()= " << baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior) << std::endl;
				REPORT_ERROR("Zero orientations fall within the local angular search. Increase the sigma-value(s) on the orientations!");
			}

		}
		CUDA_CPU_TOC("nonZeroProb");

		// Get the image and recimg data
		if (baseMLO->do_parallel_disc_io)
		{
			CUDA_CPU_TIC("setXmippOrigin");

			// If all slaves had preread images into RAM: get those now
			if (baseMLO->do_preread_images)
			{
				img() = baseMLO->mydata.particles[part_id].img;
			}
			else
			{
				// Read from disc
				FileName fn_img;
				std::istringstream split(baseMLO->exp_fn_img);
				for (int i = 0; i <= istop; i++)
					getline(split, fn_img);

				img.read(fn_img);
				img().setXmippOrigin();
			}
			if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
			{
				FileName fn_recimg;
				std::istringstream split2(baseMLO->exp_fn_recimg);
				// Get the right line in the exp_fn_img string
				for (int i = 0; i <= istop; i++)
					getline(split2, fn_recimg);
				rec_img.read(fn_recimg);
				rec_img().setXmippOrigin();
			}
			CUDA_CPU_TOC("setXmippOrigin");
		}
		else
		{
			CUDA_CPU_TIC("setXmippOrigin");
			// Unpack the image from the imagedata
			if (baseMLO->mymodel.data_dim == 3)
			{
				img().resize(baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size,baseMLO-> mymodel.ori_size);
				// Only allow a single image per call of this function!!! nr_pool needs to be set to 1!!!!
				// This will save memory, as we'll need to store all translated images in memory....
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(img())
				{
					DIRECT_A3D_ELEM(img(), k, i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, k, i, j);
				}
				img().setXmippOrigin();

				if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
				{
					rec_img().resize(baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size,baseMLO-> mymodel.ori_size);
					int offset = (baseMLO->do_ctf_correction) ? 2 * baseMLO->mymodel.ori_size : baseMLO->mymodel.ori_size;
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(rec_img())
					{
						DIRECT_A3D_ELEM(rec_img(), k, i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, offset + k, i, j);
					}
					rec_img().setXmippOrigin();

				}

			}
			else
			{
				img().resize(baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size);
				FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(img())
				{
					DIRECT_A2D_ELEM(img(), i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, op.metadata_offset + ipart, i, j);
				}
				img().setXmippOrigin();
				if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images)
				{

					////////////// TODO: think this through for no-threads here.....
					rec_img().resize(baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size);
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(rec_img())
					{
						DIRECT_A2D_ELEM(rec_img(), i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, baseMLO->exp_nr_images + op.metadata_offset + ipart, i, j);
					}
					rec_img().setXmippOrigin();
				}
			}
			CUDA_CPU_TOC("setXmippOrigin");
		}

		CUDA_CPU_TIC("selfTranslate");

		// Apply (rounded) old offsets first
		my_old_offset.selfROUND();

		int img_size = img.data.nzyxdim;
		CudaGlobalPtr<XFLOAT> d_img(img_size,0,cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> temp(img_size,0,cudaMLO->devBundle->allocator);
		d_img.device_alloc();
		temp.device_alloc();
		d_img.device_init(0);

		for (int i=0; i<img_size; i++)
			temp[i] = img.data.data[i];

		temp.cp_to_device();
		temp.streamSync();

		int STBsize = ( (int) ceilf(( float)img_size /(float)BLOCK_SIZE));
		// Apply the norm_correction term
		if (baseMLO->do_norm_correction)
		{
			CUDA_CPU_TIC("norm_corr");
			cuda_kernel_multi<<<STBsize,BLOCK_SIZE>>>(
									~temp,
									(XFLOAT)(baseMLO->mymodel.avg_norm_correction / normcorr),
									img_size);
			temp.streamSync();
			CUDA_CPU_TOC("norm_corr");
		}

		CUDA_CPU_TIC("kenrel_translate");
		cuda_kernel_translate2D<<<STBsize,BLOCK_SIZE>>>(
								~temp,  // translate from temp...
								~d_img, // ... into d_img
								img_size,
								img.data.xdim,
								img.data.ydim,
								XX(my_old_offset),
								YY(my_old_offset));
		CUDA_CPU_TOC("kenrel_translate");
//		d_img.cp_to_host();
//		d_img.streamSync();
//		for (int i=0; i<img_size; i++)
//			img.data.data[i] = d_img[i];

//		selfTranslate(img(), my_old_offset, DONT_WRAP);
		if (baseMLO->has_converged && baseMLO->do_use_reconstruct_images) //rec_img is NOT norm_corrected in the CPU-code, so nor do we.
		{
			for (int i=0; i<img_size; i++)
				temp[i] = rec_img.data.data[i];
			temp.cp_to_device();
			temp.streamSync();
			cuda_kernel_translate2D<<<STBsize,BLOCK_SIZE>>>(
											~temp,  // translate from temp...
											~d_img, // ... into d_img
											img_size,
											img.data.xdim,
											img.data.ydim,
											XX(my_old_offset),
											YY(my_old_offset));

//			d_img.cp_to_host();
//			d_img.streamSync();
//
//			for (int i=0; i<img_size; i++)
//				rec_img.data.data[i] = d_img[i];
//			selfTranslate(rec_img(), my_old_offset, DONT_WRAP);
		}

		op.old_offset[ipart] = my_old_offset;
		// Also store priors on translations
		op.prior[ipart] = my_prior;

		CUDA_CPU_TOC("selfTranslate");

//		// Always store FT of image without mask (to be used for the reconstruction)
//		MultidimArray<RFLOAT> img_aux;
//		img_aux = (baseMLO->has_converged && baseMLO->do_use_reconstruct_images) ? rec_img() : img();

		CUDA_CPU_TIC("calcFimg");
		unsigned current_size_x = baseMLO->mymodel.current_size / 2 + 1;
		unsigned current_size_y = baseMLO->mymodel.current_size;

		cudaMLO->transformer1.setSize(img().xdim,img().ydim);

		d_img.cp_on_device(cudaMLO->transformer1.reals);
//		for (int i = 0; i < img_aux.nzyxdim; i ++)
//			cudaMLO->transformer1.reals[i] = (XFLOAT) img_aux.data[i];
//
//		cudaMLO->transformer1.reals.cp_to_device();

		runCenterFFT(
				cudaMLO->transformer1.reals,
				(int)cudaMLO->transformer1.xSize,
				(int)cudaMLO->transformer1.ySize,
				false
				);

		cudaMLO->transformer1.forward();
		int FMultiBsize = ( (int) ceilf(( float)cudaMLO->transformer1.fouriers.getSize()*2/(float)BLOCK_SIZE));
		cuda_kernel_multi<<<FMultiBsize,BLOCK_SIZE>>>(
						(XFLOAT*)~cudaMLO->transformer1.fouriers,
						(XFLOAT)1/((XFLOAT)(cudaMLO->transformer1.reals.getSize())),
						cudaMLO->transformer1.fouriers.getSize()*2);

		CudaGlobalPtr<cufftComplex> d_Fimg(current_size_x * current_size_y, cudaMLO->devBundle->allocator);
		d_Fimg.device_alloc();

		windowFourierTransform2(
				cudaMLO->transformer1.fouriers,
				d_Fimg,
				cudaMLO->transformer1.xFSize,cudaMLO->transformer1.yFSize, 1, //Input dimensions
				current_size_x, current_size_y, 1 //Output dimensions
				);
		CUDA_CPU_TOC("calcFimg");

		CUDA_CPU_TIC("cpFimg2Host");
		d_Fimg.cp_to_host();
		d_Fimg.streamSync();

		Fimg.initZeros(current_size_y, current_size_x);
		for (int i = 0; i < Fimg.nzyxdim; i ++)
		{
			Fimg.data[i].real = (RFLOAT) d_Fimg[i].x;
			Fimg.data[i].imag = (RFLOAT) d_Fimg[i].y;
		}
		CUDA_CPU_TOC("cpFimg2Host");

		CUDA_CPU_TIC("selfApplyBeamTilt");
		// Here apply the beamtilt correction if necessary
		// This will only be used for reconstruction, not for alignment
		// But beamtilt only affects very high-resolution components anyway...
		//
		RFLOAT beamtilt_x = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_BEAMTILT_X);
		RFLOAT beamtilt_y = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_BEAMTILT_Y);
		RFLOAT Cs = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_CS);
		RFLOAT V = 1000. * DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_VOLTAGE);
		RFLOAT lambda = 12.2643247 / sqrt(V * (1. + V * 0.978466e-6));
		if (ABS(beamtilt_x) > 0. || ABS(beamtilt_y) > 0.)
			selfApplyBeamTilt(Fimg, beamtilt_x, beamtilt_y, lambda, Cs,baseMLO->mymodel.pixel_size, baseMLO->mymodel.ori_size);

		op.Fimgs_nomask.at(ipart) = Fimg;

		CUDA_CPU_TOC("selfApplyBeamTilt");

		CUDA_CPU_TIC("zeroMask");
		MultidimArray<RFLOAT> Mnoise;
		if (!baseMLO->do_zero_mask)
		{
			// Make a noisy background image with the same spectrum as the sigma2_noise

			// Different MPI-distributed subsets may otherwise have different instances of the random noise below,
			// because work is on an on-demand basis and therefore variable with the timing of distinct nodes...
			// Have the seed based on the part_id, so that each particle has a different instant of the noise
			if (baseMLO->do_realign_movies)
				init_random_generator(baseMLO->random_seed + part_id);
			else
				init_random_generator(baseMLO->random_seed + my_ori_particle); // This only serves for exact reproducibility tests with 1.3-code...

			// If we're doing running averages, then the sigma2_noise was already adjusted for the running averages.
			// Undo this adjustment here in order to get the right noise in the individual frames
			MultidimArray<RFLOAT> power_noise = baseMLO->sigma2_fudge * baseMLO->mymodel.sigma2_noise[group_id];
			if (baseMLO->do_realign_movies)
				power_noise *= (2. * baseMLO->movie_frame_running_avg_side + 1.);

			// Create noisy image for outside the mask
			MultidimArray<Complex > Fnoise;
			Mnoise.resize(img());
			cudaMLO->transformer.setReal(Mnoise);
			cudaMLO->transformer.getFourierAlias(Fnoise);
			// Fill Fnoise with random numbers, use power spectrum of the noise for its variance
			FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fnoise)
			{
				int ires = ROUND( sqrt( (RFLOAT)(kp * kp + ip * ip + jp * jp) ) );
				if (ires >= 0 && ires < XSIZE(Fnoise))
				{
					RFLOAT sigma = sqrt(DIRECT_A1D_ELEM(power_noise, ires));
					DIRECT_A3D_ELEM(Fnoise, k, i, j).real = rnd_gaus(0., sigma);
					DIRECT_A3D_ELEM(Fnoise, k, i, j).imag = rnd_gaus(0., sigma);
				}
				else
				{
					DIRECT_A3D_ELEM(Fnoise, k, i, j) = 0.;
				}
			}
			// Back to real space Mnoise
			CUDA_CPU_TIC("inverseFourierTransform");
			cudaMLO->transformer.inverseFourierTransform();
			CUDA_CPU_TOC("inverseFourierTransform");

			CUDA_CPU_TIC("setXmippOrigin");
			Mnoise.setXmippOrigin();
			CUDA_CPU_TOC("setXmippOrigin");

			CUDA_CPU_TIC("softMaskOutsideMap");
			d_img.cp_to_host();
			d_img.streamSync();
			for (int i=0; i<img_size; i++)
				img.data.data[i] = d_img[i];
			softMaskOutsideMap(img(), baseMLO->particle_diameter / (2. * baseMLO->mymodel.pixel_size), (RFLOAT)baseMLO->width_mask_edge, &Mnoise);
			CUDA_CPU_TOC("softMaskOutsideMap");
		}
		else
		{
			CUDA_CPU_TIC("softMaskOutsideMap");

			XFLOAT cosine_width = baseMLO->width_mask_edge;
			XFLOAT radius = (XFLOAT)((RFLOAT)baseMLO->particle_diameter / (2. *baseMLO-> mymodel.pixel_size));
			if (radius < 0)
				radius = ((RFLOAT)img.data.xdim)/2.;
			XFLOAT radius_p = radius + cosine_width;


			bool do_softmaskOnGpu = true;
			if(do_softmaskOnGpu)
			{
//				CudaGlobalPtr<XFLOAT,false> dev_img(img().nzyxdim);
//				dev_img.device_alloc();
//				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img())
//					dev_img[n]=(XFLOAT)img.data.data[n];
//
//				dev_img.cp_to_device();
				dim3 block_dim = 1; //TODO
				cuda_kernel_softMaskOutsideMap<<<block_dim,SOFTMASK_BLOCK_SIZE>>>(	~d_img,
																					img().nzyxdim,
																					img.data.xdim,
																					img.data.ydim,
																					img.data.zdim,
																					img.data.xdim/2,
																					img.data.ydim/2,
																					img.data.zdim/2,
																					true,
																					radius,
																					radius_p,
																					cosine_width);

				d_img.cp_to_host();
				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

				FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img())
				{
					img.data.data[n]=(RFLOAT)d_img[n];
				}
			}
			else
				softMaskOutsideMap(img(), radius, (RFLOAT)cosine_width);

//			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(img())
//			{
//				std::cout << img.data.data[n] << std::endl;
//			}
//			exit(0);
			CUDA_CPU_TOC("softMaskOutsideMap");
		}
		CUDA_CPU_TOC("zeroMask");

		CUDA_CPU_TIC("setSize");
		cudaMLO->transformer2.setSize(img().xdim,img().ydim);
		CUDA_CPU_TOC("setSize");

		CUDA_CPU_TIC("transform");

		d_img.cp_on_device(cudaMLO->transformer2.reals);
//		for (int i = 0; i < img().nzyxdim; i ++)
//			cudaMLO->transformer2.reals[i] = (XFLOAT) img().data[i];
//		cudaMLO->transformer2.reals.cp_to_device();

		runCenterFFT(
				cudaMLO->transformer2.reals,
				(int)cudaMLO->transformer2.xSize,
				(int)cudaMLO->transformer2.ySize,
				false
				);

		cudaMLO->transformer2.forward();
		int FMultiBsize2 = ( (int) ceilf(( float)cudaMLO->transformer2.fouriers.getSize()*2/(float)BLOCK_SIZE));
		cuda_kernel_multi<<<FMultiBsize2,BLOCK_SIZE>>>(
						(XFLOAT*)~cudaMLO->transformer2.fouriers,
						(XFLOAT)1/((XFLOAT)(cudaMLO->transformer2.reals.getSize())),
						cudaMLO->transformer2.fouriers.getSize()*2);

		CUDA_CPU_TOC("transform");

		bool powerClassOnGPU(true); //keep it like this until stable

		if(!powerClassOnGPU)
		{
			CUDA_CPU_TIC("cpResults");
			cudaMLO->transformer2.fouriers.cp_to_host();
			cudaMLO->transformer2.fouriers.streamSync();

			Faux.initZeros(img().ydim, img().xdim/2+1);
			for (int i = 0; i < Faux.nzyxdim; i ++)
			{
				Faux.data[i].real = (RFLOAT) cudaMLO->transformer2.fouriers[i].x;
				Faux.data[i].imag = (RFLOAT) cudaMLO->transformer2.fouriers[i].y;
			}
			CUDA_CPU_TOC("cpResults");
		}

		CUDA_CPU_TIC("powerClass");
		// Store the power_class spectrum of the whole image (to fill sigma2_noise between current_size and ori_size
		if (baseMLO->mymodel.current_size < baseMLO->mymodel.ori_size)
		{
			if(!powerClassOnGPU)
			{
				MultidimArray<RFLOAT> spectrum;
				spectrum.initZeros(baseMLO->mymodel.ori_size/2 + 1);
				RFLOAT highres_Xi2 = 0.;
				FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Faux)
				{
					int ires = ROUND( sqrt( (RFLOAT)(kp*kp + ip*ip + jp*jp) ) );
					// Skip Hermitian pairs in the x==0 column

					if (ires > 0 && ires < baseMLO->mymodel.ori_size/2 + 1 && !(jp==0 && ip < 0) )
					{
						RFLOAT normFaux = norm(DIRECT_A3D_ELEM(Faux, k, i, j));
						DIRECT_A1D_ELEM(spectrum, ires) += normFaux;
						// Store sumXi2 from current_size until ori_size
						if (ires >= baseMLO->mymodel.current_size/2 + 1)
							highres_Xi2 += normFaux;
					}
				}

				// Let's use .at() here instead of [] to check whether we go outside the vectors bounds
				op.power_imgs.at(ipart) = spectrum;
				op.highres_Xi2_imgs.at(ipart) = highres_Xi2;
			}
			else
			{
				CudaGlobalPtr<XFLOAT> dev_spectrum(baseMLO->mymodel.ori_size/2 + 1,0,cudaMLO->devBundle->allocator);
				CudaGlobalPtr<XFLOAT> dev_Xi2(1, 0, cudaMLO->devBundle->allocator);
				dev_spectrum.device_alloc();
				dev_spectrum.device_init(0);
				dev_Xi2.device_alloc();
				dev_Xi2.device_init(0);
				dev_spectrum.streamSync();

				dim3 gridSize = CEIL((float)(cudaMLO->transformer2.fouriers.getSize()) / (float)POWERCLASS_BLOCK_SIZE);
				cuda_kernel_powerClass2D<<<gridSize,POWERCLASS_BLOCK_SIZE,0,0>>>(
						~cudaMLO->transformer2.fouriers,
						~dev_spectrum,
						cudaMLO->transformer2.fouriers.getSize(),
						dev_spectrum.getSize(),
						cudaMLO->transformer2.xFSize,
						cudaMLO->transformer2.yFSize,
						(baseMLO->mymodel.current_size/2)+1,
						~dev_Xi2);

				dev_spectrum.streamSync();
				dev_spectrum.cp_to_host();
				dev_Xi2.cp_to_host();
				dev_spectrum.streamSync();

				op.power_imgs.at(ipart).resize(baseMLO->mymodel.ori_size/2 + 1);

				for (int i = 0; i < baseMLO->mymodel.ori_size/2 + 1; i ++)
					op.power_imgs.at(ipart).data[i] = dev_spectrum[i];
				op.highres_Xi2_imgs.at(ipart) = dev_Xi2[0];
			}
		}
		else
		{
			op.highres_Xi2_imgs.at(ipart) = 0.;
		}
		CUDA_CPU_TOC("powerClass");
		// We never need any resolutions higher than current_size
		// So resize the Fourier transforms
		CUDA_CPU_TIC("windowFourierTransform2");
		//windowFourierTransform(Faux, Fimg, baseMLO->mymodel.current_size);
		windowFourierTransform2(
				cudaMLO->transformer2.fouriers,
				d_Fimg,
				(int)cudaMLO->transformer2.xFSize,(int)cudaMLO->transformer2.yFSize, 1, //Input dimensions
				(int)current_size_x, (int)current_size_y, 1  //Output dimensions
				);
		CUDA_CPU_TOC("windowFourierTransform2");
		// Also store its CTF
		CUDA_CPU_TIC("ctfCorr");
		CUDA_CPU_TIC("cpFimg2Host_2");
		d_Fimg.streamSync();
		d_Fimg.cp_to_host();
		d_Fimg.streamSync();
		for (int i = 0; i < Fimg.nzyxdim; i ++)
		{
			Fimg.data[i].real = (RFLOAT) d_Fimg[i].x;
			Fimg.data[i].imag = (RFLOAT) d_Fimg[i].y;
		}
		CUDA_CPU_TOC("cpFimg2Host_2");

		Fctf.resize(Fimg);
		// Now calculate the actual CTF
		if (baseMLO->do_ctf_correction)
		{
			if (baseMLO->mymodel.data_dim == 3)
			{
				Image<RFLOAT> Ictf;
				if (baseMLO->do_parallel_disc_io)
				{
					// Read CTF-image from disc
					FileName fn_ctf;
					std::istringstream split(baseMLO->exp_fn_ctf);
					// Get the right line in the exp_fn_img string
					for (int i = 0; i <= istop; i++)
						getline(split, fn_ctf);
					Ictf.read(fn_ctf);
				}
				else
				{
					// Unpack the CTF-image from the exp_imagedata array
					Ictf().resize(baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size);
					FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(Ictf())
					{
						DIRECT_A3D_ELEM(Ictf(), k, i, j) = DIRECT_A3D_ELEM(baseMLO->exp_imagedata, baseMLO->mymodel.ori_size + k, i, j);
					}
				}
				// Set the CTF-image in Fctf
				Ictf().setXmippOrigin();
				FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fctf)
				{
					// Use negative kp,ip and jp indices, because the origin in the ctf_img lies half a pixel to the right of the actual center....
					DIRECT_A3D_ELEM(Fctf, k, i, j) = A3D_ELEM(Ictf(), -kp, -ip, -jp);
				}
			}
			else
			{
				CTF ctf;
				ctf.setValues(DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_DEFOCUS_U),
							  DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_DEFOCUS_V),
							  DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_DEFOCUS_ANGLE),
							  DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_VOLTAGE),
							  DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_CS),
							  DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_Q0),
							  DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CTF_BFAC));

				ctf.getFftwImage(Fctf, baseMLO->mymodel.ori_size, baseMLO->mymodel.ori_size, baseMLO->mymodel.pixel_size,
						baseMLO->ctf_phase_flipped, baseMLO->only_flip_phases, baseMLO->intact_ctf_first_peak, true);
			}
		}
		else
		{
			Fctf.initConstant(1.);
		}
		CUDA_CPU_TOC("ctfCorr");
		// Store Fimg and Fctf
		op.Fimgs.at(ipart) = Fimg;
		op.Fctfs.at(ipart) = Fctf;

	} // end loop ipart
	//cudaMLO->transformer.clear();
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_FT);
#endif
}

void getAllSquaredDifferencesCoarse(
		unsigned exp_ipass,
		OptimisationParamters &op,
		SamplingParameters &sp,
		MlOptimiser *baseMLO,
		MlOptimiserCuda *cudaMLO,
	 	CudaGlobalPtr<XFLOAT> &Mweight)
{

#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF1);
#endif

	CUSTOM_ALLOCATOR_REGION_NAME("DIFF_COARSE");

	CUDA_CPU_TIC("diff_pre_gpu");

	unsigned long weightsPerPart(baseMLO->mymodel.nr_classes * sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.nr_oversampled_rot * sp.nr_oversampled_trans);

	op.min_diff2.clear();
	op.min_diff2.resize(sp.nr_particles, LARGE_NUMBER);

	std::vector<MultidimArray<Complex > > dummy;
	baseMLO->precalculateShiftedImagesCtfsAndInvSigma2s(false, op.my_ori_particle, sp.current_image_size, sp.current_oversampling, op.metadata_offset, // inserted SHWS 12112015
			sp.itrans_min, sp.itrans_max, op.Fimgs, dummy, op.Fctfs, op.local_Fimgs_shifted, dummy,
			op.local_Fctfs, op.local_sqrtXi2, op.local_Minvsigma2s);

	unsigned image_size = op.local_Minvsigma2s[0].nzyxdim;

	CUDA_CPU_TOC("diff_pre_gpu");

	std::vector<CudaProjectorPlan> projectorPlans(0, cudaMLO->devBundle->allocator);

	//If particle specific sampling plan required
	if (cudaMLO->devBundle->generateProjectionPlanOnTheFly)
	{
		CUDA_CPU_TIC("generateProjectionSetupCoarse");

		projectorPlans.resize(baseMLO->mymodel.nr_classes, cudaMLO->devBundle->allocator);

		for (int iclass = sp.iclass_min; iclass <= sp.iclass_max; iclass++)
		{
			if (baseMLO->mymodel.pdf_class[iclass] > 0.)
			{
				projectorPlans[iclass].setup(
						baseMLO->sampling,
						op.directions_prior,
						op.psi_prior,
						op.pointer_dir_nonzeroprior,
						op.pointer_psi_nonzeroprior,
						NULL, //Mcoarse_significant
						baseMLO->mymodel.pdf_class,
						baseMLO->mymodel.pdf_direction,
						sp.nr_dir,
						sp.nr_psi,
						sp.idir_min,
						sp.idir_max,
						sp.ipsi_min,
						sp.ipsi_max,
						sp.itrans_min,
						sp.itrans_max,
						0, //current_oversampling
						1, //nr_oversampled_rot
						iclass,
						true, //coarse
						!IS_NOT_INV,
						baseMLO->do_skip_align,
						baseMLO->do_skip_rotate,
						baseMLO->mymodel.orientational_prior_mode
						);
			}
		}
		CUDA_CPU_TOC("generateProjectionSetupCoarse");
	}
	else
		projectorPlans = cudaMLO->devBundle->coarseProjectionPlans;

	// Loop only from sp.iclass_min to sp.iclass_max to deal with seed generation in first iteration
	size_t allWeights_size(0);
	for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		allWeights_size += projectorPlans[exp_iclass].orientation_num * sp.nr_trans*sp.nr_oversampled_trans * sp.nr_particles;

	CudaGlobalPtr<XFLOAT> allWeights(allWeights_size,cudaMLO->devBundle->allocator);
	allWeights.device_alloc();

	long int allWeights_pos=0;	bool do_CC = (baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc;

	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];
		long int group_id = baseMLO->mydata.getGroupId(part_id);

		/*====================================
				Generate Translations
		======================================*/

		CUDA_CPU_TIC("translation_1");

		long unsigned translation_num((sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans);

		CudaGlobalPtr<XFLOAT> trans_x(cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> trans_y(cudaMLO->devBundle->allocator);

		CudaGlobalPtr<XFLOAT> Fimgs_real(cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> Fimgs_imag(cudaMLO->devBundle->allocator);

		if (do_CC)
		{
			Fimgs_real.device_alloc(image_size * translation_num);
			Fimgs_imag.device_alloc(image_size * translation_num);

			if (baseMLO->do_shifts_onthefly)
			{
				CudaTranslator::Plan transPlan(
						op.local_Fimgs_shifted[ipart].data,
						image_size,
						sp.itrans_min * sp.nr_oversampled_trans,
						( sp.itrans_max + 1) * sp.nr_oversampled_trans,
						cudaMLO->devBundle->allocator,
						0, //stream
						baseMLO->do_scale_correction ? baseMLO->mymodel.scale_correction[group_id] : 1,
						baseMLO->do_ctf_correction && baseMLO->refs_are_ctf_corrected ? op.local_Fctfs[ipart].data : NULL);

				if (sp.current_oversampling == 0)
				{
					if (op.local_Minvsigma2s[0].ydim == baseMLO->coarse_size)
						cudaMLO->translator_coarse1.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
					else
						cudaMLO->translator_current1.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
				}
				else
				{
					if (baseMLO->strict_highres_exp > 0.)
						cudaMLO->translator_coarse2.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
					else
						cudaMLO->translator_current2.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
				}
			}
			else
			{
				Fimgs_real.host_alloc();
				Fimgs_imag.host_alloc();

				unsigned long k = 0;
				for (unsigned i = 0; i < op.local_Fimgs_shifted.size(); i ++)
				{
					for (unsigned j = 0; j < op.local_Fimgs_shifted[i].nzyxdim; j ++)
					{
						Fimgs_real[k] = op.local_Fimgs_shifted[i].data[j].real;
						Fimgs_imag[k] = op.local_Fimgs_shifted[i].data[j].imag;
						k++;
					}
				}

				Fimgs_real.cp_to_device();
				Fimgs_imag.cp_to_device();
			}
		}
		else
		{

			trans_x.setSize(translation_num);
			trans_y.setSize(translation_num);
			trans_x.host_alloc();
			trans_y.host_alloc();

			Fimgs_real.setSize(image_size);
			Fimgs_imag.setSize(image_size);
			Fimgs_real.host_alloc();
			Fimgs_imag.host_alloc();

			std::vector<RFLOAT> oversampled_translations_x, oversampled_translations_y, oversampled_translations_z;
			XFLOAT scale_correction = baseMLO->do_scale_correction ? baseMLO->mymodel.scale_correction[group_id] : 1;

			for (long int itrans = 0; itrans < translation_num; itrans++)
			{
				baseMLO->sampling.getTranslations(itrans, 0, oversampled_translations_x,
						oversampled_translations_y, oversampled_translations_z);

				trans_x[itrans] = -2 * PI * oversampled_translations_x[0] / (double)baseMLO->mymodel.ori_size;
				trans_y[itrans] = -2 * PI * oversampled_translations_y[0] / (double)baseMLO->mymodel.ori_size;
			}

			for (unsigned i = 0; i < op.local_Fimgs_shifted[ipart].nzyxdim; i ++)
			{
				XFLOAT pixel_correction = scale_correction;
				if (baseMLO->do_ctf_correction && baseMLO->refs_are_ctf_corrected)
					pixel_correction /= op.local_Fctfs[ipart].data[i];

				Fimgs_real[i] = op.local_Fimgs_shifted[ipart].data[i].real * pixel_correction;
				Fimgs_imag[i] = op.local_Fimgs_shifted[ipart].data[i].imag * pixel_correction;
			}


			trans_x.put_on_device();
			trans_y.put_on_device();
			Fimgs_real.put_on_device();
			Fimgs_imag.put_on_device();
		}
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		CUDA_CPU_TOC("translation_1");


		// To speed up calculation, several image-corrections are grouped into a single pixel-wise "filter", or image-correciton
		CudaGlobalPtr<XFLOAT> corr_img(image_size, cudaMLO->devBundle->allocator);
		corr_img.device_alloc();

		buildCorrImage(baseMLO,op,corr_img,ipart,group_id);
		corr_img.cp_to_device();

		deviceInitValue(allWeights, (XFLOAT) (op.highres_Xi2_imgs[ipart] / 2.));

		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			if ( projectorPlans[exp_iclass].orientation_num > 0 )
			{
				/*====================================
				    	   Kernel Call
				======================================*/

				CudaProjectorKernel projKernel = CudaProjectorKernel::makeKernel(
						cudaMLO->devBundle->cudaProjectors[exp_iclass],
						op.local_Minvsigma2s[0].xdim,
						op.local_Minvsigma2s[0].ydim,
						op.local_Minvsigma2s[0].xdim-1);

				runDiff2KernelCoarse(
						projKernel,
						~trans_x,
						~trans_y,
						~corr_img,
						~Fimgs_real,
						~Fimgs_imag,
						~projectorPlans[exp_iclass].eulers,
						&allWeights(allWeights_pos),
						op,
						baseMLO,
						projectorPlans[exp_iclass].orientation_num,
						translation_num,
						image_size,
						ipart,
						group_id,
						exp_iclass,
						cudaMLO->classStreams[exp_iclass],
						do_CC);

				mapAllWeightsToMweights(
						~projectorPlans[exp_iclass].iorientclasses,
						&allWeights(allWeights_pos),
						&Mweight(ipart*weightsPerPart),
						projectorPlans[exp_iclass].orientation_num,
						translation_num,
						cudaMLO->classStreams[exp_iclass]
						);

				/*====================================
				    	   Retrieve Results
				======================================*/
				allWeights_pos += projectorPlans[exp_iclass].orientation_num*translation_num;

			}
		}

		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaMLO->classStreams[exp_iclass]));

		op.min_diff2[ipart] = getMinOnDevice(allWeights);

	} // end loop ipart

#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF1);
#endif
}

void getAllSquaredDifferencesFine(unsigned exp_ipass,
		 	 	 	 	 	 	  OptimisationParamters &op,
		 	 	 	 	 	 	  SamplingParameters &sp,
		 	 	 	 	 	 	  MlOptimiser *baseMLO,
		 	 	 	 	 	 	  MlOptimiserCuda *cudaMLO,
		 	 	 	 	 	 	  std::vector<IndexedDataArray> &FinePassWeights,
		 	 	 	 	 	 	  std::vector<std::vector< IndexedDataArrayMask > > &FPCMasks,
		 	 	 	 	 	 	  std::vector<ProjectionParams> &FineProjectionData,
		 	 	 	 	 	 	  std::vector<cudaStager<unsigned long> > &stagerD2)
{
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2);
#endif

	CUSTOM_ALLOCATOR_REGION_NAME("DIFF_FINE");
	CUDA_CPU_TIC("diff_pre_gpu");

	op.min_diff2.clear();
	op.min_diff2.resize(sp.nr_particles, LARGE_NUMBER);
	CUDA_CPU_TIC("precalculateShiftedImagesCtfsAndInvSigma2s");
	std::vector<MultidimArray<Complex > > dummy;
	baseMLO->precalculateShiftedImagesCtfsAndInvSigma2s(false, op.my_ori_particle, sp.current_image_size, sp.current_oversampling, op.metadata_offset, // inserted SHWS 12112015
			sp.itrans_min, sp.itrans_max, op.Fimgs, dummy, op.Fctfs, op.local_Fimgs_shifted, dummy,
			op.local_Fctfs, op.local_sqrtXi2, op.local_Minvsigma2s);
	CUDA_CPU_TOC("precalculateShiftedImagesCtfsAndInvSigma2s");
	MultidimArray<Complex > Fref;
	Fref.resize(op.local_Minvsigma2s[0]);

	unsigned image_size = op.local_Minvsigma2s[0].nzyxdim;

	CUDA_CPU_TOC("diff_pre_gpu");

	/*=======================================================================================
										  Particle Iteration
	=========================================================================================*/
	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		// Reset size without de-allocating: we will append everything significant within
		// the current allocation and then re-allocate the then determined (smaller) volume

		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];
		long int group_id = baseMLO->mydata.getGroupId(part_id);

		/*====================================
				Generate Translations
		======================================*/

		CUDA_CPU_TIC("translation_2");

		long unsigned translation_num((sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans);

		CudaGlobalPtr<XFLOAT> Fimgs_real(cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> Fimgs_imag(cudaMLO->devBundle->allocator);

		Fimgs_real.device_alloc(image_size * translation_num);
		Fimgs_imag.device_alloc(image_size * translation_num);

		if (baseMLO->do_shifts_onthefly)
		{
			CudaTranslator::Plan transPlan(
					op.local_Fimgs_shifted[ipart].data,
					image_size,
					sp.itrans_min * sp.nr_oversampled_trans,
					( sp.itrans_max + 1) * sp.nr_oversampled_trans,
					cudaMLO->devBundle->allocator,
					0, //stream
					baseMLO->do_scale_correction ? baseMLO->mymodel.scale_correction[group_id] : 1,
					baseMLO->do_ctf_correction && baseMLO->refs_are_ctf_corrected ? op.local_Fctfs[ipart].data : NULL);

			if (sp.current_oversampling == 0)
			{
				if (op.local_Minvsigma2s[0].ydim == baseMLO->coarse_size)
					cudaMLO->translator_coarse1.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
				else
					cudaMLO->translator_current1.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
			}
			else
			{
				if (baseMLO->strict_highres_exp > 0.)
					cudaMLO->translator_coarse2.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
				else
					cudaMLO->translator_current2.translate(transPlan, ~Fimgs_real, ~Fimgs_imag);
			}
		}
		else
		{
			Fimgs_real.host_alloc();
			Fimgs_imag.host_alloc();

			unsigned long k = 0;
			for (unsigned i = 0; i < op.local_Fimgs_shifted.size(); i ++)
			{
				for (unsigned j = 0; j < op.local_Fimgs_shifted[i].nzyxdim; j ++)
				{
					Fimgs_real[k] = op.local_Fimgs_shifted[i].data[j].real;
					Fimgs_imag[k] = op.local_Fimgs_shifted[i].data[j].imag;
					k++;
				}
			}

			Fimgs_real.cp_to_device();
			Fimgs_imag.cp_to_device();
		}

		CUDA_CPU_TOC("translation_2");


		CUDA_CPU_TIC("kernel_init_1");

		CudaGlobalPtr<XFLOAT> corr_img(image_size, cudaMLO->devBundle->allocator);
		corr_img.device_alloc();
		buildCorrImage(baseMLO,op,corr_img,ipart,group_id);

		corr_img.cp_to_device();

		CUDA_CPU_TOC("kernel_init_1");
		std::vector< CudaGlobalPtr<XFLOAT> > eulers((sp.iclass_max-sp.iclass_min+1), cudaMLO->devBundle->allocator);
		cudaStager<XFLOAT> AllEulers(cudaMLO->devBundle->allocator,9*FineProjectionData[ipart].orientationNumAllClasses);
		AllEulers.prepare_device();
		unsigned long newDataSize(0);
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			FPCMasks[ipart][exp_iclass].weightNum=0;

			if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) && (FineProjectionData[ipart].class_entries[exp_iclass] > 0) )
			{
				// use "slice" constructor with class-specific parameters to retrieve a temporary ProjectionParams with data for this class
				ProjectionParams thisClassProjectionData(	FineProjectionData[ipart],
															FineProjectionData[ipart].class_idx[exp_iclass],
															FineProjectionData[ipart].class_idx[exp_iclass]+FineProjectionData[ipart].class_entries[exp_iclass]);
				// since we retrieved the ProjectionParams for *the whole* class the orientation_num is also equal.

				thisClassProjectionData.orientation_num[0] = FineProjectionData[ipart].class_entries[exp_iclass];
				long unsigned orientation_num  = thisClassProjectionData.orientation_num[0];

				if(orientation_num==0)
					continue;

				CUDA_CPU_TIC("pair_list_1");
				long unsigned significant_num(0);
				long int nr_over_orient = baseMLO->sampling.oversamplingFactorOrientations(sp.current_oversampling);
				long int nr_over_trans = baseMLO->sampling.oversamplingFactorTranslations(sp.current_oversampling);
				// Prepare the mask of the weight-array for this class
				if (FPCMasks[ipart][exp_iclass].weightNum==0)
					FPCMasks[ipart][exp_iclass].firstPos = newDataSize;

				long unsigned ihidden(0);
				std::vector< long unsigned > iover_transes, ihiddens;

				for (long int itrans = sp.itrans_min; itrans <= sp.itrans_max; itrans++, ihidden++)
				{
					for (long int iover_trans = 0; iover_trans < sp.nr_oversampled_trans; iover_trans++)
					{
						ihiddens.push_back(ihidden);
						iover_transes.push_back(iover_trans);
					}
				}

				// Do more significance checks on translations and create jobDivision
				significant_num = makeJobsForDiff2Fine(	op,	sp,												// alot of different type inputs...
														orientation_num, translation_num,
														thisClassProjectionData,
														iover_transes, ihiddens,
														nr_over_orient, nr_over_trans, ipart,
														FinePassWeights[ipart],
														FPCMasks[ipart][exp_iclass]);                // ..and output into index-arrays mask

				// extend size by number of significants found this class
				newDataSize += significant_num;
				FPCMasks[ipart][exp_iclass].weightNum = significant_num;
				FPCMasks[ipart][exp_iclass].lastPos = FPCMasks[ipart][exp_iclass].firstPos + significant_num;
				CUDA_CPU_TOC("pair_list_1");

				CUDA_CPU_TIC("IndexedArrayMemCp2");
//				FPCMasks[ipart][exp_iclass].jobOrigin.cp_to_device();
//				FPCMasks[ipart][exp_iclass].jobExtent.cp_to_device();
				stagerD2[ipart].stage(FPCMasks[ipart][exp_iclass].jobOrigin);
				stagerD2[ipart].stage(FPCMasks[ipart][exp_iclass].jobExtent);
				CUDA_CPU_TOC("IndexedArrayMemCp2");

				CUDA_CPU_TIC("generateEulerMatrices");
				eulers[exp_iclass-sp.iclass_min].setSize(9*FineProjectionData[ipart].class_entries[exp_iclass]);
				eulers[exp_iclass-sp.iclass_min].host_alloc();
				generateEulerMatrices(
						baseMLO->mymodel.PPref[exp_iclass].padding_factor,
						thisClassProjectionData,
						&(eulers[exp_iclass-sp.iclass_min])[0],
						!IS_NOT_INV);
				AllEulers.stage(eulers[exp_iclass-sp.iclass_min]);
				CUDA_CPU_TOC("generateEulerMatrices");
			}
		}
		// copy stagers to device
		stagerD2[ipart].cp_to_device();
		AllEulers.cp_to_device();

		FinePassWeights[ipart].rot_id.cp_to_device(); //FIXME this is not used
		FinePassWeights[ipart].rot_idx.cp_to_device();
		FinePassWeights[ipart].trans_idx.cp_to_device();
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) && (FineProjectionData[ipart].class_entries[exp_iclass] > 0) )
			{
				long unsigned orientation_num  = FineProjectionData[ipart].class_entries[exp_iclass];
				if(orientation_num==0)
					continue;

				long unsigned significant_num(FPCMasks[ipart][exp_iclass].weightNum);
				if(significant_num==0)
					continue;

				CUDA_CPU_TIC("Diff2MakeKernel");
				CudaProjectorKernel projKernel = CudaProjectorKernel::makeKernel(
						cudaMLO->devBundle->cudaProjectors[exp_iclass],
						op.local_Minvsigma2s[0].xdim,
						op.local_Minvsigma2s[0].ydim,
						op.local_Minvsigma2s[0].xdim-1);
				CUDA_CPU_TOC("Diff2MakeKernel");

				// Use the constructed mask to construct a partial class-specific input
				IndexedDataArray thisClassFinePassWeights(FinePassWeights[ipart],FPCMasks[ipart][exp_iclass], cudaMLO->devBundle->allocator);

				CUDA_CPU_TIC("Diff2CALL");

				runDiff2KernelFine(
						projKernel,
						~corr_img,
						~Fimgs_real,
						~Fimgs_imag,
						~(eulers[exp_iclass-sp.iclass_min]),
						~thisClassFinePassWeights.rot_id,
						~thisClassFinePassWeights.rot_idx,
						~thisClassFinePassWeights.trans_idx,
						~FPCMasks[ipart][exp_iclass].jobOrigin,
						~FPCMasks[ipart][exp_iclass].jobExtent,
						~thisClassFinePassWeights.weights,
						op,
						baseMLO,
						orientation_num,
						translation_num,
						significant_num,
						image_size,
						ipart,
						exp_iclass,
						cudaMLO->classStreams[exp_iclass],
						FPCMasks[ipart][exp_iclass].jobOrigin.getSize(),
						((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
						);

//				DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));
				CUDA_CPU_TOC("Diff2CALL");

			} // end if class significant
		} // end loop iclass

		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaMLO->classStreams[exp_iclass]));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		FinePassWeights[ipart].setDataSize( newDataSize );

		CUDA_CPU_TIC("collect_data_1");
		op.min_diff2[ipart] = std::min(op.min_diff2[ipart],(RFLOAT)getMinOnDevice(FinePassWeights[ipart].weights));
		CUDA_CPU_TOC("collect_data_1");
//		std::cerr << "  fine pass minweight  =  " << op.min_diff2[ipart] << std::endl;

	}// end loop ipart
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2);
#endif
}


void convertAllSquaredDifferencesToWeights(unsigned exp_ipass,
											OptimisationParamters &op,
											SamplingParameters &sp,
											MlOptimiser *baseMLO,
											MlOptimiserCuda *cudaMLO,
											std::vector< IndexedDataArray> &PassWeights,
											std::vector< std::vector< IndexedDataArrayMask > > &FPCMasks,
											CudaGlobalPtr<XFLOAT> &Mweight) // FPCMasks = Fine-Pass Class-Masks
{
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
	{
		if (exp_ipass == 0) baseMLO->timer.tic(baseMLO->TIMING_ESP_WEIGHT1);
		else baseMLO->timer.tic(baseMLO->TIMING_ESP_WEIGHT2);
	}
#endif


	op.sum_weight.clear();
	op.sum_weight.resize(sp.nr_particles, 0.);

	// Ready the "prior-containers" for all classes (remake every ipart)
	CudaGlobalPtr<XFLOAT>  pdf_orientation((sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi, cudaMLO->devBundle->allocator);
	CudaGlobalPtr<XFLOAT>  pdf_offset((sp.iclass_max-sp.iclass_min+1)*sp.nr_trans, cudaMLO->devBundle->allocator);

	RFLOAT pdf_orientation_mean(0);
	unsigned pdf_orientation_count(0);

	CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_PDF");

	pdf_orientation.device_alloc();
	pdf_offset.device_alloc();

	// pdf_orientation is ipart-independent, so we keep it above ipart scope
	CUDA_CPU_TIC("get_orient_priors");
	for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		for (long int idir = sp.idir_min, iorientclass = (exp_iclass-sp.iclass_min) * sp.nr_dir * sp.nr_psi; idir <=sp.idir_max; idir++)
			for (long int ipsi = sp.ipsi_min; ipsi <= sp.ipsi_max; ipsi++, iorientclass++)
			{
				RFLOAT pdf(0);

				if (baseMLO->do_skip_align || baseMLO->do_skip_rotate)
					pdf = baseMLO->mymodel.pdf_class[exp_iclass];
				else if (baseMLO->mymodel.orientational_prior_mode == NOPRIOR)
					pdf = DIRECT_MULTIDIM_ELEM(baseMLO->mymodel.pdf_direction[exp_iclass], idir);
				else
					pdf = op.directions_prior[idir] * op.psi_prior[ipsi];

				pdf_orientation[iorientclass] = pdf;
				pdf_orientation_mean += pdf;
				pdf_orientation_count ++;
			}


	pdf_orientation_mean /= (RFLOAT) pdf_orientation_count;

	//If mean is non-zero bring all values closer to 1 to improve numerical accuracy
	//This factor is over all classes and is thus removed in the final normalization
	if (pdf_orientation_mean != 0.)
		for (int i = 0; i < pdf_orientation.getSize(); i ++)
			pdf_orientation[i] /= pdf_orientation_mean;

	pdf_orientation.cp_to_device();
	CUDA_CPU_TOC("get_orient_priors");

	// loop over all particles inside this ori_particle
	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];

		RFLOAT old_offset_z;
		RFLOAT old_offset_x = XX(op.old_offset[ipart]);
		RFLOAT old_offset_y = YY(op.old_offset[ipart]);
		if (baseMLO->mymodel.data_dim == 3)
			old_offset_z = ZZ(op.old_offset[ipart]);

		if ((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
		{

			if(exp_ipass==0)
			{
				int nr_coarse_weights = (sp.iclass_max-sp.iclass_min+1)*sp.nr_particles * sp.nr_dir * sp.nr_psi * sp.nr_trans;
				PassWeights[ipart].weights.setDevPtr(&Mweight(ipart*nr_coarse_weights));
				PassWeights[ipart].weights.setHstPtr(&Mweight[ipart*nr_coarse_weights]);
				PassWeights[ipart].weights.setSize(nr_coarse_weights);
			}
			PassWeights[ipart].weights.h_do_free=false;

			std::pair<int, XFLOAT> min_pair=getArgMinOnDevice(PassWeights[ipart].weights);
			PassWeights[ipart].weights.cp_to_host();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

			//Set all device-located weights to zero, and only the smallest one to 1.
			DEBUG_HANDLE_ERROR(cudaMemsetAsync(~(PassWeights[ipart].weights), 0.f, PassWeights[ipart].weights.getSize()*sizeof(XFLOAT),0));

			XFLOAT unity=1;
			DEBUG_HANDLE_ERROR(cudaMemcpyAsync( &(PassWeights[ipart].weights(min_pair.first) ), &unity, sizeof(XFLOAT), cudaMemcpyHostToDevice, 0));

			PassWeights[ipart].weights.cp_to_host();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));
//
//				// Binarize the squared differences array to skip marginalisation
//				RFLOAT mymindiff2 = 99.e10;
//				long int myminidx = -1;
//				// Find the smallest element in this row of op.Mweight
//				for (long int i = 0; i < XSIZE(op.Mweight); i++)
//				{
//
//					RFLOAT cc = DIRECT_A2D_ELEM(op.Mweight, ipart, i);
//					// ignore non-determined cc
//					if (cc == -999.)
//						continue;
//
//					// just search for the maximum
//					if (cc < mymindiff2)
//					{
//						mymindiff2 = cc;
//						myminidx = i;
//					}
//				}
//				// Set all except for the best hidden variable to zero and the smallest element to 1
//				for (long int i = 0; i < XSIZE(op.Mweight); i++)
//					DIRECT_A2D_ELEM(op.Mweight, ipart, i)= 0.;
//
//				DIRECT_A2D_ELEM(op.Mweight, ipart, myminidx)= 1.;

			op.sum_weight[ipart] += 1.;

		}
		else
		{
			long int sumRedSize=0;
			for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
				sumRedSize+= (exp_ipass==0) ? ceilf((float)(sp.nr_dir*sp.nr_psi)/(float)SUMW_BLOCK_SIZE) : ceil((float)FPCMasks[ipart][exp_iclass].jobNum / (float)SUMW_BLOCK_SIZE);

			// loop through making translational priors for all classes this ipart - then copy all at once - then loop through kernel calls ( TODO: group kernel calls into one big kernel)
			CUDA_CPU_TIC("get_offset_priors");

			RFLOAT pdf_offset_mean(0);
			unsigned pdf_offset_count(0);

			for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
			{
				/*=========================================
						Fetch+generate Translation data
				===========================================*/
				RFLOAT myprior_x, myprior_y, myprior_z;
				if (baseMLO->mymodel.ref_dim == 2)
				{
					myprior_x = XX(baseMLO->mymodel.prior_offset_class[exp_iclass]);
					myprior_y = YY(baseMLO->mymodel.prior_offset_class[exp_iclass]);
				}
				else
				{
					myprior_x = XX(op.prior[ipart]);
					myprior_y = YY(op.prior[ipart]);
					if (baseMLO->mymodel.data_dim == 3)
						myprior_z = ZZ(op.prior[ipart]);
				}

				for (long int itrans = sp.itrans_min; itrans <= sp.itrans_max; itrans++)
				{
					RFLOAT pdf(0);
					RFLOAT offset_x = old_offset_x - myprior_x + baseMLO->sampling.translations_x[itrans];
					RFLOAT offset_y = old_offset_y - myprior_y + baseMLO->sampling.translations_y[itrans];
					RFLOAT tdiff2 = offset_x * offset_x + offset_y * offset_y;

					if (baseMLO->mymodel.data_dim == 3)
					{
						RFLOAT offset_z = old_offset_z - myprior_z + baseMLO->sampling.translations_z[itrans];
						tdiff2 += offset_z * offset_z;
					}

					// P(offset|sigma2_offset)
					// This is the probability of the offset, given the model offset and variance.
					if (baseMLO->mymodel.sigma2_offset < 0.0001)
						pdf = ( tdiff2 > 0.) ? 0. : 1.;
					else
						pdf = exp ( tdiff2 / (-2. * baseMLO->mymodel.sigma2_offset) ) / ( 2. * PI * baseMLO->mymodel.sigma2_offset );

					pdf_offset[(exp_iclass-sp.iclass_min)*sp.nr_trans + itrans] = pdf;
					pdf_offset_mean += pdf;
					pdf_offset_count ++;
				}
			}

			pdf_offset_mean /= (RFLOAT) pdf_offset_count;

			//If mean is non-zero bring all values closer to 1 to improve numerical accuracy
			//This factor is over all classes and is thus removed in the final normalization
			if (pdf_offset_mean != 0.)
				for (int i = 0; i < pdf_offset.getSize(); i ++)
					pdf_offset[i] /= pdf_offset_mean;

			pdf_offset.cp_to_device();
			CUDA_CPU_TOC("get_offset_priors");

			for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++) // TODO could use classStreams
			{
				CUDA_CPU_TIC("sumweight1");
				long int block_num;

				if(exp_ipass==0)  //use Mweight for now - FIXME use PassWeights.weights (ignore indexArrays)
				{
					CudaGlobalPtr<XFLOAT>  classMweight(
							Mweight,
							ipart * op.Mweight.xdim + exp_iclass * sp.nr_dir * sp.nr_psi * sp.nr_trans,
							sp.nr_dir * sp.nr_psi * sp.nr_trans);

					CudaGlobalPtr<XFLOAT>  pdf_orientation_class(&(pdf_orientation[(exp_iclass-sp.iclass_min)*sp.nr_dir*sp.nr_psi]), &( pdf_orientation((exp_iclass-sp.iclass_min)*sp.nr_dir*sp.nr_psi) ), sp.nr_dir*sp.nr_psi);
					CudaGlobalPtr<XFLOAT>  pdf_offset_class(&(pdf_offset[(exp_iclass-sp.iclass_min)*sp.nr_trans]), &( pdf_offset((exp_iclass-sp.iclass_min)*sp.nr_trans) ), sp.nr_trans);

					block_num = ceilf((float)(sp.nr_dir*sp.nr_psi)/(float)SUMW_BLOCK_SIZE);
					dim3 block_dim(block_num);

//					CUDA_GPU_TIC("cuda_kernel_sumweight");

					cuda_kernel_exponentiate_weights_coarse<<<block_dim,SUMW_BLOCK_SIZE,0,cudaMLO->classStreams[exp_iclass]>>>(
							~pdf_orientation_class,
							~pdf_offset_class,
							~classMweight,
							(XFLOAT)op.min_diff2[ipart],
							sp.nr_dir*sp.nr_psi,
							sp.nr_trans);

//					CUDA_GPU_TAC("cuda_kernel_sumweight");
				}
				else if ((baseMLO->mymodel.pdf_class[exp_iclass] > 0.) && (FPCMasks[ipart][exp_iclass].weightNum > 0) )
				{
					// Use the constructed mask to build a partial (class-specific) input
					// (until now, PassWeights has been an empty placeholder. We now create class-paritals pointing at it, and start to fill it with stuff)
					IndexedDataArray thisClassPassWeights(PassWeights[ipart],FPCMasks[ipart][exp_iclass], cudaMLO->devBundle->allocator);
					CudaGlobalPtr<XFLOAT>  pdf_orientation_class(&(pdf_orientation[(exp_iclass-sp.iclass_min)*sp.nr_dir*sp.nr_psi]), &( pdf_orientation((exp_iclass-sp.iclass_min)*sp.nr_dir*sp.nr_psi) ), sp.nr_dir*sp.nr_psi);
					CudaGlobalPtr<XFLOAT>  pdf_offset_class(&(pdf_offset[(exp_iclass-sp.iclass_min)*sp.nr_trans]), &( pdf_offset((exp_iclass-sp.iclass_min)*sp.nr_trans) ), sp.nr_trans);

					block_num = ceil((float)FPCMasks[ipart][exp_iclass].jobNum / (float)SUMW_BLOCK_SIZE); //thisClassPassWeights.rot_idx.getSize() / SUM_BLOCK_SIZE;
					dim3 block_dim(block_num);

//					CUDA_GPU_TIC("cuda_kernel_sumweight");
					cuda_kernel_exponentiate_weights_fine<<<block_dim,SUMW_BLOCK_SIZE,0,cudaMLO->classStreams[exp_iclass]>>>(
							~pdf_orientation_class,
							~pdf_offset_class,
							~thisClassPassWeights.weights,
							(XFLOAT)op.min_diff2[ipart],
							sp.nr_oversampled_rot,
							sp.nr_oversampled_trans,
							~thisClassPassWeights.rot_id,
							~thisClassPassWeights.trans_idx,
							~FPCMasks[ipart][exp_iclass].jobOrigin,
							~FPCMasks[ipart][exp_iclass].jobExtent,
							FPCMasks[ipart][exp_iclass].jobNum);
//					CUDA_GPU_TAC("cuda_kernel_sumweight");
				}
				CUDA_CPU_TOC("sumweight1");
			} // end loop exp_iclass

			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

			if(exp_ipass!=0)
				PassWeights[ipart].weights.cp_to_host(); // note that the host-pointer is shared: we're copying to Mweight.
		}
	} // end loop ipart

	if (exp_ipass==0)
		op.Mcoarse_significant.resizeNoCp(1,1,sp.nr_particles, XSIZE(op.Mweight));

	CUDA_CPU_TIC("convertPostKernel");
	// Now, for each particle,  find the exp_significant_weight that encompasses adaptive_fraction of op.sum_weight

	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];

		XFLOAT my_significant_weight;

		if ((baseMLO->iter == 1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
		{
			Mweight.cp_to_host();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(Mweight.getStream()));

			my_significant_weight = 0.999;
			DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NR_SIGN) = (RFLOAT) 1.;
			if (exp_ipass==0) // TODO better memset, 0 => false , 1 => true
				for (int ihidden = 0; ihidden < XSIZE(op.Mcoarse_significant); ihidden++)
					if (DIRECT_A2D_ELEM(op.Mweight, ipart, ihidden) >= my_significant_weight)
						DIRECT_A2D_ELEM(op.Mcoarse_significant, ipart, ihidden) = true;
					else
						DIRECT_A2D_ELEM(op.Mcoarse_significant, ipart, ihidden) = false;
		}
		else if (exp_ipass!=0)
		{
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));
			size_t weightSize = PassWeights[ipart].weights.getSize();

			CudaGlobalPtr<XFLOAT> sorted(weightSize, cudaMLO->devBundle->allocator);
			CudaGlobalPtr<XFLOAT> cumulative_sum(weightSize, cudaMLO->devBundle->allocator);

			CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_FINE");

			sorted.device_alloc();
			cumulative_sum.device_alloc();

			sortOnDevice(PassWeights[ipart].weights, sorted);
			scanOnDevice(sorted, cumulative_sum);

			op.sum_weight[ipart] = cumulative_sum.getDeviceAt(cumulative_sum.getSize() - 1);

			CUDA_CPU_TOC("sort");

			size_t thresholdIdx = findThresholdIdxInCumulativeSum(cumulative_sum, (1 - baseMLO->adaptive_fraction) * op.sum_weight[ipart]);
			my_significant_weight = sorted.getDeviceAt(thresholdIdx);
		}
		else
		{
			CUDA_CPU_TIC("sort");
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

			//Wrap the current ipart data in a new pointer
			CudaGlobalPtr<XFLOAT> unsorted_ipart(Mweight,
					ipart * op.Mweight.xdim + sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.iclass_min,
					(sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi * sp.nr_trans);

			CudaGlobalPtr<XFLOAT> filtered(unsorted_ipart.getSize(), cudaMLO->devBundle->allocator);

			CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_SORTSUM");

			filtered.device_alloc();

			MoreThanCubOpt<XFLOAT> moreThanOpt(0.);
			size_t filteredSize = filterOnDevice(unsorted_ipart, filtered, moreThanOpt);
			if (filteredSize == 0)
			{
				std::cerr << std::endl;
				std::cerr << " exp_fn_img= " << baseMLO->exp_fn_img << std::endl;
				std::cerr << " ipart= " << ipart << " adaptive_fraction= " << baseMLO->adaptive_fraction << std::endl;
				std::cerr << " threshold= " << (1 - baseMLO->adaptive_fraction) * op.sum_weight[ipart]  << std::endl;
				std::cerr << " my_significant_weight= " << my_significant_weight << std::endl;
				std::cerr << " op.sum_weight[ipart]= " << op.sum_weight[ipart] << std::endl;

				pdf_orientation.dump_device_to_file("error_dump_pdf_orientation");
				pdf_offset.dump_device_to_file("error_dump_pdf_offset");
				unsorted_ipart.dump_device_to_file("error_dump_filtered");

				std::cerr << "Dumped data: error_dump_pdf_orientation, error_dump_pdf_orientation and error_dump_unsorted." << std::endl;

				REPORT_ERROR("filteredSize == 0");
			}
			filtered.setSize(filteredSize);

			CudaGlobalPtr<XFLOAT> sorted(filteredSize, cudaMLO->devBundle->allocator);
			CudaGlobalPtr<XFLOAT> cumulative_sum(filteredSize, cudaMLO->devBundle->allocator);
			sorted.device_alloc();
			cumulative_sum.device_alloc();

			sortOnDevice(filtered, sorted);
			scanOnDevice(sorted, cumulative_sum);

			op.sum_weight[ipart] = cumulative_sum.getDeviceAt(cumulative_sum.getSize() - 1);

			CUDA_CPU_TOC("sort");

			size_t thresholdIdx = findThresholdIdxInCumulativeSum(cumulative_sum, (1 - baseMLO->adaptive_fraction) * op.sum_weight[ipart]);
			my_significant_weight = sorted.getDeviceAt(thresholdIdx);
			long int my_nr_significant_coarse_samples = filteredSize - thresholdIdx;

			if (my_nr_significant_coarse_samples == 0)
			{
				std::cerr << std::endl;
				std::cerr << " exp_fn_img= " << baseMLO->exp_fn_img << std::endl;
				std::cerr << " ipart= " << ipart << " adaptive_fraction= " << baseMLO->adaptive_fraction << std::endl;
				std::cerr << " threshold= " << (1 - baseMLO->adaptive_fraction) * op.sum_weight[ipart] << " thresholdIdx= " << thresholdIdx << std::endl;
				std::cerr << " my_significant_weight= " << my_significant_weight << std::endl;
				std::cerr << " op.sum_weight[ipart]= " << op.sum_weight[ipart] << std::endl;

				unsorted_ipart.dump_device_to_file("error_dump_unsorted");
				filtered.dump_device_to_file("error_dump_filtered");
				sorted.dump_device_to_file("error_dump_sorted");
				cumulative_sum.dump_device_to_file("error_dump_cumulative_sum");

				std::cerr << "Written error_dump_unsorted, error_dump_filtered, error_dump_sorted, and error_dump_cumulative_sum." << std::endl;

				REPORT_ERROR("my_nr_significant_coarse_samples == 0");
			}

			// Store nr_significant_coarse_samples for this particle
			DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NR_SIGN) = (RFLOAT) my_nr_significant_coarse_samples;

			CudaGlobalPtr<bool> Mcoarse_significant(
					&op.Mcoarse_significant.data[ipart * op.Mweight.xdim + sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.iclass_min],
					(sp.iclass_max-sp.iclass_min+1) * sp.nr_dir * sp.nr_psi * sp.nr_trans,
					cudaMLO->devBundle->allocator);

			CUSTOM_ALLOCATOR_REGION_NAME("CASDTW_SIG");
			Mcoarse_significant.device_alloc();

			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));
			arrayOverThreshold<XFLOAT>(unsorted_ipart, Mcoarse_significant, (XFLOAT) my_significant_weight);
			Mcoarse_significant.cp_to_host();
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		}

		op.significant_weight.clear();
		op.significant_weight.resize(sp.nr_particles, 0.);
		op.significant_weight[ipart] = (RFLOAT) my_significant_weight;
		//std::cerr << "@sort op.significant_weight[ipart]= " << (XFLOAT)op.significant_weight[ipart] << std::endl;

	} // end loop ipart

	CUDA_CPU_TOC("convertPostKernel");
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
	{
		if (exp_ipass == 0) baseMLO->timer.toc(baseMLO->TIMING_ESP_WEIGHT1);
		else baseMLO->timer.toc(baseMLO->TIMING_ESP_WEIGHT2);
	}
#endif
}

void storeWeightedSums(OptimisationParamters &op, SamplingParameters &sp,
						MlOptimiser *baseMLO,
						MlOptimiserCuda *cudaMLO,
						std::vector<IndexedDataArray> &FinePassWeights,
						std::vector<ProjectionParams> &ProjectionData,
						std::vector<std::vector<IndexedDataArrayMask> > &FPCMasks,
	 	 	 	 	 	std::vector<cudaStager<unsigned long> > &stagerSWS)
{
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_WSUM);
#endif
	CUDA_CPU_TIC("store_init");

	// Re-do below because now also want unmasked images AND if (stricht_highres_exp >0.) then may need to resize
	baseMLO->precalculateShiftedImagesCtfsAndInvSigma2s(true, op.my_ori_particle, sp.current_image_size, sp.current_oversampling, op.metadata_offset, // inserted SHWS 12112015
			sp.itrans_min, sp.itrans_max, op.Fimgs, op.Fimgs_nomask, op.Fctfs, op.local_Fimgs_shifted, op.local_Fimgs_shifted_nomask,
			op.local_Fctfs, op.local_sqrtXi2, op.local_Minvsigma2s);

	// In doThreadPrecalculateShiftedImagesCtfsAndInvSigma2s() the origin of the op.local_Minvsigma2s was omitted.
	// Set those back here
	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];
		int group_id = baseMLO->mydata.getGroupId(part_id);
		DIRECT_MULTIDIM_ELEM(op.local_Minvsigma2s[ipart], 0) = 1. / (baseMLO->sigma2_fudge * DIRECT_A1D_ELEM(baseMLO->mymodel.sigma2_noise[group_id], 0));
	}

	// Initialise the maximum of all weights to a negative value
	op.max_weight.clear();
	op.max_weight.resize(sp.nr_particles, -1.);

	// For norm_correction and scale_correction of all particles of this ori_particle
	std::vector<RFLOAT> exp_wsum_norm_correction;
	std::vector<MultidimArray<RFLOAT> > exp_wsum_scale_correction_XA, exp_wsum_scale_correction_AA;
	std::vector<MultidimArray<RFLOAT> > thr_wsum_signal_product_spectra, thr_wsum_reference_power_spectra;
	exp_wsum_norm_correction.resize(sp.nr_particles, 0.);

	// For scale_correction
	if (baseMLO->do_scale_correction)
	{
		MultidimArray<RFLOAT> aux;
		aux.initZeros(baseMLO->mymodel.ori_size/2 + 1);
		exp_wsum_scale_correction_XA.resize(sp.nr_particles, aux);
		exp_wsum_scale_correction_AA.resize(sp.nr_particles, aux);
		thr_wsum_signal_product_spectra.resize(baseMLO->mymodel.nr_groups, aux);
		thr_wsum_reference_power_spectra.resize(baseMLO->mymodel.nr_groups, aux);
	}

	std::vector<RFLOAT> oversampled_translations_x, oversampled_translations_y, oversampled_translations_z;
	bool have_warned_small_scale = false;

	// Make local copies of weighted sums (except BPrefs, which are too big)
	// so that there are not too many mutex locks below
	std::vector<MultidimArray<RFLOAT> > thr_wsum_sigma2_noise, thr_wsum_pdf_direction;
	std::vector<RFLOAT> thr_wsum_norm_correction, thr_sumw_group, thr_wsum_pdf_class, thr_wsum_prior_offsetx_class, thr_wsum_prior_offsety_class;
	RFLOAT thr_wsum_sigma2_offset;
	MultidimArray<RFLOAT> thr_metadata, zeroArray;
	// Wsum_sigma_noise2 is a 1D-spectrum for each group
	zeroArray.initZeros(baseMLO->mymodel.ori_size/2 + 1);
	thr_wsum_sigma2_noise.resize(baseMLO->mymodel.nr_groups, zeroArray);
	// wsum_pdf_direction is a 1D-array (of length sampling.NrDirections()) for each class
	zeroArray.initZeros(baseMLO->sampling.NrDirections());
	thr_wsum_pdf_direction.resize(baseMLO->mymodel.nr_classes, zeroArray);
	// sumw_group is a RFLOAT for each group
	thr_sumw_group.resize(baseMLO->mymodel.nr_groups, 0.);
	// wsum_pdf_class is a RFLOAT for each class
	thr_wsum_pdf_class.resize(baseMLO->mymodel.nr_classes, 0.);
	if (baseMLO->mymodel.ref_dim == 2)
	{
		thr_wsum_prior_offsetx_class.resize(baseMLO->mymodel.nr_classes, 0.);
		thr_wsum_prior_offsety_class.resize(baseMLO->mymodel.nr_classes, 0.);
	}
	// wsum_sigma2_offset is just a RFLOAT
	thr_wsum_sigma2_offset = 0.;
	unsigned image_size = op.Fimgs[0].nzyxdim;

	CUDA_CPU_TOC("store_init");

	/*=======================================================================================
	                           COLLECT 2 AND SET METADATA
	=======================================================================================*/

	CUDA_CPU_TIC("collect_data_2");
	int nr_transes = sp.nr_trans*sp.nr_oversampled_trans;
	int nr_fake_classes = (sp.iclass_max-sp.iclass_min+1);
	int oversamples = sp.nr_oversampled_trans * sp.nr_oversampled_rot;
	std::vector<long int> block_nums(sp.nr_particles*nr_fake_classes);

	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		// Allocate space for all classes, so that we can pre-calculate data for all classes, copy in one operation, call kenrels on all classes, and copy back in one operation
		CudaGlobalPtr<XFLOAT>          oo_otrans_x(nr_fake_classes*nr_transes, cudaMLO->devBundle->allocator); // old_offset_oversampled_trans_x
		CudaGlobalPtr<XFLOAT>          oo_otrans_y(nr_fake_classes*nr_transes, cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> myp_oo_otrans_x2y2z2(nr_fake_classes*nr_transes, cudaMLO->devBundle->allocator); // my_prior_old_offs....x^2*y^2*z^2
		oo_otrans_x.device_alloc();
		oo_otrans_y.device_alloc();
		myp_oo_otrans_x2y2z2.device_alloc();

		int sumBlockNum =0;
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];
		int group_id = baseMLO->mydata.getGroupId(part_id);
		CUDA_CPU_TIC("collect_data_2_pre_kernel");
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			int fake_class = exp_iclass-sp.iclass_min; // if we only have the third class to do, the third class will be the "first" we do, i.e. the "fake" first.
			if ((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[ipart].class_entries[exp_iclass] == 0) )
				continue;

			// Use the constructed mask to construct a partial class-specific input
			IndexedDataArray thisClassFinePassWeights(FinePassWeights[ipart],FPCMasks[ipart][exp_iclass], cudaMLO->devBundle->allocator);

			// Re-define the job-partition of the indexedArray of weights so that the collect-kernel can work with it.
			block_nums[nr_fake_classes*ipart + fake_class] = makeJobsForCollect(thisClassFinePassWeights, FPCMasks[ipart][exp_iclass], ProjectionData[ipart].orientation_num[exp_iclass]);

			stagerSWS[ipart].stage(FPCMasks[ipart][exp_iclass].jobOrigin);
			stagerSWS[ipart].stage(FPCMasks[ipart][exp_iclass].jobExtent);

			sumBlockNum+=block_nums[nr_fake_classes*ipart + fake_class];

			RFLOAT myprior_x, myprior_y, myprior_z;
			RFLOAT old_offset_x = XX(op.old_offset[ipart]);
			RFLOAT old_offset_y = YY(op.old_offset[ipart]);
			RFLOAT old_offset_z;

			if (baseMLO->mymodel.ref_dim == 2)
			{
				myprior_x = XX(baseMLO->mymodel.prior_offset_class[exp_iclass]);
				myprior_y = YY(baseMLO->mymodel.prior_offset_class[exp_iclass]);
			}
			else
			{
				myprior_x = XX(op.prior[ipart]);
				myprior_y = YY(op.prior[ipart]);
				if (baseMLO->mymodel.data_dim == 3)
				{
					myprior_z = ZZ(op.prior[ipart]);
					old_offset_z = ZZ(op.old_offset[ipart]);
				}
			}

			/*======================================================
								COLLECT 2
			======================================================*/

			//Pregenerate oversampled translation objects for kernel-call
			for (long int itrans = 0, iitrans = 0; itrans < sp.nr_trans; itrans++)
			{
				baseMLO->sampling.getTranslations(itrans, baseMLO->adaptive_oversampling,
						oversampled_translations_x, oversampled_translations_y, oversampled_translations_z);
				for (long int iover_trans = 0; iover_trans < sp.nr_oversampled_trans; iover_trans++, iitrans++)
				{
					oo_otrans_x[fake_class*nr_transes+iitrans] = old_offset_x + oversampled_translations_x[iover_trans];
					oo_otrans_y[fake_class*nr_transes+iitrans] = old_offset_y + oversampled_translations_y[iover_trans];
					RFLOAT diffx = myprior_x - oo_otrans_x[fake_class*nr_transes+iitrans];
					RFLOAT diffy = myprior_y - oo_otrans_y[fake_class*nr_transes+iitrans];
					if (baseMLO->mymodel.data_dim == 3)
					{
						RFLOAT diffz = myprior_z - (old_offset_z + oversampled_translations_z[iover_trans]);
						myp_oo_otrans_x2y2z2[fake_class*nr_transes+iitrans] = diffx*diffx + diffy*diffy + diffz*diffz ;
					}
					else
					{
						myp_oo_otrans_x2y2z2[fake_class*nr_transes+iitrans] = diffx*diffx + diffy*diffy ;
					}
				}
			}
		}

		stagerSWS[ipart].cp_to_device();
		oo_otrans_x.cp_to_device();
		oo_otrans_y.cp_to_device();
		myp_oo_otrans_x2y2z2.cp_to_device();
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		CudaGlobalPtr<XFLOAT>                      p_weights(sumBlockNum, cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> p_thr_wsum_prior_offsetx_class(sumBlockNum, cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> p_thr_wsum_prior_offsety_class(sumBlockNum, cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT>       p_thr_wsum_sigma2_offset(sumBlockNum, cudaMLO->devBundle->allocator);
		p_weights.device_alloc();
		p_thr_wsum_prior_offsetx_class.device_alloc();
		p_thr_wsum_prior_offsety_class.device_alloc();
		p_thr_wsum_sigma2_offset.device_alloc();
		CUDA_CPU_TOC("collect_data_2_pre_kernel");
		int partial_pos=0;
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			int fake_class = exp_iclass-sp.iclass_min; // if we only have the third class to do, the third class will be the "first" we do, i.e. the "fake" first.
			if ((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[ipart].class_entries[exp_iclass] == 0) )
				continue;

			// Use the constructed mask to construct a partial class-specific input
			IndexedDataArray thisClassFinePassWeights(FinePassWeights[ipart],FPCMasks[ipart][exp_iclass], cudaMLO->devBundle->allocator);

			int cpos=fake_class*nr_transes;
			int block_num = block_nums[nr_fake_classes*ipart + fake_class];
			dim3 grid_dim_collect2 = block_num;
			cuda_kernel_collect2jobs<<<grid_dim_collect2,SUMW_BLOCK_SIZE>>>(
						&(oo_otrans_x(cpos) ),          // otrans-size -> make const
						&(oo_otrans_y(cpos) ),          // otrans-size -> make const
						&(myp_oo_otrans_x2y2z2(cpos) ), // otrans-size -> make const
						~thisClassFinePassWeights.weights,
					(XFLOAT)op.significant_weight[ipart],
					(XFLOAT)op.sum_weight[ipart],
					sp.nr_trans,
					sp.nr_oversampled_trans,
					sp.nr_oversampled_rot,
					oversamples,
					(baseMLO->do_skip_align || baseMLO->do_skip_rotate ),
						&p_weights(partial_pos),
						&p_thr_wsum_prior_offsetx_class(partial_pos),
						&p_thr_wsum_prior_offsety_class(partial_pos),
						&p_thr_wsum_sigma2_offset(partial_pos),
					~thisClassFinePassWeights.rot_idx,
					~thisClassFinePassWeights.trans_idx,
					~FPCMasks[ipart][exp_iclass].jobOrigin,
					~FPCMasks[ipart][exp_iclass].jobExtent
						);
			partial_pos+=block_num;
		}
		CUDA_CPU_TIC("collect_data_2_post_kernel");
		p_weights.cp_to_host();
		p_thr_wsum_prior_offsetx_class.cp_to_host();
		p_thr_wsum_prior_offsety_class.cp_to_host();
		p_thr_wsum_sigma2_offset.cp_to_host();

		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));
		int iorient = 0;
		partial_pos=0;
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			int fake_class = exp_iclass-sp.iclass_min; // if we only have the third class to do, the third class will be the "first" we do, i.e. the "fake" first.
			if ((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[ipart].class_entries[exp_iclass] == 0) )
				continue;
			int block_num = block_nums[nr_fake_classes*ipart + fake_class];

			for (long int n = partial_pos; n < partial_pos+block_num; n++)
			{
				iorient= FinePassWeights[ipart].rot_id[FPCMasks[ipart][exp_iclass].jobOrigin[n-partial_pos]];

				long int mydir, idir=floor(iorient/sp.nr_psi);
				if (baseMLO->mymodel.orientational_prior_mode == NOPRIOR)
					mydir = idir;
				else
					mydir = op.pointer_dir_nonzeroprior[idir];

				// store partials according to indices of the relevant dimension
				DIRECT_MULTIDIM_ELEM(thr_wsum_pdf_direction[exp_iclass], mydir) += p_weights[n];
				thr_sumw_group[group_id]                 						+= p_weights[n];
				thr_wsum_pdf_class[exp_iclass]           						+= p_weights[n];
				thr_wsum_sigma2_offset                   						+= p_thr_wsum_sigma2_offset[n];

				if (baseMLO->mymodel.ref_dim == 2)
				{
					thr_wsum_prior_offsetx_class[exp_iclass] += p_thr_wsum_prior_offsetx_class[n];
					thr_wsum_prior_offsety_class[exp_iclass] += p_thr_wsum_prior_offsety_class[n];
				}
			}
			partial_pos+=block_num;
		} // end loop iclass
		CUDA_CPU_TOC("collect_data_2_post_kernel");
	} // end loop ipart

	/*======================================================
	                     SET METADATA
	======================================================*/

	std::vector< RFLOAT> oversampled_rot, oversampled_tilt, oversampled_psi;
	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		CUDA_CPU_TIC("setMetadata");

		CUDA_CPU_TIC("getArgMaxOnDevice");
		std::pair<int, XFLOAT> max_pair = getArgMaxOnDevice(FinePassWeights[ipart].weights);
		CUDA_CPU_TOC("getArgMaxOnDevice");

		Indices max_index;
		max_index.fineIdx = FinePassWeights[ipart].ihidden_overs[max_pair.first];
		op.max_weight[ipart] = max_pair.second;

		//std::cerr << "max val = " << op.max_weight[ipart] << std::endl;
		//std::cerr << "max index = " << max_index.fineIdx << std::endl;
		max_index.fineIndexToFineIndices(sp); // set partial indices corresponding to the found max_index, to be used below

		baseMLO->sampling.getTranslations(max_index.itrans, baseMLO->adaptive_oversampling,
				oversampled_translations_x, oversampled_translations_y, oversampled_translations_z);

		//TODO We already have rot, tilt and psi don't calculated them again
		baseMLO->sampling.getOrientations(max_index.idir, max_index.ipsi, baseMLO->adaptive_oversampling, oversampled_rot, oversampled_tilt, oversampled_psi,
				op.pointer_dir_nonzeroprior, op.directions_prior, op.pointer_psi_nonzeroprior, op.psi_prior);

		RFLOAT rot = oversampled_rot[max_index.ioverrot];
		RFLOAT tilt = oversampled_tilt[max_index.ioverrot];
		RFLOAT psi = oversampled_psi[max_index.ioverrot];
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_ROT) = rot;
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_TILT) = tilt;
		if (psi>180.)
			psi-=360.;
		else if ( psi<-180.)
			psi+=360.;
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_PSI) = psi;
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_XOFF) = XX(op.old_offset[ipart]) + oversampled_translations_x[max_index.iovertrans];
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_YOFF) = YY(op.old_offset[ipart]) + oversampled_translations_y[max_index.iovertrans];

		if (baseMLO->mymodel.data_dim == 3)
			DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_ZOFF) = ZZ(op.old_offset[ipart]) + oversampled_translations_z[max_index.iovertrans];
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_CLASS) = (RFLOAT)max_index.iclass + 1;
			DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_PMAX) = op.max_weight[ipart]/op.sum_weight[ipart];

		CUDA_CPU_TOC("setMetadata");
	}

	CUDA_CPU_TOC("collect_data_2");





	/*=======================================================================================
	                                   MAXIMIZATION
	=======================================================================================*/

	CUDA_CPU_TIC("maximization");

	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];
		int group_id = baseMLO->mydata.getGroupId(part_id);

		/*======================================================
		                     TRANSLATIONS
		======================================================*/

		CUSTOM_ALLOCATOR_REGION_NAME("TRANS_3");

		CUDA_CPU_TIC("translation_3");

		long unsigned translation_num((sp.itrans_max - sp.itrans_min + 1) * sp.nr_oversampled_trans);

		CudaGlobalPtr<XFLOAT> Fimgs_real(cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> Fimgs_imag(cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> Fimgs_nomask_real(cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> Fimgs_nomask_imag(cudaMLO->devBundle->allocator);

		Fimgs_real.device_alloc(image_size * translation_num);
		Fimgs_imag.device_alloc(image_size * translation_num);
		Fimgs_nomask_real.device_alloc(image_size * translation_num);
		Fimgs_nomask_imag.device_alloc(image_size * translation_num);

		if (baseMLO->do_shifts_onthefly)
		{
			CudaTranslator::Plan planMask(
					op.local_Fimgs_shifted[ipart].data,
					image_size,
					sp.itrans_min * sp.nr_oversampled_trans,
					( sp.itrans_max + 1) * sp.nr_oversampled_trans,
					cudaMLO->devBundle->allocator,
					0 //stream
					);

			CudaTranslator::Plan planNomask(
					op.local_Fimgs_shifted_nomask[ipart].data,
					image_size,
					sp.itrans_min * sp.nr_oversampled_trans,
					( sp.itrans_max + 1) * sp.nr_oversampled_trans,
					cudaMLO->devBundle->allocator,
					0 //stream
					);

			if (baseMLO->adaptive_oversampling == 0)
			{
				cudaMLO->translator_current1.translate(planMask,   ~Fimgs_real,        ~Fimgs_imag);
				cudaMLO->translator_current1.translate(planNomask, ~Fimgs_nomask_real, ~Fimgs_nomask_imag);
			}
			else
			{
				cudaMLO->translator_current2.translate(planMask,   ~Fimgs_real,        ~Fimgs_imag);
				cudaMLO->translator_current2.translate(planNomask, ~Fimgs_nomask_real, ~Fimgs_nomask_imag);
			}
		}
		else
		{
			Fimgs_real.host_alloc();
			Fimgs_imag.host_alloc();
			Fimgs_nomask_real.host_alloc();
			Fimgs_nomask_imag.host_alloc();

			unsigned long k = 0;
			for (unsigned i = 0; i < op.local_Fimgs_shifted.size(); i ++)
			{
				for (unsigned j = 0; j < op.local_Fimgs_shifted[i].nzyxdim; j ++)
				{
					Fimgs_real[k] = op.local_Fimgs_shifted[i].data[j].real;
					Fimgs_imag[k] = op.local_Fimgs_shifted[i].data[j].imag;
					Fimgs_nomask_real[k] = op.local_Fimgs_shifted_nomask[i].data[j].real;
					Fimgs_nomask_imag[k] = op.local_Fimgs_shifted_nomask[i].data[j].imag;
					k++;
				}
			}

			Fimgs_real.cp_to_device();
			Fimgs_imag.cp_to_device();
			Fimgs_nomask_real.cp_to_device();
			Fimgs_nomask_imag.cp_to_device();
		}

		CUDA_CPU_TOC("translation_3");


		/*======================================================
		                       SCALE
		======================================================*/

		XFLOAT part_scale(1.);

		if (baseMLO->do_scale_correction)
		{
			part_scale = baseMLO->mymodel.scale_correction[group_id];
			if (part_scale > 10000.)
			{
				std::cerr << " rlnMicrographScaleCorrection= " << part_scale << " group= " << group_id + 1 << std::endl;
				REPORT_ERROR("ERROR: rlnMicrographScaleCorrection is very high. Did you normalize your data?");
			}
			else if (part_scale < 0.001)
			{
				if (!have_warned_small_scale)
				{
					std::cout << " WARNING: ignoring group " << group_id + 1 << " with very small or negative scale (" << part_scale <<
							"); Use larger groups for more stable scale estimates." << std::endl;
					have_warned_small_scale = true;
				}
				part_scale = 0.001;
			}
		}

		CudaGlobalPtr<XFLOAT> ctfs(image_size, cudaMLO->devBundle->allocator); //TODO Same size for all iparts, should be allocated once

		if (baseMLO->do_ctf_correction)
		{
			for (unsigned i = 0; i < image_size; i++)
				ctfs[i] = (XFLOAT) op.local_Fctfs[ipart].data[i] * part_scale;
		}
		else //TODO should be handled by memset
			for (unsigned i = 0; i < image_size; i++)
				ctfs[i] = part_scale;

		ctfs.put_on_device();

		/*======================================================
		                       MINVSIGMA
		======================================================*/

		CudaGlobalPtr<XFLOAT> Minvsigma2s(image_size, cudaMLO->devBundle->allocator); //TODO Same size for all iparts, should be allocated once

		if (baseMLO->do_map)
			for (unsigned i = 0; i < image_size; i++)
				Minvsigma2s[i] = op.local_Minvsigma2s[ipart].data[i];
		else
			for (unsigned i = 0; i < image_size; i++)
				Minvsigma2s[i] = 1;

		Minvsigma2s.put_on_device();

		/*======================================================
		                      CLASS LOOP
		======================================================*/

		CUSTOM_ALLOCATOR_REGION_NAME("wdiff2s");

		CudaGlobalPtr<XFLOAT> wdiff2s_AA(baseMLO->mymodel.nr_classes*image_size, 0, cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> wdiff2s_XA(baseMLO->mymodel.nr_classes*image_size, 0, cudaMLO->devBundle->allocator);
		CudaGlobalPtr<XFLOAT> wdiff2s_sum(image_size, 0, cudaMLO->devBundle->allocator);

		wdiff2s_AA.device_alloc();
		wdiff2s_AA.device_init(0.f);
		wdiff2s_XA.device_alloc();
		wdiff2s_XA.device_init(0.f);

		unsigned long AAXA_pos=0;

		wdiff2s_sum.device_alloc();
		wdiff2s_sum.device_init(0.f);

		CUSTOM_ALLOCATOR_REGION_NAME("BP_data");

		// Loop from iclass_min to iclass_max to deal with seed generation in first iteration
		CudaGlobalPtr<XFLOAT> sorted_weights(ProjectionData[ipart].orientationNumAllClasses * translation_num, 0, cudaMLO->devBundle->allocator);
		std::vector<CudaGlobalPtr<XFLOAT> > eulers(baseMLO->mymodel.nr_classes, cudaMLO->devBundle->allocator);

		int classPos = 0;

		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			if((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[ipart].class_entries[exp_iclass] == 0))
				continue;

			// Use the constructed mask to construct a partial class-specific input
			IndexedDataArray thisClassFinePassWeights(FinePassWeights[ipart],FPCMasks[ipart][exp_iclass], cudaMLO->devBundle->allocator);

			CUDA_CPU_TIC("thisClassProjectionSetupCoarse");
			// use "slice" constructor with class-specific parameters to retrieve a temporary ProjectionParams with data for this class
			ProjectionParams thisClassProjectionData(	ProjectionData[ipart],
														ProjectionData[ipart].class_idx[exp_iclass],
														ProjectionData[ipart].class_idx[exp_iclass]+ProjectionData[ipart].class_entries[exp_iclass]);

			thisClassProjectionData.orientation_num[0] = ProjectionData[ipart].orientation_num[exp_iclass];
			CUDA_CPU_TOC("thisClassProjectionSetupCoarse");

			long unsigned orientation_num(thisClassProjectionData.orientation_num[0]);

			/*======================================================
								PROJECTIONS
			======================================================*/

			eulers[exp_iclass].setSize(orientation_num * 9);
			eulers[exp_iclass].setStream(cudaMLO->classStreams[exp_iclass]);
			eulers[exp_iclass].host_alloc();

			CUDA_CPU_TIC("generateEulerMatricesProjector");

			generateEulerMatrices(
					baseMLO->mymodel.PPref[exp_iclass].padding_factor,
					thisClassProjectionData,
					&eulers[exp_iclass][0],
					!IS_NOT_INV);

			eulers[exp_iclass].device_alloc();
			eulers[exp_iclass].cp_to_device();

			CUDA_CPU_TOC("generateEulerMatricesProjector");


			/*======================================================
								 MAP WEIGHTS
			======================================================*/

			CUDA_CPU_TIC("pre_wavg_map");

			for (long unsigned i = 0; i < orientation_num*translation_num; i++)
				sorted_weights[classPos+i] = -999.;

			for (long unsigned i = 0; i < thisClassFinePassWeights.weights.getSize(); i++)
				sorted_weights[classPos+(thisClassFinePassWeights.rot_idx[i]) * translation_num + thisClassFinePassWeights.trans_idx[i] ]
								= thisClassFinePassWeights.weights[i];

			classPos+=orientation_num*translation_num;
			CUDA_CPU_TOC("pre_wavg_map");
		}
		sorted_weights.put_on_device();

		// These syncs are necessary (for multiple ranks on the same GPU), and (assumed) low-cost.
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaMLO->classStreams[exp_iclass]));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		classPos = 0;
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			if((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[ipart].class_entries[exp_iclass] == 0))
			continue;
			/*======================================================
								 KERNEL CALL
			======================================================*/

			long unsigned orientation_num(ProjectionData[ipart].orientation_num[exp_iclass]);

			CudaProjectorKernel projKernel = CudaProjectorKernel::makeKernel(
					cudaMLO->devBundle->cudaProjectors[exp_iclass],
					op.local_Minvsigma2s[0].xdim,
					op.local_Minvsigma2s[0].ydim,
					op.local_Minvsigma2s[0].xdim-1);

			runWavgKernel(
					projKernel,
					~eulers[exp_iclass],
					~Fimgs_real,
					~Fimgs_imag,
					~Fimgs_nomask_real,
					~Fimgs_nomask_imag,
					&sorted_weights.d_ptr[classPos],
					~ctfs,
					~wdiff2s_sum,
					&wdiff2s_AA(AAXA_pos),
					&wdiff2s_XA(AAXA_pos),
					op,
					baseMLO,
					orientation_num,
					translation_num,
					image_size,
					ipart,
					group_id,
					exp_iclass,
					part_scale,
					cudaMLO->classStreams[exp_iclass]);

			/*======================================================
								BACKPROJECTION
			======================================================*/

#ifdef TIMING
			if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
				baseMLO->timer.tic(baseMLO->TIMING_WSUM_BACKPROJ);
#endif

			CUDA_CPU_TIC("backproject");

			cudaMLO->devBundle->cudaBackprojectors[exp_iclass].backproject(
				~Fimgs_nomask_real,
				~Fimgs_nomask_imag,
				&sorted_weights.d_ptr[classPos],
				~Minvsigma2s,
				~ctfs,
				translation_num,
				(XFLOAT) op.significant_weight[ipart],
				(XFLOAT) op.sum_weight[ipart],
				~eulers[exp_iclass],
				op.local_Minvsigma2s[0].xdim,
				op.local_Minvsigma2s[0].ydim,
				orientation_num,
				cudaMLO->classStreams[exp_iclass]);

			AAXA_pos += image_size;
			classPos += orientation_num*translation_num;

			CUDA_CPU_TOC("backproject");

#ifdef TIMING
			if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
				baseMLO->timer.toc(baseMLO->TIMING_WSUM_BACKPROJ);
#endif
		} // end loop iclass

		CUSTOM_ALLOCATOR_REGION_NAME("UNSET");

		// NOTE: We've never seen that this sync is necessary, but it is needed in principle, and
		// its absence in other parts of the code has caused issues. It is also very low-cost.
		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
			DEBUG_HANDLE_ERROR(cudaStreamSynchronize(cudaMLO->classStreams[exp_iclass]));
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		wdiff2s_AA.cp_to_host();
		wdiff2s_XA.cp_to_host();
		wdiff2s_sum.cp_to_host();
		DEBUG_HANDLE_ERROR(cudaStreamSynchronize(0));

		AAXA_pos=0;

		for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
		{
			if((baseMLO->mymodel.pdf_class[exp_iclass] == 0.) || (ProjectionData[ipart].class_entries[exp_iclass] == 0))
				continue;
			for (long int j = 0; j < image_size; j++)
			{
				int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine, j);
				if (ires > -1 && baseMLO->do_scale_correction &&
						DIRECT_A1D_ELEM(baseMLO->mymodel.data_vs_prior_class[exp_iclass], ires) > 3.)
				{
					DIRECT_A1D_ELEM(exp_wsum_scale_correction_AA[ipart], ires) += wdiff2s_AA[AAXA_pos+j];
					DIRECT_A1D_ELEM(exp_wsum_scale_correction_XA[ipart], ires) += wdiff2s_XA[AAXA_pos+j];
				}
			}
			AAXA_pos += image_size;
		} // end loop iclass
		for (long int j = 0; j < image_size; j++)
		{
			int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine, j);
			if (ires > -1)
			{
				thr_wsum_sigma2_noise[group_id].data[ires] += (RFLOAT) wdiff2s_sum[j];
				exp_wsum_norm_correction[ipart] += (RFLOAT) wdiff2s_sum[j]; //TODO could be gpu-reduced
			}
		}
	} // end loop ipart
	CUDA_CPU_TOC("maximization");


	CUDA_CPU_TIC("store_post_gpu");

	// Extend norm_correction and sigma2_noise estimation to higher resolutions for all particles
	// Also calculate dLL for each particle and store in metadata
	// loop over all particles inside this ori_particle
	RFLOAT thr_avg_norm_correction = 0.;
	RFLOAT thr_sum_dLL = 0., thr_sum_Pmax = 0.;
	for (long int ipart = 0; ipart < sp.nr_particles; ipart++)
	{
		long int part_id = baseMLO->mydata.ori_particles[op.my_ori_particle].particles_id[ipart];
		int group_id = baseMLO->mydata.getGroupId(part_id);

		// If the current images were smaller than the original size, fill the rest of wsum_model.sigma2_noise with the power_class spectrum of the images
		for (int ires = baseMLO->mymodel.current_size/2 + 1; ires < baseMLO->mymodel.ori_size/2 + 1; ires++)
		{
			DIRECT_A1D_ELEM(thr_wsum_sigma2_noise[group_id], ires) += DIRECT_A1D_ELEM(op.power_imgs[ipart], ires);
			// Also extend the weighted sum of the norm_correction
			exp_wsum_norm_correction[ipart] += DIRECT_A1D_ELEM(op.power_imgs[ipart], ires);
		}

		// Store norm_correction
		// Multiply by old value because the old norm_correction term was already applied to the image
		if (baseMLO->do_norm_correction)
		{
			RFLOAT old_norm_correction = DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NORM);
			old_norm_correction /= baseMLO->mymodel.avg_norm_correction;
			// The factor two below is because exp_wsum_norm_correctiom is similar to sigma2_noise, which is the variance for the real/imag components
			// The variance of the total image (on which one normalizes) is twice this value!
			RFLOAT normcorr = old_norm_correction * sqrt(exp_wsum_norm_correction[ipart] * 2.);
			thr_avg_norm_correction += normcorr;

			// Now set the new norm_correction in the relevant position of exp_metadata
			DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NORM) = normcorr;


			// Print warning for strange norm-correction values
			if (!(baseMLO->iter == 1 && baseMLO->do_firstiter_cc) && DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NORM) > 10.)
			{
				std::cout << " WARNING: norm_correction= "<< DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_NORM)
						<< " for particle " << part_id << " in group " << group_id + 1
						<< "; Are your groups large enough? Or is the reference on the correct greyscale?" << std::endl;
			}

		}

		// Store weighted sums for scale_correction
		if (baseMLO->do_scale_correction)
		{
			// Divide XA by the old scale_correction and AA by the square of that, because was incorporated into Fctf
			exp_wsum_scale_correction_XA[ipart] /= baseMLO->mymodel.scale_correction[group_id];
			exp_wsum_scale_correction_AA[ipart] /= baseMLO->mymodel.scale_correction[group_id] * baseMLO->mymodel.scale_correction[group_id];

			thr_wsum_signal_product_spectra[group_id] += exp_wsum_scale_correction_XA[ipart];
			thr_wsum_reference_power_spectra[group_id] += exp_wsum_scale_correction_AA[ipart];
		}

		// Calculate DLL for each particle
		RFLOAT logsigma2 = 0.;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(baseMLO->Mresol_fine)
		{
			int ires = DIRECT_MULTIDIM_ELEM(baseMLO->Mresol_fine, n);
			// Note there is no sqrt in the normalisation term because of the 2-dimensionality of the complex-plane
			// Also exclude origin from logsigma2, as this will not be considered in the P-calculations
			if (ires > 0)
				logsigma2 += log( 2. * PI * DIRECT_A1D_ELEM(baseMLO->mymodel.sigma2_noise[group_id], ires));
		}
		if (op.sum_weight[ipart]==0)
		{
			std::cerr << std::endl;
			std::cerr << " exp_fn_img= " << baseMLO->exp_fn_img << std::endl;
			std::cerr << " part_id= " << part_id << std::endl;
			std::cerr << " ipart= " << ipart << std::endl;
			std::cerr << " op.min_diff2[ipart]= " << op.min_diff2[ipart] << std::endl;
			std::cerr << " logsigma2= " << logsigma2 << std::endl;
			int group_id = baseMLO->mydata.getGroupId(part_id);
			std::cerr << " group_id= " << group_id << std::endl;
			std::cerr << " ml_model.scale_correction[group_id]= " << baseMLO->mymodel.scale_correction[group_id] << std::endl;
			std::cerr << " exp_significant_weight[ipart]= " << op.significant_weight[ipart] << std::endl;
			std::cerr << " exp_max_weight[ipart]= " << op.max_weight[ipart] << std::endl;
			std::cerr << " ml_model.sigma2_noise[group_id]= " << baseMLO->mymodel.sigma2_noise[group_id] << std::endl;
			REPORT_ERROR("ERROR: op.sum_weight[ipart]==0");
		}
		RFLOAT dLL;
		if ((baseMLO->iter==1 && baseMLO->do_firstiter_cc) || baseMLO->do_always_cc)
			dLL = -op.min_diff2[ipart];
		else
			dLL = log(op.sum_weight[ipart]) - op.min_diff2[ipart] - logsigma2;

		// Store dLL of each image in the output array, and keep track of total sum
		DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_DLL) = dLL;
		thr_sum_dLL += dLL;

		// Also store sum of Pmax
		thr_sum_Pmax += DIRECT_A2D_ELEM(baseMLO->exp_metadata, op.metadata_offset + ipart, METADATA_PMAX);

	}

	// Now, inside a global_mutex, update the other weighted sums among all threads
	if (!baseMLO->do_skip_maximization)
	{
		pthread_mutex_lock(&global_mutex);
		for (int n = 0; n < baseMLO->mymodel.nr_groups; n++)
		{
			baseMLO->wsum_model.sigma2_noise[n] += thr_wsum_sigma2_noise[n];
			baseMLO->wsum_model.sumw_group[n] += thr_sumw_group[n];
			if (baseMLO->do_scale_correction)
			{
				baseMLO->wsum_model.wsum_signal_product_spectra[n] += thr_wsum_signal_product_spectra[n];
				baseMLO->wsum_model.wsum_reference_power_spectra[n] += thr_wsum_reference_power_spectra[n];
			}
		}
		for (int n = 0; n < baseMLO->mymodel.nr_classes; n++)
		{
			baseMLO->wsum_model.pdf_class[n] += thr_wsum_pdf_class[n];
			if (baseMLO->mymodel.ref_dim == 2)
			{
				XX(baseMLO->wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsetx_class[n];
				YY(baseMLO->wsum_model.prior_offset_class[n]) += thr_wsum_prior_offsety_class[n];
			}

			if (!(baseMLO->do_skip_align || baseMLO->do_skip_rotate) )
				baseMLO->wsum_model.pdf_direction[n] += thr_wsum_pdf_direction[n];
		}
		baseMLO->wsum_model.sigma2_offset += thr_wsum_sigma2_offset;
		if (baseMLO->do_norm_correction)
			baseMLO->wsum_model.avg_norm_correction += thr_avg_norm_correction;
		baseMLO->wsum_model.LL += thr_sum_dLL;
		baseMLO->wsum_model.ave_Pmax += thr_sum_Pmax;
		pthread_mutex_unlock(&global_mutex);
	} // end if !do_skip_maximization

	CUDA_CPU_TOC("store_post_gpu");
#ifdef TIMING
	if (op.my_ori_particle == baseMLO->exp_my_first_ori_particle)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_WSUM);
#endif
}

MlDeviceBundle::MlDeviceBundle(MlOptimiser *baseMLOptimiser, int dev_id) :
		baseMLO(baseMLOptimiser),
		generateProjectionPlanOnTheFly(false)
{
	unsigned nr_classes = baseMLOptimiser->mymodel.nr_classes;

	/*======================================================
					DEVICE MEM OBJ SETUP
	======================================================*/

	device_id = dev_id;

	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));

	if(dev_id >= devCount)
	{
		std::cerr << " using device_id=" << dev_id << " (device no. " << dev_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		REPORT_ERROR("ERROR: Assigning a thread to a non-existent device (index likely too high)");
	}
	else
		HANDLE_ERROR(cudaSetDevice(dev_id));

	refIs3D = baseMLO->mymodel.ref_dim == 3;

	cudaProjectors.resize(nr_classes);
	cudaBackprojectors.resize(nr_classes);

	/*======================================================
	                    CUSTOM ALLOCATOR
	======================================================*/

#ifdef CUDA_NO_CUSTOM_ALLOCATION
	printf(" DEBUG: Custom allocator is disabled.\n");
	allocator = new CudaCustomAllocator(0, 1);
#else
	size_t allocationSize(0);

	size_t free, total;
	HANDLE_ERROR(cudaMemGetInfo( &free, &total ));

	if (baseMLO->available_gpu_memory > 0)
		allocationSize = baseMLO->available_gpu_memory * (1000*1000*1000);
	else
		allocationSize = (float)free * .7;

	if (allocationSize > free)
	{
		printf(" WARNING: Required memory per thread, via \"--gpu_memory_per_mpi_rank\", not available on device. (Defaulting to less)\n");
		printf("  Required size        %zu MB\n", (size_t) baseMLO->available_gpu_memory*1000);
		printf("  Total available size %zu MB\n", free/(1000*1000));
		allocationSize = (float)free * .7; //Lets leave some for other processes for now
	}

	int memAlignmentSize;
	cudaDeviceGetAttribute ( &memAlignmentSize, cudaDevAttrTextureAlignment, dev_id );

#ifdef DEBUG_CUDA
	printf(" DEBUG: Custom allocator assigned %zu MB on device id %d.\n", allocationSize / (1000*1000), dev_id);
#endif

	allocator = new CudaCustomAllocator(allocationSize, memAlignmentSize);
#endif
};

void MlDeviceBundle::resetData()
{
	unsigned nr_classes = baseMLO->mymodel.nr_classes;
	HANDLE_ERROR(cudaSetDevice(device_id));

	/*======================================================
	   PROJECTOR, PROJECTOR PLAN AND BACKPROJECTOR SETUP
	======================================================*/

	//Can we pre-generate projector plan and corresponding euler matrices for all particles
	if (baseMLO->do_skip_align || baseMLO->do_skip_rotate || baseMLO->do_auto_refine || baseMLO->mymodel.orientational_prior_mode != NOPRIOR)
		generateProjectionPlanOnTheFly = true;
	else
		generateProjectionPlanOnTheFly = false;

	coarseProjectionPlans.clear();

#ifdef DEBUG_CUDA

	allocator->syncReadyEvents();
	allocator->freeReadyAllocs();
	if (allocator->getNumberOfAllocs() != 0)
	{
		printf("DEBUG_ERROR: Non-zero allocation count encountered in custom allocator between iterations.\n");
		allocator->printState();
		fflush(stdout);
		raise(SIGSEGV);
	}

#endif

	coarseProjectionPlans.resize(nr_classes, allocator);

	//Loop over classes
	for (int iclass = 0; iclass < nr_classes; iclass++)
	{
		cudaProjectors[iclass].setMdlDim(
				baseMLO->mymodel.PPref[iclass].data.xdim,
				baseMLO->mymodel.PPref[iclass].data.ydim,
				baseMLO->mymodel.PPref[iclass].data.zdim,
				baseMLO->mymodel.PPref[iclass].data.yinit,
				baseMLO->mymodel.PPref[iclass].data.zinit,
				baseMLO->mymodel.PPref[iclass].r_max,
				baseMLO->mymodel.PPref[iclass].padding_factor);

		cudaProjectors[iclass].initMdl(baseMLO->mymodel.PPref[iclass].data.data);

		cudaBackprojectors[iclass].setMdlDim(
				baseMLO->wsum_model.BPref[iclass].data.xdim,
				baseMLO->wsum_model.BPref[iclass].data.ydim,
				baseMLO->wsum_model.BPref[iclass].data.zdim,
				baseMLO->wsum_model.BPref[iclass].data.yinit,
				baseMLO->wsum_model.BPref[iclass].data.zinit,
				baseMLO->wsum_model.BPref[iclass].r_max,
				baseMLO->wsum_model.BPref[iclass].padding_factor);

		cudaBackprojectors[iclass].initMdl();

		//If doing predefined projector plan at all and is this class significant
		if (!generateProjectionPlanOnTheFly && baseMLO->mymodel.pdf_class[iclass] > 0.)
		{
			std::vector<int> exp_pointer_dir_nonzeroprior;
			std::vector<int> exp_pointer_psi_nonzeroprior;
			std::vector<RFLOAT> exp_directions_prior;
			std::vector<RFLOAT> exp_psi_prior;

			long unsigned itrans_max = baseMLO->sampling.NrTranslationalSamplings() - 1;
			long unsigned nr_idir = baseMLO->sampling.NrDirections(0, &exp_pointer_dir_nonzeroprior);
			long unsigned nr_ipsi = baseMLO->sampling.NrPsiSamplings(0, &exp_pointer_psi_nonzeroprior );

			coarseProjectionPlans[iclass].setup(
					baseMLO->sampling,
					exp_directions_prior,
					exp_psi_prior,
					exp_pointer_dir_nonzeroprior,
					exp_pointer_psi_nonzeroprior,
					NULL, //Mcoarse_significant
					baseMLO->mymodel.pdf_class,
					baseMLO->mymodel.pdf_direction,
					nr_idir,
					nr_ipsi,
					0, //idir_min
					nr_idir - 1, //idir_max
					0, //ipsi_min
					nr_ipsi - 1, //ipsi_max
					0, //itrans_min
					itrans_max,
					0, //current_oversampling
					1, //nr_oversampled_rot
					iclass,
					true, //coarse
					!IS_NOT_INV,
					baseMLO->do_skip_align,
					baseMLO->do_skip_rotate,
					baseMLO->mymodel.orientational_prior_mode
					);
		}
	}
};

MlOptimiserCuda::MlOptimiserCuda(MlOptimiser *baseMLOptimiser, int dev_id, MlDeviceBundle* bundle) :
		baseMLO(baseMLOptimiser), transformer1(0, bundle->allocator), transformer2(0, bundle->allocator)
{
	unsigned nr_classes = baseMLOptimiser->mymodel.nr_classes;

	/*======================================================
					DEVICE MEM OBJ SETUP
	======================================================*/

	device_id = dev_id;

	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));

	if(dev_id >= devCount)
	{
		std::cerr << " using device_id=" << dev_id << " (device no. " << dev_id+1 << ") which is higher than the available number of devices=" << devCount << std::endl;
		REPORT_ERROR("ERROR: Assigning a thread to a non-existent device (index likely too high)");
	}
	else
		HANDLE_ERROR(cudaSetDevice(dev_id));

	devBundle = bundle;

	HANDLE_ERROR(cudaStreamCreate(&stream1));
	HANDLE_ERROR(cudaStreamCreate(&stream2));

	classStreams.resize(nr_classes, 0);
	for (int i = 0; i < nr_classes; i++)
		HANDLE_ERROR(cudaStreamCreate(&classStreams[i]));

	refIs3D = baseMLO->mymodel.ref_dim == 3;
};

void MlOptimiserCuda::resetData()
{
	HANDLE_ERROR(cudaSetDevice(device_id));
	/*======================================================
	                  TRANSLATIONS SETUP
	======================================================*/

	if (baseMLO->do_shifts_onthefly)
	{
		if (baseMLO->global_fftshifts_ab_coarse.size() > 0)
			translator_coarse1.setShifters(baseMLO->global_fftshifts_ab_coarse);
		else
			translator_coarse1.clear();

		if (baseMLO->global_fftshifts_ab2_coarse.size() > 0)
			translator_coarse2.setShifters(baseMLO->global_fftshifts_ab2_coarse);
		else
			translator_coarse2.clear();

		if (baseMLO->global_fftshifts_ab_current.size() > 0)
			translator_current1.setShifters(baseMLO->global_fftshifts_ab_current);
		else
			translator_current1.clear();

		if (baseMLO->global_fftshifts_ab2_current.size() > 0)
			translator_current2.setShifters(baseMLO->global_fftshifts_ab2_current);
		else
			translator_current2.clear();
	}

	transformer1.clear();
	transformer2.clear();
};

void MlOptimiserCuda::doThreadExpectationSomeParticles(int thread_id)
{
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_THR);
#endif
//	CUDA_CPU_TOC("interParticle");
	CUDA_CPU_TIC("oneTask");
	DEBUG_HANDLE_ERROR(cudaSetDevice(device_id));
	//std::cerr << " calling on device " << device_id << std::endl;
	//put mweight allocation here
	size_t first_ipart = 0, last_ipart = 0;

	while (baseMLO->exp_ipart_ThreadTaskDistributor->getTasks(first_ipart, last_ipart))
	{
		for (long unsigned ipart = first_ipart; ipart <= last_ipart; ipart++)
		{
			CUDA_CPU_TIC("oneParticle");
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_A);
#endif
			unsigned my_ori_particle = baseMLO->exp_my_first_ori_particle + ipart;
			SamplingParameters sp;
			sp.nr_particles = baseMLO->mydata.ori_particles[my_ori_particle].particles_id.size();

			OptimisationParamters op(sp.nr_particles, my_ori_particle);

			// In the first iteration, multiple seeds will be generated
			// A single random class is selected for each pool of images, and one does not marginalise over the orientations
			// The optimal orientation is based on signal-product (rather than the signal-intensity sensitive Gaussian)
			// If do_firstiter_cc, then first perform a single iteration with K=1 and cross-correlation criteria, afterwards

			// Decide which classes to integrate over (for random class assignment in 1st iteration)
			sp.iclass_max = baseMLO->mymodel.nr_classes - 1;
			// low-pass filter again and generate the seeds
			if (baseMLO->do_generate_seeds)
			{
				if (baseMLO->do_firstiter_cc && baseMLO->iter == 1)
				{
					// In first (CC) iter, use a single reference (and CC)
					sp.iclass_min = sp.iclass_max = 0;
				}
				else if ( (baseMLO->do_firstiter_cc && baseMLO->iter == 2) ||
						(!baseMLO->do_firstiter_cc && baseMLO->iter == 1))
				{
					// In second CC iter, or first iter without CC: generate the seeds
					// Now select a single random class
					// exp_part_id is already in randomized order (controlled by -seed)
					// WARNING: USING SAME iclass_min AND iclass_max FOR SomeParticles!!
					sp.iclass_min = sp.iclass_max = divide_equally_which_group(baseMLO->mydata.numberOfOriginalParticles(), baseMLO->mymodel.nr_classes, op.my_ori_particle);
				}
			}
			// Global exp_metadata array has metadata of all ori_particles. Where does my_ori_particle start?
			for (long int iori = baseMLO->exp_my_first_ori_particle; iori <= baseMLO->exp_my_last_ori_particle; iori++)
			{
				if (iori == my_ori_particle) break;
				op.metadata_offset += baseMLO->mydata.ori_particles[iori].particles_id.size();
			}
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_A);
#endif
			CUDA_CPU_TIC("getFourierTransformsAndCtfs");
			getFourierTransformsAndCtfs(my_ori_particle, op, sp, baseMLO, this);
			CUDA_CPU_TOC("getFourierTransformsAndCtfs");

			if (baseMLO->do_realign_movies && baseMLO->movie_frame_running_avg_side > 0)
			{
				baseMLO->calculateRunningAveragesOfMovieFrames(my_ori_particle, op.Fimgs, op.power_imgs, op.highres_Xi2_imgs);
			}

			// To deal with skipped alignments/rotations
			if (baseMLO->do_skip_align)
			{
				sp.itrans_min = sp.itrans_max = sp.idir_min = sp.idir_max = sp.ipsi_min = sp.ipsi_max =
						my_ori_particle - baseMLO->exp_my_first_ori_particle;
			}
			else
			{
				sp.itrans_min = 0;
				sp.itrans_max = baseMLO->sampling.NrTranslationalSamplings() - 1;

				if (baseMLO->do_skip_rotate)
				{
					sp.idir_min = sp.idir_max = sp.ipsi_min = sp.ipsi_max =
							my_ori_particle - baseMLO->exp_my_first_ori_particle;
				}
				else
				{
					sp.idir_min = sp.ipsi_min = 0;
					sp.idir_max = baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior) - 1;
					sp.ipsi_max = baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior ) - 1;
				}
			}

			// Initialise significant weight to minus one, so that all coarse sampling points will be handled in the first pass
			op.significant_weight.resize(sp.nr_particles, -1.);

			// Only perform a second pass when using adaptive oversampling
			//int nr_sampling_passes = (baseMLO->adaptive_oversampling > 0) ? 2 : 1;
			// But on the gpu the data-structures are different between passes, so we need to make a symbolic pass to set the weights up for storeWS
			int nr_sampling_passes = 2;

			/// -- This is a iframe-indexed vector, each entry of which is a dense data-array. These are replacements to using
			//    Mweight in the sparse (Fine-sampled) pass, coarse is unused but created empty input for convert ( FIXME )
			std::vector <IndexedDataArray> CoarsePassWeights(1, devBundle->allocator) ,FinePassWeights(sp.nr_particles, devBundle->allocator);
			// -- This is a iframe-indexed vector, each entry of which is a class-indexed vector of masks, one for each
			//    class in FinePassWeights
			std::vector < std::vector <IndexedDataArrayMask> > FinePassClassMasks(sp.nr_particles, std::vector <IndexedDataArrayMask>(baseMLO->mymodel.nr_classes, devBundle->allocator));
			// -- This is a iframe-indexed vector, each entry of which is parameters used in the projection-operations *after* the
			//    coarse pass, declared here to keep scope to storeWS
			std::vector < ProjectionParams > FineProjectionData(sp.nr_particles, baseMLO->mymodel.nr_classes);

			std::vector < cudaStager<unsigned long> > stagerD2(sp.nr_particles,devBundle->allocator), stagerSWS(sp.nr_particles,devBundle->allocator);

			for (int ipass = 0; ipass < nr_sampling_passes; ipass++)
			{
				CUDA_CPU_TIC("weightPass");
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_B);
#endif
				if (baseMLO->strict_highres_exp > 0.)
					// Use smaller images in both passes and keep a maximum on coarse_size, just like in FREALIGN
					sp.current_image_size = baseMLO->coarse_size;
				else if (baseMLO->adaptive_oversampling > 0)
					// Use smaller images in the first pass, larger ones in the second pass
					sp.current_image_size = (ipass == 0) ? baseMLO->coarse_size : baseMLO->mymodel.current_size;
				else
					sp.current_image_size = baseMLO->mymodel.current_size;

				// Use coarse sampling in the first pass, oversampled one the second pass
				sp.current_oversampling = (ipass == 0) ? 0 : baseMLO->adaptive_oversampling;

				sp.nr_dir = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.NrDirections(0, &op.pointer_dir_nonzeroprior);
				sp.nr_psi = (baseMLO->do_skip_align || baseMLO->do_skip_rotate) ? 1 : baseMLO->sampling.NrPsiSamplings(0, &op.pointer_psi_nonzeroprior);
				sp.nr_trans = (baseMLO->do_skip_align) ? 1 : baseMLO->sampling.NrTranslationalSamplings();
				sp.nr_oversampled_rot = baseMLO->sampling.oversamplingFactorOrientations(sp.current_oversampling);
				sp.nr_oversampled_trans = baseMLO->sampling.oversamplingFactorTranslations(sp.current_oversampling);
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_B);
#endif
				if (ipass == 0)
				{
					unsigned long weightsPerPart(baseMLO->mymodel.nr_classes * sp.nr_dir * sp.nr_psi * sp.nr_trans * sp.nr_oversampled_rot * sp.nr_oversampled_trans);

					op.Mweight.resizeNoCp(1,1,sp.nr_particles, weightsPerPart);

					CudaGlobalPtr<XFLOAT> Mweight(devBundle->allocator);
					Mweight.setSize(sp.nr_particles * weightsPerPart);
					Mweight.setHstPtr(op.Mweight.data);
					Mweight.device_alloc();
					deviceInitValue<XFLOAT>(Mweight, -999.);
					Mweight.streamSync();

					CUDA_CPU_TIC("getAllSquaredDifferencesCoarse");
					getAllSquaredDifferencesCoarse(ipass, op, sp, baseMLO, this, Mweight);
					CUDA_CPU_TOC("getAllSquaredDifferencesCoarse");

					CUDA_CPU_TIC("convertAllSquaredDifferencesToWeightsCoarse");
					convertAllSquaredDifferencesToWeights(ipass, op, sp, baseMLO, this, CoarsePassWeights, FinePassClassMasks, Mweight);
					CUDA_CPU_TOC("convertAllSquaredDifferencesToWeightsCoarse");
				}
				else
				{
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_D);
#endif
//					// -- go through all classes and generate projectionsetups for all classes - to be used in getASDF and storeWS below --
//					// the reason to do this globally is subtle - we want the orientation_num of all classes to estimate a largest possible
//					// weight-array, which would be insanely much larger than necessary if we had to assume the worst.
					for (long int iframe = 0; iframe < sp.nr_particles; iframe++)
					{
						FineProjectionData[iframe].orientationNumAllClasses = 0;
						for (int exp_iclass = sp.iclass_min; exp_iclass <= sp.iclass_max; exp_iclass++)
						{
							if(exp_iclass>0)
								FineProjectionData[iframe].class_idx[exp_iclass] = FineProjectionData[iframe].rots.size();
							FineProjectionData[iframe].class_entries[exp_iclass] = 0;

							CUDA_CPU_TIC("generateProjectionSetup");
							FineProjectionData[iframe].orientationNumAllClasses += generateProjectionSetupFine(
									op,
									sp,
									baseMLO,
									exp_iclass,
									FineProjectionData[iframe]);
							CUDA_CPU_TOC("generateProjectionSetup");

						}
						//set a maximum possible size for all weights (to be reduced by significance-checks)
						FinePassWeights[iframe].setDataSize(FineProjectionData[iframe].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans);
						FinePassWeights[iframe].dual_alloc_all();
						stagerD2[iframe].size= 2*(FineProjectionData[iframe].orientationNumAllClasses*sp.nr_trans*sp.nr_oversampled_trans);
						stagerD2[iframe].prepare();
					}
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_D);
#endif
//					printf("Allocator used space before 'getAllSquaredDifferencesFine': %.2f MiB\n", (float)devBundle->allocator->getTotalUsedSpace()/(1024.*1024.));

					CUDA_CPU_TIC("getAllSquaredDifferencesFine");
					getAllSquaredDifferencesFine(ipass, op, sp, baseMLO, this, FinePassWeights, FinePassClassMasks, FineProjectionData, stagerD2);
					CUDA_CPU_TOC("getAllSquaredDifferencesFine");

					CudaGlobalPtr<XFLOAT> Mweight(devBundle->allocator); //DUMMY

					CUDA_CPU_TIC("convertAllSquaredDifferencesToWeightsFine");
					convertAllSquaredDifferencesToWeights(ipass, op, sp, baseMLO, this, FinePassWeights, FinePassClassMasks, Mweight);
					CUDA_CPU_TOC("convertAllSquaredDifferencesToWeightsFine");

				}

				CUDA_CPU_TOC("weightPass");
			}
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.tic(baseMLO->TIMING_ESP_DIFF2_E);
#endif

			// For the reconstruction step use mymodel.current_size!
			sp.current_image_size = baseMLO->mymodel.current_size;
			for (long int iframe = 0; iframe < sp.nr_particles; iframe++)
			{
				stagerSWS[iframe].size= 2*(FineProjectionData[iframe].orientationNumAllClasses);
				stagerSWS[iframe].prepare();
			}
#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_DIFF2_E);
#endif
			CUDA_CPU_TIC("storeWeightedSums");
			storeWeightedSums(op, sp, baseMLO, this, FinePassWeights, FineProjectionData, FinePassClassMasks, stagerSWS);
			CUDA_CPU_TOC("storeWeightedSums");
		}
	}
	CUDA_CPU_TOC("oneTask");
//	CUDA_CPU_TIC("interParticle");
//	exit(0);

#ifdef TIMING
	// Only time one thread
	if (thread_id == 0)
		baseMLO->timer.toc(baseMLO->TIMING_ESP_THR);
#endif
}

