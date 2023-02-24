#ifdef _WIN32
//until NVIDIA adds 1.2 on win
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif

#include <d4d/digitizer_cl.h>
#include <d4d/PinHoleCamera.h>

#include "WandRawData.h"
#include <d4d/Scan3d.h>
#include <d4d/open_cl_imp.h>
#include <d4d/kdtree.h>
#include <d4d/range_image.h>

#include <sstream>
#include <limits>
#include <fstream>
#include <iostream>
#include <limits>
#include <atomic>
#include <math.h>

#include <boost/property_tree/ptree.hpp>

//#include <flann/flann.hpp>

using namespace std;

namespace d4d
{
	DigitizerCL::DigitizerCL(
        const boost::property_tree::ptree& pt,
        const std::vector<uint8_t>& calibration,
        float focusDistance,
        ScannerType scannerType,
        bool useMaxRange)
	{
		BOOST_LOG_FUNCTION();

		open_cl_implementation = std::unique_ptr<OpenCLImplementation>( new OpenCLImplementation() );
        kernel_stash_ = std::unique_ptr<KernelStash>(new KernelStash());
		cl_sources_path_ = "OpenCL/digitizer.cl";
		n_max_centroids_ = lines_per_image_*2;

		data_xyz_size_ = 0;

		do_corrention_ = false;
        save_debug_data_ = false;
		bilateral_radius_ = 0;
		bilateral_cull_check_radius_ = 5;
        normal_radius_ = 0;
		bilateral_gauss_delta_ = 0;
		bilateral_z_delta_ = 0;
		bilateral_z_delta_y_zero_ = 0;
		bilateral_percentage_min_cutoff_ = 0;

		minimum_scan_points_ = 100;

		normal_radius_ = 0;
        open_cl_initialized_ = false;
        median_filter_z_tolerance_ = 0.3;
        median_filter_min_hits_ = 6;
        step_ = 2;

        cl_profiling_ = false;
        enable_kernel_profiling_ = false;

        enable_motion_detection_ = false;
		force_nvidia_ = true;

		open_cl_implementation->events_.reserve(100);

		scannerType_ = ScannerType::Emerald; // default guess, will be overwritten when calibration is set

        //InitCL();
        Configure(pt);
        SetCalibration(calibration.data(), focusDistance, scannerType);
        ScanTypeCustomization(pt, useMaxRange);
	}

	DigitizerCL::~DigitizerCL() {}

    void DigitizerCL::Configure(const boost::property_tree::ptree& pt)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        set_centroid_alphax(pt.get<float>("digitizer.centroid_alphax"));
        set_centroid_alphay(pt.get<float>("digitizer.centroid_alphay"));
        set_centroid_threshold(pt.get<float>("digitizer.centroid_threshold"));

        set_search_rad(pt.get<int>("digitizer.search_rad"));
        set_search_scale(pt.get<float>("digitizer.search_scale"));
        set_motion_rms_threshold(pt.get<double>("digitizer.motion_rms_threshold"));
        set_motion_overlap_threshold(pt.get<double>("digitizer.motion_overlap_threshold"));
        enable_motion_detection_ = pt.get<bool>("digitizer.enable_motion_detection", false);

        bilateral_radius_ = pt.get<int>("digitizer.bilateral_radius");
        bilateral_cull_check_radius_ = pt.get<int>("digitizer.bilateral_cull_check_radius", 5);
        bilateral_gauss_delta_ = pt.get<float>("digitizer.bilateral_gauss_delta");
        bilateral_z_delta_ = pt.get<float>("digitizer.bilateral_z_delta");
        bilateral_z_delta_y_zero_ = pt.get<float>("digitizer.bilateral_z_delta_y_zero", 0.2);
        bilateral_percentage_min_cutoff_ = pt.get<float>("digitizer.bilateral_percentage_min_cutoff", 0.5);
        normal_radius_ = pt.get<int>("digitizer.normal_radius");

        x_crop_start_ = pt.get<float>("digitizer.x_crop_start");
        y_crop_start_ = pt.get<float>("digitizer.y_crop_start");

        x_crop_end_ = pt.get<float>("digitizer.x_crop_end");
        y_crop_end_ = pt.get<float>("digitizer.y_crop_end");

        implement_LUT = pt.get<bool>("digitizer.implement_LUT", true);
        run_region_filter = pt.get<bool>("digitizer.run_region_filter", false);
        remove_region_filter_depth = pt.get<float>("digitizer.remove_region_filter_depth", 0.3);
        z_depth_start_planscan_ = pt.get<float>("digitizer.z_depth_start_planscan");
        z_depth_end_planscan_ = pt.get<float>("digitizer.z_depth_end_planscan");

        z_depth_start_emerald_ = pt.get<float>("digitizer.z_depth_start_emerald");
        z_depth_end_emerald_ = pt.get<float>("digitizer.z_depth_end_emerald");

        // Default z depth start and end is planscan
        z_depth_start_ = z_depth_start_planscan_;
        z_depth_end_ = z_depth_end_planscan_;

        minimum_scan_points_ = pt.get<int>("digitizer.minimum_scan_points");
        cl_sources_path_ = pt.get<std::string>("Resources", ".") + "/OpenCL/digitizer.cl";

        median_filter_z_tolerance_ = pt.get<float>("digitizer.median_filter_z_tolerance");
        median_filter_min_hits_ = pt.get<int>("digitizer.median_filter_min_hits");
        save_debug_data_ = pt.get<bool>("digitizer.save_debug_data", false);
        cl_profiling_ = pt.get<bool>("digitizer.opencl_profiling", false);
        enable_kernel_profiling_ = pt.get<bool>("digitizer.enable_kernel_profiling", false);
        force_nvidia_ = pt.get<bool>("digitizer.opencl_force_nvidia", true);

        local_dotp_cutoff_negative_ = pt.get<float>("digitizer.local_dotp_cutoff_negative", -1.0);
        local_dotp_cutoff_positive_ = pt.get<float>("digitizer.local_dotp_cutoff_positive", 1.0);
        normal_y_crop_left_ = pt.get<float>("digitizer.normal_y_crop_left", 0.0);
        normal_y_crop_right_ = pt.get<float>("digitizer.normal_y_crop_right", 1.0);

        exclude_blue = pt.get<bool>("digitizer.exclude_blue", false);
        enable_color_correction_ = pt.get<bool>("digitizer.enable_color_correction", true);

        // Badslam aligner uses frames in their original camera coordinate system, but
        // pointgridaligner assumes focal distance at the origin.
        focal_origin_ = true;// !pt.get<bool>("badslamaligner.enable", true);

        if (!open_cl_initialized_)
        {
            InitCL();
            if (open_cl_initialized_) ConnectKernelIO();
        }

        //init bilateral gaussian kernel
        vector<float> bilateral_gaussian_(2 * bilateral_radius_ + 1);
        for (int i = 0; i < 2 * bilateral_radius_ + 1; i++)
        {
            int x = i - bilateral_radius_;
            bilateral_gaussian_[i] = exp(-(x * x) /
                (2 * bilateral_gauss_delta_ * bilateral_gauss_delta_));
        }

        open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->bilateral_gauss_kernel_, true, 0, sizeof(float) * (2 * bilateral_radius_ + 1), &bilateral_gaussian_[0]);

    }

    void DigitizerCL::ScanTypeCustomization(const boost::property_tree::ptree& pt, bool useMaxRange) {

        if (useMaxRange)
        {
            z_depth_end_ = pt.get<float>("scanTypeCustomization.scanBody_z_depth_end");;
        }
    }

    void DigitizerCL::InitCL()
	{
		BOOST_LOG_FUNCTION();

        OpenCLManagerInfo info;
        info.force_nvidia_ = force_nvidia_;
        info.cl_profiling_ = cl_profiling_;
        info.enable_kernel_profiling_ = enable_kernel_profiling_;
        info.source_paths_.clear();
		// Do not push the sources_paths to prevent file loading. Mechanism left here for future use
        //info.source_paths_.push_back(cl_sources_path_);
        open_cl_manager_.reset(new OpenCLManager(info));

		AllocateGPUResources();

        open_cl_initialized_ = true;
	}

    void DigitizerCL::ConnectKernelIO()
    {
        if (!open_cl_initialized_) return;

        auto &kernels = open_cl_manager_->getKernels();
        auto &o = open_cl_implementation;

        FrameInfo frame_info(width_, height_, n_images_, x_crop_start_, x_crop_end_, y_crop_start_, y_crop_end_);
        CommonKernelInfo common_kernel_info;
        common_kernel_info.cl_queue_ = &(open_cl_manager_->getQueue());
        common_kernel_info.cl_context_ = open_cl_manager_->getContext();
        common_kernel_info.cl_profiling_ = cl_profiling_;
        common_kernel_info.enable_kernel_profiling_ = enable_kernel_profiling_;
        common_kernel_info.kernels_ = kernels;
        common_kernel_info.frame_info_ = frame_info;

        kernel_stash_->clear_mask_kernel_ = ClearMaskKernel(common_kernel_info);
        kernel_stash_->char_to_float_kernel_ = CharToFloatKernel(common_kernel_info, open_cl_implementation->data_in_, open_cl_implementation->data_tmp1_);
		kernel_stash_->char_to_float_straight_kernel_ = CharToFloatStraightKernel(common_kernel_info, open_cl_implementation->data_in_, open_cl_implementation->data_tmp1_);
		kernel_stash_->filter_x_kernel_ = FilterXKernel(common_kernel_info, o->data_tmp1_, o->data_tmp2_, centroid_alphax_, centroid_alphay_);
        kernel_stash_->filter_y_kernel_ = FilterYKernel(common_kernel_info, o->data_tmp1_, o->data_tmp2_, centroid_alphax_, centroid_alphay_);
        kernel_stash_->transpose_images_kernel_ = TransposeImagesKernel(common_kernel_info, o->data_tmp2_, o->data_tmp1_);
/////////////////////////////////////////////////new segment # 2 //////////////
        {
            NonMaximaSuppressionKernel::KernelInput k;
            auto &ocl = open_cl_implementation;
            k.buffer_in_ = ocl->data_tmp1_;
            k.buffer_out_ = ocl->data_tmp2_;
            k.centroid_count_ = ocl->centroid_count_;
            k.centroid_threshold_ = centroid_threshold_;
            k.n_max_centroids_ = n_max_centroids_;
            k.centroid_weights_ = ocl->centroid_weights_;
            kernel_stash_->non_maxima_supression_kernel_ = NonMaximaSuppressionKernel(common_kernel_info, k);
        }

        kernel_stash_->zero_guesses_kernel_ = ZeroGuessesKernel(common_kernel_info, o->centroid_ids_, width_*n_images_*n_max_centroids_);
        kernel_stash_->profiler_sampling_kernel_ = ProfilerSampleKernel(common_kernel_info);
        {
            CentroidMarkerKernel::KernelInput k;
            k.in_ = o->data_tmp2_;
            k.centroid_nums_ = o->centroid_count_;
            k.output_data_ = o->centroid_ids_;
            k.search_radius_ = search_rad_;
            k.shift_scale_ = search_scale_;
            k.images_in_sequence_ = n_seq_images_;
            k.max_centroids_ = n_max_centroids_;
            k.y_start_ = y_crop_start_;
            k.y_end_ = y_crop_end_;
            kernel_stash_->centroid_marker_kernel_ = CentroidMarkerKernel(common_kernel_info, k);
        }

        {
            CopyCentroidsKernel::KernelInput k;
            k.in_ = o->data_tmp2_;
            k.centroid_weights_ = o->centroid_weights_;
            k.centroid_count_ = o->centroid_count_;
            k.centroid_ids_ = o->centroid_ids_;
            k.centroid_indxs_ = o->centroid_indxs_;
            k.data_x_ = o->data_x_;
            k.data_y_ = o->data_y_;
            k.data_z_ = o->data_z_;
            k.data_weights_ = o->data_weights_;
            k.lines_per_image_ = lines_per_image_;
            k.n_max_centroids_ = n_max_centroids_;
            kernel_stash_->copy_centroids_kernel_ = CopyCentroidsKernel(common_kernel_info, k);
        }

        {
            CreateCentroidMaskKernel::KernelInput k;
            k.centroid_count_ = o->centroid_count_;
            k.centroid_indxs_ = o->centroid_indxs_;
            k.data_x_ = o->data_x_;
            k.data_y_ = o->data_y_;
            k.data_z_ = o->data_z_;
            k.image_ = o->data_tmp1_;
            k.centroid_mask_list_ = o->centroid_mask_list_;
            k.centroid_mask_image_ = kernel_stash_->clear_mask_kernel_.getCentroidMaskImage();
            k.centroid_index_image_ = o->centroid_index_image_;
            kernel_stash_->create_centroid_mask_kernel_ = CreateCentroidMaskKernel(common_kernel_info, k);
        }

        {
            CentroidFilterKernel::KernelInput k;
            k.centroid_mask_list_ = o->centroid_mask_list_;
            k.centroid_mask_image_in_ = kernel_stash_->clear_mask_kernel_.getCentroidMaskImage();
            k.centroid_index_image_ = o->centroid_index_image_;
            k.image_ = o->data_in_;
            k.centroid_mask_image_out_ = kernel_stash_->clear_mask_kernel_.getCentroidMaskImage2();
            k.data_z_ = o->data_z_;
            k.centroid_filter_radius_ = centroid_filter_parameters_.radius_;
            k.min_intensity_range_ = centroid_filter_parameters_.min_intensity_range_;
            k.min_neighbor_centroids_ = centroid_filter_parameters_.min_neighbor_centroids_;
			k.max_intensity_ = centroid_filter_parameters_.max_intensity_;
            kernel_stash_->centroid_filter_kernel_ = CentroidFilterKernel(common_kernel_info, k);
        }

        {
            ComputeZsKernel::KernelInput k;
            k.centr_num_ = 0;
            k.data_x_ = o->data_x_;
            k.data_y_ = o->data_y_;
            k.data_z_ = o->data_z_;
            k.lens_param_ = o->lens_param_;
            k.lens_matrix_ = o->lens_matrix_;
            k.laser_cal_ = o->laser_cal_;
            k.z_depth_start_ = z_depth_start_;
            k.z_depth_end_ = z_depth_end_;
            k.data_x_new_ = o->data_x_new_;
            k.data_y_new_ = o->data_y_new_;
            k.centroid_mask_list_ = o->centroid_mask_list_;
            k.use_mask_ = centroid_filter_parameters_.radius_ > 0;
			k.calibration_offset_ = 0;
            k.undistorted_x_ = o->undistorted_x;
            k.undistorted_y_ = o->undistorted_y;
            k.back_projected_x_0 = o->back_projected_x_0;
            k.back_projected_y_0 = o->back_projected_y_0;
            k.back_projected_x_10 = o->back_projected_x_10;
            k.back_projected_y_10 = o->back_projected_y_10;
            k.use_LUT = implement_LUT? 1 : 0;
            kernel_stash_->compute_zs_kernel_ = ComputeZsKernel(common_kernel_info, k);
        }


        kernel_stash_->init_range_image_kernel_ = InitRangeImageKernel(common_kernel_info, o->image_range_, o->image_range_median_, o->image_range_bilateral_, o->image_normals_);
		kernel_stash_->init_range_image_anisotropic_kernel_ = InitRangeImageAnisotropicKernel(common_kernel_info,
			o->image_range_,
			o->image_range_anisotropic_,
			o->image_range_median_anisotropic_,
			o->image_range_bilateral_,
			o->image_normals_);

        {
            CopyToRangeImageKernel::KernelInput k(o->image_range_);
            k.data_x_new_ = o->data_x_new_;
            k.data_y_new_ = o->data_y_new_;
            k.data_z_ = o->data_z_;
            k.data_x_ = o->data_x_;
            k.data_y_ = o->data_y_;
            k.centr_num_ = 0;
            kernel_stash_->copy_to_range_image_kernel_ = CopyToRangeImageKernel(common_kernel_info, k);
        }

        {
            NoiseMedianFilterKernel::KernelInput k(o->image_range_, o->image_range_median_);
            k.lens_param_ = o->lens_param_;
            k.z_depth_start_ = z_depth_start_;
            k.z_depth_end_ = z_depth_end_;
            k.median_filter_z_tolerance_ = median_filter_z_tolerance_;
            k.median_filter_min_hits_ = median_filter_min_hits_;
			k.kernel_width_ = 2;// centroid_filter_parameters_.noise_removal_kernel_width_;
			k.kernel_height_ = 2;// centroid_filter_parameters_.noise_removal_kernel_height_;
            kernel_stash_->noise_median_filter_kernel_ = NoiseMedianFilterKernel(common_kernel_info, k);
        }

        {
            BilateralFilterKernel::KernelInput k(o->image_range_median_, o->image_range_bilateral_);
            k.bilateral_radius_ = bilateral_radius_;
            k.bilateral_cull_check_radius_ = bilateral_cull_check_radius_;
            k.bilateral_z_delta_ = bilateral_z_delta_;
            k.bilateral_z_delta_y_zero_ = bilateral_z_delta_y_zero_;
            k.bilateral_percentage_min_cutoff_ = bilateral_percentage_min_cutoff_;
            k.bilateral_gauss_kernel_ = o->bilateral_gauss_kernel_;
            k.step_ = step_;
            kernel_stash_->bilateral_filter_kernel_ = BilateralFilterKernel(common_kernel_info, k);
        }

		{
			CopyToRangeImageAnisotropicKernel::KernelInput k(o->image_range_, o->image_range_anisotropic_);
			kernel_stash_->copy_to_range_image_anisotropic_kernel_ = CopyToRangeImageAnisotropicKernel(common_kernel_info, k);
		}

		{
			NoiseMedianFilterAnisotropicKernel::KernelInput k(o->image_range_anisotropic_, o->image_range_median_anisotropic_);
			k.lens_param_ = o->lens_param_;
			k.z_depth_start_ = z_depth_start_;
			k.z_depth_end_ = z_depth_end_;
			k.median_filter_z_tolerance_ = median_filter_z_tolerance_;
			k.median_filter_min_hits_ = median_filter_min_hits_;
			k.kernel_width_ = centroid_filter_parameters_.noise_removal_kernel_width_;
			k.kernel_height_ = centroid_filter_parameters_.noise_removal_kernel_height_;
			kernel_stash_->noise_median_filter_anisotropic_kernel_ = NoiseMedianFilterAnisotropicKernel(common_kernel_info, k);
		}

		{
			BilateralFilterAnisotropicKernel::KernelInput k(o->image_range_median_anisotropic_, o->image_range_bilateral_);
			k.bilateral_radius_ = bilateral_radius_;
			k.bilateral_cull_check_radius_ = bilateral_cull_check_radius_;
			k.bilateral_z_delta_ = bilateral_z_delta_;
			k.bilateral_z_delta_y_zero_ = bilateral_z_delta_y_zero_;
			k.bilateral_percentage_min_cutoff_ = bilateral_percentage_min_cutoff_;
			k.bilateral_gauss_kernel_ = o->bilateral_gauss_kernel_;
			k.step_ = step_;
			kernel_stash_->bilateral_filter_anisotropic_kernel_ = BilateralFilterAnisotropicKernel(common_kernel_info, k);
		}

		{
			MeanFilterAnisotropicKernel::KernelInput k(o->image_range_median_anisotropic_, o->image_range_bilateral_);
			k.bilateral_radius_ = bilateral_radius_;
			k.bilateral_cull_check_radius_ = bilateral_cull_check_radius_;
			k.bilateral_z_delta_ = bilateral_z_delta_;
			k.bilateral_z_delta_y_zero_ = bilateral_z_delta_y_zero_;
			k.bilateral_percentage_min_cutoff_ = bilateral_percentage_min_cutoff_;
			k.bilateral_gauss_kernel_ = o->bilateral_gauss_kernel_;
			k.step_ = step_;
			kernel_stash_->mean_filter_anisotropic_kernel_ = MeanFilterAnisotropicKernel(common_kernel_info, k);
		}

        {
            NormalsComputationKernel::KernelInput k(o->image_range_bilateral_, o->image_normals_);
            k.normal_radius_ = normal_radius_;
            k.distance_ = (float)((normal_radius_*0.15)*(normal_radius_*0.15));
            k.step_ = step_;
            k.local_dotp_cutoff_negative_ = local_dotp_cutoff_negative_;
            k.local_dotp_cutoff_positive_ = local_dotp_cutoff_positive_;
            k.normal_y_crop_left_ = normal_y_crop_left_;
            k.normal_y_crop_right_ = normal_y_crop_right_;
            kernel_stash_->normals_computation_kernel_ = NormalsComputationKernel(common_kernel_info, k);
        }

        {
            DownSampleBufferToImage::KernelInput k(o->color_data_in_, o->color_image_);
            k.n_rows_ = height_;
            k.n_columns_ = width_;
            kernel_stash_->copy_color_data_to_image_ = DownSampleBufferToImage(common_kernel_info, k);
        }

        {
            MaskZeroKernel::KernelInput k(o->color_mask_);
            k.n_rows_ = height_/2;
            k.n_columns_ = width_/2;
            kernel_stash_->mask_zero_ = MaskZeroKernel(common_kernel_info, k);
        }

        {
            ColorMaskUnion::KernelInput k(o->color_image_,
                                          o->color_mask_gloves_colors_centers_,
                                          o->color_mask_gloves_colors_radii_,
                                          o->color_mask_);
            k.inside_value_ = 0;
            k.outside_value_ = 1;
            k.n_balls_ = o->color_mask_gloves_colors_n_;
            k.n_rows_ = height_/2;
            k.n_columns_ = width_/2;
            kernel_stash_->create_mask_ = ColorMaskUnion(common_kernel_info, k);
        }

        {
            MaskApplyKernel::KernelInput k(o->color_mask_, o->image_range_bilateral_);
            k.n_rows_ = height_/2;
            k.n_columns_ = width_/2;
            kernel_stash_->mask_apply_ = MaskApplyKernel(common_kernel_info, k);
        }

    }

	void DigitizerCL::AllocateGPUResources()
	{
/////////////////////////////////////////////////new segment # 3 //////////////
        auto c = open_cl_manager_->getContext();

		open_cl_implementation->image_in_ = cl::Image2D(*c, CL_MEM_READ_ONLY,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_*n_images_/4, height_);
		open_cl_implementation->image_tmp1_ = cl::Image2D(*c, CL_MEM_READ_WRITE,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_*n_images_/4, height_);
		open_cl_implementation->image_tmp2_ = cl::Image2D(*c, CL_MEM_READ_WRITE,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_*n_images_/4, height_);
		open_cl_implementation->image_range_ = cl::Image2D(*c, CL_MEM_READ_WRITE,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_, height_);
		open_cl_implementation->image_range_median_ = cl::Image2D(*c, CL_MEM_READ_WRITE,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_, height_);
        open_cl_implementation->image_range_bilateral_ = cl::Image2D(*c, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width_/step_, height_/step_);
		open_cl_implementation->image_normals_ = cl::Image2D(*c, CL_MEM_READ_WRITE,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_ / step_, height_ / step_);

		open_cl_implementation->image_range_anisotropic_ = cl::Image2D(*c, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width_, height_/2);
		open_cl_implementation->image_range_median_anisotropic_ = cl::Image2D(*c, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), width_, height_/2);


		open_cl_implementation->centroid_index_image_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(unsigned int)*width_*height_*(n_images_));
		open_cl_implementation->data_in_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(char)*width_*height_*(n_images_));
		open_cl_implementation->data_tmp1_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)*width_*height_*(n_images_));
		open_cl_implementation->data_tmp2_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)*width_*height_*(n_images_));
		open_cl_implementation->centroid_count_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(unsigned int)*width_*(n_images_));
		open_cl_implementation->centroid_ids_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(unsigned int)*width_*n_images_*n_max_centroids_);
        open_cl_implementation->centroid_weights_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)*width_*n_images_*n_max_centroids_);

		open_cl_implementation->centroid_indxs_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(unsigned int)*width_*n_images_);

		open_cl_implementation->laser_cal_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(float)*lines_per_image_*n_images_*12);
		open_cl_implementation->lens_param_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(float)*10);
		open_cl_implementation->lens_matrix_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(float)*16 );

		open_cl_implementation->bilateral_gauss_kernel_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(float)*(2 * bilateral_radius_ + 1) );

        open_cl_implementation->color_data_in_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(char)*width_*height_*3);
        open_cl_implementation->color_image_ = cl::Image2D(*c, CL_MEM_READ_WRITE,cl::ImageFormat(CL_RGBA,CL_FLOAT),width_ / step_, height_ / step_);
        open_cl_implementation->color_mask_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(unsigned int)*width_ / 2 *height_ / 2);

        open_cl_implementation->color_mask_gloves_colors_n_ = 4;
        float some_glove_colors[12] = { 58.0f,77.0f,118.0f,77.0f,94.0f,150.0f,118.0f,130.0f,200.0f, 184.3f, 229.3f, 251.3f };
        float some_glove_radii[4] = { 75.0f,  75.0f, 120.0f, 35.0f };
        for( int i=0; i<open_cl_implementation->color_mask_gloves_colors_n_; i++){
            some_glove_radii[i] = some_glove_radii[i]*some_glove_radii[i];
        }
        open_cl_implementation->color_mask_gloves_colors_centers_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(float)*3*(open_cl_implementation->color_mask_gloves_colors_n_));
        open_cl_implementation->color_mask_gloves_colors_radii_ = cl::Buffer(*c, CL_MEM_READ_ONLY, sizeof(float)*(open_cl_implementation->color_mask_gloves_colors_n_));


        open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->color_mask_gloves_colors_centers_, true, 0, sizeof(float)*3*(open_cl_implementation->color_mask_gloves_colors_n_), &some_glove_colors );
        open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->color_mask_gloves_colors_radii_, true, 0, sizeof(float)*(open_cl_implementation->color_mask_gloves_colors_n_), &some_glove_radii);

    }

	void DigitizerCL::SetCalibration(const void* data, double focusDistance, ScannerType scannerType)
	{
		if(!data)
		{
			BOOST_LOG_SEV(log_,Logger::severity_level::critical) << "DigitizerCL::SetCalibration input data is null";
			throw std::runtime_error("DigitizerCL::SetCalibration input data is null");
		}
		const float* data_c = (const float*)data;

		std::copy(data_c, data_c+18, lens_calibration_.begin());
		double rq,iq,jq,kq;
		iq = data_c[0];
		jq = data_c[1];
		kq = data_c[2];
		rq = data_c[3];

		Quaternion<> q(rq,iq,jq,kq);
		SmallVector<> v;

		v[0] = data_c[4];
		v[1] = data_c[5];
		v[2] = data_c[6];

		double focal;
		double cx,cy;
		double k1,k2,k3,t1,t2;
		double dx,dy,dz;



		focal = data_c[7];
		cx = data_c[8];
		cy = data_c[9];
		k1 = data_c[10];
		k2 = data_c[11];
		k3 = data_c[12];
		t1 = data_c[13];
		t2 = data_c[14];
		dx = data_c[15];
		dy = data_c[16];
		dz = data_c[17];

		camera_ = PinHoleCamera(q, v, focal, focal, cx, cy, k1, k2, k3, t1, t2,dx,dy);
		float buff_lens_mat[16];
		float buff_lens_param[10];
		SmallSquareMatrix<double,4> sm;
		sm = camera_.m;
		sm.Transpose();
		for(int i = 0; i < 16; i++)
			buff_lens_mat[i] = ((double*)&(sm))[i];

		buff_lens_param[0] = camera_.alpha;
		buff_lens_param[1] = camera_.cx;
		buff_lens_param[2] = camera_.cy;
		buff_lens_param[3] = camera_.k1;
		buff_lens_param[4] = camera_.k2;
		buff_lens_param[5] = camera_.k3;
		buff_lens_param[6] = camera_.t1;
		buff_lens_param[7] = camera_.t2;
		buff_lens_param[8] = camera_.dist_center_shift_x;
		buff_lens_param[9] = camera_.dist_center_shift_y;
		open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->lens_matrix_, true, 0, 16*sizeof(float), buff_lens_mat);
		open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->lens_param_, true, 0, 10*sizeof(float), buff_lens_param);
		open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->laser_cal_,true,0,lines_per_image_*n_images_*12*sizeof(float),data_c+18);

		focusDistance_ = focusDistance;

		if (scannerType == ScannerType::Planscan) {
			z_depth_start_ = z_depth_start_planscan_;
			z_depth_end_ = z_depth_end_planscan_;
		}
		else {
			z_depth_start_ = z_depth_start_emerald_;
			z_depth_end_ = z_depth_end_emerald_;
		}


		scannerType_ = scannerType;

		ConnectKernelIO();


	}

    void Save_buffer_to_disk(int width, int height, int n_images, OpenCLManager* open_cl_manager, cl::Buffer buffer, vector<cl::Event> events, std::string path)
    {
        std::vector<float> tmp1_out(width * height * (n_images));
        RangeImage tmp1(width, height);

        open_cl_manager->getQueue().enqueueReadBuffer(buffer, false, 0,
            sizeof(float) * width * height * (n_images), &tmp1_out[0], 0, &events.back());

        for (int i = 0; i < width; i++)
            for (int j = 0, sz = height; j < sz; j++)
                tmp1.value(i, j) = tmp1_out[(j * width + i)];

        tmp1.SaveAsPGM(path);
    }

    void Save_image2d_to_disk(int width, int height, int n_images, OpenCLManager* open_cl_manager, cl::Image2D image, vector<cl::Event> events, std::string path)
    {
        cl::array<cl::size_type, 3> origin{ 0,0,0 }, region;
        region[0] = width;
        region[1] = height;
        region[2] = 1;

        vector<float> tmp_image(width * height * 4);
        events.push_back(cl::Event());
        open_cl_manager->getQueue().enqueueReadImage(image,
            true, origin, region, 0, 0, &tmp_image[0], 0, &events.back());

        RangeImage tmp(width, height);
        for (int i = 0; i < width; i++)
            for (int j = 0, sz = height; j < sz; j++)
                tmp.value(i, j) = tmp_image[(j * width + i) * 4 + 2];

        tmp.SaveAsPGM(path);
    }

    std::shared_ptr<Scan3D> DigitizerCL::ProcessData(const std::shared_ptr<const WandRawData>& wand_raw_data)
    {
        PROFILER_AUTO_FUNC();

        std::lock_guard<std::mutex> lock(mutex_);
        BOOST_LOG_FUNCTION();
        BOOST_LOG_SCOPED_LOGGER_ATTR(log_, "Timer", boost::log::attributes::timer());
        BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "Begin Digitizer Frame " << Scan3D::getNextScanId();

        const unsigned char* data = &(wand_raw_data->digitizer_frames()[0]);

        const unsigned int transpose_block_size = 16;

        //Get the number of images captured for generating one frame. Can be 12 for regular scan settings, 6 for insane settings
        int n_images = wand_raw_data->CurrentDigitizingFrameCounter();
        bool insane_mode = n_images == 6;

        if (wand_raw_data->GetFrameID() & 1)
        {
            kernel_stash_->compute_zs_kernel_.setCalibrationOffset(lines_per_image_ * 6);
        }

        vector<cl::Event> events;
        events.reserve(100);
        vector<std::string> event_descr;
        //copy data to GPU
        events.push_back(cl::Event());

        if (save_debug_data_)
        {
            RangeImage tmp(width_, height_);
            for (int i = 0; i < width_; i++)
                for (int j = 0, sz = height_; j < sz; j++)
                    tmp.value(i, j) = data[(j * width_ + i)];

            tmp.SaveAsPGM("c:/temp/rawData.pgm");
        }
/////////////////////////////////////////////////new segment # 4 //////////////
        //Populate buffers to be sent to GPU   ????  SEGMENT FOUR ######
        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizelCl Copy Images to GPU");
        open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->data_in_, false, 0, width_ * height_ * n_images * sizeof(unsigned char), data, 0, &events.back());
        event_descr.push_back("Copy images to GPU");

        kernel_stash_->profiler_sampling_kernel_.execute_end(); // Kernel: DigitizelCl Copy Images to GPU
        std::vector<RangeImage> range_images;

        vector<cl::Event> c2f_ev(1, events.back());

        for (auto k : kernel_stash_->getAllKernels())
        {
            k->resetEvents();
        }

        kernel_stash_->char_to_float_straight_kernel_.execute();

        //Filter data to remove noise.
        //2nd derivative is computed for data in both x and y directions

        //1st derivative
        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizerCL Filter Image");

        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizerCL Filter Image 1st Derivative");

        kernel_stash_->filter_y_kernel_.execute();
        kernel_stash_->transpose_images_kernel_.execute(false);
        kernel_stash_->filter_x_kernel_.execute();
        kernel_stash_->transpose_images_kernel_.execute(true);
        kernel_stash_->profiler_sampling_kernel_.execute_end(); // Kernel DigitizerCL Filter Image1

        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizerCL Filter Image 2nd Derivative");

        //2nd Derivative
        kernel_stash_->filter_y_kernel_.execute();
        kernel_stash_->transpose_images_kernel_.execute(false);
        kernel_stash_->filter_x_kernel_.execute();
        kernel_stash_->transpose_images_kernel_.execute(true);

        kernel_stash_->profiler_sampling_kernel_.execute_end(); // Kernel: DigitizerCL Filter Image 2
        kernel_stash_->profiler_sampling_kernel_.execute_end(); // Kernel: DigitizerCL Filter Image

        if (save_debug_data_)
        {
            Save_buffer_to_disk(width_, height_, n_images_, open_cl_manager_.get(),
                open_cl_implementation->data_tmp1_, events, "C:/temp/convolution.pgm");
        }

        //Non maxima suppression is like threshold filtering. It rejects values smaller than a specified value.
        kernel_stash_->non_maxima_supression_kernel_.execute_after_preceding({ events.back() });

        if (save_debug_data_)
        {
            Save_buffer_to_disk(width_, height_, n_images_, open_cl_manager_.get(),
                open_cl_implementation->data_tmp2_, events, "C:/temp/rawData_non_maxima.pgm");
        }


        //Centroids are calculated from the filtered data obtained from the above executions.
        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizerCL copy centroids counts to CPU");

        std::vector<unsigned int> centroids_count(width_ * n_images);
        events.push_back(cl::Event());
        open_cl_manager_->getQueue().enqueueReadBuffer(open_cl_implementation->centroid_count_, false, 0, width_ * n_images * sizeof(int), &centroids_count[0], 0, &events.back());
        if (cl_profiling_)
            event_descr.push_back("copy centroids counts to CPU");
        auto copy_centroid_count_event = events.back();

        kernel_stash_->profiler_sampling_kernel_.execute_end();

        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizerCL zero_guesses and centroid_marker");

        kernel_stash_->zero_guesses_kernel_.execute_after_preceding({ events.back() });
        kernel_stash_->centroid_marker_kernel_.execute_after_preceding({ events.back() });

        kernel_stash_->profiler_sampling_kernel_.execute_end();

        //copy_centroid_count_event.wait();
        unsigned int centr_num = 0;
        for (int i = 0, sz = width_ * n_images; i < sz; i++)
        {
            unsigned int tmp = centroids_count[i];
            centroids_count[i] = centr_num;
            centr_num += tmp;
        }

        BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "CPU centroids number is " << centr_num;

        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizerCL copy CPU centroids counts to GPU");

        events.push_back(cl::Event());
        open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->centroid_indxs_, false, 0, width_ * n_images * sizeof(int), &centroids_count[0], 0, &events.back());
        auto copy_cpu_centroid_count_to_gpu = events.back();

        if (cl_profiling_)
            event_descr.push_back("copy CPU centroids counts to GPU");

        kernel_stash_->profiler_sampling_kernel_.execute_end();

        if (centr_num < 100) {
            auto scan = std::make_shared<Scan3D>(Scan3D::POOR_DATA);

            BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "Insuficient centroid count, aborting digitizer";
            BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "End Digitizer Frame " << scan->getScanId();

            return scan;
        }

        //reallocate buffers
        if (data_xyz_size_ < centr_num)
        {
            size_t initialSize = data_xyz_size_;

            if (data_xyz_size_ == 0)
                data_xyz_size_ = 1;
            while (data_xyz_size_ < centr_num)
                data_xyz_size_ *= 2;

            BOOST_LOG_SEV(log_, Logger::severity_level::info) << "Resizing centroid data buffers (" << initialSize << " to " << data_xyz_size_ << ")";

            auto c = open_cl_manager_->getContext();

            open_cl_implementation->data_x_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float) * data_xyz_size_);
            open_cl_implementation->data_y_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float) * data_xyz_size_);

            open_cl_implementation->data_x_new_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float) * data_xyz_size_);
            open_cl_implementation->data_y_new_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float) * data_xyz_size_);

            open_cl_implementation->data_z_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float) * data_xyz_size_);
            open_cl_implementation->data_weights_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float) * data_xyz_size_);

            open_cl_implementation->centroid_mask_list_ = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(int) * data_xyz_size_);

            ConnectKernelIO();
        }
        vector<cl::Event> cntr_indx(1, copy_cpu_centroid_count_to_gpu);

        kernel_stash_->copy_centroids_kernel_.setPredecessorEvents(cntr_indx);
        kernel_stash_->copy_centroids_kernel_.execute_after_preceding({ events.back() });

        //centroid filtering
        if (centroid_filter_parameters_.radius_ > 0)
        {
            kernel_stash_->clear_mask_kernel_.execute_after_preceding({ events.back() });
            kernel_stash_->create_centroid_mask_kernel_.execute_after_preceding({ events.back() });
            kernel_stash_->centroid_filter_kernel_.execute_after_preceding({ events.back() });
        }

        std::vector<float> old_xs(centr_num), old_ys(centr_num), old_numbers(centr_num), old_weights(centr_num);
        std::vector<float> new_xs(centr_num), new_ys(centr_num), new_zs(centr_num);
        std::vector<unsigned char> centroid_mask_image_cpu_container(width_ * height_ * n_images);

        //Compute z points for every detected centroid
        //z calculation consists of the following steps:
        //Use lens equation to compute the true location of each centroid in x-y plane
        //Back project from each centroid towards the camera
        //Compute ray intersections with calibration planes(these planes pass through scan lines and might be curved. Planes are generated during calibration)
        //Convert computed intersection points from image space to camera space.

        centr_num = centroids_count[width_ * n_images - 1];
        kernel_stash_->compute_zs_kernel_.setCentroidCount(centr_num); // do not include last image (it used only for motion detection)
        kernel_stash_->compute_zs_kernel_.execute_after_preceding({ events.back() });

        kernel_stash_->init_range_image_kernel_.execute_after_preceding({ events.back() });
        kernel_stash_->init_range_image_anisotropic_kernel_.execute_after_preceding({ events.back() });

        kernel_stash_->copy_to_range_image_kernel_.setCentroidNum(centr_num);
        kernel_stash_->copy_to_range_image_kernel_.execute_after_preceding({ events.back() });

        kernel_stash_->copy_to_range_image_anisotropic_kernel_.execute_after_preceding({ events.back() });

        if (save_debug_data_)
        {
            Save_image2d_to_disk(width_, height_ / 2, 1, open_cl_manager_.get(),
                open_cl_implementation->image_range_anisotropic_, events, "c:/temp/anisotropic.pgm");
        }

        kernel_stash_->noise_median_filter_anisotropic_kernel_.execute_after_preceding({ events.back() });

        if (save_debug_data_)
        {
            Save_image2d_to_disk(width_, height_ / 2, 1, open_cl_manager_.get(),
                open_cl_implementation->image_range_median_anisotropic_, events, "c:/temp/median_anisotropic.pgm");
        }

        //Filter the obtained points to remove outliers.
        //These filters are used to remove artifacts like feathering.
        //More can be learned about them in Canary knowledge transfer videos (Part 1 - 3)
        if (centroid_filter_parameters_.use_mean_filter_)
        {
            kernel_stash_->mean_filter_anisotropic_kernel_.execute_after_preceding({ events.back() });
        }
        else
        {
            kernel_stash_->bilateral_filter_anisotropic_kernel_.execute_after_preceding({ events.back() });
        }
                 ////////////// SEGMENT FIVE/////////
        //Compute normals for the generated scan
       //The generated scan is just for one frame. BADSLAM is used to merge it with the cumulated point cloud.
        kernel_stash_->normals_computation_kernel_.execute_after_preceding({ events.back() });


        if( exclude_blue ){
            const unsigned char* color_data = &(wand_raw_data->live_view_frame()[0]);

            // Copy raw color data to GPU.
            kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel: DigitizelCl Copy Live View Images to GPU");
            open_cl_manager_->getQueue().enqueueWriteBuffer(open_cl_implementation->color_data_in_, false, 0, width_ * height_ * 3 * sizeof(unsigned char), color_data, 0, &events.back());
            event_descr.push_back("Copy color images to GPU");
            kernel_stash_->profiler_sampling_kernel_.execute_end();

            // Create image from color data.
            kernel_stash_->copy_color_data_to_image_.execute();

            // Zero the color mask
            kernel_stash_->mask_zero_.execute();

            // Create the new mask
            kernel_stash_->create_mask_.execute();

            // Apply the mask
            kernel_stash_->mask_apply_.execute();

            if (save_debug_data_)
            {
                cl::array<cl::size_type, 3> origin{ 0,0,0 }, region;
                region[0] = width_ / 2;
                region[1] = height_ / 2;
                region[2] = 1;

                vector<float> tmp_image(width_ / 2 * height_ / 2 * 4);
                events.push_back(cl::Event());
                open_cl_manager_->getQueue().enqueueReadImage(open_cl_implementation->color_image_, true, origin, region, 0, 0, &tmp_image[0], 0, &events.back());

                vector<int> mask(width_ / 2 * height_ / 2);
                open_cl_manager_->getQueue().enqueueReadBuffer(open_cl_implementation->color_mask_, false, 0, sizeof(unsigned int)*width_ / 2 *height_ / 2, &mask[0], 0, &events.back());

                vector<float> tmp_image_bilateral(width_ / 2 * height_ / 2 * 4);
                events.push_back(cl::Event());
                open_cl_manager_->getQueue().enqueueReadImage(open_cl_implementation->image_range_bilateral_, true, origin, region, 0, 0, &tmp_image_bilateral[0], 0, &events.back());

                {
                    ofstream fout("c:/temp/color_raw", ios::out | ios::binary);
                    fout.write((char*)color_data, wand_raw_data->live_view_frame().size() * sizeof(unsigned char));
                    fout.close();
                }
                {
                    ofstream fout("c:/temp/color", ios::out | ios::binary);
                    fout.write((char*)&tmp_image[0], tmp_image.size() * sizeof(float));
                    fout.close();
                }
                {
                    ofstream fout("c:/temp/color_mask", ios::out | ios::binary);
                    fout.write((char*)&mask[0], mask.size() * sizeof(unsigned int));
                    fout.close();
                }
                {
                    ofstream fout("c:/temp/pos_image", ios::out | ios::binary);
                    fout.write((char*)&tmp_image_bilateral[0], tmp_image_bilateral.size() * sizeof(float));
                    fout.close();
                }
            }
        }
//////////////////////////////////new segment #5 //////////////////////////////

        cl::array<cl::size_type, 3> origin{ 0,0,0 }, region;
        region[0] = width_ / step_;
        region[1] = height_ / step_;
        region[2] = 1;

        //Read depth buffer from GPU to run region filtering on CPU.
        //Run region remove filter before reading into these bufffers
        points_image_.resize(width_ * height_ * 4 / (step_ * step_));
        normal_image_.resize(width_ * height_ * 4 / (step_ * step_));

        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel DigitizerCL Copy Image to Cpu 2");
        events.push_back(cl::Event());
        open_cl_manager_->getQueue().enqueueReadImage(open_cl_implementation->image_range_bilateral_, true, origin, region, 0, 0, points_image_.data(), 0, &events.back());
        if (cl_profiling_)
            event_descr.push_back("copy points image to CPU");

        kernel_stash_->profiler_sampling_kernel_.execute_end();

        kernel_stash_->profiler_sampling_kernel_.execute_begin("Kernel DigitizerCL Copy Normals to Cpu 2");

        events.push_back(cl::Event());
        open_cl_manager_->getQueue().enqueueReadImage(open_cl_implementation->image_normals_, true, origin, region, 0, 0, normal_image_.data(), 0, &events.back());
        if (cl_profiling_)
            event_descr.push_back("copy normals image to CPU");

        kernel_stash_->profiler_sampling_kernel_.execute_end(); // CopyImagesToCPU

        events.back().wait();
        (events.end() - 2)->wait();

        int i = 0;
        size_t total_gpu_time = 0;
        if (cl_profiling_)
        {
            for (vector<cl::Event>::iterator it = events.begin(); it != events.end(); i++, ++it)
            {
                BOOST_LOG_SEV(gpu_prof_log_, Logger::severity_level::debug) << "Event " << i << " " << event_descr[i] << " time " << (it->getProfilingInfo<CL_PROFILING_COMMAND_END>() - it->getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 10e-6;
                total_gpu_time += (it->getProfilingInfo<CL_PROFILING_COMMAND_END>() - it->getProfilingInfo<CL_PROFILING_COMMAND_START>());
            }
            BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "Getting  GPU profiling info time ";
            BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "Total GPU time " << total_gpu_time * 10e-6;
        }

        BOOST_LOG_SEV(log_, Logger::severity_level::trace) << "GPU processing done";

        range_tmp_.SetDimentions(width_ / step_, height_ / step_);

        for (int i = 0; i < (width_ / step_) * (height_ / step_); i++)
        {
            const auto& z = points_image_[i * 4 + 2];

            const auto& nx = normal_image_[i * 4 + 0];

            //If normal is NaN, continue.
            if (nx != nx  )
                continue;

            range_tmp_.value(i) = z;
        }

        //NOTE: This method needs to be moved to GPU. It currently runs on CPU and takes up half of the frame processing time.
        //It filters out small islands of scan data that might not contribute to the final scan

        if (run_region_filter) {
            range_tmp_.NoiseRegionsRemoveFilter(remove_region_filter_depth, 300);
        }

        BOOST_LOG_SEV(log_, Logger::severity_level::trace) << "Done noise region filter";

        auto startTime = std::chrono::high_resolution_clock::now();

        points_.clear();
        colors_.clear();

        grid_.resize((width_ / step_) * (height_ / step_), -1);
        int n_pts(0);
        int k(0);

        //Saving color data from the scanned image(x-y space)
        float maxDist = std::min(height_ * 0.5f / step_ * height_ * 0.5f / step_, width_ * 0.5f / step_ * width_ * 0.5f / step_);
        float smallestValueAtBorder = 0.01f;
        //e^(-fallOff * maxdist) = smallestValueAtBorder
        float fallOff = log(smallestValueAtBorder) / (-maxDist);

        //Camera world coordinates assumed to be at origin
        std::vector<int> cameraPosition = { 0, 0, 0 };

        //Rotating the camera direction vector by the angle to get lighting direction
        float rotation_angle = 6.6;

        float sin_rotation_angle = sin(rotation_angle);
        float cos_rotation_angle = cos(rotation_angle);

        float dim_factor = 0.874;

        for (int j = 0; j < height_ / step_; j++)
        {
            //float dj = j - height_ / step_ * 0.5;
            for (int i = 0; i < width_ / step_; k++, i++)
            {
                //float di = i - width_ * 0.5f / step_;
                int res = -1;
                if (range_tmp_.value(k) != range_tmp_.value(k))
                {
                }
                else
                {
                    const auto& x = points_image_[k * 4 + 0];
                    const auto& y = points_image_[k * 4 + 1];
                    const auto& z = points_image_[k * 4 + 2];

                    //normals are already normalized
                    float nx = normal_image_[k * 4 + 0];
                    float ny = normal_image_[k * 4 + 1];
                    float nz = normal_image_[k * 4 + 2];

                    if (
                        (x == x) &&
                        (y == y) &&
                        (z == z) &&
                        (nx == nx) &&
                        (ny == ny) &&
                        (nz == nz)
                        )
                    {
                        res = points_.size();
                        float color_b = (wand_raw_data->live_view_frame())[((height_ - j * step_ - 1) * width_ + i * step_) * 3] / 255.f;
                        float color_g = (wand_raw_data->live_view_frame())[((height_ - j * step_ - 1) * width_ + i * step_) * 3 + 1] / 255.f;
                        float color_r = (wand_raw_data->live_view_frame())[((height_ - j * step_ - 1) * width_ + i * step_) * 3 + 2] / 255.f;

                        if (enable_color_correction_)
                        {
                            //Camera direction (Viewing direction) vector assuming camera at origin
                            float viewing_x = (cameraPosition[0] - x);
                            float viewing_y = (cameraPosition[1] - y);
                            float viewing_z = (cameraPosition[2] - z);

                            float norm_v = sqrt((viewing_x * viewing_x) + (viewing_y * viewing_y) + (viewing_z * viewing_z));
                            viewing_x /= norm_v;
                            viewing_y /= norm_v;
                            viewing_z /= norm_v;

                            //Rotating the camera direction vector to get lighting direction
                            float lighting_x = (viewing_x * cos_rotation_angle) - (viewing_y * sin_rotation_angle);
                            float lighting_y = (viewing_y * sin_rotation_angle) + (viewing_y * cos_rotation_angle);
                            float lighting_z = viewing_z;

                            float norm_l = sqrt((lighting_x * lighting_x) + (lighting_y * lighting_y) + (lighting_z * lighting_z));
                            lighting_x /= norm_l;
                            lighting_y /= norm_l;
                            lighting_z /= norm_l;

                            //Dot product between the lighting-viewing direction and normal
                            float n_dot_l = (lighting_x * nx) + (lighting_y * ny) + (lighting_z * nz);
                            float n_dot_v = (viewing_x * nx) + (viewing_y * ny) + (viewing_z * nz);

                            float fresnel_factor = 0.7;
                            float color_corrector = ((n_dot_l * n_dot_v) + (1 - (n_dot_l * n_dot_v))) / 2;

                            if (color_corrector < 0.3) { color_corrector = 0.3; }

                            //Color correction
                            color_b = (color_b * dim_factor * fresnel_factor) / color_corrector;
                            color_g = (color_g * dim_factor * fresnel_factor) / color_corrector;
                            color_r = (color_r * dim_factor * fresnel_factor) / color_corrector;

                            if (color_b > 1.f) { color_b = 1.f; }
                            if (color_g > 1.f) { color_g = 1.f; }
                            if (color_r > 1.f) { color_r = 1.f; }
                        }

                        colors_.emplace_back(color_b, color_g, color_r);

                        float weight = 1.0; // (color_r + color_g + color_b) / 3.0;

                        if (focal_origin_)
                            points_.emplace_back(DummyPointFloat(x, y, z - focusDistance_), DummyPointFloat(nx, ny, nz), weight);
                        else
                            points_.emplace_back(DummyPointFloat(x, y, z), DummyPointFloat(nx, ny, nz), weight);
                        //colors_.emplace_back(weight, weight, weight);
                    }
                }
                grid_[k] = res;
            }
        }

        BOOST_LOG_SEV(log_, Logger::severity_level::trace) << "End color correction filter";

        //Copy data from GPU buffers back to CPU and return the processed (filtered)
        //point cloud data to be used for BADSLAM
        //Reject data if scanned points are less than the specified value
        std::shared_ptr<Scan3D> scan;
        {
            if (points_.size() < minimum_scan_points_)
            {
                scan = std::make_shared<Scan3D>(Scan3D::POOR_DATA);
                BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "Insuficient point count, aborting digitizer";
            }
            else
            {
                scan = std::make_shared<Scan3D>(width_ / step_, height_ / step_, grid_, points_, colors_);
                scan->set_lens_calibration(lens_calibration());
                scan->set_focus_distance(focusDistance_);
                scan->set_z_depth(z_depth_start_ - focusDistance_, z_depth_end_ - focusDistance_);
            }
            //TOD: need only live view frame, rest is only for debug
            scan->set_raw_data(wand_raw_data);

            if (save_debug_data_)
            {
                scan->set_centroid_indx(centroids_count);
                scan->set_centroids_xs(old_xs);
                scan->set_centroids_ys(old_ys);
                scan->set_centroids_nums(old_numbers);
                if (centroid_filter_parameters_.radius_ > 0)
                {
                    scan->set_centroids_mask_image(centroid_mask_image_cpu_container);
                }
                scan->set_centroids_3d_xs(new_xs);
                scan->set_centroids_3d_ys(new_ys);
                scan->set_centroids_3d_zs(new_zs);
            }
            scan->set_raw_data(wand_raw_data);
        }
        std::array<unsigned short, 4> crop_vals;
        crop_vals[0] = x_crop_start_;
        crop_vals[1] = x_crop_end_;
        crop_vals[2] = y_crop_start_;
        crop_vals[3] = y_crop_end_;
        scan->setCropValues(crop_vals);

        BOOST_LOG_SEV(log_, Logger::severity_level::debug) << "End Digitizer Frame " << scan->getScanId();

        return scan;
    }

    //Generate matrix for mapping distorted to undistorted data
    //This function is called in ScanningProcess before starting the scanning loop
    void DigitizerCL::CreateUndistortionData() {
        auto c = open_cl_manager_->getContext();
        auto& o = open_cl_implementation;
        auto& kernels = open_cl_manager_->getKernels();

        o->undistorted_x = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)* width_ * height_);
        o->undistorted_y = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)* height_ * width_);
        o->back_projected_x_0 = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)* height_ * width_);
        o->back_projected_y_0 = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)* height_ * width_);
        o->back_projected_x_10 = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)* height_ * width_);
        o->back_projected_y_10 = cl::Buffer(*c, CL_MEM_READ_WRITE, sizeof(float)* height_ * width_);

        FrameInfo frame_info(width_, height_, n_images_, x_crop_start_, x_crop_end_, y_crop_start_, y_crop_end_);

        CommonKernelInfo common_kernel_info;
        common_kernel_info.cl_queue_ = &(open_cl_manager_->getQueue());
        common_kernel_info.cl_context_ = open_cl_manager_->getContext();
        common_kernel_info.cl_profiling_ = cl_profiling_;
        common_kernel_info.kernels_ = kernels;
        common_kernel_info.frame_info_ = frame_info;

        {
            UndistortionKernel::KernelInput k;
            k.centr_num_ = 0;
            k.undistorted_x = o->undistorted_x;
            k.undistorted_y = o->undistorted_y;
            k.lens_param_ = o->lens_param_;
            k.lens_matrix_ = o->lens_matrix_;
            k.back_projected_x_0 = o->back_projected_x_0;
            k.back_projected_y_0 = o->back_projected_y_0;
            k.back_projected_x_10 = o->back_projected_x_10;
            k.back_projected_y_10 = o->back_projected_y_10;
            k.implement_LUT = implement_LUT ? 1 : 0;
            kernel_stash_->undistortion_kernel = UndistortionKernel(common_kernel_info, k);
        }

        kernel_stash_->undistortion_kernel.setCentroidCount(width_ * height_);
        kernel_stash_->undistortion_kernel.execute();
    }
}
