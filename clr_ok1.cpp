#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

std::vector<int> cameraPosition = { 0, 0, 0 };
float rotation_angle = 6.6;
float sin_rotation_angle = sin(rotation_angle);
float cos_rotation_angle = cos(rotation_angle);
float dim_factor = 0.750;
float edge_weight_ = 0.5; // Modify edge_weight_ as needed
float log_sigma_ = 2.0; // Modify log_sigma_ as needed

int main(int argc, char** argv)
{
    // Load point cloud from file
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>("input_cloud.pcd", *cloud);

    // Apply the modified code to the point cloud
    for (size_t i = 0; i < cloud->size(); ++i)
    {
        const auto& point = cloud->at(i);
        float x = point.x;
        float y = point.y;
        float z = point.z;

        // Normals are already normalized
        float nx = point.normal_x;
        float ny = point.normal_y;
        float nz = point.normal_z;

        // Camera direction (Viewing direction) vector assuming camera at origin
        float viewing_x = (cameraPosition[0] - x);
        float viewing_y = (cameraPosition[1] - y);
        float viewing_z = (cameraPosition[2] - z);

        float norm_v = sqrt((viewing_x * viewing_x) + (viewing_y * viewing_y) + (viewing_z * viewing_z));
        viewing_x /= norm_v;
        viewing_y /= norm_v;
        viewing_z /= norm_v;

        // Rotating the camera direction vector to get lighting direction
        float lighting_x = (viewing_x * cos_rotation_angle) - (viewing_y * sin_rotation_angle);
        float lighting_y = (viewing_y * sin_rotation_angle) + (viewing_y * cos_rotation_angle);
        float lighting_z = viewing_z;

        float norm_l = sqrt((lighting_x * lighting_x) + (lighting_y * lighting_y) + (lighting_z * lighting_z));
        lighting_x /= norm_l;
        lighting_y /= norm_l;
        lighting_z /= norm_l;

        // Dot product between the lighting-viewing direction and normal
        float n_dot_l = (lighting_x * nx) + (lighting_y * ny) + (lighting_z * nz);
        float n_dot_v = (viewing_x * nx) + (viewing_y * ny) + (viewing_z * nz);

        float fresnel_factor = 0.7;

        float n_dot_l_v = n_dot_l * n_dot_v;
        //float color_corrector = (n_dot_l_v + (1.0 - n_dot_l_v)) / 2.0

        float color_corrector = (n_dot_l_v + (1.0 - n_dot_l_v)) / 2.0;

        // Compute Laplacian of Gaussian (LoG) correction factor
        float sigma = 2.0;
        float x2 = x * x + y * y + z * z;
        float LoG = -1.0 / (M_PI * pow(sigma, 4)) * (1 - x2 / (2 * sigma * sigma)) * exp(-x2 / (2 * sigma * sigma));
        float LoG_factor = 1.0 + 0.2 * LoG;


        float color_corrected_b = (color_b * dim_factor * fresnel_factor) / (color_corrector * LoG_factor);
        float color_corrected_g = (color_g * dim_factor * fresnel_factor) / (color_corrector * LoG_factor);
        float color_corrected_r = (color_r * dim_factor * fresnel_factor) / (color_corrector * LoG_factor);

        // Update the point cloud with the corrected color
        point.b = color_corrected_b * 255;
        point.g = color_corrected_g * 255;
        point.r = color_corrected_r * 255;

        // Update the point cloud with the new point
        cloud->at(i) = point;
        }

        // Visualize the point cloud
        pcl::visualization::CloudViewer viewer("Point Cloud Viewer");
        viewer.showCloud(cloud);

        while (!viewer.wasStopped())
        {
        // Do nothing, just wait for the viewer to be stopped
        }

        return 0;
      }
