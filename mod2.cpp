std::vector<int> cameraPosition = { 0, 0, 0 };

       //Rotating the camera direction vector by this angle to get lighting direction
       float rotation_angle = 6.6;

       float sin_rotation_angle = sin(rotation_angle);
       float cos_rotation_angle = cos(rotation_angle);

       float dim_factor = 0.750;

       float edge_weight_ = 0.5; // Modify edge_weight_ as needed
       float log_sigma_ = 2.0; // Modify log_sigma_ as needed

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

                       //Normalizing the normals
                       float norm_n = sqrt((nx * nx) + (ny * ny) + (nz * nz));
                       nx /= norm_n;
                       ny /= norm_n;
                       nz /= norm_n;

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

                       float n_dot_l_v = n_dot_l * n_dot_v;
                       float color_corrector = (n_dot_l_v + (1.0 - n_dot_l_v)) / 2.0;

                       float color_corrector = (n_dot_l_v + (1.0 - n_dot_l_v)) / 2.0;

                       float log_weight = exp(-abs(log(range_tmp_.value(k))) / (2.0 * log_sigma_ * log_sigma_));
                       color_corrector = (1.0 - edge_weight_) * color_corrector + edge_weight_ * log_weight;


                       //   float color_corrector = ((n_dot_l * n_dot_v) + (1 - (n_dot_l * n_dot_v))) / 2;
                      //   color_corrector = std::max(color_corrector, 0.3f);
                        // This uses the std::max function to set color_corrector to the maximum of its current value and 0.3.
                        // If color_corrector is already greater than 0.3, it remains unchanged. Otherwise, it is set to 0.3.
                        // This is equivalent to the original code's behavior of clamping color_corrector to a minimum of 0.3.
                       if (color_corrector < 0.3) { color_corrector = 0.3; }

                       //Color correction
                       color_b = (color_b * dim_factor * fresnel_factor) / color_corrector;
                       color_g = (color_g * dim_factor * fresnel_factor) / color_corrector;
                       color_r = (color_r * dim_factor * fresnel_factor) / color_corrector;

                       if (color_b > 1.f) { color_b = 1.f; }
                       if (color_g > 1.f) { color_g = 1.f; }
                       if (color_r > 1.f) { color_r = 1.f; }

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
