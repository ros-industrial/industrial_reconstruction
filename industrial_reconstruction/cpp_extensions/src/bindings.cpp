#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <opencv2/opencv.hpp>

#include "image_buffer.h"
#include "pose_calculator.h"
#include "memory_manager.h"

namespace py = pybind11;

// Helper function to convert numpy array to cv::Mat
cv::Mat numpy_to_cv_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 2 && buf_info.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be 2 or 3");
    }
    
    int type;
    if (buf_info.ndim == 2) {
        type = CV_8UC1;
    } else if (buf_info.ndim == 3 && buf_info.shape[2] == 3) {
        type = CV_8UC3;
    } else if (buf_info.ndim == 3 && buf_info.shape[2] == 1) {
        type = CV_8UC1;
    } else {
        throw std::runtime_error("Unsupported image format");
    }
    
    return cv::Mat(buf_info.shape[0], buf_info.shape[1], type, (unsigned char*)buf_info.ptr);
}

// Helper function to convert cv::Mat to numpy array
py::array_t<uint8_t> cv_mat_to_numpy(const cv::Mat& mat) {
    if (mat.channels() == 1) {
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols},
            {sizeof(uint8_t) * mat.cols, sizeof(uint8_t)},
            mat.data
        );
    } else if (mat.channels() == 3) {
        return py::array_t<uint8_t>(
            {mat.rows, mat.cols, 3},
            {sizeof(uint8_t) * mat.cols * 3, sizeof(uint8_t) * 3, sizeof(uint8_t)},
            mat.data
        );
    } else {
        throw std::runtime_error("Unsupported number of channels");
    }
}

PYBIND11_MODULE(industrial_reconstruction_cpp, m) {
    m.doc() = "Industrial Reconstruction C++ Extensions";
    
    // ImageBuffer bindings
    py::class_<industrial_reconstruction::ImageBuffer>(m, "ImageBuffer")
        .def(py::init<size_t>(), py::arg("max_size") = 100)
        .def("push", [](industrial_reconstruction::ImageBuffer& self, 
                       py::array_t<uint8_t> depth_img, 
                       py::array_t<uint8_t> color_img, 
                       double timestamp) {
            cv::Mat depth_mat = numpy_to_cv_mat(depth_img);
            cv::Mat color_mat = numpy_to_cv_mat(color_img);
            auto time_point = std::chrono::time_point<std::chrono::high_resolution_clock>(
                std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
                    std::chrono::duration<double>(timestamp)));
            self.push(depth_mat, color_mat, time_point);
        })
        .def("pop", [](industrial_reconstruction::ImageBuffer& self, int timeout_ms) -> py::tuple {
            industrial_reconstruction::ImageData data;
            bool success = self.pop(data, timeout_ms);
            if (success) {
                return py::make_tuple(
                    true,
                    cv_mat_to_numpy(data.depth_image),
                    cv_mat_to_numpy(data.color_image),
                    data.timestamp
                );
            } else {
                return py::make_tuple(false, py::none(), py::none(), py::none());
            }
        })
        .def("try_pop", [](industrial_reconstruction::ImageBuffer& self) -> py::tuple {
            industrial_reconstruction::ImageData data;
            bool success = self.try_pop(data);
            if (success) {
                return py::make_tuple(
                    true,
                    cv_mat_to_numpy(data.depth_image),
                    cv_mat_to_numpy(data.color_image),
                    data.timestamp
                );
            } else {
                return py::make_tuple(false, py::none(), py::none(), py::none());
            }
        })
        .def("clear", &industrial_reconstruction::ImageBuffer::clear)
        .def("size", &industrial_reconstruction::ImageBuffer::size)
        .def("empty", &industrial_reconstruction::ImageBuffer::empty)
        .def("full", &industrial_reconstruction::ImageBuffer::full)
        .def("set_max_size", &industrial_reconstruction::ImageBuffer::set_max_size)
        .def("enable_compression", &industrial_reconstruction::ImageBuffer::enable_compression);
    
    // TransformData bindings
    py::class_<industrial_reconstruction::TransformData>(m, "TransformData")
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3d&, const Eigen::Quaterniond&, double>())
        .def_property("translation", 
                     [](const industrial_reconstruction::TransformData& self) -> py::array_t<double> {
                         return py::array_t<double>({3}, self.translation.data());
                     },
                     [](industrial_reconstruction::TransformData& self, py::array_t<double> value) {
                         py::buffer_info buf = value.request();
                         if (buf.ndim != 1 || buf.shape[0] != 3) {
                             throw std::runtime_error("Translation must be a 3-element array");
                         }
                         self.translation = Eigen::Vector3d(static_cast<double*>(buf.ptr));
                     })
        .def_property("rotation",
                     [](const industrial_reconstruction::TransformData& self) -> py::array_t<double> {
                         return py::array_t<double>({4}, self.rotation.coeffs().data());
                     },
                     [](industrial_reconstruction::TransformData& self, py::array_t<double> value) {
                         py::buffer_info buf = value.request();
                         if (buf.ndim != 1 || buf.shape[0] != 4) {
                             throw std::runtime_error("Rotation must be a 4-element quaternion array");
                         }
                         self.rotation = Eigen::Quaterniond(static_cast<double*>(buf.ptr));
                     })
        .def_readwrite("timestamp", &industrial_reconstruction::TransformData::timestamp);
    
    // PoseCalculator bindings
    py::class_<industrial_reconstruction::PoseCalculator>(m, "PoseCalculator")
        .def(py::init<>())
        .def("create_transformation_matrix", 
             [](industrial_reconstruction::PoseCalculator& self,
                py::array_t<double> translation_array,
                py::array_t<double> rotation_array) -> py::array_t<double> {
            // Convert numpy arrays to Eigen types
            py::buffer_info trans_buf = translation_array.request();
            py::buffer_info rot_buf = rotation_array.request();
            
            if (trans_buf.ndim != 1 || trans_buf.shape[0] != 3) {
                throw std::runtime_error("Translation must be a 3-element array");
            }
            if (rot_buf.ndim != 1 || rot_buf.shape[0] != 4) {
                throw std::runtime_error("Rotation must be a 4-element quaternion array");
            }
            
            Eigen::Vector3d translation(static_cast<double*>(trans_buf.ptr));
            Eigen::Quaterniond rotation(static_cast<double*>(rot_buf.ptr));
            
            Eigen::Matrix4d transform = self.create_transformation_matrix(translation, rotation);
            return py::array_t<double>({4, 4}, transform.data());
        })
        .def("invert_transformation",
             [](industrial_reconstruction::PoseCalculator& self,
                py::array_t<double> transform_array) -> py::array_t<double> {
            py::buffer_info buf = transform_array.request();
            if (buf.ndim != 2 || buf.shape[0] != 4 || buf.shape[1] != 4) {
                throw std::runtime_error("Transform must be a 4x4 matrix");
            }
            Eigen::Matrix4d transform;
            std::memcpy(transform.data(), buf.ptr, sizeof(double) * 16);
            Eigen::Matrix4d inverse = self.invert_transformation(transform);
            return py::array_t<double>({4, 4}, inverse.data());
        })
        .def("calculate_translation_distance", 
             [](industrial_reconstruction::PoseCalculator& self,
                py::array_t<double> pos1_array,
                py::array_t<double> pos2_array) -> double {
            py::buffer_info buf1 = pos1_array.request();
            py::buffer_info buf2 = pos2_array.request();
            
            if (buf1.ndim != 1 || buf1.shape[0] != 3 || buf2.ndim != 1 || buf2.shape[0] != 3) {
                throw std::runtime_error("Positions must be 3-element arrays");
            }
            
            Eigen::Vector3d pos1(static_cast<double*>(buf1.ptr));
            Eigen::Vector3d pos2(static_cast<double*>(buf2.ptr));
            
            return self.calculate_translation_distance(pos1, pos2);
        })
        .def("calculate_rotation_distance", 
             [](industrial_reconstruction::PoseCalculator& self,
                py::array_t<double> rot1_array,
                py::array_t<double> rot2_array) -> double {
            py::buffer_info buf1 = rot1_array.request();
            py::buffer_info buf2 = rot2_array.request();
            
            if (buf1.ndim != 1 || buf1.shape[0] != 4 || buf2.ndim != 1 || buf2.shape[0] != 4) {
                throw std::runtime_error("Rotations must be 4-element quaternion arrays");
            }
            
            Eigen::Quaterniond rot1(static_cast<double*>(buf1.ptr));
            Eigen::Quaterniond rot2(static_cast<double*>(buf2.ptr));
            
            return self.calculate_rotation_distance(rot1, rot2);
        })
        .def("should_process_pose", &industrial_reconstruction::PoseCalculator::should_process_pose)
        .def("interpolate_poses", &industrial_reconstruction::PoseCalculator::interpolate_poses)
        .def("process_pose_sequence",
             [](industrial_reconstruction::PoseCalculator& self,
                const std::vector<industrial_reconstruction::TransformData>& poses,
                double translation_threshold,
                double rotation_threshold) -> py::list {
            auto transforms = self.process_pose_sequence(poses, translation_threshold, rotation_threshold);
            py::list result;
            for (const auto& transform : transforms) {
                result.append(py::array_t<double>({4, 4}, transform.data()));
            }
            return result;
        })
        .def("add_pose", &industrial_reconstruction::PoseCalculator::add_pose)
        .def("get_pose_at_index", &industrial_reconstruction::PoseCalculator::get_pose_at_index)
        .def("get_pose_count", &industrial_reconstruction::PoseCalculator::get_pose_count)
        .def("clear_poses", &industrial_reconstruction::PoseCalculator::clear_poses);
    
    // ImageMemoryManager bindings
    py::class_<industrial_reconstruction::ImageMemoryManager>(m, "ImageMemoryManager")
        .def(py::init<>())
        .def("allocate_image", 
             [](industrial_reconstruction::ImageMemoryManager& self, int rows, int cols, int type) -> py::array_t<uint8_t> {
            cv::Mat image = self.allocate_image(rows, cols, type);
            return cv_mat_to_numpy(image);
        })
        .def("get_temp_depth_image", 
             [](industrial_reconstruction::ImageMemoryManager& self) -> py::array_t<uint8_t> {
            cv::Mat image = self.get_temp_depth_image();
            return cv_mat_to_numpy(image);
        })
        .def("get_temp_color_image",
             [](industrial_reconstruction::ImageMemoryManager& self) -> py::array_t<uint8_t> {
            cv::Mat image = self.get_temp_color_image();
            return cv_mat_to_numpy(image);
        })
        .def("return_temp_image", 
             [](industrial_reconstruction::ImageMemoryManager& self, py::array_t<uint8_t> image_array) {
            cv::Mat image = numpy_to_cv_mat(image_array);
            self.return_temp_image(image);
        })
        .def("get_allocated_memory", &industrial_reconstruction::ImageMemoryManager::get_allocated_memory)
        .def("get_pool_memory", &industrial_reconstruction::ImageMemoryManager::get_pool_memory)
        .def("print_memory_stats", &industrial_reconstruction::ImageMemoryManager::print_memory_stats)
        .def("preallocate_buffers", &industrial_reconstruction::ImageMemoryManager::preallocate_buffers)
        .def("clear_unused_buffers", &industrial_reconstruction::ImageMemoryManager::clear_unused_buffers);
    
    // ScopedImageBuffer bindings
    py::class_<industrial_reconstruction::ScopedImageBuffer>(m, "ScopedImageBuffer")
        .def(py::init<industrial_reconstruction::ImageMemoryManager&, int, int, int>())
        .def("get", 
             [](industrial_reconstruction::ScopedImageBuffer& self) -> py::array_t<uint8_t> {
            return cv_mat_to_numpy(self.get());
        });
    
    // GlobalMemoryManager bindings
    py::class_<industrial_reconstruction::GlobalMemoryManager>(m, "GlobalMemoryManager")
        .def_static("get_image_manager", &industrial_reconstruction::GlobalMemoryManager::get_image_manager,
                   py::return_value_policy::reference)
        .def_static("initialize", &industrial_reconstruction::GlobalMemoryManager::initialize)
        .def_static("cleanup", &industrial_reconstruction::GlobalMemoryManager::cleanup);
    
    // Utility functions
    m.def("numpy_to_cv_mat", &numpy_to_cv_mat, "Convert numpy array to OpenCV Mat");
    m.def("cv_mat_to_numpy", &cv_mat_to_numpy, "Convert OpenCV Mat to numpy array");
}
