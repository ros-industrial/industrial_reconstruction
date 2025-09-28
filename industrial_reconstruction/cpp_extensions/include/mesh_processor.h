#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace industrial_reconstruction {

struct MeshQualityMetrics {
    double vertex_count;
    double triangle_count;
    double surface_area;
    double volume;
    double average_edge_length;
    double min_edge_length;
    double max_edge_length;
    double aspect_ratio;
    double dihedral_angle_quality;
    double hole_count;
    double non_manifold_edge_count;
    double quality_score; // Overall quality score 0-1
};

class MeshProcessor {
public:
    MeshProcessor();
    ~MeshProcessor() = default;
    
    // Main processing pipeline
    cv::Mat processMesh(const cv::Mat& raw_mesh_vertices, 
                       const cv::Mat& raw_mesh_triangles,
                       const cv::Mat& raw_mesh_colors = cv::Mat());
    
    // Individual processing steps
    cv::Mat removeOutliers(const cv::Mat& vertices, const cv::Mat& triangles, 
                          double std_dev_threshold = 2.0);
    cv::Mat smoothMesh(const cv::Mat& vertices, const cv::Mat& triangles, 
                      int iterations = 10, double lambda = 0.5);
    cv::Mat fillHoles(const cv::Mat& vertices, const cv::Mat& triangles, 
                     double max_hole_diameter = 0.1);
    cv::Mat decimateMesh(const cv::Mat& vertices, const cv::Mat& triangles, 
                        double reduction_ratio = 0.5);
    cv::Mat remeshUniform(const cv::Mat& vertices, const cv::Mat& triangles, 
                         double target_edge_length = 0.01);
    
    // Quality improvement
    cv::Mat improveMeshQuality(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat fixNonManifoldEdges(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat optimizeTopology(const cv::Mat& vertices, const cv::Mat& triangles);
    
    // Advanced filtering
    cv::Mat bilateralSmoothing(const cv::Mat& vertices, const cv::Mat& triangles, 
                              int iterations = 5, double sigma_s = 0.1, double sigma_r = 0.1);
    cv::Mat curvatureBasedSmoothing(const cv::Mat& vertices, const cv::Mat& triangles, 
                                   int iterations = 10);
    cv::Mat laplacianSmoothing(const cv::Mat& vertices, const cv::Mat& triangles, 
                              int iterations = 20, double lambda = 0.1);
    
    // Quality assessment
    MeshQualityMetrics calculateQualityMetrics(const cv::Mat& vertices, const cv::Mat& triangles);
    double calculateMeshQuality(const cv::Mat& vertices, const cv::Mat& triangles);
    bool isMeshValid(const cv::Mat& vertices, const cv::Mat& triangles, 
                    double min_quality_threshold = 0.6);
    
    // Configuration
    void setSmoothingIterations(int iterations);
    void setSmoothingLambda(double lambda);
    void setOutlierThreshold(double threshold);
    void setQualityThreshold(double threshold);
    void setTargetEdgeLength(double length);
    
    // Advanced processing
    cv::Mat adaptiveProcessing(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat multiResolutionProcessing(const cv::Mat& vertices, const cv::Mat& triangles, 
                                    int num_levels = 3);
    
private:
    // Processing parameters
    int smoothing_iterations_;
    double smoothing_lambda_;
    double outlier_threshold_;
    double quality_threshold_;
    double target_edge_length_;
    
    // Helper functions
    cv::Mat calculateVertexNormals(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat calculateFaceNormals(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat calculateCurvature(const cv::Mat& vertices, const cv::Mat& triangles);
    std::vector<std::vector<int>> buildVertexAdjacency(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat findHoles(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat detectNonManifoldEdges(const cv::Mat& vertices, const cv::Mat& triangles);
    
    // Geometric calculations
    double calculateTriangleArea(const cv::Vec3f& v1, const cv::Vec3f& v2, const cv::Vec3f& v3);
    double calculateEdgeLength(const cv::Vec3f& v1, const cv::Vec3f& v2);
    double calculateAspectRatio(const cv::Vec3f& v1, const cv::Vec3f& v2, const cv::Vec3f& v3);
    double calculateDihedralAngle(const cv::Vec3f& n1, const cv::Vec3f& n2);
    cv::Vec3f calculateTriangleNormal(const cv::Vec3f& v1, const cv::Vec3f& v2, const cv::Vec3f& v3);
    
    // Mesh operations
    cv::Mat subdivideMesh(const cv::Mat& vertices, const cv::Mat& triangles);
    cv::Mat collapseEdges(const cv::Mat& vertices, const cv::Mat& triangles, 
                         const std::vector<int>& edges_to_collapse);
    cv::Mat flipEdges(const cv::Mat& vertices, const cv::Mat& triangles, 
                     const std::vector<int>& edges_to_flip);
};

} // namespace industrial_reconstruction
