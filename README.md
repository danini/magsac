# The MAGSAC algorithm for robust model fitting without using an inlier-outlier threshold

The MAGSAC algorithm proposed in paper: Daniel Barath, Jana Noskova and Jiri Matas; MAGSAC: Marginalizing sample consensus, Conference on Computer Vision and Pattern Recognition, 2019. 
It is available at https://arxiv.org/pdf/1803.07469.pdf

Made in OpenCV 3.46.

To run the executable with the examples, copy the "data" folder next to the executable or set the path in the main() function.


# The MAGSAC algorithm for robust model fitting without using an inlier-outlier threshold

The MAGSAC algorithm proposed in paper: Daniel Barath, Jana Noskova and Jiri Matas; MAGSAC: marginalizing sample consensus, Conference on Computer Vision and Pattern Recognition, 2019. 
It is available at http://openaccess.thecvf.com/content_CVPR_2019/papers/Barath_MAGSAC_Marginalizing_Sample_Consensus_CVPR_2019_paper.pdf

When using the algorithm, please cite `Barath, Daniel, and Noskova, Jana and Matas, Jiří. "MAGSAC: marginalizing sample consensus" Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019`.

# Installation

To build and install `MAGSAC`, clone or download this repository and, also, the sub-modules. Then build the project by CMAKE. 

# Example project

To build the sample project showing examples of fundamental matrix, homography and essential matrix fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 

Next to the executable, copy the `data` folder and, also, create a `results` folder. 

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- A modern compiler with C++17 support
