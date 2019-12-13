# The MAGSAC algorithm for robust model fitting without using an inlier-outlier threshold

The MAGSAC and MAGSAC++ algorithms proposed for robust model estimation without a single inlier-outlier threshold.

Made in OpenCV 3.46.
To run the executable with the examples, copy the "data" folder next to the executable or set the path in the main() function.


If you use the algorithm, please cite

```
@inproceedings{barath2019magsac,
	author = {Barath, Daniel and Matas, Jiri and Noskova, Jana},
	title = {MAGSAC: marginalizing sample consensus},
	booktitle = {Conference on Computer Vision and Pattern Recognition},
	year = {2019},
}

@inproceedings{barath2019magsacplusplus,
	author = {Barath, Daniel and Noskova, Jana and Ivashechkin, Maksym and Matas, Jiri},
	title = {MAGSAC++, a fast, reliable and accurate robust estimator},
	booktitle = {arXiv preprint:1912.05909},
	year = {2019},
}
```

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
