This archive contains dataset of image pairs with manually annotated ground truth in form of corresponding points homogeneous coordinates (6×N, usually ~10 points, A image coordinates first) in Matlab structures (validation.pts). The naming convention is as follows: first image nameA.ext, second image nameB.ext, validation points name_vpts.mat. For some pairs there may be additional information in the validation structure.

This dataset does not contain purely planar scenes. Thus can be used for testing of epipolar geometry estimation.

Images were downloaded from:
http://www.robots.ox.ac.uk/~vgg/data.html
http://homes.esat.kuleuven.be/~tuytelaa/
http://www.cs.unc.edu/~marc/
http://cmp.felk.cvut.cz/projects/is3d/
http://cmp.felk.cvut.cz/~cechj/SCV/

Correspondences were annotated manually on zoomed-in images. Then they were
checked by several homographies with high RANSAC score and corrected if
necessary.

