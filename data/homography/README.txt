This archive contains dataset of image pairs with manually annotated ground truth in form of corresponding points homogeneous coordinates (6×N, usually ~10 points, A image coordinates first) in Matlab structures (validation.pts). The naming convention is as follows: first image nameA.ext, second image nameB.ext, validation points name_vpts.mat. For some pairs there may be additional information in the validation structure.

This dataset contains planar scenes! Thus cannot be used for testing of epipolar geometry estimation.

Images comes from:
http://www.vision.cs.rpi.edu/gdbicp/dataset/
http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html
http://www.robots.ox.ac.uk/~vgg/data5.html

Correspondences were annotated manually on zoomed-in images. Then they were
checked by several homographies with high RANSAC score and corrected if
necessary. Finally, they were refined by modified Bundle Adjustment of
Lourakis (www.ics.forth.gr/~lourakis/sba/). For the non-bundled correspondences
please use the files from the vpts_old directory.


