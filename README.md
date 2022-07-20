# **"Image Stitching - Implemented Algorithm from scratch (without OpenCV)"**

## The task is to stitch two given images, left.jpg and right.jpg to construct a panorama image.

### The Image stiching task involves following steps

* Find key points in the given images
* Match Keypoint based on euclidean distance
* Ratio test to extract coordinates of good matching points
* Implementing Ransac Algorithm to remove outliers and get best homography matrix
* Stitching Image using best homography matrix

Please find more about them implementation [here](https://github.com/AkashSDE/ImageStitching/blob/dcae41f50202eab68e246a2219c4a18ba761ef4d/Report/ProjectReport.pdf)

### Steps to run the code

* Run Stitch.py file inside the code folder using command >> python Stitch.py
* The stiched image can be found inside Results folder