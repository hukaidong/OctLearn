
Trajectory is an array of points like [[x0, y0], [x1, y1]...]. Our target 
is to create a binary image array. Image rectangle is defined in [xmin, xmax,
ymin, ymax], and resolution of each pixel. Assume the image is reprsented
in [y, x] (row first)

The index of each point is calculated by 
index(x, y) = round((point-pointmin)/resolution),
the array index is indexnum = index .* [1, xsize] 