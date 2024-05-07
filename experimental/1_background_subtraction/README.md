Experiment in background subtraction.

https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html

hatch run shiny run experimental/1_background_subtraction/app.py

# conclusion

 - Knn is best, with lower distance threshold.

 - Tends to only detect highly contrasting objects.

 - Use open & close kernel to denoise and blob. 

 - resolution is king, bigger is better

 - pre-calculating background doesn't work