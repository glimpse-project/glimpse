Early playground for Glimpse Project

# Status

The early direction to get started with this project is to look at technically
implementing some of the capabilities demoed in this teaser video:

https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6

The first aspect looked at was supporting real-time (frontal) face detection
which we now have working well enough to move on:

https://medium.com/impossible/building-make-believe-tech-glimpse-in-progress-ecb9bbc113a1

There are still lots of opportunities to improve what we do for face tracking
but it's good enough to work with for now so we can start looking at the more
tricky problem of skeletal tracking.

The current focus is on skeletal tracking. The current aim is to reproduce the
R&D done by Microsoft for skeleton tracking with their Kinect cameras, which
provide similar data to Google Tango phones. Their research was published as a paper titled: [Real-Time Human Pose Recognition in Parts from Single Depth Images](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf)

In terms of understanding that paper then:

* Some reminders about set theory notation: http://byjus.com/maths/set-theory-symbols/
    * Also a reminder on set-builder notation: http://www.mathsisfun.com/sets/set-builder-notation.html
* How to calculate Shannon Entropy: http://www.bearcave.com/misl/misl_tech/wavelets/compression/shannon.html
* A key referenced paper on randomized trees for keypoint recognition: https://pdfs.semanticscholar.org/f637/a3357444112d0d2c21765949409848a5cba3.pdf
    * A related technical report (more implementation details): http://cvlabwww.epfl.ch/~lepetit/papers/lepetit_tr04.pdf
* Referenced paper on using meanshift: https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf
    * Simple intuitive description of meanshift: http://docs.opencv.org/trunk/db/df8/tutorial_py_meanshift.html
    * Comparison of mean shift tracking methods (inc. camshift): http://old.cescg.org/CESCG-2008/papers/Hagenberg-Artner-Nicole.pdf


# Building Dependencies

See build-third-party/build-third-party.sh

TODO: explain how to use


# References

https://github.com/betars/Face-Resources

This repo has a scalar implementation of the same feature extraction algorithm used
in DLib that's a bit simpler to review for understanding what it's doing:
https://github.com/rbgirshick/voc-dpm (see features/features.cc) (though also
note there's the scalar code in fhog.h that has to handle the border pixels that
don't neatly fit into simd registers)

## Papers

[Histograms of Oriented Gradients for Human Detection by Navneet Dalal and Bill Triggs, CVPR 2005](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf)

[Object Detection with Discriminatively Trained Part Based Models by P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, Sep. 2010](https://cs.brown.edu/~pff/papers/lsvm-pami.pdf)

[Real-Time Human Pose Recognition in Parts from Single Depth Images](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf)
