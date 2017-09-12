# Setting up Blender

To work with the glimpse-training.blend file it's first necessary to enable a
number of addons...

Follow the instructions here to install the makehuman BlenderTools addons:
http://www.makehumancommunity.org/wiki/Documentation:Getting_and_installing_BlenderTools

Within Blender's User Preferences -> File tab:

Point the 'Scripts:' entry to the glimpse-sdk/blender/ directory

Press 'Save User Settings' and quit and reopen Blender

Under User Preferences -> Addons now enable these Addons:

* Make Walk
* Make Clothes
* Make Target
* Glimpse Rig Paint
* Glimpse Training Data Generor

Note: to run Blender non-interactively on a headless machine then the
userpref.blend file that's updated as a result of the above steps can be copied
to the headless machine (assuming paths are compatible). As an alternative to
needing an absolute 'Scripts:' path in the userpref.blend file then the
glimpse-sdk/blender/addon/xyz directories can be symlinked under
.blender/<version>/scripts/addons/xyz instead.
