# Animated Drawings

![Sequence 02](https://user-images.githubusercontent.com/6675724/219223438-2c93f9cb-d4b5-45e9-a433-149ed76affa6.gif)

**_Update 09-03-25: This project has been a joy to share with you all. Thanks to this community for your creativity and support along the way. I'm moving on to new adventures and won't be able to maintain this repository anymore, so I've chosen to archive it. If you have questions or want to say hello in the future, come find me at [www.hjessmith.com](http://www.hjessmith.com)._**
 
This repo contains an implementation of the algorithm described in the paper, [A Method for Animating Children's Drawings of the Human Figure](https://dl.acm.org/doi/10.1145/3592788). In addition, this repo aims to be a useful creative tool in its own right, allowing you to flexibly create animations starring your own drawn characters. Here's a [video overview](https://www.youtube.com/watch?v=WsMUKQLVsOI) of the project. If you do create something fun with this, let us know! Use hashtag **#FAIRAnimatedDrawings**, or tag me on twitter: [@hjessmith](https://twitter.com/hjessmith/).




## Installation
This branch is set up for Apple Silicon macOS using Python 3.9 and `uv`.
Avoid building the local TorchServe stack from a Conda-backed Python on M-series Macs; native extensions used by the pose-estimator workers can pick up the wrong architecture.

From a fresh checkout:

````bash
# optional but recommended for video preview/transcoding
brew install uv ffmpeg

uv python install 3.9
uv venv --python 3.9 .venv
uv pip install -e ".[dev]"

# sanity check the local video workflow
.venv/bin/python -m pytest tests/test_video_app.py tests/test_video_pose.py
````

Use the virtual environment's Python directly:

````bash
.venv/bin/python examples/video_app.py --check
.venv/bin/python examples/video_app.py --port 5060
````

If you prefer activating the environment:

````bash
source .venv/bin/activate
````

Conda can still work for the basic renderer, but the maintained Apple Silicon path for this branch is the `uv` setup above.

## Using Animated Drawings

### Quick Start
Now that everything's set up, let's animate some drawings! To get started, follow these steps:
1. Open a terminal and activate the local virtual environment:
````bash
~ % source .venv/bin/activate
````

2. Ensure you're in the root directory of AnimatedDrawings:
````bash
(.venv) ~ % cd {location of AnimatedDrawings on your computer}
````

3. Start up a Python interpreter:
````bash
(.venv) AnimatedDrawings % python
````

4. Copy and paste the follow two lines into the interpreter:
````python
from animated_drawings import render
render.start('./examples/config/mvc/interactive_window_example.yaml')
````

If everything is installed correctly, an interactive window should appear on your screen.
(Use spacebar to pause/unpause the scene, arrow keys to move back and forth in time, and q to close the screen.)

<img src='./media/interactive_window_example.gif' width="256" height="256" /> </br></br></br>

There's a lot happening behind the scenes here. Characters, motions, scenes, and more are all controlled by configuration files, such as `interactive_window_example.yaml`. Below, we show how different effects can be achieved by varying the config files. You can learn more about the [config files here](examples/config/README.md).

### Export MP4 video

Suppose you'd like to save the animation as a video file instead of viewing it directly in a window. Specify a different example config by copying these lines into the Python interpreter:

````python
from animated_drawings import render
render.start('./examples/config/mvc/export_mp4_example.yaml')
````

Instead of an interactive window, the animation was saved to a file, video.mp4, located in the same directory as your script.

<img src='./media/mp4_export_video.gif' width="256" height="256" /> </br></br></br>

### Export transparent .gif

Perhaps you'd like a transparent .gif instead of an .mp4? Copy these lines in the Python interpreter instead:

````python
from animated_drawings import render
render.start('./examples/config/mvc/export_gif_example.yaml')
````

Instead of an interactive window, the animation was saved to a file, video.gif, located in the same directory as your script.

<img src='./media/gif_export_video.gif' width="256" height="256" /> </br></br></br>

### Headless Rendering

If you'd like to generate a video headlessly (e.g. on a remote server accessed via ssh), you'll need to specify `USE_MESA: True` within the `view` section of the config file.

````yaml
    view:
      USE_MESA: True
````

### Animating Your Own Drawing

All of the examples above use drawings with pre-existing annotations.
To understand what we mean by *annotations* here, look at one of the 'pre-rigged' character's [annotation files](examples/characters/char1/).
You can use whatever process you'd like to create those annotations files and, as long as they are valid, AnimatedDrawings will give you an animation.

So you'd like to animate your own drawn character.
I wouldn't want you to create those annotation files manually. That would be tedious.
To make it fast and easy, we've trained a drawn humanoid figure detector and pose estimator and provided scripts to automatically generate annotation files from the model predictions.
There are currently two options for setting this up.

#### Option 1: Docker
To get it working, you'll need to set up a Docker container that runs TorchServe.
This allows us to quickly show your image to our machine learning models and receive their predictions.

To set up the container, follow these steps:

1. [Install Docker Desktop](https://docs.docker.com/get-docker/)
2. Ensure Docker Desktop is running.
3. Run the following commands, starting from the Animated Drawings root directory:

````bash
    (animated_drawings) AnimatedDrawings % cd torchserve

    # build the docker image... this takes a while (~5-7 minutes on Macbook Pro 2021)
    (animated_drawings) torchserve % docker build -t docker_torchserve .

    # start the docker container and expose the necessary ports
    (animated_drawings) torchserve % docker run -d --name docker_torchserve -p 8080:8080 -p 8081:8081 docker_torchserve
````

Wait ~10 seconds, then ensure Docker and TorchServe are working by pinging the server:

````bash
    (animated_drawings) torchserve % curl http://localhost:8080/ping

    # should return:
    # {
    #   "status": "Healthy"
    # }
````

If, after waiting, the response is `curl: (52) Empty reply from server`, one of two things is likely happening.
1. Torchserve hasn't finished initializing yet, so wait another 10 seconds and try again.
2. Torchserve is failing because it doesn't have enough RAM.  Try [increasing the amount of memory available to your Docker containers](https://docs.docker.com/desktop/settings/mac/#advanced) to 16GB by modifying Docker Desktop's settings.

With that set up, you can now go directly from image -> animation with a single command:

````bash
    (animated_drawings) torchserve % cd ../examples
    (animated_drawings) examples % python image_to_animation.py drawings/garlic.png garlic_out
````

As you waited, the image located at `drawings/garlic.png` was analyzed, the character detected, segmented, and rigged, and it was animated using BVH motion data from a human actor.
The resulting animation was saved as `./garlic_out/video.gif`.

<img src='./examples/drawings/garlic.png' height="256" /><img src='./media/garlic.gif' width="256" height="256" /></br></br></br>

#### Option 2: Running locally on macOS

Getting Docker working can be complicated, and it's unnecessary if you just want to play around with this locally.
The local setup script prepares TorchServe and the OpenMMLab pose-estimator stack inside `./.venv`.
Run it from a `uv` Python 3.9 environment, not from Conda.

```bash
brew install uv
brew install openjdk@17
brew install ffmpeg
uv python install 3.9
uv venv --python 3.9 .venv
uv pip install -e .
cd torchserve
./setup_macos.sh
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
../.venv/bin/torchserve --start --disable-token-auth --ts-config config.local.properties --foreground

# in another terminal, verify TorchServe is ready before running the example
curl http://127.0.0.1:8080/ping
```

If your existing `./.venv/bin/python` points into Miniconda or Anaconda, recreate `./.venv` with the commands above before running `setup_macos.sh`. The TorchServe pose-estimator workers load `xtcocotools`, and that native extension has been failing on macOS when the `uv` environment is built on top of a Conda Python.

The macOS command above explicitly uses `--disable-token-auth`. Without that flag, current TorchServe releases enable token auth by default, `curl http://localhost:8080/ping` returns HTTP `400`, and the local example scripts in this repo do not send the required auth headers.

The local macOS config binds HTTP and gRPC listeners to `127.0.0.1`, and pins TorchServe's internal gRPC listeners away from the defaults `7070/7071`, which are already occupied on some machines. If you have customized `torchserve/config.local.properties`, make sure the listener addresses stay local-only unless you also add authentication, and make sure `grpc_inference_port` and `grpc_management_port` do not conflict with another local service.

`setup_macos.sh` also installs the local `animated_drawings` package into `./.venv` and adds the renderer/example dependencies used by `image_to_animation.py`, including `scikit-image`, `glfw`, and `PyOpenGL`.

With torchserve running locally like this, you can use the same command as before to make the garlic dance:

```bash 
cd ../examples
../.venv/bin/python image_to_animation.py drawings/garlic.png garlic_out
```
### Fixing bad predictions
You may notice that, when you ran `python image_to_animation.py drawings/garlic.png garlic_out`, there were additional non-video files within `garlic_out`.
`mask.png`, `texture.png`, and `char_cfg.yaml` contain annotation results of the image character analysis step. These annotations were created from our model predictions.
If the mask predictions are incorrect, you can edit the mask with an image editing program like Paint or Photoshop.
If the joint predictions are incorrect, you can run `python fix_annotations.py` to launch a web interface to visualize, correct, and update the annotations. Pass it the location of the folder containing incorrect joint predictions (here we use `garlic_out/` as an example):

````bash
    (animated_drawings) examples % python fix_annotations.py garlic_out/
    ...
     * Running on http://127.0.0.1:5050
    Press CTRL+C to quit
````

Navigate to `http://127.0.0.1:5050` in your browser to access the web interface. Drag the joints into the appropriate positions, and hit `Submit` to save your edits.

Once you've modified the annotations, you can render an animation using them like so:

````bash
    # specify the folder where the fixed annoations are located
    (animated_drawings) examples % python annotations_to_animation.py garlic_out
````

### Adding multiple characters to scene
Multiple characters can be added to a video by specifying multiple entries within the config scene's 'ANIMATED_CHARACTERS' list.
To see for yourself, run the following commands from a Python interpreter within the AnimatedDrawings root directory:

````python
from animated_drawings import render
render.start('./examples/config/mvc/multiple_characters_example.yaml')
````
<img src='./examples/characters/char1/texture.png' height="256" /> <img src='./examples/characters/char2/texture.png' height="256" /> <img src='./media/multiple_characters_example.gif' height="256" />

### Adding a background image
Suppose you'd like to add a background to the animation. You can do so by specifying the image path within the config.
Run the following commands from a Python interpreter within the AnimatedDrawings root directory:

````python
from animated_drawings import render
render.start('./examples/config/mvc/background_example.yaml')
````

<img src='./examples/characters/char4/texture.png' height="256" /> <img src='./examples/characters/char4/background.png' height="256" /> <img src='./media/background_example.gif' height="256" />

### Using BVH Files with Different Skeletons
You can use any motion clip you'd like, as long as it is in BVH format.

If the BVH's skeleton differs from the examples used in this project, you'll need to create a new motion config file and retarget config file.
Once you've done that, you should be good to go.
The following code and resulting clip uses a BVH with completely different skeleton.
Run the following commands from a Python interpreter within the AnimatedDrawings root directory:

````python
from animated_drawings import render
render.start('./examples/config/mvc/different_bvh_skeleton_example.yaml')
````

<img src='./media/different_bvh_skeleton_example.gif' height="256" />

### Creating Your Own BVH Files
You may be wondering how you can create BVH files of your own.
You used to need a motion capture studio.
But now, thankfully, there are simple and accessible options for getting 3D motion data from a single RGB video.
For example, I created this Readme's banner animation by:
1. Recording myself doing a silly dance with my phone's camera.
2. Using [Rokoko](https://www.rokoko.com/) to export a BVH from my video.
3. Creating a new [motion config file](examples/config/README.md#motion) and [retarget config file](examples/config/README.md#retarget) to fit the skeleton exported by Rokoko.
4. Using AnimatedDrawings to animate the characters and export a transparent animated gif.
5. Combining the animated gif, original video, and original drawings in Adobe Premiere.
<img src='https://user-images.githubusercontent.com/6675724/219223438-2c93f9cb-d4b5-45e9-a433-149ed76affa6.gif' height="256" />

Here is an example of the configs I used apply my motion to a character. To use these config files, ensure that the Rokoko exports the BVH with the Mixamo skeleton preset:

 ````python
from animated_drawings import render
render.start('./examples/config/mvc/rokoko_motion_example.yaml')
 ````

It will show this in a new window:

![Sequence 01](https://user-images.githubusercontent.com/6675724/233157474-1506d219-c085-49f9-a537-43d6c1bae93a.gif)

### Creating Motion From Your Own Video
This repo also includes an experimental local video pose workflow. It uses MediaPipe to estimate a human pose from a short video, writes a MediaPipe-style BVH file, and then uses the existing Animated Drawings retargeter.

Videos are limited to 10 seconds in this first version.

To convert a video into a motion config from the command line:

````bash
python examples/video_to_motion.py path/to/video.mp4 ./video_motion_out --max-seconds 10
````

This writes `pose_sequence.json`, `pose_overlay.mp4`, `motion.bvh`, and `motion.yaml` in the output directory. You can use `motion.yaml` anywhere a normal motion config is accepted. It pairs with `examples/config/retarget/mediapipe_pfp.yaml`.

To use the browser-based local GUI:

````bash
python examples/video_app.py --port 5060
````

Open `http://127.0.0.1:5060`. The app can record or upload a short video, upload a MediaPipe-compatible BVH file, select an existing motion, select one of the bundled character rigs, upload a drawing, preview estimated joints, and render synchronized source/animation playback.

Bundled characters work without TorchServe. Uploading a new drawing still uses the existing image-to-annotations path, so TorchServe must be running and healthy before using that part of the app.




### Adding Addition Character Skeletons
All of the example animations above depict "human-like" characters; they have two arms and two legs.
Our method is primarily designed with these human-like characters in mind, and the provided pose estimation model assumes a human-like skeleton is present.
But you can manually specify a different skeletons within the `character config` and modify the specified `retarget config` to support it.
If you're interested, look at the configuration files specified in the two examples below.


````python
from animated_drawings import render
render.start('./examples/config/mvc/six_arms_example.yaml')
````

<img src='https://user-images.githubusercontent.com/6675724/223584962-925ee5aa-11de-47e5-ace2-a6d5940b34ae.png' height="256" /><img src='https://user-images.githubusercontent.com/6675724/223585000-dc8acf4e-974d-4cae-998b-94543f5f42c8.gif' width="256" height="256" /></br></br></br>

````python
from animated_drawings import render
render.start('./examples/config/mvc/four_legs_example.yaml')
````

<img src='https://user-images.githubusercontent.com/6675724/223585033-f11e4e66-0443-405a-80e5-09b6aa0e335d.png' height="256" /><img src='https://user-images.githubusercontent.com/6675724/223585043-7ce9eac0-bb4c-4547-b038-c63ca2852ef2.gif' width="256" height="256" /></br></br></br>

If you're interested in animating quadrupeds specifically, you may want to check out [the quadruped example directory](examples/quadruped).

### Creating Your Own Config Files
If you want to create your own config files, see the [configuration file documentation](examples/config/README.md).

## Browser-Based Demo

If you'd like to animate a drawing of your own, but don't want to deal with downloading code and using the command line, check out our browser-based demo:

[www.sketch.metademolab.com](https://sketch.metademolab.com/)

## Paper & Citation
 If you find the resources in this repo helpful, please consider citing the accompanying paper, [A Method for Animating Children's Drawings of The Human Figure](https://dl.acm.org/doi/10.1145/3592788)).

Citation:

```
@article{10.1145/3592788,
author = {Smith, Harrison Jesse and Zheng, Qingyuan and Li, Yifei and Jain, Somya and Hodgins, Jessica K.},
title = {A Method for Animating Children’s Drawings of the Human Figure},
year = {2023},
issue_date = {June 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {3},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592788},
doi = {10.1145/3592788},
abstract = {Children’s drawings have a wonderful inventiveness, creativity, and variety to them. We present a system that automatically animates children’s drawings of the human figure, is robust to the variance inherent in these depictions, and is simple and straightforward enough for anyone to use. We demonstrate the value and broad appeal of our approach by building and releasing the Animated Drawings Demo, a freely available public website that has been used by millions of people around the world. We present a set of experiments exploring the amount of training data needed for fine-tuning, as well as a perceptual study demonstrating the appeal of a novel twisted perspective retargeting technique. Finally, we introduce the Amateur Drawings Dataset, a first-of-its-kind annotated dataset, collected via the public demo, containing over 178,000 amateur drawings and corresponding user-accepted character bounding boxes, segmentation masks, and joint location annotations.},
journal = {ACM Trans. Graph.},
month = {jun},
articleno = {32},
numpages = {15},
keywords = {2D animation, motion retargeting, motion stylization, Skeletal animation}
}
```

## Amateur Drawings Dataset

To obtain the Amateur Drawings Dataset, run the following two commands from the command line:

````bash
# download annotations (~275Mb)
wget https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings_annotations.json

# download images (~50Gb)
wget https://dl.fbaipublicfiles.com/amateur_drawings/amateur_drawings.tar
````

If you'd like higher res images, they can be found on the releases (ad_orig_img_fs). They've been split into multiple chunks using the split cli. They are released under the same license as the original dataset. 

If you have feedback about the dataset, please fill out [this form](https://forms.gle/kE66yskh9uhtLbFz9).

## ChildlikeSHAPES

If you want this data set, construct the full archive from the chunks in the release page
````cat datachunk_* > full_archive.7z  #(pw=An1m8dR3610Ns)````


## Trained Model Weights

Trained model weights for human-like figure detection and pose estimation are included in the [repo releases](https://github.com/facebookresearch/AnimatedDrawings/releases). Model weights are released under [MIT license](https://github.com/facebookresearch/AnimatedDrawings/blob/main/LICENSE). The .mar files were generated using the OpenMMLab framework ([OpenMMDet Apache 2.0 License](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE), [OpenMMPose Apache 2.0 License](https://github.com/open-mmlab/mmpose/blob/main/LICENSE))

## As-Rigid-As-Possible Shape Manipulation

These characters are deformed using [As-Rigid-As-Possible (ARAP) shape manipulation](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf).
We have a Python implementation of the algorithm, located [here](https://github.com/fairinternal/AnimatedDrawings/blob/main/animated_drawings/model/arap.py), that might be of use to other developers.

## License
Animated Drawings code, model weights, and Amateur Drawings dataset is released under the [MIT license](https://github.com/fairinternal/AnimatedDrawings/blob/main/LICENSE). ChildlikeSHAPES dataset is released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. 
