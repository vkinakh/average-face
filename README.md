# average-face
Algorithm for calculating average face from multiple images

To detect faces, face landmarks and save them to file run
` python face_landmark_detector.py -p <path to Dlib shape predictor model> -f <path to folder with images>`

To run face averaging
`python face_average.py -p <path to folder with images>`

The code is based on this [post](https://javewa.github.io/2018/08/23/face/) and this [post](https://www.learnopencv.com/average-face-opencv-c-python-tutorial/)
