import argparse
from pathlib import Path

import dlib


def main(args):
    image_pattern = '*.jpg'

    predictor_path = args.predictor_path
    faces_folder_path = Path(args.folder)
    display = args.display

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    if display:
        win = dlib.image_window()

    for c in faces_folder_path.glob(image_pattern):
        with open(str(c) + '.txt', 'w') as coord:
            print(f'Processing file: {c}')
            img = dlib.load_rgb_image(str(c))

            if display:
                win.clear_overlay()
                win.set_image(img)

            # Detect bounding boxes of each face on the image
            # Second argument in detector is upscale ratio
            dets = detector(img, 1)
            print(f'Number of faces detected: {len(dets)}')

            for k, d in enumerate(dets):
                print(f'Detection {k}; Left: {d.left()}, Top: {d.top()}, Right: {d.right()}, Bottom: {d.bottom()}')
                # Get the landmarks for the face in box d.
                shape = predictor(img, d)

                if display:
                    win.add_overlay(shape)

                for i in shape.parts():
                    coord.write(f'{i.x} {i.y}\n')

        if display:
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect face landmarks on the images and save them to text files')
    parser.add_argument('-p', '--predictor_path', action='store', dest='predictor_path', type=str,
                        default='./shape_predictor_68_face_landmarks.dat')
    parser.add_argument('-f', '--folder',  action='store', dest='folder', type=str, default='./images')
    parser.add_argument('-d', '--display', action='store_true')
    args = parser.parse_args()
    main(args)
