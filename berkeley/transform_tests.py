import cv2
import math
import numpy as np

def image_prototype():
    first_turn_markers =  [(30, 197), (364, 197), (197, 364), (197, 30)]
    inner_markers = [(row, col) for row in [90, 144, 197, 251, 305] for col in [90, 144, 197, 251, 305]]
    return first_turn_markers + inner_markers

def cartesian_to_homogeneous(point):
    return np.array([point[0], point[1], 1])

def homogeneous_to_cartesian(point):
    return (round(point[0]), round(point[1]))

def translation_matrix(deltaX, deltaY):
    return np.array([
             [1, 0, deltaX],
             [0, 1, deltaY],
             [0, 0, 1]
           ])

def rotation_matrix(theta):
    theta = math.radians(theta)
    return np.array([
             [math.cos(theta), -math.sin(theta), 0],
             [math.sin(theta), math.cos(theta), 0],
             [0, 0, 1]
           ])

def scaling_matrix(scaleX, scaleY):
    return np.array([
             [scaleX, 0, 0],
             [0, scaleY, 0],
             [0, 0, 1]
           ])

def shearing_matrix(shearX, shearY):
    return np.array([
             [1, shearX, 0],
             [shearY, 1, 0],
             [0, 0, 1]
           ])

def draw_transformation(transformation_matrix, filename):
    visualization = np.zeros((1000, 1000, 3))
    prototype = image_prototype()
    for point in prototype:
        transformed = homogeneous_to_cartesian(np.matmul(transformation_matrix, cartesian_to_homogeneous(point)))
        cv2.circle(visualization, transformed, 4, (255, 255, 255), -1)
    cv2.imwrite(f'results/transformation_visualizations/{filename}', visualization)




def main():
    translation_m = translation_matrix(300, 300)
    for rotation in [0, 5, 10, 15]:
        rotation_m = rotation_matrix(rotation)
        for scalingXY in [0.25, 0.5, 0.75]:
            scaling_m = scaling_matrix(scalingXY, scalingXY)
            for shearingX in [0, 0.5, 1]:
                for shearingY in [0, 0.5, 1]:
                    if shearingX == 1 and shearingY == 1:
                        continue
                    shearing_m = shearing_matrix(shearingX, shearingY)
                    total_matrix = np.matmul(translation_m, np.matmul(scaling_m, np.matmul(shearing_m, rotation_m)))
                    draw_transformation(total_matrix, f'rot{rotation}_sca{scalingXY}_sheX{shearingX}_sheY{shearingY}'.replace('.', '-') + '.png')


if __name__=='__main__':
    main()
