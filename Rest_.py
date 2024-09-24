import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

rotated_histograms = []
template_histograms = []

def perform_edge_detection_line_fitting_and_plot(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1=50, threshold2=150)
            lines = cv2.HoughLinesP(edges, rho=5,theta=np.pi / 360, threshold=50, minLineLength=10,maxLineGap=10) #many trial and error is used to find parameters

            line_data = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                angle = np.arctan2(y2 - y1, x2 - x1)

                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                line_data.append((angle, length))

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img)

            # sorting yields errory prediction so commented out


            angles, lengths = zip(*line_data)

            bin_edges = np.arange(-100, 101, 20)

            hist = np.zeros(len(bin_edges) - 1)

            for angle, length in zip(angles, lengths):
                bin_index = int((np.degrees(angle) + 100) // 20)
                hist[bin_index] += length

            bin_centers = bin_edges[:-1] +10
            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, hist, width=20, color='red', align='center')
            plt.title(f'Line Length Histogram - {filename}')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Sum of Lengths')
            plt.xticks(bin_edges)
            plt.grid(True)
            plt.tight_layout()
            #plt.show()
            if input_dir=="rotated_edges":
                rotated_histograms.append(hist)
            else:
                template_histograms.append(hist)

rotated_edges_dir = "rotated_edges"
template_edges_dir = "template_edges"
output_rotated_lines_dir = "rotated_lines"
output_template_lines_dir = "template_lines"

perform_edge_detection_line_fitting_and_plot(rotated_edges_dir, output_rotated_lines_dir)

print(len(rotated_histograms))
perform_edge_detection_line_fitting_and_plot(template_edges_dir, output_template_lines_dir)

print(len(template_histograms))


def find_original_books(rotated_histograms, template_histograms):
    rotation_angles = []

    for rotated_hist in rotated_histograms:
        min_distance = float('inf')
        best_template_idx = None
        best_shift = 0

        for idx, template_hist in enumerate(template_histograms):
            for shift in range(len(rotated_hist)):
                shifted_hist_forward = np.roll(rotated_hist, shift) #circular matrix shift
                distance_forward = np.linalg.norm(shifted_hist_forward - template_hist) #euclidian norm of difference of shifted and original

                if distance_forward < min_distance:
                    min_distance = distance_forward
                    best_shift = shift
                    best_template_idx = idx

            for shift in range(len(rotated_hist)):
                shifted_hist_backward = np.roll(rotated_hist, -shift)
                distance_backward = np.linalg.norm(shifted_hist_backward - template_hist)

                if distance_backward < min_distance:
                    min_distance = distance_backward
                    best_shift = -shift
                    best_template_idx = idx

        estimated_angle = best_shift * 20 #shift counter * angles corresponding to bins (approximation)
        rotation_angles.append((estimated_angle, best_template_idx))

    return rotation_angles

"""
The below commented out code block is only used for testing and observation of results, not included in the assignment requirements, irrelevant

"""
def rotate_images(template_folder, rotation_angles, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, filename in enumerate(os.listdir(template_folder)):
        template_path = os.path.join(template_folder, filename)
        if os.path.isfile(template_path):
            image = cv2.imread(template_path)
            rotated_image = rotate_image(image, rotation_angles[i])
            output_filename = f"rotated_{i}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, rotated_image)

def rotate_image(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle[0], 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    return rotated_image

template_folder = "template_edges"
output_folder = "rotated_edges"

rotation_angles = find_original_books(rotated_histograms,template_histograms)
rotate_images(template_folder, rotation_angles, output_folder)


print(find_original_books(rotated_histograms,template_histograms))

