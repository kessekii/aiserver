import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import cv2
import numpy as np
import heapq
   

import heapq
import cv2
def create_depth_map():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open("C:/Users/hamme/Downloads/123.jpg")

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")

    # prepare image for the d
    inputs = image_processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.show()

    depth.save("depth_map.png")


def load_and_invert_depth_map(filepath):
    depth_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    inverted_depth_map = cv2.bitwise_not(depth_map)
    return inverted_depth_map

def find_path(depth_map, start, end):
    # Placeholder for pathfinding logic
    # Implement BFS or another pathfinding algorithm here
    # For simplicity, returning a dummy path
    return [start, end]

def draw_path(depth_map, path):
    for i in range(len(path) - 1):
        cv2.line(depth_map, path[i], path[i+1], (0, 0, 255), 2)
    return depth_map



def find_bottom_center_point(depth_map):
    height, width = depth_map.shape[:2]
    # Center by height (middle row)
    center_column = width // 2
    # Bottom by width (last column)
    bottom_row = height - 1
    # The desired point
    point = (center_column, bottom_row)
    return point

def find_highest_point_in_inverted_map(depth_map, start_point):
    # Start from the specified start point and move upwards
    x, y = start_point
    previous_depth = depth_map[y, x]
    target_depth = 200  # Example threshold for "high" depth in an inverted map
    depth_change_threshold = 20  # Example threshold for a significant depth change 
    while y > 0:  # Ensure we don't go out of bounds
        y -= 1  # Move one pixel up
        current_depth = depth_map[y, x]
        depth_difference = abs(current_depth - previous_depth)

        if depth_difference > depth_change_threshold:
            return (x, y + 1)  # Return the last point before the significant change

        previous_depth = current_depth

    return start_point  # Return the start point if no significant change is found

def heuristic(a, b):
    # Using Manhattan distance as the heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(depth_map, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-way connectivity
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < depth_map.shape[0]:
                if 0 <= neighbor[1] < depth_map.shape[1]:                
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue
                    
                    if  tentative_g_score < gscore.get(neighbor, np.inf) or neighbor not in [i[1]for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
                else:
                    # Neighbor is out of bounds
                    continue
            else:
                # Neighbor is out of bounds
                continue

    return False

# Assuming the rest of the provided code is here, including the definitions of find_lowest_center_point and find_furthest_center_point
def load_and_invert_depth_map(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    depth_map = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        print(f"Failed to load the depth map from {filepath}")
        return None
    return cv2.bitwise_not(depth_map)

def visualize_path(depth_map, path, start, goal):
    path_image = depth_map.copy()
    if path:
        for point in path:
            cv2.circle(path_image, point, 1, (0, 255, 0), -1)
    else:
        print(depth_map)
    print(goal)
    # Highlight start and goal points distinctly
    cv2.circle(path_image, start, 5, (255, 0, 0), -1)  # Start in blue
    print(start)
    cv2.circle(path_image, goal, 5, (0, 0, 255), -1)  # Goal in red
    cv2.imshow("Path on Depth Map", path_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(filepath):
    create_depth_map()
    inverted_depth_map = load_and_invert_depth_map(filepath)
    cv2.imwrite('inverted_depth_map.png', inverted_depth_map)
    if inverted_depth_map is None:
        return
    start = find_bottom_center_point(inverted_depth_map)
    print(start)
    # goal = find_highest_point_in_inverted_map(inverted_depth_map, start)
    start = (691, 944)
    goal = (1125, 529)
     
    path = a_star_search(inverted_depth_map, start, goal)
    visualize_path(inverted_depth_map, path, start, goal)

if __name__ == "__main__":
    filepath = './depth_map.png'
    main(filepath)

    