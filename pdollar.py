import sys
import math
import pickle
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Gesture:
    def __init__(self, points, name=""):
        self.points = points
        self.name = name
        self.SAMPLING_RESOLUTION = 64
        #print(f"Initializing gesture {name} with {len(points)} points")
        if len(points) < 2:
        #    print(f"Warning: Gesture {name} has less than 2 points. Duplicating the single point.")
            self.points = [points[0]] * self.SAMPLING_RESOLUTION
        else:
            self.resample()
        self.scale()
        self.translate_to_origin()
        self.compute_shape_context()
        #print(f"Gesture {name} processed. Final point count: {len(self.points)}")

    def resample(self):
        #print(f"Resampling {self.name}")
        path_length = self.path_length()
        if path_length == 0:
        #    print(f"Warning: Path length is 0 for {self.name}. Duplicating points.")
            self.points = [self.points[0]] * self.SAMPLING_RESOLUTION
        else:
            interval = path_length / (self.SAMPLING_RESOLUTION - 1)
            D = 0
            new_points = [self.points[0]]
            for i in range(1, len(self.points)):
                d = self.distance(self.points[i - 1], self.points[i])
                if D + d >= interval:
                    qx = self.points[i - 1].x + ((interval - D) / d) * (self.points[i].x - self.points[i - 1].x)
                    qy = self.points[i - 1].y + ((interval - D) / d) * (self.points[i].y - self.points[i - 1].y)
                    q = Point(qx, qy)
                    new_points.append(q)
                    self.points.insert(i, q)
                    D = 0
                else:
                    D += d
            while len(new_points) < self.SAMPLING_RESOLUTION:
                new_points.append(self.points[-1])
            self.points = new_points
        #print(f"Resampled {self.name} to {len(self.points)} points")

    def scale(self):
        #print(f"Scaling {self.name}")
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)
        scale_x = max_x - min_x
        scale_y = max_y - min_y
        scale = max(scale_x, scale_y)
        if scale == 0:
        #    print(f"Warning: Cannot scale {self.name} (all points are the same)")
            return
        for p in self.points:
            p.x = (p.x - min_x) / scale
            p.y = (p.y - min_y) / scale
        #print(f"Scaled {self.name}")

    def translate_to_origin(self):
        #print(f"Translating {self.name} to origin")
        centroid = self.centroid()
        for p in self.points:
            p.x -= centroid.x
            p.y -= centroid.y
        #print(f"Translated {self.name} to origin")

    def compute_shape_context(self):
        #print(f"Computing shape context for {self.name}")
        points = np.array([(p.x, p.y) for p in self.points])
        pairwise_distances = cdist(points, points)
        
        num_bins_r = 5
        num_bins_theta = 12
        
        r_inner = 0.1250
        r_outer = 2.0
        
        r_array = np.logspace(np.log10(r_inner), np.log10(r_outer), num=num_bins_r)
        
        self.shape_contexts = []
        
        for i in range(len(points)):
            point = points[i]
            other_points = np.delete(points, i, axis=0)
            
            delta_x = other_points[:, 0] - point[0]
            delta_y = other_points[:, 1] - point[1]
            
            r = np.sqrt(delta_x**2 + delta_y**2)
            theta = np.arctan2(delta_y, delta_x)
            
            r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), num=num_bins_r+1)
            theta_bin_edges = np.linspace(-np.pi, np.pi, num=num_bins_theta+1)
            
            hist, _, _ = np.histogram2d(r, theta, bins=(r_bin_edges, theta_bin_edges))
            self.shape_contexts.append(hist.flatten())
        
        self.shape_contexts = np.array(self.shape_contexts)
        #print(f"Computed shape contexts for {self.name}")

    def centroid(self):
        x = sum(p.x for p in self.points) / len(self.points)
        y = sum(p.y for p in self.points) / len(self.points)
        return Point(x, y)

    def path_length(self):
        d = 0
        for i in range(1, len(self.points)):
            d += self.distance(self.points[i - 1], self.points[i])
        return d

    @staticmethod
    def distance(p1, p2):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx * dx + dy * dy)
    

    def compute_features(self):
        #print(f"Computing features for {self.name}")
        self.features = []
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            p3 = self.points[(i + 2) % len(self.points)]
            
            # Compute angle
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            
            # Compute curvature
            dx1, dy1 = p2.x - p1.x, p2.y - p1.y
            dx2, dy2 = p3.x - p2.x, p3.y - p2.y
            denominator = (math.sqrt(dx1*dx1 + dy1*dy1) * math.sqrt(dx2*dx2 + dy2*dy2))
            if denominator == 0:
                curvature = 0  # Handle the case where points are the same
            else:
                curvature = (dx1 * dy2 - dy1 * dx2) / denominator
            
            self.features.append((angle, curvature))
        #print(f"Computed {len(self.features)} features for {self.name}")

    def compute_angles(self):
        self.angles = []
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            self.angles.append(angle)


class PointCloudRecognizer:
    @staticmethod
    def classify(candidate, training_set):
        if not training_set:
            return "No matching gesture found"
        b = float('inf')
        result = "No matching gesture found"
        #print(f"Number of templates in training set: {len(training_set)}")
        #print(f"Number of points in candidate gesture: {len(candidate.points)}")
        for gesture in training_set:
        #    print(f"Comparing with {gesture.name} (points: {len(gesture.points)})")
            d = PointCloudRecognizer.dtw_distance(candidate.shape_contexts, gesture.shape_contexts)
        #    print(f"Distance to {gesture.name}: {d}")
            if d < b:
                b = d
                result = gesture.name
        #print(f"Best match: {result} with distance: {b}")
        return result

    @staticmethod
    def dtw_distance(s1, s2):
        n, m = len(s1), len(s2)
        dtw_matrix = np.zeros((n+1, m+1))
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.linalg.norm(s1[i-1] - s2[j-1])
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
        
        return dtw_matrix[n, m]
    
    @staticmethod
    def gesture_distance(g1, g2):
        if len(g1.features) != len(g2.features):
            return float('inf')
        
        total_distance = 0
        for f1, f2 in zip(g1.features, g2.features):
            angle_diff = min(abs(f1[0] - f2[0]), 2*math.pi - abs(f1[0] - f2[0]))
            curvature_diff = abs(f1[1] - f2[1])
            total_distance += math.sqrt(angle_diff**2 + curvature_diff**2)
        
        return total_distance / len(g1.features)
    
    def distance_at_best_angle(candidate, template):
        PHI = 0.5 * (-1.0 + math.sqrt(5.0))  # Golden Ratio
        MAX_ANGLE = math.pi / 4.0
        MIN_ANGLE = -MAX_ANGLE
        a = MIN_ANGLE
        b = MAX_ANGLE
        x1 = PHI * a + (1 - PHI) * b
        f1 = PointCloudRecognizer.distance_at_angle(candidate, template, x1)
        x2 = (1 - PHI) * a + PHI * b
        f2 = PointCloudRecognizer.distance_at_angle(candidate, template, x2)

        while abs(b - a) > 0.01:
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = PHI * a + (1 - PHI) * b
                f1 = PointCloudRecognizer.distance_at_angle(candidate, template, x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = (1 - PHI) * a + PHI * b
                f2 = PointCloudRecognizer.distance_at_angle(candidate, template, x2)

        return min(f1, f2)
    
    @staticmethod
    def distance_at_angle(candidate, template, angle):
        new_points = PointCloudRecognizer.rotate_by(candidate.points, angle)
        d = PointCloudRecognizer.cloud_distance(new_points, template.points)
        return d

    @staticmethod
    def rotate_by(points, angle):
        new_points = []
        for p in points:
            qx = p.x * math.cos(angle) - p.y * math.sin(angle)
            qy = p.x * math.sin(angle) + p.y * math.cos(angle)
            new_points.append(Point(qx, qy))
        return new_points
    
    @staticmethod
    def greedy_cloud_match(points1, points2):
        e = 0.50
        step = math.floor(math.pow(len(points1), 1 - e))
        min_distance = float('inf')
        for i in range(0, len(points1), step):
            d1 = PointCloudRecognizer.cloud_distance(points1, points2, i)
            d2 = PointCloudRecognizer.cloud_distance(points2, points1, i)
            min_distance = min(min_distance, min(d1, d2))
        return min_distance

    @staticmethod
    def cloud_distance(points1, points2):
        n = len(points1)
        matched = [False] * n
        sum_distance = 0
        for i in range(n):
            min_dist = float('inf')
            index = -1
            for j in range(n):
                if not matched[j]:
                    d = Gesture.distance(points1[i], points2[j])
                    if d < min_dist:
                        min_dist = d
                        index = j
            matched[index] = True
            sum_distance += min_dist
        return sum_distance / n

def print_help():
    print("Usage:")
    print("pdollar -t <gesturefile>   : Add gesture template")
    print("pdollar -r                 : Clear all templates")
    print("pdollar <eventstream>      : Recognize gestures from event stream")

def add_template(gesture_file):
    with open(gesture_file, 'r') as f:
        lines = f.readlines()
    
    name = lines[0].strip()
    gestures = []
    current_gesture = []
    
    for line in lines[1:]:
        line = line.strip()
        if line == "BEGIN":
            current_gesture = []
        elif line == "END":
            if current_gesture:
                gestures.append(current_gesture)
        else:
            try:
                x, y = map(float, line.split(','))
                current_gesture.append(Point(x, y))
            except ValueError:
                #print(f"Warning: Skipping invalid line: {line}")
                continue

    for i, gesture_points in enumerate(gestures):
        gesture = Gesture(gesture_points, f"{name}")  # Remove the _i+1 suffix
        training_set.append(gesture)
        print(f"Added template: {gesture.name} (Version {i+1})")

    save_training_set()

    if not gestures:
        print(f"No valid gestures found in file: {gesture_file}")

def clear_templates():
    global training_set
    training_set = []
    save_training_set()
    print("All templates cleared")

def recognize_gestures(event_stream_file):
    with open(event_stream_file, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines[1:-2]:  # Skip MOUSEDOWN and MOUSEUP/RECOGNIZE
        if line.strip() in ["MOUSEDOWN", "MOUSEUP"]:
            continue
        try:
            x, y = map(float, line.strip().split(','))
            points.append(Point(x, y))
        except ValueError:
            print(f"Warning: Skipping invalid line: {line.strip()}")
    
    if not points:
        print("No valid points found in the event stream.")
        return
    
    #print(f"Number of points in the event stream: {len(points)}")
    candidate = Gesture(points)
    
    if not training_set:
        print("No templates available. Please add templates using the -t option.")
        return
    
    result = PointCloudRecognizer.classify(candidate, training_set)
    print(f"Recognized gesture: {result}")

def save_training_set():
    with open('training_set.pkl', 'wb') as f:
        pickle.dump(training_set, f)

def load_training_set():
    global training_set
    try:
        with open('training_set.pkl', 'rb') as f:
            training_set = pickle.load(f)
        #print(f"Loaded {len(training_set)} templates from file.")
    except FileNotFoundError:
        training_set = []
        print("No existing training set found. Starting with an empty set.")

def main(args):
    load_training_set()

    if len(args) == 0:
        print_help()
        return

    try:
        if args[0] == '-t':
            if len(args) < 2:
                print("Error: Missing gesture file path")
                print_help()
            else:
                add_template(args[1])
        elif args[0] == '-r':
            clear_templates()
        else:
            recognize_gestures(args[0])
    except FileNotFoundError:
        print(f"Error: File not found - {args[1] if len(args) > 1 else args[0]}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Initialize global training set
training_set = []

if __name__ == "__main__":
    main(sys.argv[1:])