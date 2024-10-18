# 🖐️ $P Point-Cloud Gesture Recognizer
An enhanced implementation of the $P gesture recognizer algorithm for robust 2D gesture recognition.

## 🌟 Features
- **Shape Context Descriptors**: Capture global shape information for improved gesture discrimination
- **Dynamic Time Warping (DTW)**: Handle temporal variations in gesture execution
- **Multi-Template Support**: Store multiple versions of each gesture for natural variation handling
- **Scale and Position Invariant**: Recognize gestures regardless of size or location

## 🧠 Algorithm Overview
1. **Resampling**: Convert input to 64 equidistant points
   - Converts input gestures into a fixed number of points (64 in this implementation).
   - Ensures consistent representation regardless of input speed or sampling rate.
     
2. **Scaling & Translation**: Normalize to unit square and common origin
   - Scales gestures to a unit square.
   - Translates to a common origin (0,0).
   - Makes recognition scale and position invariant.
     
3. **Shape Context Computation**: Calculate rich shape descriptors
   - Computes a shape context descriptor for each point.
   - Captures the distribution of other points in log-polar space.
   - Provides a rich, discriminative representation of shape.
     
4. **DTW Matching**: Find optimal alignment between gesture and templates
   - Uses DTW to find optimal alignment between two shape context sequences.
   - Allows for non-linear temporal alignment, handling variations in gesture execution speed.
 
## 🚀 Quick Start
### Prerequisites
- Python 3.7+

### Installation
Install the packages mentioned in the file "requirements.txt"

### File Format:
#### gesture file format (examples provided):
GestureName

BEGIN

x,y <- List of points, a point per line

…

x,y

END



#### event stream file format (examples provided):

MOUSEDOWN

x,y <- List of points, a point per line

MOUSEUP

RECOGNIZE <- When you see this, you should output the result. (In UI Specifically, avoid in command line)

### Usage
1. Add a gesture template:
python .\pdollar.py -t <path_to_gesture_file>

2. Upload an event file to recognize a gesture:
python .\pdollar.py <path_to_event_file>

3. Clear all templates
python .\pdollar.py -r

#### Example Run:
> python .\pdollar.py -r                                                                                               
All templates cleared

> python .\pdollar.py -t "<path_to_gesture_file>"     
Added template: arrowhead (Version 1)
Added template: arrowhead (Version 2)

> python .\pdollar.py "<path_to_event_file>"
Recognized gesture: arrowhead

## 📊 Performance
The enhanced $P recognizer shows improved accuracy over the original algorithm, especially for:
- Similar-looking gestures (e.g., 'arrowhead' vs 'five_point_star' vs 'exclamation_mark')

## 🛠️ Extending the Recognizer (future aspects)
- **Rotation Invariance**: Implement circular shift matching of shape contexts
- **Multi-Stroke Support**: Extend to handle complex, multi-stroke gestures
- **User Adaptation**: Implement online learning for personalized recognition
- **GUI Interface**: Develop real-time input and recognition interface
- **Cross-Platform Support**: Extend beyond Windows OS

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments
- Original $P algorithm by [Vatavu et al.](http://depts.washington.edu/madlab/proj/dollar/pdollar.html)

## Contact
[Revathi Dhotre] - [LinkedIn](https://www.linkedin.com/in/revathi-dhotre/) - dhotrerevathi1@gmail.com
